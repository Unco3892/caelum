#!/usr/bin/env python3
"""Does the A/E decomposition carry interpretable signal? Two validations.

The paper demotes lambda* and R(x) = log(lambda* U_epi + eps) - log(U_ale + eps)
to a "regime read-out" after the weighting proved inert (negative result #1).
This script tests whether the decomposition nevertheless has a *validated*
per-instance meaning, using only cached results (no GPU, no APIs):

(1) Ambiguity validation. AmbigQA (100 ambiguous / 100 unambiguous) and
    AmbigInst (184/120) carry per-item ``is_ambig`` labels. For every judged
    cell with label variation we compute the AUROC of U_ale (and of the
    aleatoric share U_ale/U_total, -R variants, U_epi, U_total) for
    separating ambiguous from unambiguous questions, plus a corpus-wide
    per-(model, dataset) mean-U_ale table.

(2) Escalation routing. Aligning each base model's judged cache with the
    Qwen3-14B / Qwen3-32B fallback caches by question id, we restrict to the
    base model's support failures (correct_cluster_idx < 0 -- the items the
    abstention trigger targets) and ask whether the base's decomposition
    predicts which of them the fallback's support rescues (fallback
    correct_cluster_idx >= 0). If aleatoric-dominated failures are rescued
    less often, R(x) is a *routing* diagnostic: clarify vs escalate/abstain.

Outputs ``results/decomp_interpret.json``; add ``--write-markdown`` for ``.md``. Read-only w.r.t. all
cached results.

Usage:
    python -u src/run_decomp_interpret.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import warnings as _warnings
from pathlib import Path

import numpy as np

_warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from caelum.paths import GENERATIONS_DIR, JUDGED_DIR, RESULTS_DIR  # noqa: E402

from caelum.abstention import attach_refusal_fractions  # noqa: E402
from caelum.evaluation import compute_auroc  # noqa: E402

EPS = 1e-6

MODEL_ORDER = ["qwen3_8b", "qwen3_14b", "qwen3_32b",
               "ministral_3_8b_instruct_2512", "gemma_4_e4b_it"]
DATASET_ORDER = ["triviaqa", "ambigqa", "opennq", "ambiginst", "simpleqa",
                 "popqa"]

FALLBACK_PAIRS = {
    "qwen3_14b": ["qwen3_8b", "ministral_3_8b_instruct_2512",
                  "gemma_4_e4b_it"],
    "qwen3_32b": ["qwen3_8b", "qwen3_14b", "ministral_3_8b_instruct_2512",
                  "gemma_4_e4b_it"],
}


def _setup_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S", force=True)
    return logging.getLogger("decomp_interpret")


def load_judged(dataset: str, model: str) -> list[dict] | None:
    """Load the judged unc cache for one cell (largest n if several)."""
    hits = sorted(JUDGED_DIR.glob(f"unc_{dataset}_{model}_vne_*.json"),
                  key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
    if not hits:
        return None
    with open(hits[-1], "r", encoding="utf-8") as f:
        data = json.load(f)
    gen_name = hits[-1].name.replace("unc_", "gen_").replace("_vne", "")
    attach_refusal_fractions(data, GENERATIONS_DIR / gen_name)
    return data


def load_lambda_stars() -> dict[str, float]:
    """Per-cell mean lambda* from the headline multiseed record."""
    path = RESULTS_DIR / "headline_multiseed.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        rec = json.load(f)
    return {
        key: float(cell["methods"]["CAELUM"]["lam_mean"])
        for key, cell in rec["cells"].items()
    }


def item_key(item: dict) -> str:
    qid = item.get("question_id", item.get("datapoint_id"))
    return str(qid)


def signals_for(items: list[dict], lam_star: float) -> dict[str, np.ndarray]:
    """Per-item decomposition signals, oriented so higher = more ambiguous
    (part 1) / for part 2 the orientation is reported per signal."""
    u_ale = np.array([float(it["U_ale"]) for it in items])
    u_epi = np.array([float(it["U_epi"]) for it in items])
    u_tot = np.array([float(it["U_total"]) for it in items])
    r_free = np.log(u_epi + EPS) - np.log(u_ale + EPS)
    r_star = np.log(lam_star * u_epi + EPS) - np.log(u_ale + EPS)
    return {
        "U_ale": u_ale,
        "U_epi": u_epi,
        "U_total": u_tot,
        "ale_share": u_ale / np.maximum(u_tot, EPS),
        "neg_R_free": -r_free,
        "neg_R_lamstar": -r_star,
        "refusal_frac": np.array(
            [float(it.get("refusal_frac", 0.0)) for it in items]),
    }


# ---------------------------------------------------------------------------
# (1) Ambiguity validation
# ---------------------------------------------------------------------------

def ambiguity_block(lam_stars: dict[str, float], logger) -> dict:
    out: dict[str, dict] = {"per_cell": {}, "mean_table": {}}
    for model in MODEL_ORDER:
        for dataset in DATASET_ORDER:
            items = load_judged(dataset, model)
            if items is None:
                continue
            key = f"{model}__{dataset}"
            lam = lam_stars.get(key, 0.0)
            sig = signals_for(items, lam)
            y = np.array([1 if it.get("is_ambig") else 0 for it in items])
            out["mean_table"][key] = {
                "n": len(items),
                "mean_U_ale": float(sig["U_ale"].mean()),
                "mean_U_epi": float(sig["U_epi"].mean()),
                "mean_ale_share": float(sig["ale_share"].mean()),
                "frac_ambig": float(y.mean()),
            }
            if len(np.unique(y)) < 2:
                continue
            aurocs = {name: float(compute_auroc(y, s))
                      for name, s in sig.items()}
            # mean U_ale by label
            m1 = float(sig["U_ale"][y == 1].mean())
            m0 = float(sig["U_ale"][y == 0].mean())
            out["per_cell"][key] = {
                "n": len(items), "frac_ambig": float(y.mean()),
                "lam_star": lam, "auroc": aurocs,
                "mean_U_ale_ambig": m1, "mean_U_ale_unambig": m0,
            }
            logger.info(
                "  ambiguity %s: AUROC U_ale=%.3f share=%.3f -R=%.3f "
                "U_epi=%.3f (n=%d, %.0f%% ambig)",
                key, aurocs["U_ale"], aurocs["ale_share"],
                aurocs["neg_R_free"], aurocs["U_epi"], len(items),
                100 * y.mean(),
            )
    # dataset-level aggregates of per-cell AUROCs
    agg: dict[str, dict[str, float]] = {}
    for dataset in ("ambigqa", "ambiginst"):
        cells = {k: v for k, v in out["per_cell"].items()
                 if k.endswith("__" + dataset)}
        if not cells:
            continue
        agg[dataset] = {
            name: float(np.mean([c["auroc"][name] for c in cells.values()]))
            for name in next(iter(cells.values()))["auroc"]
        }
        agg[dataset]["n_cells"] = len(cells)
    out["dataset_mean_auroc"] = agg
    return out


# ---------------------------------------------------------------------------
# (2) Escalation routing
# ---------------------------------------------------------------------------

def routing_block(lam_stars: dict[str, float], logger) -> dict:
    out: dict[str, dict] = {}
    for fb_model, bases in FALLBACK_PAIRS.items():
        for base_model in bases:
            for dataset in DATASET_ORDER:
                base_items = load_judged(dataset, base_model)
                fb_items = load_judged(dataset, fb_model)
                if base_items is None or fb_items is None:
                    continue
                fb_by_qid = {item_key(it): it for it in fb_items}
                pairs = [(b, fb_by_qid[item_key(b)]) for b in base_items
                         if item_key(b) in fb_by_qid]
                # stratum: base support failures (what abstention flags)
                fails = [(b, f) for b, f in pairs
                         if int(b.get("correct_cluster_idx", -1)) < 0]
                if len(fails) < 10:
                    continue
                y = np.array([
                    1 if int(f.get("correct_cluster_idx", -1)) >= 0 else 0
                    for _, f in fails
                ])
                if len(np.unique(y)) < 2:
                    continue
                key = f"{base_model}->{fb_model}__{dataset}"
                lam = lam_stars.get(f"{base_model}__{dataset}", 0.0)
                sig = signals_for([b for b, _ in fails], lam)
                # orientation: higher signal value -> predicted RESCUED.
                # For R variants the routing hypothesis is "epistemic-
                # dominated failures are rescuable", i.e. +R (not -R).
                aurocs = {
                    "R_free": float(compute_auroc(y, -sig["neg_R_free"])),
                    "R_lamstar": float(compute_auroc(y, -sig["neg_R_lamstar"])),
                    "U_ale": float(compute_auroc(y, sig["U_ale"])),
                    "U_epi": float(compute_auroc(y, sig["U_epi"])),
                    "U_total": float(compute_auroc(y, sig["U_total"])),
                    "ale_share": float(compute_auroc(y, sig["ale_share"])),
                    "refusal_frac": float(
                        compute_auroc(y, sig["refusal_frac"])),
                }
                out[key] = {
                    "n_failures": len(fails), "rescue_rate": float(y.mean()),
                    "auroc_for_rescued": aurocs,
                }
                logger.info(
                    "  routing %s: n_fail=%d rescue=%.2f | R_free=%.3f "
                    "ale_share=%.3f U_ale=%.3f refusal=%.3f",
                    key, len(fails), y.mean(), aurocs["R_free"],
                    aurocs["ale_share"], aurocs["U_ale"],
                    aurocs["refusal_frac"],
                )
    # aggregates per fallback
    agg: dict[str, dict[str, float]] = {}
    for fb_model in FALLBACK_PAIRS:
        cells = {k: v for k, v in out.items() if f"->{fb_model}__" in k}
        if not cells:
            continue
        names = next(iter(cells.values()))["auroc_for_rescued"].keys()
        agg[fb_model] = {
            name: float(np.mean(
                [c["auroc_for_rescued"][name] for c in cells.values()]))
            for name in names
        }
        agg[fb_model]["n_pairs"] = len(cells)
        agg[fb_model]["mean_rescue_rate"] = float(
            np.mean([c["rescue_rate"] for c in cells.values()]))
    return {"per_pair": out, "aggregate": agg}


# ---------------------------------------------------------------------------
# Ambiguity -> rescue cross-check (is_ambig vs rescued, labeled cells only)
# ---------------------------------------------------------------------------

def ambig_rescue_crosscheck(logger) -> dict:
    out: dict[str, dict] = {}
    for fb_model, bases in FALLBACK_PAIRS.items():
        for base_model in bases:
            for dataset in ("ambigqa", "ambiginst"):
                base_items = load_judged(dataset, base_model)
                fb_items = load_judged(dataset, fb_model)
                if base_items is None or fb_items is None:
                    continue
                fb_by_qid = {item_key(it): it for it in fb_items}
                fails = [
                    (b, fb_by_qid[item_key(b)]) for b in base_items
                    if item_key(b) in fb_by_qid
                    and int(b.get("correct_cluster_idx", -1)) < 0
                ]
                if len(fails) < 10:
                    continue
                resc = np.array([
                    1 if int(f.get("correct_cluster_idx", -1)) >= 0 else 0
                    for _, f in fails
                ])
                amb = np.array(
                    [1 if b.get("is_ambig") else 0 for b, _ in fails])
                if len(np.unique(amb)) < 2:
                    continue
                key = f"{base_model}->{fb_model}__{dataset}"
                out[key] = {
                    "n_failures": len(fails),
                    "rescue_rate_ambig": float(resc[amb == 1].mean()),
                    "rescue_rate_unambig": float(resc[amb == 0].mean()),
                    "n_ambig": int(amb.sum()),
                    "n_unambig": int((1 - amb).sum()),
                }
                logger.info(
                    "  crosscheck %s: rescue ambig=%.2f (n=%d) vs "
                    "unambig=%.2f (n=%d)", key,
                    out[key]["rescue_rate_ambig"], out[key]["n_ambig"],
                    out[key]["rescue_rate_unambig"], out[key]["n_unambig"],
                )
    return out


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-markdown", action="store_true",
                        help="Also write the Markdown report.")
    args = parser.parse_args()

    logger = _setup_logging()
    t0 = time.time()
    lam_stars = load_lambda_stars()
    logger.info("decomp interpret: lambda* loaded for %d cells",
                len(lam_stars))

    ambiguity = ambiguity_block(lam_stars, logger)
    routing = routing_block(lam_stars, logger)
    crosscheck = ambig_rescue_crosscheck(logger)

    result = {
        "ambiguity": ambiguity,
        "routing": routing,
        "ambig_rescue_crosscheck": crosscheck,
    }
    with open(RESULTS_DIR / "decomp_interpret.json", "w",
              encoding="utf-8") as f:
        json.dump(result, f, indent=1, default=float)

    # ---- markdown ----
    lines = ["# Decomposition interpretability validation", ""]
    lines += ["## (1) U_ale vs per-item ambiguity labels "
              "(AmbigQA 100/100, AmbigInst 184/120)", "",
              "| Cell | n | AUROC U_ale | ale_share | -R_free | -R_lam* | "
              "U_epi | U_total | refusal |", "|---|---|---|---|---|---|---|---|---|"]
    for key, c in ambiguity["per_cell"].items():
        a = c["auroc"]
        lines.append(
            f"| {key} | {c['n']} | {a['U_ale']:.3f} | {a['ale_share']:.3f} "
            f"| {a['neg_R_free']:.3f} | {a['neg_R_lamstar']:.3f} "
            f"| {a['U_epi']:.3f} | {a['U_total']:.3f} "
            f"| {a['refusal_frac']:.3f} |")
    lines += ["", "### Dataset means of per-cell AUROC", "",
              "```json",
              json.dumps(ambiguity["dataset_mean_auroc"], indent=1), "```"]

    lines += ["", "## Corpus mean U_ale / U_epi per cell", "",
              "| Cell | mean U_ale | mean U_epi | ale share | frac ambig |",
              "|---|---|---|---|---|"]
    for key, m in ambiguity["mean_table"].items():
        lines.append(
            f"| {key} | {m['mean_U_ale']:.3f} | {m['mean_U_epi']:.3f} "
            f"| {m['mean_ale_share']:.3f} | {m['frac_ambig']:.2f} |")

    lines += ["", "## (2) Routing: among base support failures, does the "
              "decomposition predict fallback rescue?", "",
              "| Pair | n_fail | rescue rate | R_free | R_lam* | U_ale | "
              "U_epi | ale_share | refusal |",
              "|---|---|---|---|---|---|---|---|---|"]
    for key, c in routing["per_pair"].items():
        a = c["auroc_for_rescued"]
        lines.append(
            f"| {key} | {c['n_failures']} | {c['rescue_rate']:.2f} "
            f"| {a['R_free']:.3f} | {a['R_lamstar']:.3f} | {a['U_ale']:.3f} "
            f"| {a['U_epi']:.3f} | {a['ale_share']:.3f} "
            f"| {a['refusal_frac']:.3f} |")
    lines += ["", "### Aggregate per fallback", "", "```json",
              json.dumps(routing["aggregate"], indent=1), "```"]

    lines += ["", "## (3) Cross-check: rescue rate by is_ambig", "",
              "| Pair | n_fail | rescue (ambig) | rescue (unambig) |",
              "|---|---|---|---|"]
    for key, c in crosscheck.items():
        lines.append(
            f"| {key} | {c['n_failures']} "
            f"| {c['rescue_rate_ambig']:.2f} (n={c['n_ambig']}) "
            f"| {c['rescue_rate_unambig']:.2f} (n={c['n_unambig']}) |")

    if args.write_markdown:
        with open(RESULTS_DIR / "decomp_interpret.md", "w",
                  encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    logger.info("done in %.1f min", (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
