#!/usr/bin/env python3
"""Aggregate Stage 3 results with full baseline comparison.

For each (model, dataset) cell, runs:
  - CAELUM       : lambda* via argmin-APSS
  - Vanilla CP   : lambda = 0 (aleatoric only)
  - Equal weight : lambda = 1
  - Epistemic-only: lambda = 1e6
  - Semantic Entropy : ranking with U_total (no prediction sets)
  - VNE (= U_total ranking AUROC; same as semantic entropy here since we use VNE)

With --write-markdown, also emits Markdown tables:
  - headline_with_baselines.md : APSS + Coverage per method
  - coverage_by_method.md      : Coverage per method
  - auroc_baselines.md         : AUROC per signal (U_total, U_epi, U_ale, R(x))

Plus a JSON dump of all rows for plotting.

Usage:
    python src/aggregate_with_baselines.py
"""

import argparse
import json
import re
import sys
import logging
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from caelum.paths import GENERATIONS_DIR, RESULTS_DIR  # noqa: E402

from caelum.calibration import CAELUMCalibrator
from caelum.baselines import ConUStyleCalibrator, SemanticEntropyCalibrator
from caelum.utils import split_data


SKIP_DATASETS = {"condambigqa"}
SKIP_NSAMPLE_RE = re.compile(r"_50\.json$")

# The accepted 18-cell benchmark uses these three LLMs. Larger-model caches
# (qwen3_14b, qwen3_32b) share results, so they are skipped unless --models
# names them explicitly.
ACCEPTED_MODELS = ("qwen3_8b", "ministral_3_8b_instruct_2512", "gemma_4_e4b_it")

MODEL_PRETTY = {
    "qwen3_8b": "Qwen3-8B",
    "ministral_3_8b_instruct_2512": "Ministral-3-8B",
    "gemma_4_e4b_it": "Gemma-4-E4B-it",
}
DATASET_PRETTY = {
    "triviaqa": "TriviaQA",
    "ambigqa": "AmbigQA",
    "opennq": "OpenNQ",
    "ambiginst": "AmbigInst",
    "simpleqa": "SimpleQA",
    "popqa": "PopQA",
}
MODEL_ORDER = ["qwen3_8b", "ministral_3_8b_instruct_2512", "gemma_4_e4b_it"]
DATASET_ORDER = ["triviaqa", "ambigqa", "opennq", "ambiginst", "simpleqa", "popqa"]


def manual_auroc(scores, labels):
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    ranks = rankdata(s)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


UNC_RE = re.compile(r"^unc_([a-z0-9]+)_(.+)_vne_(\d+)\.json$")

def parse_unc_filename(name):
    m = UNC_RE.match(name)
    if not m:
        return None
    return {"dataset": m.group(1), "model": m.group(2), "n_samples": int(m.group(3))}


def evaluate_one_calibrator(cal, val1, val2, test, label="caelum"):
    """Calibrate and evaluate one calibrator on the given splits."""
    cal.calibrate(val1, val2, verbose=False)
    test_covered = [cal.is_covered(it) for it in test]
    test_sizes = [cal.predict_set_size(it) for it in test]
    return {
        "label": label,
        "lambda_star": cal.optimal_lambda,
        "tau_hat": cal.tau_hat,
        "coverage": float(np.mean(test_covered)),
        "apss": float(np.mean(test_sizes)),
    }


def evaluate_row(unc_path, alpha=0.1, seed=42):
    with open(unc_path, encoding="utf-8") as f:
        items = json.load(f)
    n_total = len(items)
    n_matched = sum(1 for it in items if it.get("correct_cluster_idx", -1) >= 0)
    match_rate = n_matched / n_total if n_total else 0.0

    val1, val2, test = split_data(items, val1_frac=0.4, val2_frac=0.3, seed=seed)

    # Run the six calibrator variants
    methods = {}
    methods["CAELUM"] = evaluate_one_calibrator(
        CAELUMCalibrator(alpha=alpha, seed=seed), val1, val2, test, "CAELUM"
    )
    methods["Vanilla"] = evaluate_one_calibrator(
        CAELUMCalibrator.vanilla_conformal(alpha=alpha), val1, val2, test, "Vanilla"
    )
    methods["EqualWeight"] = evaluate_one_calibrator(
        CAELUMCalibrator.equal_weight(alpha=alpha), val1, val2, test, "EqualWeight"
    )
    methods["EpistemicOnly"] = evaluate_one_calibrator(
        CAELUMCalibrator.epistemic_only(alpha=alpha), val1, val2, test, "EpistemicOnly"
    )
    # External baselines
    methods["ConU"] = evaluate_one_calibrator(
        ConUStyleCalibrator(alpha=alpha, seed=seed), val1, val2, test, "ConU"
    )
    methods["SemanticEntropyCP"] = evaluate_one_calibrator(
        SemanticEntropyCalibrator(alpha=alpha, seed=seed), val1, val2, test, "SemanticEntropyCP"
    )

    # AUROCs (use CAELUM's R(x))
    cal_caelum = CAELUMCalibrator(alpha=alpha, seed=seed)
    cal_caelum.calibrate(val1, val2, verbose=False)
    incorrect = [int(it["correct_cluster_idx"] < 0) for it in test]
    u_total = [it.get("U_total", it["U_ale"] + it["U_epi"]) for it in test]
    u_epi = [it["U_epi"] for it in test]
    u_ale = [it["U_ale"] for it in test]
    r_vals = [cal_caelum.diagnostic_ratio(it) for it in test]

    aurocs = {
        "U_total": manual_auroc(u_total, incorrect),
        "U_epi": manual_auroc(u_epi, incorrect),
        "U_ale": manual_auroc(u_ale, incorrect),
        "R(x)": manual_auroc(r_vals, incorrect),
    }

    return {
        "n_samples": n_total,
        "n_test": len(test),
        "match_rate": match_rate,
        "methods": methods,
        "aurocs": aurocs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations_dir", default=GENERATIONS_DIR, type=Path,
                    help="Folder with the unc_* caches.")
    ap.add_argument("--output_dir", default=RESULTS_DIR, type=Path)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", nargs="+", default=list(ACCEPTED_MODELS),
                    help="model keys to include (default: the accepted 18-cell set)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing results file")
    ap.add_argument("--write-markdown", action="store_true",
                    help="Write an optional Markdown summary.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    log = logging.getLogger("aggregate")

    json_path = args.output_dir / "stage3_results.json"
    if json_path.exists() and not args.overwrite:
        raise SystemExit(f"{json_path} already exists and backs reported results; "
                         "pass --overwrite or write elsewhere.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = {}
    files = sorted(args.generations_dir.glob("unc_*.json"))
    for f in files:
        if SKIP_NSAMPLE_RE.search(f.name):
            continue
        info = parse_unc_filename(f.name)
        if (info is None or info["dataset"] in SKIP_DATASETS
                or info["model"] not in args.models):
            continue
        log.info("Evaluating: %s", f.name)
        try:
            metrics = evaluate_row(f, alpha=args.alpha, seed=args.seed)
        except Exception as e:
            log.warning("FAILED on %s: %s", f.name, e)
            continue
        rows[(info["model"], info["dataset"])] = {
            "model": info["model"], "dataset": info["dataset"], **metrics,
        }

    # ---- write JSON ----
    out = {f"{m}__{d}": v for (m, d), v in rows.items()}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info("Wrote: %s", json_path)

    # ---- markdown table 1: APSS / Coverage per method (6 methods) ----
    lines = [
        "# Stage 3 — Coverage & APSS by method (single seed=42)",
        "",
        "Six conformal-set methods, all calibrated on the same 40/30/30 splits at α=0.1:",
        "",
        "1. **CAELUM** — argmin-APSS λ\\* on Val₁ (our method)",
        "2. **Vanilla CP** (λ=0) — pure aleatoric weighting",
        "3. **Equal weight** (λ=1) — neutral",
        "4. **Epistemic-only** (λ=10⁶) — opposite extreme",
        "5. **ConU** — Wang et al., EMNLP 2024: nonconformity = 1 - max cluster prop (consistency)",
        "6. **Semantic Entropy CP** — Kuhn et al. 2023: APS with weight 1 + U_total",
        "",
    ]
    headers = ["Model", "Dataset", "N", "Match%",
               "Cov<sub>CAELUM</sub>", "APSS<sub>CAELUM</sub>",
               "APSS<sub>Vanilla</sub>", "APSS<sub>Equal</sub>", "APSS<sub>EpiOnly</sub>",
               "APSS<sub>ConU</sub>", "APSS<sub>SE-CP</sub>",
               "λ<sub>CAELUM</sub>"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for model in MODEL_ORDER:
        for dataset in DATASET_ORDER:
            if (model, dataset) not in rows:
                continue
            r = rows[(model, dataset)]
            m = r["methods"]
            lines.append(
                f"| {MODEL_PRETTY.get(model, model)} "
                f"| {DATASET_PRETTY.get(dataset, dataset)} "
                f"| {r['n_samples']} "
                f"| {100*r['match_rate']:.1f} "
                f"| {m['CAELUM']['coverage']:.3f} "
                f"| {m['CAELUM']['apss']:.2f} "
                f"| {m['Vanilla']['apss']:.2f} "
                f"| {m['EqualWeight']['apss']:.2f} "
                f"| {m['EpistemicOnly']['apss']:.2f} "
                f"| {m['ConU']['apss']:.2f} "
                f"| {m['SemanticEntropyCP']['apss']:.2f} "
                f"| {m['CAELUM']['lambda_star']:.3g} |"
            )
    md1 = args.output_dir / "headline_with_baselines.md"
    if args.write_markdown:
        md1.write_text("\n".join(lines) + "\n", encoding="utf-8")
        log.info("Wrote: %s", md1)

    # ---- markdown table 1b: Coverage by method ----
    lines = [
        "# Stage 3 — Coverage by method (single seed=42)",
        "",
        "All methods target 1−α = 0.9 marginal coverage. Coverage is bounded by the match rate.",
        "",
    ]
    headers = ["Model", "Dataset", "Match%",
               "Cov<sub>CAELUM</sub>", "Cov<sub>Vanilla</sub>", "Cov<sub>Equal</sub>",
               "Cov<sub>EpiOnly</sub>", "Cov<sub>ConU</sub>", "Cov<sub>SE-CP</sub>"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for model in MODEL_ORDER:
        for dataset in DATASET_ORDER:
            if (model, dataset) not in rows:
                continue
            r = rows[(model, dataset)]
            m = r["methods"]
            lines.append(
                f"| {MODEL_PRETTY.get(model, model)} "
                f"| {DATASET_PRETTY.get(dataset, dataset)} "
                f"| {100*r['match_rate']:.1f} "
                f"| {m['CAELUM']['coverage']:.3f} "
                f"| {m['Vanilla']['coverage']:.3f} "
                f"| {m['EqualWeight']['coverage']:.3f} "
                f"| {m['EpistemicOnly']['coverage']:.3f} "
                f"| {m['ConU']['coverage']:.3f} "
                f"| {m['SemanticEntropyCP']['coverage']:.3f} |"
            )
    md1b = args.output_dir / "coverage_by_method.md"
    if args.write_markdown:
        md1b.write_text("\n".join(lines) + "\n", encoding="utf-8")
        log.info("Wrote: %s", md1b)

    # ---- markdown table 2: AUROC per signal ----
    lines = ["# Stage 3 — Hallucination AUROC by uncertainty signal (single seed=42)\n"]
    lines.append(
        "| Model | Dataset | Match% | "
        "U_total | U_epi | U_ale | R(x) | "
        "Δ(U_epi - U_total) | Δ(R(x) - U_total) |"
    )
    lines.append("|" + "|".join(["---"] * 9) + "|")
    for model in MODEL_ORDER:
        for dataset in DATASET_ORDER:
            if (model, dataset) not in rows:
                continue
            r = rows[(model, dataset)]
            au = r["aurocs"]
            d_epi = au["U_epi"] - au["U_total"] if not (np.isnan(au["U_epi"]) or np.isnan(au["U_total"])) else float("nan")
            d_r = au["R(x)"] - au["U_total"] if not (np.isnan(au["R(x)"]) or np.isnan(au["U_total"])) else float("nan")
            lines.append(
                f"| {MODEL_PRETTY.get(model, model)} | "
                f"{DATASET_PRETTY.get(dataset, dataset)} | "
                f"{100*r['match_rate']:.1f} | "
                f"{au['U_total']:.3f} | {au['U_epi']:.3f} | {au['U_ale']:.3f} | {au['R(x)']:.3f} | "
                f"{d_epi:+.3f} | {d_r:+.3f} |"
            )
    md2 = args.output_dir / "auroc_baselines.md"
    if args.write_markdown:
        md2.write_text("\n".join(lines) + "\n", encoding="utf-8")
        log.info("Wrote: %s", md2)


if __name__ == "__main__":
    main()
