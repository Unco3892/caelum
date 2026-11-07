#!/usr/bin/env python3
"""Multi-seed judged re-evaluation runner (headline + conditional + abstention).

Re-evaluates all 18 (model, dataset) cells at nominal alpha=0.1 across seeds
42-45 with randomized APS, consuming LLM-judged correctness caches when
available and falling back per-cell to the heuristic ``unc_*_vne_*.json`` files
otherwise. It produces three families of artefacts:

(a) Headline coverage / APSS / lambda* per cell x method under BOTH full-V2
    and support-conditional calibration (Prop. 1). Methods: CAELUM
    (standardized weight, set_score selection), Vanilla CP, TrueAPS, SE-CP,
    Equal-weight, Epistemic-only.

(b) Conditional coverage in U_epi and U_ale quartile bins for CAELUM vs
    Vanilla vs TrueAPS -- the adaptivity evidence. The headline number is the
    per-method mean coverage gap between the bottom and top uncertainty
    quartile (does CAELUM flatten conditional coverage?).

(c) Abstention-augmented sets: a pre-generation logistic matchability
    classifier (trained on a V0 split) triggers an explicit ABSTAIN element;
    coverage is counted over the augmented output space. Reports augmented
    coverage, abstain rate, and abstain precision/recall against true
    unmatchability.

Outputs (all under ``results`` / ``results/plots``; Markdown copies only with
``--write-markdown``):
    - judged_rerun_headline.json
    - judged_rerun_conditional_coverage.json + plot
    - judged_rerun_abstention.json

Usage::

    # Smoke test on a few cells against the heuristic caches
    python -u src/run_judged_rerun.py \
        --unc_dir results/generations \
        --cells qwen3_8b__triviaqa ministral_3_8b_instruct_2512__ambigqa

    # Full 18-cell x 4-seed run against the heuristic caches
    python -u src/run_judged_rerun.py \
        --unc_dir results/generations
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import warnings as _warnings
from pathlib import Path

import numpy as np

_warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from caelum.paths import GENERATIONS_DIR, JUDGED_DIR, PLOTS_DIR, RESULTS_DIR  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from caelum.abstention import (  # noqa: E402
    augmented_metrics,
    calibrate_abstain_threshold,
    train_matchability_classifier,
)
from caelum.baselines import SemanticEntropyCalibrator  # noqa: E402
from caelum.calibration import CAELUMCalibrator  # noqa: E402
from caelum.utils import SentenceEncoder, split_data  # noqa: E402

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_UNC_DIR = JUDGED_DIR
LOG_PATH = RESULTS_DIR / "judged_rerun.txt"

ALPHA = 0.1
SEEDS = [42, 43, 44, 45]
N_BINS = 4

# Headline methods. Each is evaluated under support_conditional False and True.
HEADLINE_METHODS = ["CAELUM", "Vanilla", "TrueAPS", "SE-CP", "Equal", "EpiOnly"]
# Methods used in the conditional-coverage adaptivity analysis.
CONDITIONAL_METHODS = ["CAELUM", "Vanilla", "TrueAPS"]

UNC_RE = re.compile(r"unc_(.+?)_(.+?)_vne_(\d+)\.json")
SKIP_DATASETS = {"condambigqa"}
MODEL_PRETTY = {
    "qwen3_8b": "Qwen3-8B",
    "ministral_3_8b_instruct_2512": "Ministral-3-8B",
    "gemma_4_e4b_it": "Gemma-4-E4B-it",
    "qwen3_14b": "Qwen3-14B",
    "qwen3_32b": "Qwen3-32B",
    "qwen3_235b_a22b_instruct_2507": "Qwen3-235B-A22B",
    "gpt_oss_120b": "GPT-OSS-120B",
}
DATASET_PRETTY = {
    "triviaqa": "TriviaQA", "ambigqa": "AmbigQA", "opennq": "OpenNQ",
    "ambiginst": "AmbigInst", "simpleqa": "SimpleQA", "popqa": "PopQA",
}
MODEL_ORDER = ["qwen3_8b", "ministral_3_8b_instruct_2512", "gemma_4_e4b_it",
               "qwen3_14b", "qwen3_32b", "qwen3_235b_a22b_instruct_2507",
               "gpt_oss_120b"]
DATASET_ORDER = ["triviaqa", "ambigqa", "opennq", "ambiginst", "simpleqa", "popqa"]


# ---------------------------------------------------------------------------
# Logging (Tee to terminal + file, matches run_llm_judge_rematch.py)
# ---------------------------------------------------------------------------

class Tee:
    """Redirect writes to both terminal and a log file."""

    def __init__(self, filepath: str | Path) -> None:
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass
        self.terminal = sys.stdout
        self.log = open(str(filepath), "w", encoding="utf-8", buffering=1)

    def write(self, msg: str) -> None:
        self.terminal.write(msg)
        try:
            self.terminal.flush()
        except Exception:
            pass
        self.log.write(msg)
        self.log.flush()

    def flush(self) -> None:
        try:
            self.terminal.flush()
        except Exception:
            pass
        self.log.flush()


def _setup_logging(log_path: Path = LOG_PATH) -> logging.Logger:
    sys.stdout = Tee(log_path)
    sys.stderr = sys.stdout
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S", stream=sys.stdout, force=True,
    )
    for noisy in (
        "httpx", "httpcore", "sentence_transformers", "transformers",
        "huggingface_hub", "filelock", "urllib3", "fsspec", "matplotlib",
        "PIL", "torch", "accelerate", "safetensors",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("judged_rerun")


# ---------------------------------------------------------------------------
# Data loading + cell discovery
# ---------------------------------------------------------------------------

def _ensure_types(data: list[dict]) -> list[dict]:
    """Coerce JSON-loaded fields to the numeric types the calibrator expects."""
    for item in data:
        if "cluster_props" in item:
            item["cluster_props"] = np.asarray(item["cluster_props"], dtype=float)
        for k in ("U_ale", "U_epi", "U_total"):
            if k in item:
                item[k] = float(item[k])
        if "correct_cluster_idx" in item:
            item["correct_cluster_idx"] = int(item["correct_cluster_idx"])
    return data


def discover_cells(base_dir: Path) -> list[dict]:
    """Enumerate the (model, dataset) cells from the heuristic cache directory.

    Files follow ``unc_{dataset}_{model_tag}_vne_{n}.json``; the largest sample
    count is kept per (model, dataset). condambigqa and unknown model/dataset
    tags are skipped.
    """
    cells = []
    for f in sorted(base_dir.glob("unc_*_vne_*.json")):
        m = UNC_RE.match(f.name)
        if not m:
            continue
        ds, model, n = m.group(1), m.group(2), int(m.group(3))
        if ds in SKIP_DATASETS or model not in MODEL_PRETTY or ds not in DATASET_PRETTY:
            continue
        cells.append({
            "model": model, "dataset": ds, "n_samples": n,
            "filename": f.name, "base_path": str(f),
            "cell_key": f"{model}__{ds}",
        })
    # Dedupe (keep largest n) then order.
    best: dict[str, dict] = {}
    for c in cells:
        k = c["cell_key"]
        if k not in best or c["n_samples"] > best[k]["n_samples"]:
            best[k] = c
    ordered = sorted(best.values(), key=lambda c: (
        MODEL_ORDER.index(c["model"]), DATASET_ORDER.index(c["dataset"]),
    ))
    return ordered


def resolve_source(cell: dict, unc_dir: Path, base_dir: Path) -> tuple[str, str]:
    """Return (path, source_tag) preferring a judged file, else the heuristic.

    Args:
        cell: A discovered cell dict.
        unc_dir: Directory searched first (judged caches).
        base_dir: Heuristic fallback directory.

    Returns:
        (path, source) where source is "judged" or "heuristic".
    """
    judged = unc_dir / cell["filename"]
    if judged.exists() and judged.resolve() != Path(cell["base_path"]).resolve():
        return str(judged), "judged"
    return cell["base_path"], "heuristic"


# ---------------------------------------------------------------------------
# Calibrator factory
# ---------------------------------------------------------------------------

def make_calibrator(method: str, alpha: float, seed: int, support_conditional: bool):
    """Construct a calibrator for ``method`` with the given support mode.

    All CAELUM-family calibrators use randomized APS. CAELUM uses the
    standardized weight with the set_score selection objective; the baselines
    use single-lambda grids per their definitions.
    """
    if method == "CAELUM":
        return CAELUMCalibrator(
            alpha=alpha, seed=seed, randomize=True,
            weight_fn="standardized", selection_method="set_score",
            support_conditional=support_conditional,
        )
    if method == "Vanilla":
        return CAELUMCalibrator(
            alpha=alpha, seed=seed, randomize=True,
            lambda_grid=np.array([0.0]), weight_fn="additive",
            support_conditional=support_conditional,
        )
    if method == "TrueAPS":
        return CAELUMCalibrator(
            alpha=alpha, seed=seed, randomize=True,
            lambda_grid=np.array([0.0]), weight_fn="unit",
            support_conditional=support_conditional,
        )
    if method == "Equal":
        return CAELUMCalibrator(
            alpha=alpha, seed=seed, randomize=True,
            lambda_grid=np.array([1.0]), weight_fn="additive",
            support_conditional=support_conditional,
        )
    if method == "EpiOnly":
        return CAELUMCalibrator(
            alpha=alpha, seed=seed, randomize=True,
            lambda_grid=np.array([1e6]), weight_fn="additive",
            support_conditional=support_conditional,
        )
    if method == "SE-CP":
        return SemanticEntropyCalibrator(
            alpha=alpha, seed=seed, support_conditional=support_conditional,
        )
    raise ValueError(f"Unknown method: {method!r}")


def _matched_coverage(items: list[dict], cal) -> float:
    """Coverage restricted to matched items (correct_cluster_idx >= 0)."""
    matched = [it for it in items if int(it["correct_cluster_idx"]) >= 0]
    if not matched:
        return float("nan")
    return float(np.mean([cal.is_covered(it) for it in matched]))


# ---------------------------------------------------------------------------
# (a) Headline evaluation
# ---------------------------------------------------------------------------

def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.array([v for v in values if v is not None and np.isfinite(v)],
                   dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std())


def evaluate_headline_cell(
    data: list[dict], alpha: float, seeds: list[int],
) -> dict:
    """Per-method headline metrics (mean+/-std over seeds), both support modes."""
    match_rate = float(np.mean([
        int(d["correct_cluster_idx"]) >= 0 for d in data
    ]))

    # method -> support_mode -> metric -> list over seeds
    acc: dict[str, dict[str, dict[str, list[float]]]] = {
        m: {"full": {"cov_full": [], "cov_matched": [], "apss": [], "lam": []},
            "support": {"cov_full": [], "cov_matched": [], "apss": [], "lam": []}}
        for m in HEADLINE_METHODS
    }

    for seed in seeds:
        val1, val2, test = split_data(data, val1_frac=0.4, val2_frac=0.3, seed=seed)
        for m in HEADLINE_METHODS:
            for mode_key, sc in (("full", False), ("support", True)):
                cal = make_calibrator(m, alpha, seed, support_conditional=sc)
                cal.calibrate(val1, val2, verbose=False)
                sizes = [cal.predict_set_size(it) for it in test]
                cov_full = float(np.mean([cal.is_covered(it) for it in test]))
                cov_matched = _matched_coverage(test, cal)
                acc[m][mode_key]["cov_full"].append(cov_full)
                acc[m][mode_key]["cov_matched"].append(cov_matched)
                acc[m][mode_key]["apss"].append(float(np.mean(sizes)))
                acc[m][mode_key]["lam"].append(
                    float(getattr(cal, "optimal_lambda", 0.0))
                )

    methods: dict[str, dict] = {}
    for m in HEADLINE_METHODS:
        methods[m] = {}
        for mode_key in ("full", "support"):
            d = acc[m][mode_key]
            cf_m, cf_s = _mean_std(d["cov_full"])
            cm_m, cm_s = _mean_std(d["cov_matched"])
            ap_m, ap_s = _mean_std(d["apss"])
            lm_m, lm_s = _mean_std(d["lam"])
            methods[m][mode_key] = {
                "cov_full_mean": cf_m, "cov_full_std": cf_s,
                "cov_matched_mean": cm_m, "cov_matched_std": cm_s,
                "apss_mean": ap_m, "apss_std": ap_s,
                "lam_mean": lm_m, "lam_std": lm_s,
            }
    return {"match_rate": match_rate, "methods": methods}


# ---------------------------------------------------------------------------
# (b) Conditional coverage by uncertainty quartile
# ---------------------------------------------------------------------------

def evaluate_conditional_cell(
    data: list[dict], alpha: float, seeds: list[int],
) -> dict:
    """Per-method coverage in U_epi / U_ale quartile bins (mean over seeds).

    Uses full-V2 (marginal) calibration -- the adaptivity question is whether
    the *marginal* calibrator's coverage stays flat across uncertainty bins.
    """
    out: dict[str, dict] = {}
    for by in ("U_epi", "U_ale"):
        per_method: dict[str, dict] = {}
        for m in CONDITIONAL_METHODS:
            cov_bins: list[np.ndarray] = []
            apss_bins: list[np.ndarray] = []
            for seed in seeds:
                val1, val2, test = split_data(
                    data, val1_frac=0.4, val2_frac=0.3, seed=seed,
                )
                cal = make_calibrator(m, alpha, seed, support_conditional=False)
                cal.calibrate(val1, val2, verbose=False)
                strat = cal.stratified_coverage(test, n_bins=N_BINS, by=by)
                cov_bins.append(np.asarray(strat["coverage"], dtype=float))
                apss_bins.append(np.asarray(strat["apss"], dtype=float))
            cov_mean = np.nanmean(np.vstack(cov_bins), axis=0)
            apss_mean = np.nanmean(np.vstack(apss_bins), axis=0)
            # Gap = bottom-quartile coverage minus top-quartile coverage. A
            # large positive gap means coverage collapses in the high-U bin.
            gaps = [float(cb[0] - cb[-1]) for cb in cov_bins]
            gap_mean, gap_std = _mean_std(gaps)
            per_method[m] = {
                "cov_bins_mean": cov_mean.tolist(),
                "apss_bins_mean": apss_mean.tolist(),
                "gap_mean": gap_mean, "gap_std": gap_std,
            }
        out[by] = per_method
    return out


# ---------------------------------------------------------------------------
# (c) Abstention-augmented evaluation
# ---------------------------------------------------------------------------

def evaluate_abstention_cell(
    data: list[dict], alpha: float, seeds: list[int], encoder: SentenceEncoder,
) -> dict:
    """Abstention-augmented metrics (mean+/-std over seeds).

    Uses a 25/25/25/25 split: V0 trains the matchability classifier, V1 selects
    lambda, V2 sets the conformal threshold and the abstain threshold, Test is
    evaluated. The base semantic calibrator is CAELUM (standardized weight,
    set_score selection, support-conditional).
    """
    acc: dict[str, list[float]] = {
        k: [] for k in (
            "coverage_aug", "coverage_semantic", "apss_semantic", "apss_aug",
            "abstain_rate", "abstain_precision", "abstain_recall", "threshold",
        )
    }
    for seed in seeds:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(data))
        q = len(data) // 4
        v0 = [data[i] for i in perm[:q]]
        v1 = [data[i] for i in perm[q:2 * q]]
        v2 = [data[i] for i in perm[2 * q:3 * q]]
        test = [data[i] for i in perm[3 * q:]]

        clf = train_matchability_classifier(v0, encoder=encoder, seed=seed)
        cal = CAELUMCalibrator(
            alpha=alpha, seed=seed, randomize=True, weight_fn="standardized",
            selection_method="set_score", support_conditional=True,
        )
        cal.calibrate(v1, v2, verbose=False)
        thr = calibrate_abstain_threshold(v2, cal, clf, alpha, encoder=encoder)
        m = augmented_metrics(test, cal, clf, thr, encoder=encoder)
        for k in acc:
            if k == "threshold":
                acc[k].append(float(thr))
            else:
                acc[k].append(m[k])

    summary: dict[str, float] = {}
    for k, vals in acc.items():
        mu, sd = _mean_std(vals)
        summary[f"{k}_mean"] = mu
        summary[f"{k}_std"] = sd
    return summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_conditional_coverage(agg: dict, alpha: float, out_path: Path) -> None:
    """Plot aggregate coverage vs uncertainty quartile for the 3 methods."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    colors = {"CAELUM": "#1f77b4", "Vanilla": "#ff7f0e", "TrueAPS": "#2ca02c"}
    markers = {"CAELUM": "o", "Vanilla": "s", "TrueAPS": "^"}
    for ax, by in zip(axes, ("U_epi", "U_ale")):
        bins = np.arange(1, N_BINS + 1)
        for m in CONDITIONAL_METHODS:
            cov = agg[by][m]["cov_bins_mean"]
            ax.plot(bins, cov, marker=markers[m], color=colors[m], label=m)
        ax.axhline(1.0 - alpha, ls="--", color="gray", lw=1,
                   label=f"target {1 - alpha:.2f}")
        ax.set_xticks(bins)
        ax.set_xlabel(f"{by} quartile (1=low, {N_BINS}=high)")
        ax.set_title(f"Coverage by {by} bin")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("empirical coverage")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Conditional coverage adaptivity (aggregate over cells x seeds)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown writers
# ---------------------------------------------------------------------------

def _fmt(v: float, nd: int = 3) -> str:
    return "nan" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def write_headline_md(cells: dict, alpha: float, seeds: list[int], out: Path) -> None:
    lines = [
        f"# Judged re-evaluation: headline (alpha={alpha}, seeds {seeds})",
        "",
        "Mean +/- std over seeds. Coverage reported for the full population and",
        "the matched stratum; both full-V2 and support-conditional calibration.",
        "",
        "## APSS (mean +/- std), full-V2 calibration",
        "",
        "| Model | Dataset | Match | " + " | ".join(HEADLINE_METHODS) + " |",
        "|---|---|---|" + "---|" * len(HEADLINE_METHODS),
    ]
    for ck, r in cells.items():
        ms = r["methods"]
        row = (f"| {MODEL_PRETTY[r['model']]} | {DATASET_PRETTY[r['dataset']]} | "
               f"{_fmt(r['match_rate'], 2)} | ")
        row += " | ".join(
            f"{_fmt(ms[m]['full']['apss_mean'], 2)}±{_fmt(ms[m]['full']['apss_std'], 2)}"
            for m in HEADLINE_METHODS
        )
        lines.append(row + " |")

    lines += [
        "",
        "## Full-population coverage (mean +/- std), full-V2 calibration",
        "",
        "| Model | Dataset | " + " | ".join(HEADLINE_METHODS) + " |",
        "|---|---|" + "---|" * len(HEADLINE_METHODS),
    ]
    for ck, r in cells.items():
        ms = r["methods"]
        row = f"| {MODEL_PRETTY[r['model']]} | {DATASET_PRETTY[r['dataset']]} | "
        row += " | ".join(
            f"{_fmt(ms[m]['full']['cov_full_mean'])}" for m in HEADLINE_METHODS
        )
        lines.append(row + " |")

    lines += [
        "",
        "## Support-conditional matched-stratum coverage (mean +/- std)",
        "",
        "Target is 1-alpha on the matched stratum (Prop. 1).",
        "",
        "| Model | Dataset | " + " | ".join(HEADLINE_METHODS) + " |",
        "|---|---|" + "---|" * len(HEADLINE_METHODS),
    ]
    for ck, r in cells.items():
        ms = r["methods"]
        row = f"| {MODEL_PRETTY[r['model']]} | {DATASET_PRETTY[r['dataset']]} | "
        row += " | ".join(
            f"{_fmt(ms[m]['support']['cov_matched_mean'])}"
            for m in HEADLINE_METHODS
        )
        lines.append(row + " |")

    lines += [
        "",
        "## CAELUM lambda* (mean +/- std, full-V2)",
        "",
        "| Model | Dataset | lambda* |",
        "|---|---|---|",
    ]
    for ck, r in cells.items():
        c = r["methods"]["CAELUM"]["full"]
        lines.append(
            f"| {MODEL_PRETTY[r['model']]} | {DATASET_PRETTY[r['dataset']]} | "
            f"{_fmt(c['lam_mean'], 2)}±{_fmt(c['lam_std'], 2)} |"
        )
    out.write_text("\n".join(lines), encoding="utf-8")


def write_conditional_md(cells: dict, agg: dict, out: Path) -> None:
    lines = [
        "# Judged re-evaluation: conditional coverage (adaptivity)",
        "",
        "Coverage in equal-count uncertainty quartiles (1=low, 4=high),",
        "full-V2 calibration, aggregated across cells x seeds.",
        "Gap = coverage(bottom quartile) - coverage(top quartile); a smaller",
        "absolute gap means flatter, more adaptive conditional coverage.",
        "",
        "## Aggregate coverage by bin",
        "",
    ]
    for by in ("U_epi", "U_ale"):
        lines += [
            f"### By {by}",
            "",
            "| Method | bin1 | bin2 | bin3 | bin4 | gap (b1-b4) |",
            "|---|---|---|---|---|---|",
        ]
        for m in CONDITIONAL_METHODS:
            d = agg[by][m]
            cov = d["cov_bins_mean"]
            lines.append(
                f"| {m} | " + " | ".join(_fmt(c) for c in cov) +
                f" | {_fmt(d['gap_mean'])}±{_fmt(d['gap_std'])} |"
            )
        lines.append("")

    lines += [
        "## Per-cell U_epi gap (b1-b4)",
        "",
        "| Model | Dataset | " + " | ".join(CONDITIONAL_METHODS) + " |",
        "|---|---|" + "---|" * len(CONDITIONAL_METHODS),
    ]
    for ck, r in cells.items():
        row = f"| {MODEL_PRETTY[r['model']]} | {DATASET_PRETTY[r['dataset']]} | "
        row += " | ".join(
            _fmt(r["conditional"]["U_epi"][m]["gap_mean"])
            for m in CONDITIONAL_METHODS
        )
        lines.append(row + " |")
    out.write_text("\n".join(lines), encoding="utf-8")


def write_abstention_md(cells: dict, agg: dict, alpha: float, out: Path) -> None:
    lines = [
        f"# Judged re-evaluation: abstention-augmented sets (alpha={alpha})",
        "",
        "25/25/25/25 split (V0 classifier / V1 lambda / V2 threshold / Test).",
        "Base calibrator: CAELUM (standardized, set_score, support-conditional).",
        "Coverage_aug counts abstaining on a truly unmatched item as correct.",
        "",
        "| Model | Dataset | cov_aug | cov_sem | abstain_rate | precision | recall | apss_aug |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for ck, r in cells.items():
        a = r["abstention"]
        lines.append(
            f"| {MODEL_PRETTY[r['model']]} | {DATASET_PRETTY[r['dataset']]} | "
            f"{_fmt(a['coverage_aug_mean'])}±{_fmt(a['coverage_aug_std'])} | "
            f"{_fmt(a['coverage_semantic_mean'])} | "
            f"{_fmt(a['abstain_rate_mean'])}±{_fmt(a['abstain_rate_std'])} | "
            f"{_fmt(a['abstain_precision_mean'])} | "
            f"{_fmt(a['abstain_recall_mean'])} | "
            f"{_fmt(a['apss_aug_mean'], 2)} |"
        )
    lines += [
        "",
        "## Aggregate (mean over cells)",
        "",
        f"- coverage_aug:     **{_fmt(agg['coverage_aug'])}**",
        f"- coverage_semantic: {_fmt(agg['coverage_semantic'])}",
        f"- abstain_rate:      {_fmt(agg['abstain_rate'])}",
        f"- abstain_precision: {_fmt(agg['abstain_precision'])}",
        f"- abstain_recall:    {_fmt(agg['abstain_recall'])}",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unc_dir", type=str, default=str(DEFAULT_UNC_DIR),
        help="Directory of judged caches searched first; missing files fall "
             "back per-cell to the heuristic results/generations/unc_*.json.",
    )
    parser.add_argument(
        "--generations_dir", type=str, default=str(GENERATIONS_DIR),
        help="Heuristic cache directory used for discovery and fallback.",
    )
    parser.add_argument(
        "--cells", type=str, nargs="*", default=None,
        help="Optional cell_key filters (substring match) for smoke testing, "
             "e.g. qwen3_8b__triviaqa.",
    )
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    parser.add_argument(
        "--skip_abstention", action="store_true",
        help="Skip the abstention block (no SentenceEncoder load).",
    )
    parser.add_argument("--write-markdown", action="store_true",
                        help="Also write Markdown reports (off by default).")
    args = parser.parse_args()

    logger = _setup_logging()
    t0 = time.time()

    unc_dir = Path(args.unc_dir)
    base_dir = Path(args.generations_dir)
    cells = discover_cells(base_dir)
    if args.cells:
        cells = [c for c in cells
                 if any(f in c["cell_key"] for f in args.cells)]
    logger.info("=" * 70)
    logger.info("Judged re-evaluation: %d cells, alpha=%.2f, seeds=%s",
                len(cells), args.alpha, args.seeds)
    logger.info("unc_dir (judged): %s", unc_dir)
    logger.info("base_dir (heuristic fallback): %s", base_dir)
    if not cells:
        logger.error("No cells discovered under %s.", base_dir)
        return 1

    encoder = None
    if not args.skip_abstention:
        logger.info("Loading sentence encoder for abstention classifier...")
        encoder = SentenceEncoder(model_name="all-mpnet-base-v2")
        logger.info("Encoder ready.")

    headline_cells: dict[str, dict] = {}
    # Accumulators for conditional-coverage aggregate.
    cond_acc: dict[str, dict[str, dict[str, list[float]]]] = {
        by: {m: {"cov_bins": [], "gaps": []} for m in CONDITIONAL_METHODS}
        for by in ("U_epi", "U_ale")
    }
    abst_acc: dict[str, list[float]] = {
        k: [] for k in (
            "coverage_aug", "coverage_semantic", "abstain_rate",
            "abstain_precision", "abstain_recall",
        )
    }

    for i, cell in enumerate(cells, start=1):
        path, source = resolve_source(cell, unc_dir, base_dir)
        with open(path, encoding="utf-8") as f:
            data = _ensure_types(json.load(f))
        n = len(data)
        logger.info("-" * 70)
        logger.info("[%d/%d] %s  (n=%d, source=%s)",
                    i, len(cells), cell["cell_key"], n, source)

        head = evaluate_headline_cell(data, args.alpha, args.seeds)
        cond = evaluate_conditional_cell(data, args.alpha, args.seeds)
        rec = {
            "model": cell["model"], "dataset": cell["dataset"],
            "n_samples": n, "source": source,
            "match_rate": head["match_rate"],
            "methods": head["methods"],
            "conditional": cond,
        }

        if encoder is not None:
            rec["abstention"] = evaluate_abstention_cell(
                data, args.alpha, args.seeds, encoder,
            )
            a = rec["abstention"]
            for k in abst_acc:
                abst_acc[k].append(a[f"{k}_mean"])

        headline_cells[cell["cell_key"]] = rec

        # Feed conditional aggregate.
        for by in ("U_epi", "U_ale"):
            for m in CONDITIONAL_METHODS:
                cond_acc[by][m]["cov_bins"].append(
                    np.asarray(cond[by][m]["cov_bins_mean"], dtype=float)
                )
                cond_acc[by][m]["gaps"].append(cond[by][m]["gap_mean"])

        c_full = head["methods"]["CAELUM"]["full"]
        c_sup = head["methods"]["CAELUM"]["support"]
        v_full = head["methods"]["Vanilla"]["full"]
        logger.info(
            "    CAELUM APSS=%.2f cov=%.3f | Vanilla APSS=%.2f cov=%.3f | "
            "CAELUM support matched-cov=%.3f | lam*=%.2f",
            c_full["apss_mean"], c_full["cov_full_mean"],
            v_full["apss_mean"], v_full["cov_full_mean"],
            c_sup["cov_matched_mean"], c_full["lam_mean"],
        )
        if "abstention" in rec:
            a = rec["abstention"]
            logger.info(
                "    abstain: cov_aug=%.3f (sem=%.3f) rate=%.3f "
                "prec=%.3f rec=%.3f",
                a["coverage_aug_mean"], a["coverage_semantic_mean"],
                a["abstain_rate_mean"], a["abstain_precision_mean"],
                a["abstain_recall_mean"],
            )

    # ---- Aggregate conditional coverage ----
    cond_agg: dict[str, dict] = {}
    for by in ("U_epi", "U_ale"):
        cond_agg[by] = {}
        for m in CONDITIONAL_METHODS:
            cov_stack = np.vstack(cond_acc[by][m]["cov_bins"])
            gap_mean, gap_std = _mean_std(cond_acc[by][m]["gaps"])
            cond_agg[by][m] = {
                "cov_bins_mean": np.nanmean(cov_stack, axis=0).tolist(),
                "gap_mean": gap_mean, "gap_std": gap_std,
            }

    abst_agg = {k: _mean_std(v)[0] for k, v in abst_acc.items()}

    # ---- Write outputs ----
    headline_json = {
        "alpha": args.alpha, "seeds": args.seeds,
        "methods": HEADLINE_METHODS, "cells": headline_cells,
    }
    (RESULTS_DIR / "judged_rerun_headline.json").write_text(
        json.dumps(headline_json, indent=2), encoding="utf-8")
    if args.write_markdown:
        write_headline_md(headline_cells, args.alpha, args.seeds,
                          RESULTS_DIR / "judged_rerun_headline.md")

    conditional_json = {
        "alpha": args.alpha, "seeds": args.seeds, "n_bins": N_BINS,
        "aggregate": cond_agg,
        "cells": {ck: r["conditional"] for ck, r in headline_cells.items()},
    }
    (RESULTS_DIR / "judged_rerun_conditional_coverage.json").write_text(
        json.dumps(conditional_json, indent=2), encoding="utf-8")
    if args.write_markdown:
        write_conditional_md(headline_cells, cond_agg,
                             RESULTS_DIR / "judged_rerun_conditional_coverage.md")
    plot_conditional_coverage(cond_agg, args.alpha,
                              PLOTS_DIR / "judged_rerun_conditional_coverage.png")

    if encoder is not None:
        abstention_json = {
            "alpha": args.alpha, "seeds": args.seeds, "aggregate": abst_agg,
            "cells": {ck: r["abstention"] for ck, r in headline_cells.items()
                      if "abstention" in r},
        }
        (RESULTS_DIR / "judged_rerun_abstention.json").write_text(
            json.dumps(abstention_json, indent=2), encoding="utf-8")
        if args.write_markdown:
            write_abstention_md(headline_cells, abst_agg, args.alpha,
                                RESULTS_DIR / "judged_rerun_abstention.md")

    # ---- Compact cross-cell summary to the log ----
    logger.info("=" * 70)
    logger.info("CROSS-CELL SUMMARY")
    logger.info("=" * 70)
    logger.info(
        "%-34s %6s %6s %6s %7s %7s",
        "cell", "match", "CAELUM", "Vanil", "Ccov", "SCcov",
    )
    for ck, r in headline_cells.items():
        c = r["methods"]["CAELUM"]
        v = r["methods"]["Vanilla"]
        logger.info(
            "%-34s %6.2f %6.2f %6.2f %7.3f %7.3f",
            ck[:34], r["match_rate"],
            c["full"]["apss_mean"], v["full"]["apss_mean"],
            c["full"]["cov_full_mean"], c["support"]["cov_matched_mean"],
        )

    logger.info("-" * 70)
    logger.info("Conditional-coverage adaptivity (mean |b1-b4| gap over cells):")
    for by in ("U_epi", "U_ale"):
        parts = [
            f"{m}={cond_agg[by][m]['gap_mean']:+.3f}" for m in CONDITIONAL_METHODS
        ]
        logger.info("  by %s: %s", by, "  ".join(parts))

    # Support-conditional matched coverage aggregate.
    sc_matched = [r["methods"]["CAELUM"]["support"]["cov_matched_mean"]
                  for r in headline_cells.values()]
    sc_mu, sc_sd = _mean_std(sc_matched)
    logger.info("-" * 70)
    logger.info("Support-conditional CAELUM matched coverage: %.3f +/- %.3f "
                "(target %.2f)", sc_mu, sc_sd, 1.0 - args.alpha)

    if encoder is not None:
        logger.info("-" * 70)
        logger.info(
            "Abstention (aggregate): cov_aug=%.3f cov_sem=%.3f rate=%.3f "
            "prec=%.3f rec=%.3f",
            abst_agg["coverage_aug"], abst_agg["coverage_semantic"],
            abst_agg["abstain_rate"], abst_agg["abstain_precision"],
            abst_agg["abstain_recall"],
        )

    logger.info("=" * 70)
    logger.info("Wrote judged_rerun_{headline,conditional_coverage,abstention}"
                ".{json,md} to %s", RESULTS_DIR)
    logger.info("Plot: %s",
                PLOTS_DIR / "judged_rerun_conditional_coverage.png")
    logger.info("Total elapsed: %.1f s (%.1f min)",
                time.time() - t0, (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
