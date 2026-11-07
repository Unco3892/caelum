#!/usr/bin/env python3
"""R(x) vs U_epi-alone routing comparison (Attack 11).

Reviewer attack: "Is R(x) better than just thresholding U_epi alone?" R(x) is
the diagnostic ratio R(x) = log(lambda* * U_epi + eps) - log(U_ale + eps).
U_epi alone is freely available without conformal calibration.

This script implements the same selective-answering protocol used in
``r_protocol_experiment.py`` (rank items by score descending, hand top-k to
oracle, measure accuracy AUC over escalation rate), and adds:

    Delta_R_vs_Uepi = AUC(R(x)) - AUC(U_epi)

Aggregates across the 18 cells with mean, median, sign-test, and the count
of cells where R(x) > U_epi.

Usage (from the repository root):
    python src/run_r_vs_uepi_routing.py
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from caelum.paths import GENERATIONS_DIR, RESULTS_DIR  # noqa: E402

from caelum.calibration import CAELUMCalibrator  # noqa: E402
from caelum.utils import split_data  # noqa: E402

UNC_RE = re.compile(r"^unc_([a-z0-9]+)_(.+)_vne_(\d+)\.json$")
SKIP_DATASETS: set[str] = {"condambigqa"}
SKIP_NSAMPLE_RE = re.compile(r"_50\.json$")

# The accepted 18-cell benchmark uses these three LLMs. Larger-model caches
# (qwen3_14b, qwen3_32b) share results, so they are skipped unless --models
# names them explicitly.
ACCEPTED_MODELS = ("qwen3_8b", "ministral_3_8b_instruct_2512", "gemma_4_e4b_it")

MODEL_PRETTY = {
    "qwen3_8b": "Qwen3-8B",
    "ministral_3_8b_instruct_2512": "Ministral-3-8B",
    "gemma_4_e4b_it": "Gemma-4-E4B",
}
DATASET_PRETTY = {
    "triviaqa": "TriviaQA", "ambigqa": "AmbigQA", "opennq": "OpenNQ",
    "ambiginst": "AmbigInst", "simpleqa": "SimpleQA", "popqa": "PopQA",
}


def parse_unc_name(name: str) -> dict | None:
    """Parse the unc_*.json filename into dataset/model/n_samples."""
    m = UNC_RE.match(name)
    if m is None:
        return None
    return {
        "dataset": m.group(1),
        "model": m.group(2),
        "n_samples": int(m.group(3)),
    }


def base_accuracy_per_item(items: list[dict]) -> np.ndarray:
    """Return 1 if model's top cluster matches GT, else 0."""
    return np.array([1 if it.get("correct_cluster_idx", -1) == 0 else 0 for it in items])


def policy_accuracy_curve(
    items: list[dict], scores: np.ndarray, n_steps: int = 21,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort items by ``scores`` desc; escalate top-k to oracle (oracle accuracy=1).

    Returns ``(escalation_rates, accuracies)`` -- accuracy as a function of the
    fraction of items handed to the oracle.
    """
    base_acc = base_accuracy_per_item(items)
    # Stable sort: tied scores keep their input order on every NumPy version.
    order = np.argsort(-np.asarray(scores), kind="stable")
    n = len(items)
    rates = np.linspace(0, 1, n_steps)
    accs = []
    for r in rates:
        k = int(round(r * n))
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[order[:k]] = False
        acc_non_esc = base_acc[keep_mask].sum()
        accs.append((k + acc_non_esc) / n)
    return rates, np.array(accs)


def auc_under_curve(rates: np.ndarray, accs: np.ndarray) -> float:
    """Trapezoid AUC of accuracy vs escalation rate (higher = better)."""
    # np.trapezoid was added in NumPy 2.0; the project uses an older numpy.
    # np.trapz was renamed np.trapezoid in NumPy 2.0 (same computation).
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(accs, rates))


def evaluate_cell(
    unc_path: Path, alpha: float = 0.1, seed: int = 42,
) -> dict | None:
    """Evaluate selective-answering AUCs for one cell."""
    info = parse_unc_name(unc_path.name)
    if info is None:
        return None

    with unc_path.open(encoding="utf-8") as f:
        items = json.load(f)
    if len(items) < 30:
        return None

    val1, val2, test = split_data(items, val1_frac=0.4, val2_frac=0.3, seed=seed)

    cal = CAELUMCalibrator(alpha=alpha, seed=seed, randomize=True)
    cal.calibrate(val1, val2, verbose=False)

    r_scores = np.array([cal.diagnostic_ratio(it) for it in test])
    u_total = np.array([
        it.get("U_total", it["U_ale"] + it["U_epi"]) for it in test
    ])
    u_epi = np.array([float(it["U_epi"]) for it in test])
    u_ale = np.array([float(it["U_ale"]) for it in test])
    base_acc = base_accuracy_per_item(test)

    rng = np.random.RandomState(seed)
    incorrect_score = 1 - base_acc + rng.normal(0, 1e-3, len(test))
    random_score = rng.uniform(size=len(test))

    rates, acc_R = policy_accuracy_curve(test, r_scores)
    _, acc_Uepi = policy_accuracy_curve(test, u_epi)
    _, acc_Uale = policy_accuracy_curve(test, u_ale)
    _, acc_Utotal = policy_accuracy_curve(test, u_total)
    _, acc_random = policy_accuracy_curve(test, random_score)
    _, acc_oracle = policy_accuracy_curve(test, incorrect_score)

    auc = {
        "Random": auc_under_curve(rates, acc_random),
        "U_total": auc_under_curve(rates, acc_Utotal),
        "U_epi": auc_under_curve(rates, acc_Uepi),
        "U_ale": auc_under_curve(rates, acc_Uale),
        "R(x)": auc_under_curve(rates, acc_R),
        "Oracle": auc_under_curve(rates, acc_oracle),
    }

    return {
        "model": info["model"],
        "dataset": info["dataset"],
        "n_test": int(len(test)),
        "lambda_star": float(cal.optimal_lambda),
        "base_accuracy": float(base_acc.mean()),
        "auc": auc,
        "delta_R_vs_Uepi": auc["R(x)"] - auc["U_epi"],
        "delta_R_vs_Utotal": auc["R(x)"] - auc["U_total"],
        "delta_R_vs_Random": auc["R(x)"] - auc["Random"],
    }


def aggregate(per_cell: list[dict]) -> dict:
    """Aggregate Delta(R(x) - U_epi) across cells with sign test."""
    deltas_uepi = np.array([c["delta_R_vs_Uepi"] for c in per_cell])
    deltas_utotal = np.array([c["delta_R_vs_Utotal"] for c in per_cell])
    deltas_random = np.array([c["delta_R_vs_Random"] for c in per_cell])

    n = len(deltas_uepi)
    n_R_beats_Uepi = int(np.sum(deltas_uepi > 0))
    n_ties_uepi = int(np.sum(deltas_uepi == 0))
    # Two-sided sign test: P(X >= n_R_beats_Uepi or X <= n - n_R_beats_Uepi).
    n_eff = n - n_ties_uepi
    if n_eff > 0:
        n_pos = n_R_beats_Uepi
        sign_p = float(stats.binomtest(n_pos, n_eff, p=0.5).pvalue)
    else:
        sign_p = float("nan")

    # Wilcoxon signed-rank as a complement (paired, non-parametric).
    if n_eff > 1:
        try:
            wilcox = stats.wilcoxon(deltas_uepi, alternative="two-sided")
            wilcox_stat = float(wilcox.statistic)
            wilcox_p = float(wilcox.pvalue)
        except (ValueError, RuntimeError):
            wilcox_stat = float("nan")
            wilcox_p = float("nan")
    else:
        wilcox_stat = float("nan")
        wilcox_p = float("nan")

    by_lambda_regime: dict[str, list[float]] = {"low_lambda": [], "high_lambda": []}
    for c in per_cell:
        regime = "high_lambda" if c["lambda_star"] > 1.0 else "low_lambda"
        by_lambda_regime[regime].append(c["delta_R_vs_Uepi"])

    return {
        "n_cells": n,
        "delta_R_vs_Uepi_mean": float(np.mean(deltas_uepi)),
        "delta_R_vs_Uepi_std": float(np.std(deltas_uepi, ddof=1))
        if n > 1 else 0.0,
        "delta_R_vs_Uepi_median": float(np.median(deltas_uepi)),
        "n_cells_R_beats_Uepi": n_R_beats_Uepi,
        "n_cells_R_loses_Uepi": int(np.sum(deltas_uepi < 0)),
        "n_cells_tied": n_ties_uepi,
        "sign_test_p": sign_p,
        "wilcoxon_stat": wilcox_stat,
        "wilcoxon_p": wilcox_p,
        "delta_R_vs_Utotal_mean": float(np.mean(deltas_utotal)),
        "delta_R_vs_Random_mean": float(np.mean(deltas_random)),
        "by_lambda_regime": {
            regime: {
                "n": len(vals),
                "delta_R_vs_Uepi_mean":
                    float(np.mean(vals)) if vals else float("nan"),
                "n_R_beats_Uepi":
                    int(np.sum(np.array(vals) > 0)) if vals else 0,
            }
            for regime, vals in by_lambda_regime.items()
        },
    }


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generations_dir", default=GENERATIONS_DIR, type=Path,
        help="Folder with the unc_* caches.",
    )
    parser.add_argument("--alpha", default=0.1, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--output", default=None, type=Path)
    parser.add_argument("--models", nargs="+", default=list(ACCEPTED_MODELS),
                        help="model keys to include (default: the accepted 18-cell set)")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing results file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    log = logging.getLogger("r_vs_uepi")
    log.setLevel(logging.INFO)

    output_path = args.output or (RESULTS_DIR / "r_vs_uepi_routing.json")
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"{output_path} already exists and backs reported results; "
                         "pass --overwrite or write elsewhere.")
    cells: list[dict] = []
    for unc_file in sorted(args.generations_dir.glob("unc_*.json")):
        if SKIP_NSAMPLE_RE.search(unc_file.name):
            continue
        info = parse_unc_name(unc_file.name)
        if (info is None or info["dataset"] in SKIP_DATASETS
                or info["model"] not in args.models):
            continue
        log.info("Processing %s", unc_file.name)
        cell = evaluate_cell(unc_file, alpha=args.alpha, seed=args.seed)
        if cell is not None:
            cells.append(cell)

    if not cells:
        log.error("No cells produced; aborting.")
        sys.exit(1)

    summary = aggregate(cells)

    out = {
        "alpha": args.alpha,
        "seed": args.seed,
        "summary": summary,
        "per_cell": cells,
    }

    output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("Wrote %s", output_path)

    print("\n=== R(x) vs U_epi routing summary ===")
    print(
        f"Cells: {summary['n_cells']}, "
        f"R(x) beats U_epi: {summary['n_cells_R_beats_Uepi']}/{summary['n_cells']} "
        f"(ties={summary['n_cells_tied']})"
    )
    print(
        f"Mean Delta(R - U_epi)  = {summary['delta_R_vs_Uepi_mean']:+.4f} "
        f"(std={summary['delta_R_vs_Uepi_std']:.4f}, "
        f"median={summary['delta_R_vs_Uepi_median']:+.4f})"
    )
    print(
        f"Sign test p = {summary['sign_test_p']:.4f}, "
        f"Wilcoxon p = {summary['wilcoxon_p']:.4f}"
    )
    for regime, info in summary["by_lambda_regime"].items():
        print(
            f"  {regime} (n={info['n']}): "
            f"mean Delta = {info['delta_R_vs_Uepi_mean']:+.4f}, "
            f"R wins = {info['n_R_beats_Uepi']}/{info['n']}"
        )


if __name__ == "__main__":
    main()
