#!/usr/bin/env python3
"""DeLong + bootstrap CIs on per-cell AUROC for U_epi vs U_total (statistical significance).

Audit-synthesis statement:
    The mean +0.014 AUROC effect of U_epi over U_total has p ~ 0.17 (sign test
    only). Multi-seed re-runs are deferred. Within a single seed we can still:

      * Paired DeLong test on each cell's U_epi vs U_total AUROC
        (DeLong et al., 1988; Sun & Xu, 2014 fast implementation).
      * Bootstrap-resample test items per cell to obtain a 95% CI on
        per-cell Delta AUROC.
      * Aggregate via fixed-effect inverse-variance meta-analysis with the
        I-squared heterogeneity statistic and a DerSimonian-Laird
        random-effects estimate when heterogeneity is high.

This script computes all of the above on the cached per-test-set predictions
in ``unc_*_vne_*.json`` and saves a JSON report.

Usage (from the repository root):
    python src/run_delong_aurocs.py
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

from caelum.utils import split_data  # noqa: E402

UNC_RE = re.compile(r"^unc_([a-z0-9]+)_(.+)_vne_(\d+)\.json$")
SKIP_DATASETS: set[str] = {"condambigqa"}
SKIP_NSAMPLE_RE = re.compile(r"_50\.json$")

# The accepted 18-cell benchmark uses these three LLMs. Larger-model caches
# (qwen3_14b, qwen3_32b) share results, so they are skipped unless --models
# names them explicitly.
ACCEPTED_MODELS = ("qwen3_8b", "ministral_3_8b_instruct_2512", "gemma_4_e4b_it")


# ---------------------------------------------------------------------------
# DeLong's covariance estimator for paired AUROC
# ---------------------------------------------------------------------------

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Compute midranks used in the DeLong AUROC variance estimator.

    Args:
        x: Sample of scores, shape (n,).

    Returns:
        Midranks, shape (n,).
    """
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        # Find run of ties
        j_end = i
        while j_end < n and z[j_end] == z[i]:
            j_end += 1
        # Midrank = average rank within tie group (1-indexed in DeLong's notation)
        midrank = 0.5 * (i + j_end - 1) + 1
        t[i:j_end] = midrank
        i = j_end
    # Unsort back to original order
    out = np.empty(n, dtype=float)
    out[j] = t
    return out


def fast_delong(
    predictions_sorted_transposed: np.ndarray, label_one_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast DeLong implementation (Sun & Xu, 2014).

    The labels are assumed to be sorted: positives first, then negatives.

    Args:
        predictions_sorted_transposed: Predictions of shape (k, n) where k is
            the number of classifiers and n is the number of samples. The
            samples are sorted such that the first ``label_one_count`` are
            positives.
        label_one_count: Number of positive samples (m). The remaining
            ``n - m`` rows are negatives (n_neg).

    Returns:
        Tuple ``(aurocs, delong_cov)`` where ``aurocs`` has shape (k,) and
        ``delong_cov`` has shape (k, k) -- the DeLong covariance matrix of
        the AUROC estimates.
    """
    m = label_one_count
    n = predictions_sorted_transposed.shape[1] - m
    n_classifiers = predictions_sorted_transposed.shape[0]
    if m == 0 or n == 0:
        return (
            np.full(n_classifiers, np.nan, dtype=float),
            np.full((n_classifiers, n_classifiers), np.nan, dtype=float),
        )

    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]

    tx = np.empty((n_classifiers, m), dtype=float)
    ty = np.empty((n_classifiers, n), dtype=float)
    tz = np.empty((n_classifiers, m + n), dtype=float)

    for r in range(n_classifiers):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)

    v01 = (tz[:, :m] - tx[:, :]) / n  # (k, m)
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m  # (k, n)

    sx = np.cov(v01, ddof=1)
    sy = np.cov(v10, ddof=1)
    # When k=1, np.cov returns a scalar; reshape
    sx = np.atleast_2d(sx)
    sy = np.atleast_2d(sy)

    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def delong_test_paired(
    labels: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray,
) -> dict:
    """Run paired DeLong test for AUROC(scores_a) vs AUROC(scores_b).

    Args:
        labels: Binary labels (0/1), shape (n,).
        scores_a: Scores for classifier A (higher => positive class), shape (n,).
        scores_b: Scores for classifier B, shape (n,).

    Returns:
        Dict with ``auc_a``, ``auc_b``, ``delta``, ``var_delta``, ``z``,
        ``p_value`` (two-sided), and ``ci95_low``, ``ci95_high`` for the
        delta. Returns NaNs in degenerate cases (all positives or all
        negatives).
    """
    labels = np.asarray(labels, dtype=int)
    if len(labels) == 0 or labels.sum() == 0 or labels.sum() == len(labels):
        return {
            "auc_a": float("nan"), "auc_b": float("nan"),
            "delta": float("nan"), "var_delta": float("nan"),
            "z": float("nan"), "p_value": float("nan"),
            "ci95_low": float("nan"), "ci95_high": float("nan"),
        }

    order = np.argsort(-labels)  # positives first (label=1 -> -1 sorts first)
    labels_sorted = labels[order]
    a_sorted = np.asarray(scores_a, dtype=float)[order]
    b_sorted = np.asarray(scores_b, dtype=float)[order]
    m = int(np.sum(labels_sorted))  # number of positives

    preds = np.stack([a_sorted, b_sorted], axis=0)  # (2, n)
    aucs, cov = fast_delong(preds, m)
    auc_a, auc_b = float(aucs[0]), float(aucs[1])
    delta = auc_a - auc_b
    var_delta = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])

    if var_delta <= 0 or not np.isfinite(var_delta):
        # Degenerate -- e.g., scores are identical
        z = float("nan")
        p_value = float("nan")
        ci_low = float("nan")
        ci_high = float("nan")
    else:
        z = delta / np.sqrt(var_delta)
        p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
        ci_low = float(delta - 1.96 * np.sqrt(var_delta))
        ci_high = float(delta + 1.96 * np.sqrt(var_delta))

    return {
        "auc_a": auc_a, "auc_b": auc_b,
        "delta": float(delta), "var_delta": var_delta,
        "z": float(z) if np.isfinite(z) else float("nan"),
        "p_value": p_value,
        "ci95_low": ci_low, "ci95_high": ci_high,
    }


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def _auroc_manual(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUROC via the rank-sum (Mann-Whitney) statistic."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pos_mask = labels == 1
    n_pos = int(pos_mask.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(scores)
    rank_sum_pos = float(np.sum(ranks[pos_mask]))
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc


def bootstrap_paired_delta_ci(
    labels: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Paired bootstrap CI for AUROC(a) - AUROC(b)."""
    labels = np.asarray(labels, dtype=int)
    if len(labels) == 0 or labels.sum() == 0 or labels.sum() == len(labels):
        return {
            "delta_mean": float("nan"),
            "ci_low": float("nan"), "ci_high": float("nan"),
            "n_valid": 0,
        }
    rng = np.random.RandomState(seed)
    n = len(labels)
    deltas = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        lb = labels[idx]
        if lb.sum() == 0 or lb.sum() == n:
            continue
        a = scores_a[idx]
        b = scores_b[idx]
        try:
            auc_a = _auroc_manual(lb, a)
            auc_b = _auroc_manual(lb, b)
            if not (np.isfinite(auc_a) and np.isfinite(auc_b)):
                continue
            deltas.append(auc_a - auc_b)
        except Exception:
            continue
    if not deltas:
        return {
            "delta_mean": float("nan"),
            "ci_low": float("nan"), "ci_high": float("nan"),
            "n_valid": 0,
        }
    deltas = np.array(deltas)
    return {
        "delta_mean": float(np.mean(deltas)),
        "ci_low": float(np.quantile(deltas, alpha / 2)),
        "ci_high": float(np.quantile(deltas, 1 - alpha / 2)),
        "n_valid": int(len(deltas)),
    }


# ---------------------------------------------------------------------------
# Per-cell evaluation
# ---------------------------------------------------------------------------

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


def evaluate_cell(
    unc_path: Path, n_boot: int = 1000, seed: int = 42,
) -> dict | None:
    """Compute paired DeLong + bootstrap CIs on a single cell's test set.

    The test set is the same 30% partition used everywhere else (seed=42,
    40/30/30). Labels = (correct_cluster_idx < 0), i.e., 1 means the model
    failed to match GT (incorrect/hallucination).

    Returns None if the cell is degenerate (all positives or all negatives in
    the test set, or fewer than 30 items).
    """
    info = parse_unc_name(unc_path.name)
    if info is None:
        return None

    with unc_path.open(encoding="utf-8") as f:
        items = json.load(f)
    if len(items) < 30:
        return None

    _, _, test = split_data(items, val1_frac=0.4, val2_frac=0.3, seed=seed)
    if len(test) < 10:
        return None

    labels = np.array([
        1 if it.get("correct_cluster_idx", -1) < 0 else 0 for it in test
    ], dtype=int)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return {
            "model": info["model"],
            "dataset": info["dataset"],
            "n_test": int(len(test)),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "degenerate": True,
            "reason": "all positive" if n_neg == 0 else "all negative",
        }

    u_total = np.array([
        float(it.get("U_total", it["U_ale"] + it["U_epi"])) for it in test
    ])
    u_epi = np.array([float(it["U_epi"]) for it in test])

    # DeLong: a = U_epi, b = U_total
    delong = delong_test_paired(labels, u_epi, u_total)
    boot = bootstrap_paired_delta_ci(
        labels, u_epi, u_total, n_boot=n_boot, seed=seed,
    )

    return {
        "model": info["model"],
        "dataset": info["dataset"],
        "n_test": int(len(test)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "degenerate": False,
        "auroc_uepi": delong["auc_a"],
        "auroc_utotal": delong["auc_b"],
        "delta": delong["delta"],
        "delong_var": delong["var_delta"],
        "delong_z": delong["z"],
        "delong_p": delong["p_value"],
        "delong_ci_low": delong["ci95_low"],
        "delong_ci_high": delong["ci95_high"],
        "bootstrap_delta_mean": boot["delta_mean"],
        "bootstrap_ci_low": boot["ci_low"],
        "bootstrap_ci_high": boot["ci_high"],
        "bootstrap_n_valid": boot["n_valid"],
    }


# ---------------------------------------------------------------------------
# Meta-analysis aggregation
# ---------------------------------------------------------------------------

def meta_analysis(per_cell: list[dict]) -> dict:
    """Inverse-variance fixed-effect + DerSimonian-Laird random-effects.

    Uses DeLong variance when available; falls back to bootstrap-derived
    variance ((ci_high - ci_low) / (2 * 1.96))^2 if DeLong variance is NaN
    or non-positive. Cells with neither variance are dropped.
    """
    deltas: list[float] = []
    variances: list[float] = []
    contributions: list[dict] = []

    for c in per_cell:
        if c.get("degenerate"):
            continue
        delta = c["delta"]
        var = c["delong_var"]
        used = "delong"
        if not np.isfinite(var) or var <= 0:
            ci_low = c.get("bootstrap_ci_low")
            ci_high = c.get("bootstrap_ci_high")
            if (ci_low is not None and ci_high is not None
                    and np.isfinite(ci_low) and np.isfinite(ci_high)
                    and ci_high > ci_low):
                var = ((ci_high - ci_low) / (2.0 * 1.96)) ** 2
                used = "bootstrap"
            else:
                continue
        if not np.isfinite(delta) or var <= 0:
            continue
        deltas.append(float(delta))
        variances.append(float(var))
        contributions.append({
            "model": c["model"], "dataset": c["dataset"],
            "delta": float(delta), "var": float(var),
            "weight_source": used,
        })

    if not deltas:
        return {
            "n_cells_used": 0,
            "fixed_effect_delta": float("nan"),
            "fixed_effect_se": float("nan"),
            "fixed_effect_ci_low": float("nan"),
            "fixed_effect_ci_high": float("nan"),
            "fixed_effect_z": float("nan"),
            "fixed_effect_p": float("nan"),
            "i_squared": float("nan"),
            "tau_squared": float("nan"),
            "random_effect_delta": float("nan"),
            "random_effect_ci_low": float("nan"),
            "random_effect_ci_high": float("nan"),
            "random_effect_p": float("nan"),
            "contributions": [],
        }

    deltas_arr = np.array(deltas)
    var_arr = np.array(variances)
    weights = 1.0 / var_arr

    # Fixed-effect (inverse-variance)
    fe_delta = float(np.sum(weights * deltas_arr) / np.sum(weights))
    fe_se = float(np.sqrt(1.0 / np.sum(weights)))
    fe_ci_low = fe_delta - 1.96 * fe_se
    fe_ci_high = fe_delta + 1.96 * fe_se
    fe_z = fe_delta / fe_se if fe_se > 0 else float("nan")
    fe_p = (
        float(2.0 * (1.0 - stats.norm.cdf(abs(fe_z))))
        if np.isfinite(fe_z) else float("nan")
    )

    # Heterogeneity
    k = len(deltas_arr)
    q = float(np.sum(weights * (deltas_arr - fe_delta) ** 2))
    df = k - 1
    if df > 0:
        i_squared = float(max(0.0, (q - df) / q)) if q > 0 else 0.0
    else:
        i_squared = float("nan")

    # DerSimonian-Laird random-effects
    if df > 0:
        c_factor = float(
            np.sum(weights) - np.sum(weights ** 2) / np.sum(weights),
        )
        if c_factor > 0:
            tau_squared = max(0.0, (q - df) / c_factor)
        else:
            tau_squared = 0.0
    else:
        tau_squared = 0.0

    re_weights = 1.0 / (var_arr + tau_squared)
    re_delta = float(np.sum(re_weights * deltas_arr) / np.sum(re_weights))
    re_se = float(np.sqrt(1.0 / np.sum(re_weights)))
    re_ci_low = re_delta - 1.96 * re_se
    re_ci_high = re_delta + 1.96 * re_se
    re_z = re_delta / re_se if re_se > 0 else float("nan")
    re_p = (
        float(2.0 * (1.0 - stats.norm.cdf(abs(re_z))))
        if np.isfinite(re_z) else float("nan")
    )

    return {
        "n_cells_used": k,
        "fixed_effect_delta": fe_delta,
        "fixed_effect_se": fe_se,
        "fixed_effect_ci_low": fe_ci_low,
        "fixed_effect_ci_high": fe_ci_high,
        "fixed_effect_z": fe_z,
        "fixed_effect_p": fe_p,
        "q_statistic": q,
        "i_squared": i_squared,
        "tau_squared": float(tau_squared),
        "random_effect_delta": re_delta,
        "random_effect_se": re_se,
        "random_effect_ci_low": re_ci_low,
        "random_effect_ci_high": re_ci_high,
        "random_effect_z": re_z,
        "random_effect_p": re_p,
        "contributions": contributions,
    }


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generations_dir", default=GENERATIONS_DIR, type=Path,
        help="Folder with the unc_* caches.",
    )
    parser.add_argument("--n_boot", default=1000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--output", default=None, type=Path)
    parser.add_argument("--models", nargs="+", default=list(ACCEPTED_MODELS),
                        help="model keys to include (default: the accepted 18-cell set)")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing results file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    log = logging.getLogger("delong_aurocs")
    log.setLevel(logging.INFO)

    output_path = args.output or (RESULTS_DIR / "auroc_delong_cis.json")
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
        cell = evaluate_cell(unc_file, n_boot=args.n_boot, seed=args.seed)
        if cell is not None:
            cells.append(cell)

    summary = meta_analysis(cells)
    n_degenerate = sum(1 for c in cells if c.get("degenerate"))

    out = {
        "comparison": "U_epi (a) vs U_total (b); delta = AUROC(U_epi) - AUROC(U_total)",
        "seed": args.seed,
        "n_boot": args.n_boot,
        "n_cells_total": len(cells),
        "n_cells_degenerate": n_degenerate,
        "summary": summary,
        "per_cell": cells,
    }

    output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("Wrote %s", output_path)

    print("\n=== AUROC DeLong CI summary ===")
    print(
        f"Cells: {len(cells)} total, {n_degenerate} degenerate (excluded), "
        f"{summary['n_cells_used']} used in meta-analysis"
    )
    print(
        f"Fixed-effect Delta = {summary['fixed_effect_delta']:+.4f} "
        f"[{summary['fixed_effect_ci_low']:+.4f}, "
        f"{summary['fixed_effect_ci_high']:+.4f}], "
        f"p = {summary['fixed_effect_p']:.4f}"
    )
    print(
        f"Random-effect Delta = {summary['random_effect_delta']:+.4f} "
        f"[{summary['random_effect_ci_low']:+.4f}, "
        f"{summary['random_effect_ci_high']:+.4f}], "
        f"p = {summary['random_effect_p']:.4f}"
    )
    print(
        f"Heterogeneity: I^2 = {summary['i_squared']:.3f}, "
        f"tau^2 = {summary['tau_squared']:.5f}, "
        f"Q = {summary.get('q_statistic', float('nan')):.2f}"
    )

    n_significant = sum(
        1 for c in cells if not c.get("degenerate")
        and np.isfinite(c.get("delong_p", float("nan")))
        and c["delong_p"] < 0.05
    )
    n_positive = sum(
        1 for c in cells if not c.get("degenerate")
        and np.isfinite(c.get("delta", float("nan")))
        and c["delta"] > 0
    )
    print(
        f"Per-cell: {n_positive}/{summary['n_cells_used']} cells have positive Delta; "
        f"{n_significant} significant at p<0.05 (DeLong, uncorrected)"
    )


if __name__ == "__main__":
    main()
