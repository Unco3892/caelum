"""
Evaluation metrics and plotting for CAELUM experiments.

Metrics:
  - Coverage  (marginal and conditional)
  - APSS      (average prediction set size)
  - AUROC     (for hallucination detection / ambiguity detection)
  - AUPR      (average precision / area under precision-recall curve)
  - Lambda optimization curve
  - Uncertainty scatter plot
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_coverage(data: List[Dict], calibrator) -> float:
    """Empirical coverage: fraction of test items where GT is in prediction set."""
    covered = []
    for i, item in enumerate(data):
        covered.append(calibrator.is_covered(item))
        if len(data) > 100 and (i + 1) % 50 == 0:
            logger.info("    Coverage: processed %d/%d samples...", i + 1, len(data))
    result = float(np.mean(covered))
    logger.info("    Coverage = %.3f", result)
    return result


def compute_apss(data: List[Dict], calibrator) -> float:
    """Average prediction set size."""
    sizes = []
    for i, item in enumerate(data):
        sizes.append(calibrator.predict_set_size(item))
        if len(data) > 100 and (i + 1) % 50 == 0:
            logger.info("    APSS: processed %d/%d samples...", i + 1, len(data))
    result = float(np.mean(sizes))
    logger.info("    APSS = %.2f", result)
    return result


def compute_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUROC without sklearn dependency (manual implementation).

    Args:
        labels: binary labels (0/1), shape (n,).
        scores: continuous scores, shape (n,).  Higher = more positive.

    Returns:
        AUROC in [0, 1].
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n = len(labels)
    if n == 0 or labels.sum() == 0 or labels.sum() == n:
        return 0.5  # undefined; return chance

    # Sort by score descending
    order = np.argsort(-scores)
    labels_sorted = labels[order]

    # Count true positives and false positives
    tp = 0
    fp = 0
    tp_prev = 0
    fp_prev = 0
    auc = 0.0
    score_prev = -np.inf
    n_pos = labels.sum()
    n_neg = n - n_pos

    for i in range(n):
        if scores[order[i]] != score_prev:
            # Trapezoidal rule
            auc += 0.5 * (fp - fp_prev) * (tp + tp_prev) / max(n_pos * n_neg, 1)
            score_prev = scores[order[i]]
            fp_prev = fp
            tp_prev = tp
        if labels_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
    auc += 0.5 * (fp - fp_prev) * (tp + tp_prev) / max(n_pos * n_neg, 1)
    return float(auc)


def compute_aupr(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUPR (Average Precision) without sklearn, matching average_precision_score.

    Implements the same step-function integration as sklearn:
        AP = -sum(diff(recall) * precision[:-1])
    where recall is decreasing and a (precision=1, recall=0) point is appended.

    Args:
        labels: binary labels (0/1), shape (n,).
        scores: continuous scores, shape (n,). Higher = more positive.

    Returns:
        AUPR in [0, 1]. Returns 0.0 if no positive labels, 1.0 if all positive.
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n = len(labels)
    n_pos = int(labels.sum())

    if n == 0 or n_pos == 0:
        return 0.0
    if n_pos == n:
        return 1.0

    # Stable descending sort — matches sklearn's argsort(ascending)[::-1]
    order = np.argsort(scores, kind="mergesort")[::-1]
    labels_sorted = labels[order]
    scores_sorted = scores[order]

    # Cumulative TP and FP at each position
    tps = np.cumsum(labels_sorted).astype(float)
    fps = np.arange(1, n + 1, dtype=float) - tps

    # Group by unique threshold: last index of each unique score group
    distinct_mask = np.r_[scores_sorted[:-1] != scores_sorted[1:], True]
    distinct_value_indices = np.where(distinct_mask)[0]

    # Find first position where all positives are captured (sklearn's last_ind)
    first_full = int(np.searchsorted(tps, n_pos))
    # Find the group boundary at or after first_full
    group_idx = int(np.searchsorted(distinct_value_indices, first_full))
    if group_idx >= len(distinct_value_indices):
        group_idx = len(distinct_value_indices) - 1
    threshold_indices = distinct_value_indices[: group_idx + 1]

    tps_t = tps[threshold_indices]
    fps_t = fps[threshold_indices]
    precision = tps_t / (tps_t + fps_t)
    recall = tps_t / n_pos

    # Reverse to decreasing recall + prepend (1, 0) starting point
    precision = np.r_[precision[::-1], 1.0]
    recall = np.r_[recall[::-1], 0.0]

    return float(-np.sum(np.diff(recall) * precision[:-1]))


def evaluate_diagnostic_ratio(
    test_data: List[Dict],
    calibrator,
) -> Dict[str, float]:
    """Evaluate R(x) as a classifier for hallucination vs ambiguity.

    R(x) = log(lambda* * U_epi + eps) - log(U_ale + eps).
    Positive => epistemic-dominated (hallucination).
    Negative => aleatoric-dominated (ambiguity).

    Returns dict with:
        mean_R_correct, mean_R_incorrect, mean_R_ambiguous,
        auroc_R_hallucination, auroc_R_ambiguity,
        pct_positive_R, pct_negative_R.
    """
    r_values = np.array([calibrator.diagnostic_ratio(d) for d in test_data])
    correct_mask = np.array([d["correct_cluster_idx"] >= 0 for d in test_data])
    incorrect_mask = ~correct_mask
    ambig_mask = np.array([d.get("is_ambig", False) for d in test_data])

    mean_R_correct = float(np.mean(r_values[correct_mask])) if correct_mask.sum() > 0 else float("nan")
    mean_R_incorrect = float(np.mean(r_values[incorrect_mask])) if incorrect_mask.sum() > 0 else float("nan")
    mean_R_ambiguous = float(np.mean(r_values[ambig_mask])) if ambig_mask.sum() > 0 else float("nan")

    # AUROC: R(x) as score, incorrect as positive label
    labels_hall = incorrect_mask.astype(int)
    auroc_R_hall = compute_auroc(labels_hall, r_values)

    # AUROC for ambiguity: -R(x) as score (negative R => ambiguous), is_ambig as positive
    labels_ambig = ambig_mask.astype(int)
    auroc_R_ambig = compute_auroc(labels_ambig, -r_values)

    pct_positive = float(np.mean(r_values > 0)) * 100
    pct_negative = float(np.mean(r_values < 0)) * 100

    results = {
        "mean_R_correct": mean_R_correct,
        "mean_R_incorrect": mean_R_incorrect,
        "mean_R_ambiguous": mean_R_ambiguous,
        "auroc_R_hallucination": auroc_R_hall,
        "auroc_R_ambiguity": auroc_R_ambig,
        "pct_positive_R": pct_positive,
        "pct_negative_R": pct_negative,
    }

    logger.info("R(x) diagnostic evaluation:")
    logger.info("  mean R(correct)=%.3f  mean R(incorrect)=%.3f  mean R(ambig)=%.3f",
                mean_R_correct, mean_R_incorrect, mean_R_ambiguous)
    logger.info("  AUROC R(x)->hall: %.3f   AUROC -R(x)->ambig: %.3f",
                auroc_R_hall, auroc_R_ambig)
    logger.info("  R>0 (epistemic): %.1f%%   R<0 (aleatoric): %.1f%%",
                pct_positive, pct_negative)

    return results


def diagnostic_ratio_summary(test_data: List[Dict], calibrator) -> str:
    """Formatted summary of R(x) distribution for logging."""
    r_values = np.array([calibrator.diagnostic_ratio(d) for d in test_data])
    bins = [
        ("R < -2", r_values < -2),
        ("-2 < R < -1", (r_values >= -2) & (r_values < -1)),
        ("-1 < R < 0", (r_values >= -1) & (r_values < 0)),
        ("0 < R < 1", (r_values >= 0) & (r_values < 1)),
        ("1 < R < 2", (r_values >= 1) & (r_values < 2)),
        ("R > 2", r_values >= 2),
    ]
    n = len(r_values)
    lines = ["R(x) Distribution:"]
    for label, mask in bins:
        count = int(mask.sum())
        pct = 100.0 * count / n if n > 0 else 0.0
        bar = "#" * int(pct / 2)
        lines.append(f"  {label:>12s}: {count:4d} ({pct:5.1f}%) {bar}")
    lines.append(f"  {'median':>12s}: {float(np.median(r_values)):+.3f}")
    lines.append(f"  {'mean':>12s}: {float(np.mean(r_values)):+.3f}")
    return "\n".join(lines)


def auroc_hallucination(data: List[Dict]) -> float:
    """AUROC for hallucination detection: U_epi predicts incorrectness.

    label=1 means the model is WRONG (correct_cluster_idx == -1).
    """
    labels = np.array([1 if d["correct_cluster_idx"] < 0 else 0 for d in data])
    scores = np.array([d["U_epi"] for d in data])
    return compute_auroc(labels, scores)


def auroc_ambiguity(data: List[Dict]) -> float:
    """AUROC for ambiguity detection: U_ale predicts is_ambig=True."""
    labels = np.array([1 if d.get("is_ambig", False) else 0 for d in data])
    scores = np.array([d["U_ale"] for d in data])
    return compute_auroc(labels, scores)


def auroc_total_hallucination(data: List[Dict]) -> float:
    """Baseline AUROC: U_total predicts incorrectness."""
    labels = np.array([1 if d["correct_cluster_idx"] < 0 else 0 for d in data])
    scores = np.array([d["U_total"] for d in data])
    return compute_auroc(labels, scores)


def auroc_total_ambiguity(data: List[Dict]) -> float:
    """Baseline AUROC: U_total predicts is_ambig."""
    labels = np.array([1 if d.get("is_ambig", False) else 0 for d in data])
    scores = np.array([d["U_total"] for d in data])
    return compute_auroc(labels, scores)


def aupr_hallucination(data: List[Dict]) -> float:
    """AUPR for hallucination detection: U_epi predicts incorrectness."""
    labels = np.array([1 if d["correct_cluster_idx"] < 0 else 0 for d in data])
    scores = np.array([d["U_epi"] for d in data])
    return compute_aupr(labels, scores)


def aupr_ambiguity(data: List[Dict]) -> float:
    """AUPR for ambiguity detection: U_ale predicts is_ambig=True."""
    labels = np.array([1 if d.get("is_ambig", False) else 0 for d in data])
    scores = np.array([d["U_ale"] for d in data])
    return compute_aupr(labels, scores)


def aupr_total_hallucination(data: List[Dict]) -> float:
    """Baseline AUPR: U_total predicts incorrectness."""
    labels = np.array([1 if d["correct_cluster_idx"] < 0 else 0 for d in data])
    scores = np.array([d["U_total"] for d in data])
    return compute_aupr(labels, scores)


def aupr_total_ambiguity(data: List[Dict]) -> float:
    """Baseline AUPR: U_total predicts is_ambig."""
    labels = np.array([1 if d.get("is_ambig", False) else 0 for d in data])
    scores = np.array([d["U_total"] for d in data])
    return compute_aupr(labels, scores)


# ---------------------------------------------------------------------------
# Full evaluation report
# ---------------------------------------------------------------------------

def evaluate(
    test_data: List[Dict],
    calibrator,
    dataset_name: str = "",
    verbose: bool = True,
) -> Dict[str, float]:
    """Compute all evaluation metrics on test data.

    Returns dict of metric_name -> value.
    """
    if verbose:
        logger.info("Computing evaluation metrics for %s (%d samples)...",
                    dataset_name, len(test_data))

    logger.info("  Computing coverage...")
    coverage = compute_coverage(test_data, calibrator)
    logger.info("  Computing APSS...")
    apss = compute_apss(test_data, calibrator)
    logger.info("  Computing AUROC metrics...")
    auroc_hall = auroc_hallucination(test_data)
    auroc_ambig = auroc_ambiguity(test_data)
    auroc_total_hall = auroc_total_hallucination(test_data)
    auroc_total_ambig = auroc_total_ambiguity(test_data)
    logger.info("  Computing AUPR metrics...")
    aupr_hall = aupr_hallucination(test_data)
    aupr_ambig = aupr_ambiguity(test_data)
    aupr_total_hall = aupr_total_hallucination(test_data)
    aupr_total_ambig = aupr_total_ambiguity(test_data)

    results = {
        "coverage": coverage,
        "apss": apss,
        "auroc_epi_hallucination": auroc_hall,
        "auroc_ale_ambiguity": auroc_ambig,
        "auroc_total_hallucination": auroc_total_hall,
        "auroc_total_ambiguity": auroc_total_ambig,
        "aupr_epi_hallucination": aupr_hall,
        "aupr_ale_ambiguity": aupr_ambig,
        "aupr_total_hallucination": aupr_total_hall,
        "aupr_total_ambiguity": aupr_total_ambig,
        "optimal_lambda": calibrator.optimal_lambda,
        "tau_hat": calibrator.tau_hat,
    }

    if verbose:
        logger.info("=" * 60)
        logger.info("Evaluation results for %s", dataset_name)
        logger.info("-" * 60)
        logger.info("  Coverage:          %.3f  (target >= %.2f)",
                     coverage, 1.0 - calibrator.alpha)
        logger.info("  APSS:              %.3f", apss)
        logger.info("  λ*:                %.4f", calibrator.optimal_lambda)
        logger.info("  τ̂:                 %.4f", calibrator.tau_hat)
        logger.info("  AUROC(U_epi→hall): %.3f  (baseline U_total: %.3f)",
                     auroc_hall, auroc_total_hall)
        logger.info("  AUROC(U_ale→ambig):%.3f  (baseline U_total: %.3f)",
                     auroc_ambig, auroc_total_ambig)
        logger.info("  AUPR(U_epi→hall): %.3f  (baseline U_total: %.3f)",
                     aupr_hall, aupr_total_hall)
        logger.info("  AUPR(U_ale→ambig):%.3f  (baseline U_total: %.3f)",
                     aupr_ambig, aupr_total_ambig)
        logger.info("=" * 60)

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_lambda_curve(
    calibrator,
    save_path: Optional[str] = None,
    title: str = "Lambda Optimisation: APSS vs. λ",
):
    """Plot APSS vs. lambda (from calibrator.lambda_losses)."""
    logger.info("Generating lambda optimization curve...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plot.")
        return

    if calibrator.lambda_losses is None:
        logger.warning("No lambda losses stored — run calibrate() first.")
        return

    logger.info("  Lambda grid size: %d points", len(calibrator.lambda_grid))
    logger.info("  Optimal lambda*: %.4f", calibrator.optimal_lambda)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(calibrator.lambda_grid, calibrator.lambda_losses, "b-", linewidth=1.5)
    ax.axvline(x=calibrator.optimal_lambda, color="r", linestyle="--",
               label=f"λ* = {calibrator.optimal_lambda:.4f}")
    ax.set_xlabel("λ")
    ax.set_ylabel("Average Prediction Set Size (APSS)")
    ax.set_title(title)
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved lambda curve to %s", save_path)
    plt.close(fig)


def plot_uncertainty_scatter(
    data: List[Dict],
    color_by: str = "correctness",
    save_path: Optional[str] = None,
    title: str = "Uncertainty Decomposition",
):
    """Scatter plot: U_ale (x) vs U_epi (y), coloured by correctness or ambiguity."""
    logger.info("Generating uncertainty scatter plot (color by: %s)...", color_by)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plot.")
        return

    logger.info("  Data points: %d samples", len(data))

    ale = np.array([d["U_ale"] for d in data])
    epi = np.array([d["U_epi"] for d in data])

    if color_by == "correctness":
        labels = np.array([1 if d["correct_cluster_idx"] >= 0 else 0 for d in data])
        color_map = {0: "red", 1: "green"}
        legend_map = {0: "Incorrect (no GT cluster)", 1: "Correct"}
    else:  # ambiguity
        labels = np.array([1 if d.get("is_ambig", False) else 0 for d in data])
        color_map = {0: "blue", 1: "orange"}
        legend_map = {0: "Unambiguous", 1: "Ambiguous"}

    fig, ax = plt.subplots(figsize=(8, 6))
    for val in sorted(color_map.keys()):
        mask = labels == val
        if mask.sum() > 0:
            ax.scatter(ale[mask], epi[mask], c=color_map[val], alpha=0.6,
                       s=30, label=legend_map[val])

    ax.set_xlabel("Aleatoric Uncertainty (U_ale)")
    ax.set_ylabel("Epistemic Uncertainty (U_epi)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved scatter plot to %s", save_path)
    plt.close(fig)


def plot_coverage_vs_apss(
    data: List[Dict],
    calibrator_class,
    lambda_grid: np.ndarray,
    alphas: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "Coverage vs APSS",
):
    """Plot coverage vs APSS for varying alpha, comparing CAELUM vs vanilla."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plot.")
        return

    if alphas is None:
        alphas = np.arange(0.05, 0.55, 0.05)

    logger.info("=" * 60)
    logger.info("Generating Coverage vs APSS plot")
    logger.info("  Data size: %d samples", len(data))
    logger.info("  Alpha values: %d points from %.2f to %.2f",
                len(alphas), alphas[0], alphas[-1])
    logger.info("=" * 60)

    from .utils import split_data

    val1, val2, test = split_data(data)
    logger.info("Split complete: Val1=%d, Val2=%d, Test=%d",
                len(val1), len(val2), len(test))

    clear_coverages, clear_apss = [], []
    vanilla_coverages, vanilla_apss = [], []

    for i, alpha in enumerate(alphas):
        logger.info("[%d/%d] Processing alpha=%.3f...", i+1, len(alphas), alpha)

        # CAELUM (full lambda optimisation)
        logger.info("  Calibrating CAELUM (adaptive lambda)...")
        cal = calibrator_class(alpha=alpha, lambda_grid=lambda_grid)
        cal.calibrate(val1, val2, verbose=False)
        cov_clear = compute_coverage(test, cal)
        apss_clear = compute_apss(test, cal)
        clear_coverages.append(cov_clear)
        clear_apss.append(apss_clear)
        logger.info("  CAELUM: coverage=%.3f, APSS=%.2f, lambda*=%.4f",
                    cov_clear, apss_clear, cal.optimal_lambda)

        # Vanilla conformal (λ = 0)
        logger.info("  Calibrating Vanilla CP (lambda=0)...")
        van = calibrator_class(alpha=alpha, lambda_grid=np.array([0.0]))
        van.calibrate(val1, val2, verbose=False)
        cov_van = compute_coverage(test, van)
        apss_van = compute_apss(test, van)
        vanilla_coverages.append(cov_van)
        vanilla_apss.append(apss_van)
        logger.info("  Vanilla: coverage=%.3f, APSS=%.2f", cov_van, apss_van)
        logger.info("  Improvement: APSS reduction = %.2f (%.1f%%)",
                    apss_van - apss_clear,
                    100 * (apss_van - apss_clear) / apss_van if apss_van > 0 else 0)

    logger.info("-" * 60)
    logger.info("Creating plot...")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(clear_coverages, clear_apss, "o-", label="CAELUM", color="tab:blue")
    ax.plot(vanilla_coverages, vanilla_apss, "s--", label="Vanilla CP (λ=0)", color="tab:orange")
    ax.set_xlabel("Coverage (1 - α)")
    ax.set_ylabel("APSS")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("✓ Saved coverage vs APSS plot to %s", save_path)
    else:
        logger.info("Plot generated (not saved - no save_path provided)")
    plt.close(fig)
    logger.info("=" * 60)


def plot_diagnostic_scatter(
    test_data: List[Dict],
    calibrator,
    save_path: Optional[str] = None,
    title: str = "Diagnostic R(x) vs Correctness",
):
    """Scatter plot: R(x) on y-axis, coloured by correct/incorrect/ambiguous."""
    logger.info("Generating diagnostic scatter plot...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plot.")
        return

    r_values = np.array([calibrator.diagnostic_ratio(d) for d in test_data])
    correct = np.array([d["correct_cluster_idx"] >= 0 for d in test_data])
    ambig = np.array([d.get("is_ambig", False) for d in test_data])

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(test_data))

    # Three groups: correct, incorrect, ambiguous
    mask_correct = correct & ~ambig
    mask_incorrect = ~correct & ~ambig
    mask_ambig = ambig

    if mask_correct.sum() > 0:
        ax.scatter(x[mask_correct], r_values[mask_correct], c="green",
                   alpha=0.6, s=30, label="Correct")
    if mask_incorrect.sum() > 0:
        ax.scatter(x[mask_incorrect], r_values[mask_incorrect], c="red",
                   alpha=0.6, s=30, label="Incorrect (hallucination)")
    if mask_ambig.sum() > 0:
        ax.scatter(x[mask_ambig], r_values[mask_ambig], c="orange",
                   alpha=0.6, s=30, label="Ambiguous")

    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5,
               label="R=0 (boundary)")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("R(x) = log(λ*·U_epi + ε) - log(U_ale + ε)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved diagnostic scatter to %s", save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def format_results_table(
    results: Dict[str, Dict[str, float]],
) -> str:
    """Format results as a markdown table.

    Args:
        results: {method_dataset: {metric: value}} dict.
    """
    header = "| Method | Dataset | Coverage | APSS | AUROC (Hall.) | AUROC (Ambig.) | λ* |"
    sep = "|--------|---------|----------|------|---------------|----------------|----|"
    rows = [header, sep]

    for key, metrics in results.items():
        parts = key.split("_", 1)
        method = parts[0] if len(parts) > 0 else key
        dataset = parts[1] if len(parts) > 1 else ""
        row = (
            f"| {method} | {dataset} "
            f"| {metrics.get('coverage', 0):.3f} "
            f"| {metrics.get('apss', 0):.2f} "
            f"| {metrics.get('auroc_epi_hallucination', 0):.3f} "
            f"| {metrics.get('auroc_ale_ambiguity', 0):.3f} "
            f"| {metrics.get('optimal_lambda', 0):.3f} |"
        )
        rows.append(row)

    return "\n".join(rows)
