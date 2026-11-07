#!/usr/bin/env python3
"""Normalized Calibrated APSS (NCAPSS) — matched-coverage analogue of CLEAR's NCIW.

CLEAR (Azizi et al., 2025) defines NCIW after a per-method post-hoc rescaling to
a common target coverage. The discrete-set diagnostic for our setting is:

    NCAPSS_method(cell) = APSS_method(cell, alpha = alpha*_method(q*))

where ``alpha*_method(q*)`` is the per-method alpha that reaches target
coverage q* on the test sweep, found by linear interpolation over an alpha
sweep. The target coverage q* per cell is

    q* = min(1 - alpha_nominal, validation_match_rate)

i.e., the estimated operationally achievable coverage ceiling from calibration
data. Because interpolation itself uses the test coverage sweep, NCAPSS is a
post-hoc coverage-efficiency diagnostic rather than a deployable validation-only
selection rule.

Outputs
-------
1. ``results/ncapss_results.json`` — per-cell, per-method records.
2. ``results/ncapss_summary.md``  — optional report (``--write-markdown``).
3. ``results/plots/ncapss_comparison.{pdf,png}`` — boxplot of NCAPSS / NCAPSS_CAELUM.
4. ``results/exported_tables/ncapss_tabular.tex`` — auto-generated LaTeX table.

Methods compared (six)
----------------------
    CAELUM       : lambda* selected on Val1
    Vanilla      : lambda = 0
    Equal        : lambda = 1
    EpiOnly      : lambda = 1e6
    ConU         : self-consistency (Wang et al., EMNLP 2024)
    SE-CP        : APS weighted by 1 + U_total (Kuhn et al., 2023)

Usage
-----
    python -u src/run_ncapss_full.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from caelum.paths import GENERATIONS_DIR, PLOTS_DIR, RESULTS_DIR  # noqa: E402

from caelum.calibration import CAELUMCalibrator
from caelum.baselines import ConUStyleCalibrator, SemanticEntropyCalibrator
from caelum.evaluation import compute_apss, compute_coverage
from caelum.utils import load_json, split_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
for noisy in (
    "caelum.calibration", "caelum.evaluation", "caelum.utils",
    "caelum.baselines", "sentence_transformers", "transformers",
    "huggingface_hub", "filelock", "urllib3", "fsspec",
):
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger("ncapss")

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Export analysis tables to results/exported_tables/ (gitignored).
PAPER_TABLES_DIR = RESULTS_DIR / "exported_tables"

OUT_JSON = RESULTS_DIR / "ncapss_results.json"
OUT_MD = RESULTS_DIR / "ncapss_summary.md"
OUT_PDF = PLOTS_DIR / "ncapss_comparison.pdf"
OUT_PNG = PLOTS_DIR / "ncapss_comparison.png"
OUT_TEX = PAPER_TABLES_DIR / "ncapss_tabular.tex"

# ---------------------------------------------------------------------------
# Cell discovery and pretty-naming
# ---------------------------------------------------------------------------

UNC_RE = re.compile(r"^unc_([a-z0-9]+)_(.+)_vne_(\d+)\.json$")

MODEL_PRETTY: dict[str, str] = {
    "qwen3_8b": "Qwen3-8B",
    "ministral_3_8b_instruct_2512": "Ministral-3-8B",
    "gemma_4_e4b_it": "Gemma-4-E4B-it",
}
DATASET_PRETTY: dict[str, str] = {
    "triviaqa": "TriviaQA",
    "ambigqa": "AmbigQA",
    "opennq": "OpenNQ",
    "ambiginst": "AmbigInst",
    "simpleqa": "SimpleQA",
    "popqa": "PopQA",
}
MODEL_ORDER: list[str] = ["qwen3_8b", "ministral_3_8b_instruct_2512", "gemma_4_e4b_it"]
DATASET_ORDER: list[str] = ["triviaqa", "ambigqa", "opennq", "ambiginst",
                             "simpleqa", "popqa"]
SKIP_DATASETS: set[str] = {"condambigqa"}
SKIP_NSAMPLES: set[int] = {50}  # smoke-test caches

# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

METHOD_ORDER: list[str] = ["CAELUM", "Vanilla", "Equal", "EpiOnly",
                            "ConU", "SE-CP"]


def _make_calibrator(method: str, alpha: float, seed: int = 42):
    """Instantiate the appropriate calibrator for a given method name."""
    if method == "CAELUM":
        return CAELUMCalibrator(alpha=alpha, seed=seed)
    if method == "Vanilla":
        return CAELUMCalibrator.vanilla_conformal(alpha=alpha)
    if method == "Equal":
        return CAELUMCalibrator.equal_weight(alpha=alpha)
    if method == "EpiOnly":
        return CAELUMCalibrator.epistemic_only(alpha=alpha)
    if method == "ConU":
        return ConUStyleCalibrator(alpha=alpha, seed=seed)
    if method == "SE-CP":
        return SemanticEntropyCalibrator(alpha=alpha, seed=seed)
    raise ValueError(f"Unknown method: {method!r}")


# ---------------------------------------------------------------------------
# Type-coercion helper (cached JSON sometimes has numpy/list mixes)
# ---------------------------------------------------------------------------

def _ensure_types(data: list[dict]) -> list[dict]:
    """Coerce JSON-deserialised fields into the types the calibrators expect."""
    for item in data:
        item["cluster_props"] = [float(p) for p in item["cluster_props"]]
        item["correct_cluster_idx"] = int(item["correct_cluster_idx"])
        item["U_ale"] = float(item["U_ale"])
        item["U_epi"] = float(item["U_epi"])
        item["U_total"] = float(item.get("U_total", item["U_ale"] + item["U_epi"]))
    return data


# ---------------------------------------------------------------------------
# Cell discovery
# ---------------------------------------------------------------------------

def discover_cells() -> list[dict]:
    """Discover all 18 (model, dataset) cells from cached uncertainty files.

    Returns:
        List of cell dicts with keys ``model``, ``dataset``, ``n_samples``,
        ``path``, sorted by (model, dataset) presentation order.
    """
    cells: list[dict] = []
    for f in sorted(GENERATIONS_DIR.glob("unc_*_vne_*.json")):
        m = UNC_RE.match(f.name)
        if not m:
            continue
        ds, model, n = m.group(1), m.group(2), int(m.group(3))
        if ds in SKIP_DATASETS:
            continue
        if n in SKIP_NSAMPLES:
            continue
        if model not in MODEL_PRETTY or ds not in DATASET_PRETTY:
            continue
        cells.append({
            "model": model,
            "dataset": ds,
            "n_samples": n,
            "path": str(f),
            "cell_key": f"{model}__{ds}",
        })

    # Sort by canonical order
    def _sort_key(c: dict) -> tuple[int, int]:
        return (
            MODEL_ORDER.index(c["model"]),
            DATASET_ORDER.index(c["dataset"]),
        )
    cells.sort(key=_sort_key)
    return cells


# ---------------------------------------------------------------------------
# Alpha sweep + coverage interpolation
# ---------------------------------------------------------------------------

ALPHA_SWEEP: tuple[float, ...] = (
    0.05, 0.07, 0.10, 0.13, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40,
)


def sweep_method(
    method: str,
    val1: list[dict],
    val2: list[dict],
    test: list[dict],
    alphas: tuple[float, ...] = ALPHA_SWEEP,
    seed: int = 42,
) -> list[dict]:
    """Sweep alpha for a method and return (alpha, coverage, apss) records."""
    out: list[dict] = []
    for alpha in alphas:
        cal = _make_calibrator(method, alpha, seed=seed)
        cal.calibrate(val1, val2, verbose=False)
        cov = compute_coverage(test, cal)
        apss = compute_apss(test, cal)
        out.append({
            "alpha": float(alpha),
            "coverage": float(cov),
            "apss": float(apss),
            "lambda_star": float(getattr(cal, "optimal_lambda", 0.0)),
            "tau_hat": float(getattr(cal, "tau_hat", 0.0)),
        })
    return out


def find_apss_at_coverage(
    sweep: list[dict], q_star: float, cov_tol: float = 0.005,
) -> dict:
    """Find the APSS at target coverage q* via piecewise-linear interpolation.

    Strategy
    --------
    * Sort sweep entries by coverage (ascending).
    * If q_star is bracketed by two sweep coverages, linearly interpolate
      both APSS and alpha between the bracketing entries (status="interpolated").
    * If q_star exceeds the maximum sweep coverage by more than cov_tol, the
      method **cannot reach q*** — this is a *coverage failure*. We report the
      best achievable coverage with status="undercovers"; the reported APSS
      should be interpreted alongside `coverage_shortfall`.
    * If q_star equals the max coverage within cov_tol, treat as the boundary
      (status="boundary"); pick the alpha among the candidates achieving max
      coverage that has the lowest APSS.

    The cov_tol margin (default 0.005, i.e. half a percentage point) absorbs
    the discrete jumps in coverage that arise from APS construction with small
    test sets (e.g. n=92 means coverage steps are ~0.011).

    Args:
        sweep: List of dicts produced by ``sweep_method``.
        q_star: Target coverage in [0, 1].
        cov_tol: Tolerance for declaring "boundary" matches.

    Returns:
        Dict with keys ``apss_at_q_star``, ``alpha_at_q_star``, ``cov_actual``,
        ``coverage_shortfall``, ``status`` (interpolated|boundary|undercovers|
        extrapolated_low), ``lambda_star``, ``tau_hat``.
    """
    sorted_sweep = sorted(sweep, key=lambda e: e["coverage"])
    covs = np.array([e["coverage"] for e in sorted_sweep])
    apss = np.array([e["apss"] for e in sorted_sweep])
    alphas = np.array([e["alpha"] for e in sorted_sweep])
    lams = np.array([e["lambda_star"] for e in sorted_sweep])
    taus = np.array([e["tau_hat"] for e in sorted_sweep])

    cov_min, cov_max = float(covs[0]), float(covs[-1])
    shortfall = max(0.0, q_star - cov_max)

    # Case 1: q* below sweep range -> any alpha gives over-coverage.
    # Pick the alpha achieving cov closest to q* with lowest APSS.
    if q_star <= cov_min - cov_tol:
        # Among entries with cov >= q*, pick the smallest-APSS.
        # (All entries have cov >= q* in this case.)
        best_idx = int(np.argmin(apss))
        return {
            "apss_at_q_star": float(apss[best_idx]),
            "alpha_at_q_star": float(alphas[best_idx]),
            "cov_actual": float(covs[best_idx]),
            "coverage_shortfall": 0.0,
            "lambda_star": float(lams[best_idx]),
            "tau_hat": float(taus[best_idx]),
            "status": "extrapolated_low",
        }

    # Case 2: q* matches the sweep boundary (within cov_tol).
    if abs(q_star - cov_max) <= cov_tol:
        # Among entries achieving the max coverage (within tol), pick the
        # smallest-APSS one. The user wants the EFFICIENT alpha for q*.
        max_mask = covs >= cov_max - cov_tol
        candidate_apss = apss[max_mask]
        candidate_idx = np.where(max_mask)[0][int(np.argmin(candidate_apss))]
        return {
            "apss_at_q_star": float(apss[candidate_idx]),
            "alpha_at_q_star": float(alphas[candidate_idx]),
            "cov_actual": float(covs[candidate_idx]),
            "coverage_shortfall": 0.0,
            "lambda_star": float(lams[candidate_idx]),
            "tau_hat": float(taus[candidate_idx]),
            "status": "boundary",
        }

    # Case 3: q* truly above sweep range -> COVERAGE FAILURE.
    # Report the APSS at the max coverage (best effort).
    if q_star > cov_max:
        max_mask = covs >= cov_max - 1e-9
        candidate_apss = apss[max_mask]
        candidate_idx = np.where(max_mask)[0][int(np.argmin(candidate_apss))]
        return {
            "apss_at_q_star": float(apss[candidate_idx]),
            "alpha_at_q_star": float(alphas[candidate_idx]),
            "cov_actual": float(covs[candidate_idx]),
            "coverage_shortfall": float(shortfall),
            "lambda_star": float(lams[candidate_idx]),
            "tau_hat": float(taus[candidate_idx]),
            "status": "undercovers",
        }

    # Case 4: q* in interior of sweep range -> linear interpolation.
    uniq_cov, inv = np.unique(covs, return_inverse=True)
    if len(uniq_cov) < len(covs):
        # Coverage ties: pick the smallest APSS at each unique coverage
        # (this is the EFFICIENT alpha for that coverage, matching CLEAR's
        # idea of post-hoc rescaling to a target coverage).
        agg_apss = np.array([apss[inv == i].min() for i in range(len(uniq_cov))])
        # For the alpha/lambda/tau, use the entry that achieved that min APSS.
        agg_alpha = np.zeros(len(uniq_cov))
        agg_lam = np.zeros(len(uniq_cov))
        agg_tau = np.zeros(len(uniq_cov))
        for i in range(len(uniq_cov)):
            sub = np.where(inv == i)[0]
            best_sub = sub[int(np.argmin(apss[sub]))]
            agg_alpha[i] = alphas[best_sub]
            agg_lam[i] = lams[best_sub]
            agg_tau[i] = taus[best_sub]
    else:
        agg_apss, agg_alpha = apss, alphas
        agg_lam, agg_tau = lams, taus

    apss_q = float(np.interp(q_star, uniq_cov, agg_apss))
    alpha_q = float(np.interp(q_star, uniq_cov, agg_alpha))
    lam_q = float(np.interp(q_star, uniq_cov, agg_lam))
    tau_q = float(np.interp(q_star, uniq_cov, agg_tau))

    return {
        "apss_at_q_star": apss_q,
        "alpha_at_q_star": alpha_q,
        "cov_actual": float(q_star),
        "coverage_shortfall": 0.0,
        "lambda_star": lam_q,
        "tau_hat": tau_q,
        "status": "interpolated",
    }


# ---------------------------------------------------------------------------
# Per-cell evaluation
# ---------------------------------------------------------------------------

def evaluate_cell(cell: dict, alpha_nominal: float = 0.1, seed: int = 42) -> dict:
    """Run the full NCAPSS sweep for a single cell.

    Args:
        cell: dict with ``path``, ``model``, ``dataset``, ``n_samples``.
        alpha_nominal: nominal miscoverage. The matched-coverage target is
            ``q_star = min(1 - alpha_nominal, validation_match_rate)``.
        seed: random seed.

    Returns:
        Dict with per-method NCAPSS records and metadata.
    """
    data = _ensure_types(load_json(cell["path"]))
    n_total = len(data)
    n_matched = sum(1 for d in data if d["correct_cluster_idx"] >= 0)
    match_rate = n_matched / n_total if n_total else 0.0

    val1, val2, test = split_data(data, val1_frac=0.4, val2_frac=0.3, seed=seed)

    # Use VALIDATION (val1+val2) match rate, NOT test, to set q*.  This
    # decouples the matched-coverage TARGET from the evaluation set,
    # eliminating the selection-bias contamination of an earlier version
    # that used `test_match_rate` directly.  Test match rate is reported
    # only for diagnostic comparison.
    val_pool = list(val1) + list(val2)
    val_matched = sum(1 for d in val_pool if d["correct_cluster_idx"] >= 0)
    val_match_rate = val_matched / len(val_pool) if len(val_pool) else 0.0

    test_matched = sum(1 for d in test if d["correct_cluster_idx"] >= 0)
    test_match_rate = test_matched / len(test) if len(test) else 0.0

    # Operationally meaningful target: the achievable ceiling per cell,
    # estimated on the calibration data only (no test leakage).
    q_star = float(min(1.0 - alpha_nominal, val_match_rate))

    logger.info(
        "  cell %s | n=%d match=%.3f val_match=%.3f test_match=%.3f q*=%.3f",
        cell["cell_key"], n_total, match_rate, val_match_rate,
        test_match_rate, q_star,
    )

    methods_out: dict[str, dict] = {}
    for method in METHOD_ORDER:
        t0 = time.time()
        sweep = sweep_method(method, val1, val2, test, ALPHA_SWEEP, seed=seed)
        rec = find_apss_at_coverage(sweep, q_star)
        rec["sweep"] = sweep
        elapsed = time.time() - t0
        flag = (
            f" SHORTFALL={rec['coverage_shortfall']:.3f}"
            if rec["status"] == "undercovers" else ""
        )
        logger.info(
            "    %-8s  q*=%.3f -> APSS=%.3f (alpha=%.3f, %s)%s [%.1fs]",
            method, q_star, rec["apss_at_q_star"], rec["alpha_at_q_star"],
            rec["status"], flag, elapsed,
        )
        methods_out[method] = rec

    return {
        "model": cell["model"],
        "dataset": cell["dataset"],
        "n_samples": n_total,
        "match_rate": float(match_rate),
        "test_match_rate": float(test_match_rate),
        "target_q_star": q_star,
        "alpha_nominal": float(alpha_nominal),
        "methods": methods_out,
    }


# ---------------------------------------------------------------------------
# Aggregate normalised statistics
# ---------------------------------------------------------------------------

def compute_normalized_stats(all_results: dict[str, dict]) -> dict[str, dict]:
    """Compute NCAPSS / NCAPSS_CAELUM across cells per method.

    Two ratio sets are reported per method:
      * ``ratios_all``: every cell, regardless of whether the method reaches
        q*. The "win" here is contaminated by under-coverage on shortfall cells.
      * ``ratios_matched``: only cells where BOTH methods (the method itself
        and CAELUM) reach the validation-derived q* without under-coverage.
        This is the apples-to-apples NCAPSS diagnostic; cells that cannot reach
        q* are deliberately excluded rather than forced to count as matched.

    For the matched subset we also count cells where the method is strictly
    smaller than CAELUM (n_better_matched). This is the headline narrative
    number for the paper.

    Returns:
        Dict mapping method -> dict with keys:
          - ``ratios_all``, ``mean_all``, ``median_all``
          - ``ratios_matched``, ``mean_matched``, ``median_matched``
          - ``n_better_all``, ``n_worse_all``, ``n_tied_all``, ``n_cells_all``
          - ``n_better_matched``, ``n_worse_matched``, ``n_tied_matched``,
            ``n_cells_matched``
          - ``n_undercovers``, ``mean_shortfall``, ``max_shortfall``
    """
    cell_keys = list(all_results.keys())
    stats: dict[str, dict] = {}

    for m in METHOD_ORDER:
        ratios_all: list[float] = []
        ratios_matched: list[float] = []
        shortfalls: list[float] = []

        for ck in cell_keys:
            cell = all_results[ck]
            caelum_rec = cell["methods"]["CAELUM"]
            method_rec = cell["methods"][m]
            ncapss_caelum = caelum_rec["apss_at_q_star"]
            ncapss_method = method_rec["apss_at_q_star"]
            if ncapss_caelum <= 1e-9:
                continue
            ratio = ncapss_method / ncapss_caelum
            ratios_all.append(ratio)
            method_undercovers = method_rec["status"] == "undercovers"
            caelum_undercovers = caelum_rec["status"] == "undercovers"
            shortfalls.append(method_rec["coverage_shortfall"])
            # Matched subset: BOTH the method and the reference reach q*.
            if not method_undercovers and not caelum_undercovers:
                ratios_matched.append(ratio)

        ra = np.array(ratios_all)
        rm = np.array(ratios_matched)
        sf = np.array(shortfalls)

        def _better_worse_tied(r):
            n_b = int(np.sum(r < 1.0 - 1e-6))
            n_w = int(np.sum(r > 1.0 + 1e-6))
            return n_b, n_w, int(len(r) - n_b - n_w)

        n_b_all, n_w_all, n_t_all = _better_worse_tied(ra)
        n_b_m, n_w_m, n_t_m = _better_worse_tied(rm)
        n_undercov = int(np.sum(sf > 1e-6))

        stats[m] = {
            "ratios_all": ra.tolist(),
            "ratios_matched": rm.tolist(),
            "mean_all": float(ra.mean()) if len(ra) else float("nan"),
            "median_all": float(np.median(ra)) if len(ra) else float("nan"),
            "mean_matched": float(rm.mean()) if len(rm) else float("nan"),
            "median_matched": float(np.median(rm)) if len(rm) else float("nan"),
            "n_better_all": n_b_all,
            "n_worse_all": n_w_all,
            "n_tied_all": n_t_all,
            "n_cells_all": int(len(ra)),
            "n_better_matched": n_b_m,
            "n_worse_matched": n_w_m,
            "n_tied_matched": n_t_m,
            "n_cells_matched": int(len(rm)),
            "n_undercovers": n_undercov,
            "mean_shortfall": float(sf.mean()) if len(sf) else float("nan"),
            "max_shortfall": float(sf.max()) if len(sf) else 0.0,
        }
    return stats


# ---------------------------------------------------------------------------
# Outputs: markdown summary, LaTeX table, plot
# ---------------------------------------------------------------------------

def write_markdown(
    all_results: dict[str, dict],
    stats: dict[str, dict],
    out_path: Path,
) -> None:
    """Write the human-readable summary markdown.

    Includes (a) per-cell raw NCAPSS with under-coverage flags, (b) the
    aggregate normalized table (all-cells and matched-cells variants), and
    (c) the headline CAELUM-vs-ConU narrative respecting coverage shortfalls.
    """
    lines: list[str] = []
    lines.append("# NCAPSS - Normalized Calibrated APSS\n")
    lines.append(
        "*Per-method APSS at matched coverage "
        "`q* = min(1-alpha, validation_match_rate)`. Lower is better. "
        "CAELUM is the reference. The target is validation-derived, but APSS "
        "is interpolated on each method's test coverage sweep, so this is a "
        "post-hoc diagnostic. Cells where a method "
        "**under-covers** q* (cannot reach the matched coverage even at "
        "alpha=0.4) are flagged with `(-X.Xpp)` showing the coverage gap.*\n",
    )
    lines.append("## Per-cell raw NCAPSS\n")

    header = "| Model | Dataset | q* | " + " | ".join(METHOD_ORDER) + " |"
    sep = "|" + "---|" * (3 + len(METHOD_ORDER))
    lines.append(header)
    lines.append(sep)
    for ck, cell in all_results.items():
        cells_str: list[str] = []
        # Identify the best APSS among methods that ACTUALLY reach q*
        valid_apss: list[tuple[int, float]] = []
        for i, m in enumerate(METHOD_ORDER):
            rec = cell["methods"][m]
            if rec["status"] != "undercovers":
                valid_apss.append((i, rec["apss_at_q_star"]))
        best_valid_idx: int | None = None
        if valid_apss:
            best_valid_idx = min(valid_apss, key=lambda kv: kv[1])[0]

        for i, m in enumerate(METHOD_ORDER):
            rec = cell["methods"][m]
            v = rec["apss_at_q_star"]
            if rec["status"] == "undercovers":
                gap_pp = 100 * rec["coverage_shortfall"]
                tag = f"{v:.3f} (-{gap_pp:.1f}pp)"
            else:
                tag = f"{v:.3f}"
                if best_valid_idx is not None and i == best_valid_idx:
                    tag = f"**{tag}**"
            cells_str.append(tag)
        lines.append(
            "| {model} | {dataset} | {q:.3f} | {vals} |".format(
                model=MODEL_PRETTY[cell["model"]],
                dataset=DATASET_PRETTY[cell["dataset"]],
                q=cell["target_q_star"],
                vals=" | ".join(cells_str),
            )
        )

    lines.append(
        "\n*Note: bold = smallest APSS **among methods that reach q\\***. "
        "ConU's small APSS values frequently come at the cost of under-coverage.*\n"
    )

    lines.append("\n## Aggregate normalized NCAPSS (vs CAELUM)\n")
    lines.append(
        "Two aggregations are reported. **All-cells** uses every cell, regardless "
        "of whether the method reaches q*; this is the optimistic view for a "
        "method that systematically under-covers (since smaller-and-under-covering "
        "is recorded as a 'win'). **Matched-cells** restricts to cells where the "
        "method actually achieves the validation-derived target coverage q*.\n"
    )
    lines.append(
        "| Method | Mean (all) | Median (all) | # better/cells (all) | "
        "Mean (matched) | Median (matched) | # better/cells (matched) | "
        "# under-covers | mean shortfall |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for m in METHOD_ORDER:
        s = stats[m]
        lines.append(
            f"| {m} "
            f"| {s['mean_all']:.3f} | {s['median_all']:.3f} "
            f"| {s['n_better_all']}/{s['n_cells_all']} "
            f"| {s['mean_matched']:.3f} | {s['median_matched']:.3f} "
            f"| {s['n_better_matched']}/{s['n_cells_matched']} "
            f"| {s['n_undercovers']} | {s['mean_shortfall']:.3f} |"
        )

    if "ConU" in stats and "CAELUM" in stats:
        sc = stats["ConU"]
        lines.append("\n## CAELUM vs ConU at matched coverage\n")
        lines.append(
            f"- ConU **under-covers** q* on {sc['n_undercovers']}/"
            f"{sc['n_cells_all']} cells "
            f"(mean shortfall {sc['mean_shortfall']:.3f}, "
            f"max {sc['max_shortfall']:.3f}). On these cells ConU is "
            f"reporting APSS at a strictly smaller coverage than the matched "
            f"target, so its raw APSS advantage is partly an under-coverage "
            f"artefact.\n"
            f"- Across the {sc['n_cells_matched']}/{sc['n_cells_all']} cells "
            f"where ConU **does** reach q*, CAELUM is smaller than ConU on "
            f"{sc['n_worse_matched']} cells, ConU is smaller on "
            f"{sc['n_better_matched']} cells, with mean ratio "
            f"NCAPSS_ConU / NCAPSS_CAELUM = {sc['mean_matched']:.3f}.\n"
            f"- Across **all** {sc['n_cells_all']} cells (ignoring shortfalls), "
            f"the all-cell mean ratio is {sc['mean_all']:.3f}: ConU's apparent "
            f"win shrinks substantially once coverage is held fixed."
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote markdown summary to %s", out_path)


def write_latex(
    all_results: dict[str, dict],
    stats: dict[str, dict],
    out_path: Path,
) -> None:
    """Write the auto-generated LaTeX tabular for the paper appendix.

    Include the fragment inside a LaTeX table environment with a caption and label.

    Layout: 18 rows x (model, dataset, q*, 6 method APSS values).
    Cells where a method UNDER-COVERS q* are typeset with a small
    ``$^{\\dagger}$`` marker; bold marks the smallest APSS *among methods that
    reach q\\**.
    """
    lines: list[str] = []
    lines.append(
        "% AUTO-GENERATED by src/run_ncapss_full.py from JSON results.\n"
        "% Do not edit by hand - re-run the script after updating results.\n"
    )
    col_spec = "ll" + "r" + "r" * len(METHOD_ORDER)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    method_hdr = " & ".join(
        ["\\textbf{Model}", "\\textbf{Dataset}", "$q^\\star$"]
        + [f"\\textbf{{{m}}}" for m in METHOD_ORDER]
    )
    lines.append(method_hdr + " \\\\")
    lines.append("\\midrule")

    cells_for_model: dict[str, list[dict]] = {}
    for ck, cell in all_results.items():
        cells_for_model.setdefault(cell["model"], []).append(cell)

    for model_idx, model in enumerate(MODEL_ORDER):
        if model not in cells_for_model:
            continue
        group = cells_for_model[model]
        n_rows = len(group)
        for row_idx, cell in enumerate(group):
            apss_vals = [cell["methods"][m]["apss_at_q_star"] for m in METHOD_ORDER]
            statuses = [cell["methods"][m]["status"] for m in METHOD_ORDER]

            # Only consider methods that actually reach q* for "best" highlight
            valid_pairs = [
                (i, v) for i, (v, s) in enumerate(zip(apss_vals, statuses))
                if s != "undercovers"
            ]
            best_idx: int | None = None
            second_idx: int | None = None
            if valid_pairs:
                valid_sorted = sorted(valid_pairs, key=lambda kv: kv[1])
                best_idx = valid_sorted[0][0]
                if len(valid_sorted) >= 2 and valid_sorted[1][1] > valid_sorted[0][1] + 1e-6:
                    second_idx = valid_sorted[1][0]

            cells_str: list[str] = []
            for i, (v, s) in enumerate(zip(apss_vals, statuses)):
                if s == "undercovers":
                    cells_str.append(f"{v:.2f}$^{{\\dagger}}$")
                elif i == best_idx:
                    cells_str.append(f"\\textbf{{{v:.2f}}}")
                elif i == second_idx:
                    cells_str.append(f"\\underline{{{v:.2f}}}")
                else:
                    cells_str.append(f"{v:.2f}")

            model_cell = (
                f"\\multirow{{{n_rows}}}{{*}}{{{MODEL_PRETTY[model]}}}"
                if row_idx == 0 else ""
            )
            lines.append(
                f" {model_cell} & {DATASET_PRETTY[cell['dataset']]} "
                f"& {cell['target_q_star']:.2f} & "
                + " & ".join(cells_str)
                + " \\\\"
            )
        if model_idx < len(MODEL_ORDER) - 1 and any(
            other_model in cells_for_model
            for other_model in MODEL_ORDER[model_idx + 1:]
        ):
            lines.append("\\midrule")

    # Aggregate rows
    lines.append("\\midrule")
    mean_all_row = (
        ["\\multicolumn{2}{l}{Mean ratio vs CAELUM (all cells)}", "--"]
        + [f"{stats[m]['mean_all']:.2f}" for m in METHOD_ORDER]
    )
    lines.append(" & ".join(mean_all_row) + " \\\\")
    mean_match_row = (
        ["\\multicolumn{2}{l}{Mean ratio vs CAELUM (matched cells)}", "--"]
        + [f"{stats[m]['mean_matched']:.2f}" for m in METHOD_ORDER]
    )
    lines.append(" & ".join(mean_match_row) + " \\\\")
    wins_row = (
        ["\\multicolumn{2}{l}{\\# cells smaller than CAELUM (matched)}", "--"]
        + [f"{stats[m]['n_better_matched']}/{stats[m]['n_cells_matched']}"
           for m in METHOD_ORDER]
    )
    lines.append(" & ".join(wins_row) + " \\\\")
    undercov_row = (
        ["\\multicolumn{2}{l}{\\# cells under-covering $q^\\star$}", "--"]
        + [f"{stats[m]['n_undercovers']}/{stats[m]['n_cells_all']}"
           for m in METHOD_ORDER]
    )
    lines.append(" & ".join(undercov_row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote LaTeX tabular to %s", out_path)


def make_boxplot(
    stats: dict[str, dict],
    pdf_path: Path,
    png_path: Path,
) -> None:
    """Create the box-plot of NCAPSS / NCAPSS_CAELUM ratios across cells.

    Two side-by-side panels:
      Left:  All 18 cells per method (apparent improvement, contaminated by
             under-coverage on shortfall cells).
      Right: Matched-cells subset (the apples-to-apples NCAPSS comparison).
             Methods that under-cover q* on every cell are absent from this
             panel by construction (n_cells_matched=0).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from caelum.plotting import apply_paper_style
    apply_paper_style()

    methods = METHOD_ORDER
    method_colors = {
        "CAELUM":  "#2E86C1",
        "Vanilla": "#E74C3C",
        "Equal":   "#27AE60",
        "EpiOnly": "#8E44AD",
        "ConU":    "#F39C12",
        "SE-CP":   "#16A085",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    panel_specs = [
        (axes[0], "ratios_all", "n_cells_all",
         "All 18 cells (apparent)"),
        (axes[1], "ratios_matched", "n_cells_matched",
         "Cells where method reaches $q^\\star$ (apples-to-apples)"),
    ]

    for ax, key, n_key, title in panel_specs:
        data = [stats[m][key] for m in methods]
        n_per = [stats[m][n_key] for m in methods]
        # Replace empty datasets with [nan] so the box is drawn as empty
        plot_data = [d if d else [np.nan] for d in data]

        # MPL >= 3.9 renamed `labels` -> `tick_labels`. Pass via positional.
        box = ax.boxplot(
            plot_data, patch_artist=True, widths=0.55, showmeans=True,
            tick_labels=methods,
            meanprops=dict(marker="D", markerfacecolor="white",
                           markeredgecolor="black", markersize=6),
            medianprops=dict(color="black", linewidth=1.5),
        )
        for patch, m in zip(box["boxes"], methods):
            patch.set_facecolor(method_colors.get(m, "lightgray"))
            patch.set_alpha(0.75)

        rng = np.random.RandomState(0)
        for i, vals in enumerate(data, start=1):
            if not vals:
                continue
            x = rng.uniform(-0.12, 0.12, size=len(vals)) + i
            ax.scatter(x, vals, color="black", s=12, alpha=0.55, zorder=3)

        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1,
                   alpha=0.7, label="Parity with CAELUM")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3, which="both")
        # Annotate sample size below each box
        for i, n in enumerate(n_per, start=1):
            ax.text(i, 0.05, f"n={n}", ha="center", va="bottom", fontsize=8,
                    transform=ax.get_xaxis_transform())

    axes[0].set_ylabel("NCAPSS ratio (method / CAELUM)", fontsize=11)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(
        "Matched-coverage APSS (NCAPSS), normalized by CAELUM\n"
        "(lower = smaller prediction sets at the same coverage)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved NCAPSS box-plot to %s and %s", pdf_path, png_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run NCAPSS evaluation across all 18 cells."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace the existing NCAPSS results JSON.")
    parser.add_argument("--write-markdown", action="store_true",
                        help="Also write the Markdown report.")
    args = parser.parse_args()
    # ncapss_results.json is a frozen input of the accepted figures and tables
    if OUT_JSON.exists() and not args.overwrite:
        raise SystemExit(f"{OUT_JSON} already exists and backs reported results; "
                         "pass --overwrite to replace it.")
    PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    logger.info("=" * 72)
    logger.info("NCAPSS — Normalized Calibrated APSS evaluation")
    logger.info("=" * 72)

    cells = discover_cells()
    logger.info("Discovered %d cells:", len(cells))
    for c in cells:
        logger.info("  %s | %s | n=%d", c["model"], c["dataset"], c["n_samples"])

    all_results: dict[str, dict] = {}
    for i, cell in enumerate(cells, start=1):
        logger.info("-" * 72)
        logger.info("[%d/%d] %s", i, len(cells), cell["cell_key"])
        all_results[cell["cell_key"]] = evaluate_cell(cell)

    logger.info("=" * 72)
    logger.info("Computing aggregate normalised stats")
    stats = compute_normalized_stats(all_results)
    logger.info(
        "  %-8s  %8s  %8s  %12s  %8s  %12s  %8s",
        "method", "mean_all", "med_all", "wins_all/n",
        "mean_match", "wins_match/n", "undercov",
    )
    for m in METHOD_ORDER:
        s = stats[m]
        logger.info(
            "  %-8s  %8.3f  %8.3f  %5d/%-6d  %8.3f  %5d/%-6d  %5d/%-2d",
            m, s["mean_all"], s["median_all"],
            s["n_better_all"], s["n_cells_all"],
            s["mean_matched"],
            s["n_better_matched"], s["n_cells_matched"],
            s["n_undercovers"], s["n_cells_all"],
        )

    payload = {
        "alpha_nominal": 0.1,
        "alpha_sweep": list(ALPHA_SWEEP),
        "method_order": METHOD_ORDER,
        "cells": all_results,
        "stats": stats,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote JSON to %s", OUT_JSON)

    if args.write_markdown:
        write_markdown(all_results, stats, OUT_MD)
    write_latex(all_results, stats, OUT_TEX)
    make_boxplot(stats, OUT_PDF, OUT_PNG)

    logger.info("=" * 72)
    logger.info("Total elapsed: %.1f s", time.time() - t0)


if __name__ == "__main__":
    main()
