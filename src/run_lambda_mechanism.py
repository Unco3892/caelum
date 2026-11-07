#!/usr/bin/env python3
"""Quantify WHY lambda-weighting is inert on discrete cluster sets.

Negative result #1 of the paper says CAELUM's two-parameter weight
w(x) = 1 + U_ale + lambda * U_epi does not change APSS relative to Vanilla
(lambda = 0). The stated mechanism is qualitative: "the calibrated threshold
lands in the same gap between cumulative cluster masses regardless of the
weight". This script makes that quantitative, using the exact protocol of
``run_headline_multiseed.py`` (the script behind tab:multiseed-apss):
40/30/30 split, additive weight, full-V2 calibration, randomized APS,
alpha = 0.1, seeds 42-45.

For each (cell, seed) it calibrates ``CAELUMCalibrator`` at a fixed lambda
sweep (lambda pinned via a singleton lambda_grid, so tau_hat is recalibrated
at every lambda exactly as the real pipeline would) and measures, per test
item:

  * the effective threshold  tau_eff(x) = tau_hat_lambda * w_lambda(x);
  * the slack to the nearest set-size boundary (the cumulative cluster
    masses C_k = p_1 + ... + p_k), i.e. how far tau_eff would have to move
    to add or remove one cluster;
  * the maximum displacement of tau_eff(x) achievable over the whole sweep
    (0 <= lambda <= 1e6, far beyond the search grid's max of 100);
  * the smallest lambda at which the item's set size differs from lambda=0
    (inf if no lambda in the sweep changes it).

Outputs ``results/lambda_mechanism.json`` (and a Markdown report with
``--write-markdown``). Read-only with respect to all cached results.

Usage:
    python -u src/run_lambda_mechanism.py
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
from caelum.paths import JUDGED_DIR, RESULTS_DIR  # noqa: E402

from caelum.calibration import CAELUMCalibrator  # noqa: E402
from caelum.utils import split_data  # noqa: E402

from run_headline_multiseed import (  # noqa: E402
    ALPHA, SEEDS, UNC_RE, SKIP_DATASETS, MODEL_PRETTY, DATASET_PRETTY,
    _ensure_types, discover_cells,
)


# Two calibrator configurations, closing the "maybe lambda works elsewhere"
# loophole:
#   headline -- the tab:multiseed-apss protocol: 40/30/30 split, additive
#       weight, full-V2 threshold (support failures included in the quantile).
#   abstain  -- the abstention pipeline's base calibrator: 25/25/25/25 split
#       (V1 for lambda, V2a for tau), standardized weight, set_score
#       selection, support-conditional tau on the matched stratum. Here the
#       threshold is NOT saturated by the s=1 failure block a priori.
CONFIGS = {
    "headline": dict(randomize=True),
    "abstain": dict(randomize=True, weight_fn="standardized",
                    selection_method="set_score", support_conditional=True),
}

# Fixed-lambda sweep. The tuned grid tops out at 100; values beyond that
# answer "what lambda WOULD be needed", up to the epistemic-only limit.
LAMBDA_SWEEP = [
    0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0,
    300.0, 1e3, 3e3, 1e4, 1e5, 1e6,
]


def _setup_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S", force=True)
    return logging.getLogger("lambda_mechanism")


def fixed_lambda_calibrator(
    lam: float, seed: int, config: str = "headline",
) -> CAELUMCalibrator:
    """The chosen calibrator config with lambda pinned to a single value."""
    return CAELUMCalibrator(
        alpha=ALPHA, lambda_grid=np.array([lam]), seed=seed,
        **CONFIGS[config],
    )


def config_split(
    data: list[dict], seed: int, config: str,
) -> tuple[list[dict], list[dict], list[dict]]:
    """(lambda-selection split, threshold split, test split) per config."""
    if config == "headline":
        return split_data(data, val1_frac=0.4, val2_frac=0.3, seed=seed)
    # abstain: the 25/25/25/25 split of run_caelum_abstain (V0 unused here);
    # tau is set on V2a, the first half of V2, exactly as in the pipeline.
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(data))
    q = len(data) // 4
    v1 = [data[i] for i in perm[q:2 * q]]
    v2 = perm[2 * q:3 * q]
    v2a = [data[i] for i in v2[:len(v2) // 2]]
    test = [data[i] for i in perm[3 * q:]]
    return v1, v2a, test


def discover_judged_cells() -> list[dict]:
    """The judged 30-cell grid (skips condambigqa; largest n per cell)."""
    best: dict[str, dict] = {}
    for f in sorted(JUDGED_DIR.glob("unc_*_vne_*.json")):
        m = UNC_RE.match(f.name)
        if not m:
            continue
        ds, model, n = m.group(1), m.group(2), int(m.group(3))
        if ds in SKIP_DATASETS or model not in MODEL_PRETTY \
                or ds not in DATASET_PRETTY:
            continue
        key = f"{model}__{ds}"
        if key not in best or n > best[key]["n_samples"]:
            best[key] = {"model": model, "dataset": ds, "n_samples": n,
                         "path": str(f), "cell_key": key}
    return list(best.values())


def effective_threshold(cal: CAELUMCalibrator, item: dict) -> float:
    """tau_eff(x) = tau_hat * w_lambda(x), the per-item inclusion threshold."""
    w = cal._compute_weights_vec(
        np.asarray([item["U_ale"]]), np.asarray([item["U_epi"]]),
        cal.optimal_lambda,
    )[0]
    return float(cal.tau_hat * w)


def boundaries(props: np.ndarray) -> np.ndarray:
    """Cumulative cluster masses C_1..C_K; the set has size m iff
    tau_eff in [C_{m-1}, C_m) with C_0 = 0 (size K when tau_eff >= C_{K-1})."""
    return np.cumsum(np.asarray(props, dtype=float))


def analyze_cell(data: list[dict], seeds: list[int], logger,
                 config: str = "headline") -> dict:
    per_seed_rows = []
    for seed in seeds:
        val1, val2, test = config_split(data, seed, config)
        # Calibrate once per sweep value (tau_hat recalibrated at each
        # lambda, exactly as the pipeline's grid search would).
        cals = {lam: fixed_lambda_calibrator(lam, seed, config).calibrate(
            val1, val2, verbose=False) for lam in LAMBDA_SWEEP}

        n_test = len(test)
        sizes = {lam: np.array([c.predict_set_size(it) for it in test])
                 for lam, c in cals.items()}
        taus = {lam: np.array([effective_threshold(c, it) for it in test])
                for lam, c in cals.items()}

        size0 = sizes[0.0]
        tau0 = taus[0.0]

        # --- per-item slack to the nearest boundary at lambda = 0 ---
        # Set construction includes cluster k iff cumulative_before(k) <=
        # tau_eff, so size m occupies the gap [C_{m-1}, C_m) between
        # cumulative masses C_j = p_1 + ... + p_j (C_0 = 0). Growing to m+1
        # requires tau_eff >= C_m; shrinking to m-1 requires tau_eff < C_{m-1}.
        slack_up = np.full(n_test, np.inf)      # mass to ADD one cluster
        slack_down = np.full(n_test, np.inf)    # mass to DROP one cluster
        gap_width = np.full(n_test, np.inf)     # width of the occupied gap
        k_clusters = np.zeros(n_test, dtype=int)
        for i, it in enumerate(test):
            props = np.asarray(it["cluster_props"], dtype=float)
            k = len(props)
            k_clusters[i] = k
            cum = boundaries(props)  # cum[j] = C_{j+1}
            m = int(size0[i])
            if m < k:
                slack_up[i] = cum[m - 1] - tau0[i]
                gap_width[i] = float(props[m - 1])
            if m > 1:
                slack_down[i] = tau0[i] - cum[m - 2]
        slack_nearest = np.minimum(slack_up, slack_down)

        # --- achievable threshold displacement over the sweep ---
        tau_stack = np.stack([taus[lam] for lam in LAMBDA_SWEEP])  # (L, n)
        max_shift = np.max(np.abs(tau_stack - tau0[None, :]), axis=0)
        shift_at_100 = np.max(
            np.abs(np.stack([taus[lam] for lam in LAMBDA_SWEEP
                             if lam <= 100.0]) - tau0[None, :]), axis=0)
        # Signed shifts: what lambda can achieve in each direction.
        shift_up = np.max(tau_stack - tau0[None, :], axis=0)
        shift_down = np.max(tau0[None, :] - tau_stack, axis=0)

        # --- smallest lambda changing each item's set size ---
        min_lambda_change = np.full(n_test, np.inf)
        for lam in LAMBDA_SWEEP[1:]:
            newly = (sizes[lam] != size0) & np.isinf(min_lambda_change)
            min_lambda_change[newly] = lam
        any_change = np.isfinite(min_lambda_change)
        change_within_grid = min_lambda_change <= 100.0

        # --- APSS / coverage / change-fraction across the sweep ---
        apss_by_lambda = {lam: float(sizes[lam].mean()) for lam in LAMBDA_SWEEP}
        cidx = np.array(
            [int(it.get("correct_cluster_idx", -1)) for it in test])
        matched = cidx >= 0
        cov_by_lambda = {
            lam: float(((cidx >= 0) & (cidx < sizes[lam])).mean())
            for lam in LAMBDA_SWEEP}
        cov_matched_by_lambda = {
            lam: float((cidx[matched] < sizes[lam][matched]).mean())
            if matched.any() else float("nan") for lam in LAMBDA_SWEEP}
        apss_matched_by_lambda = {
            lam: float(sizes[lam][matched].mean())
            if matched.any() else float("nan") for lam in LAMBDA_SWEEP}
        frac_diff_by_lambda = {
            lam: float((sizes[lam] != size0).mean()) for lam in LAMBDA_SWEEP}
        frac_diff_matched_by_lambda = {
            lam: float((sizes[lam][matched] != size0[matched]).mean())
            if matched.any() else float("nan") for lam in LAMBDA_SWEEP}

        # --- full-grid lambda* under the same config (what the pipeline's
        # own grid search would pick, and whether it moves anything) ---
        cal_star = CAELUMCalibrator(
            alpha=ALPHA, seed=seed, **CONFIGS[config],
        ).calibrate(val1, val2, verbose=False)
        sizes_star = np.array([cal_star.predict_set_size(it) for it in test])
        star_row = {
            "lambda_star_grid": float(cal_star.optimal_lambda),
            "apss_star": float(sizes_star.mean()),
            "cov_star": float(((cidx >= 0) & (cidx < sizes_star)).mean()),
            "frac_diff_star_vs_0": float((sizes_star != size0).mean()),
            "apss_matched_star": float(sizes_star[matched].mean())
            if matched.any() else float("nan"),
            "cov_matched_star": float(
                (cidx[matched] < sizes_star[matched]).mean())
            if matched.any() else float("nan"),
        }

        multi = k_clusters > 1
        full_set = size0 == k_clusters
        resizable = multi & ~full_set  # a boundary exists in each direction
        # Directional feasibility: lambda changes this item's size iff the
        # achievable signed shift crosses the corresponding boundary.
        crossable = ((shift_up >= slack_up) | (shift_down > slack_down))
        finite_gaps = gap_width[np.isfinite(gap_width)]
        per_seed_rows.append({
            "seed": seed,
            "n_test": n_test,
            "frac_single_cluster": float((~multi).mean()),
            "frac_full_set": float(full_set.mean()),
            "frac_full_set_multi": float(full_set[multi].mean())
            if multi.any() else float("nan"),
            "frac_resizable": float(resizable.mean()),
            "frac_crossable": float(crossable.mean()),
            "median_slack_nearest": float(np.median(slack_nearest[multi]))
            if multi.any() else float("nan"),
            "median_slack_down_full": float(
                np.median(slack_down[multi & full_set]))
            if (multi & full_set).any() else float("nan"),
            "median_gap_width_finite": float(np.median(finite_gaps))
            if len(finite_gaps) else float("nan"),
            "frac_gap_infinite": float(np.isinf(gap_width).mean()),
            "median_max_shift": float(np.median(max_shift)),
            "median_shift_at_100": float(np.median(shift_at_100)),
            "frac_shift_below_slack": float(
                np.mean(max_shift < slack_nearest)),
            "frac_shift_below_slack_at_100": float(
                np.mean(shift_at_100 < slack_nearest)),
            "frac_any_change": float(any_change.mean()),
            "frac_change_within_grid": float(change_within_grid.mean()),
            "median_min_lambda_change": float(
                np.median(min_lambda_change[multi])) if multi.any()
            else float("inf"),
            "min_lambda_changed_items": (
                min_lambda_change[any_change].tolist()),
            **star_row,
            "apss_by_lambda": apss_by_lambda,
            "cov_by_lambda": cov_by_lambda,
            "apss_matched_by_lambda": apss_matched_by_lambda,
            "cov_matched_by_lambda": cov_matched_by_lambda,
            "frac_diff_by_lambda": frac_diff_by_lambda,
            "frac_diff_matched_by_lambda": frac_diff_matched_by_lambda,
            "tau_hat_by_lambda": {
                lam: float(cals[lam].tau_hat) for lam in LAMBDA_SWEEP},
            "mean_w_by_lambda": {
                lam: float(np.mean(cals[lam]._compute_weights_vec(
                    np.array([it["U_ale"] for it in test]),
                    np.array([it["U_epi"] for it in test]),
                    lam))) for lam in LAMBDA_SWEEP},
        })
    return {"per_seed": per_seed_rows}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="headline",
                        choices=tuple(CONFIGS))
    parser.add_argument("--write-markdown", action="store_true",
                        help="Also write the Markdown report.")
    args = parser.parse_args()
    config = args.config

    logger = _setup_logging()
    t0 = time.time()
    cells = discover_cells() if config == "headline" else discover_judged_cells()
    logger.info("lambda mechanism [%s]: %d cells, seeds=%s, sweep=%s",
                config, len(cells), SEEDS, LAMBDA_SWEEP)

    out: dict[str, dict] = {}
    for i, cell in enumerate(cells, 1):
        with open(cell["path"], encoding="utf-8") as f:
            data = _ensure_types(json.load(f))
        res = analyze_cell(data, SEEDS, logger, config)
        out[cell["cell_key"]] = res
        r0 = res["per_seed"][0]
        logger.info(
            "[%d/%d] %s: slack=%.3f max_shift=%.3f shift@100=%.3f "
            "any_change=%.2f within_grid=%.2f 1cluster=%.2f full=%.2f "
            "lam*=%.2g d_apss*=%+.3f",
            i, len(cells), cell["cell_key"],
            r0["median_slack_nearest"], r0["median_max_shift"],
            r0["median_shift_at_100"], r0["frac_any_change"],
            r0["frac_change_within_grid"], r0["frac_single_cluster"],
            r0["frac_full_set"], r0["lambda_star_grid"],
            r0["apss_star"] - r0["apss_by_lambda"][0.0],
        )

    # ---- pooled aggregates over all (cell, seed) rows ----
    rows = [r for res in out.values() for r in res["per_seed"]]
    def pool(key: str) -> float:
        vals = np.array([r[key] for r in rows], dtype=float)
        return float(np.nanmean(vals[np.isfinite(vals)]))
    def pool_median(key: str) -> float:
        vals = np.array([r[key] for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        return float(np.median(vals)) if len(vals) else float("nan")

    agg = {
        "n_cell_seed_rows": len(rows),
        "mean_frac_single_cluster": pool("frac_single_cluster"),
        "mean_frac_full_set": pool("frac_full_set"),
        "mean_frac_full_set_multi": pool("frac_full_set_multi"),
        "mean_frac_resizable": pool("frac_resizable"),
        "mean_frac_crossable": pool("frac_crossable"),
        "pooled_median_slack_nearest": pool_median("median_slack_nearest"),
        "pooled_median_slack_down_full": pool_median(
            "median_slack_down_full"),
        "pooled_median_gap_width_finite": pool_median(
            "median_gap_width_finite"),
        "mean_frac_gap_infinite": pool("frac_gap_infinite"),
        "pooled_median_max_shift": pool_median("median_max_shift"),
        "pooled_median_shift_at_100": pool_median("median_shift_at_100"),
        "mean_frac_shift_below_slack": pool("frac_shift_below_slack"),
        "mean_frac_shift_below_slack_at_100": pool(
            "frac_shift_below_slack_at_100"),
        "mean_frac_any_change": pool("frac_any_change"),
        "mean_frac_change_within_grid": pool("frac_change_within_grid"),
        "mean_lambda_star_grid": pool("lambda_star_grid"),
        "mean_frac_diff_star_vs_0": pool("frac_diff_star_vs_0"),
        "mean_apss_star_minus_0": float(np.nanmean(
            [r["apss_star"] - r["apss_by_lambda"][0.0] for r in rows])),
        "mean_cov_star_minus_0": float(np.nanmean(
            [r["cov_star"] - r["cov_by_lambda"][0.0] for r in rows])),
        "mean_apss_matched_star_minus_0": float(np.nanmean(
            [r["apss_matched_star"] - r["apss_matched_by_lambda"][0.0]
             for r in rows])),
        "mean_cov_matched_at_0": pool_median("cov_matched_star"),
    }
    logger.info("AGGREGATE: %s", json.dumps(agg, indent=1))

    suffix = "" if config == "headline" else f"_{config}"
    with open(RESULTS_DIR / f"lambda_mechanism{suffix}.json", "w",
              encoding="utf-8") as f:
        json.dump({"config": config, "lambda_sweep": LAMBDA_SWEEP,
                   "aggregate": agg, "cells": out}, f, indent=1, default=float)

    # markdown summary
    lines = [
        f"# Lambda-inertness mechanism ({config} protocol)",
        "",
        f"Sweep: {LAMBDA_SWEEP} (tuned grid max = 100).",
        "",
        "| Cell | 1-cluster | full-set | resizable | crossable | med slack "
        "| med max-shift (lam<=1e6) | frac any change | frac change "
        "lam<=100 | med min-lambda |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, res in out.items():
        r = {k: float(np.nanmean([row[k] for row in res["per_seed"]]))
             for k in ("frac_single_cluster", "frac_full_set",
                       "frac_resizable", "frac_crossable",
                       "median_slack_nearest", "median_max_shift",
                       "frac_any_change", "frac_change_within_grid",
                       "median_min_lambda_change")}
        lines.append(
            f"| {key} | {r['frac_single_cluster']:.2f} "
            f"| {r['frac_full_set']:.2f} | {r['frac_resizable']:.2f} "
            f"| {r['frac_crossable']:.2f} "
            f"| {r['median_slack_nearest']:.3f} "
            f"| {r['median_max_shift']:.3f} "
            f"| {r['frac_any_change']:.3f} "
            f"| {r['frac_change_within_grid']:.3f} "
            f"| {r['median_min_lambda_change']:.3g} |"
        )
    lines += ["", "## Aggregate", "",
              "```json", json.dumps(agg, indent=1), "```"]
    if args.write_markdown:
        with open(RESULTS_DIR / f"lambda_mechanism{suffix}.md", "w",
                  encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    logger.info("done in %.1f min", (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
