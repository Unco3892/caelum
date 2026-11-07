#!/usr/bin/env python3
"""Validation-split weight-movement audit for the 18 main cells.

For each cell, the seed-42 40/30/30 split is rebuilt and only V1 is used.
Deterministic APS prediction sets are recomputed from the cached uncertainty
records on the 4,011-point lambda grid used for selection. Per cell, the audit
reports the V1 support-failure rate, the fraction of V1 items whose set size
changes across the grid (f_move), the APSS range, the grid points attaining the
minimum, and whether the main table's selected lambda* is a V1 minimizer. No
model is run and no answers are reclustered or rematched.

Usage::

    python src/run_validation_weight_audit.py

Writes results/validation_weight_audit.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from caelum.paths import GENERATIONS_DIR, RESULTS_DIR, content_sha256, repo_relative  # noqa: E402

MODELS = ["qwen3_8b", "ministral_3_8b_instruct_2512", "gemma_4_e4b_it"]
DATASETS = ["triviaqa", "ambigqa", "opennq", "ambiginst", "simpleqa", "popqa"]


def audit_cell(path, grid, alpha=.1, seed=42):
    records = json.loads(path.read_text(encoding="utf-8"))
    permutation = np.random.RandomState(seed).permutation(len(records))
    v1_indices = permutation[:int(.4*len(records))]
    items = [records[i] for i in v1_indices]
    n = len(items)
    raw = []
    for item in items:
        props = np.asarray(item["cluster_props"], dtype=float)
        correct = int(item["correct_cluster_idx"])
        raw.append(min(float(np.sum(props)), 1.) if correct < 0 or correct >= len(props)
                   else float(np.sum(props[:correct])))
    ale = np.array([item["U_ale"] for item in items], dtype=float)
    epi = np.array([item["U_epi"] for item in items], dtype=float)
    weights = np.maximum(1 + ale[None, :] + grid[:, None]*epi[None, :], 1e-12)
    scores = np.asarray(raw)[None, :] / weights
    quantile_level = min(math.ceil((1-alpha)*(n+1))/n, 1.)
    thresholds = np.quantile(scores, quantile_level, axis=1, method="higher")
    effective = thresholds[:, None] * weights
    sizes = np.empty((len(grid), n), dtype=np.int16)
    for j, item in enumerate(items):
        props = np.asarray(item["cluster_props"], dtype=float)
        assert len(props) > 0 and np.all(props >= 0)
        boundaries = np.r_[0., np.cumsum(props[:-1])]
        sizes[:, j] = np.sum(boundaries[None, :] <= effective[:, j, None], axis=1)
    # Integer sums make exact equality of APSS minima unambiguous.
    size_sums = np.sum(sizes, axis=1, dtype=np.int64)
    minimum = int(size_sums.min())
    maximum = int(size_sums.max())
    minimizers = np.flatnonzero(size_sums == minimum)
    movement = sizes.min(axis=0) < sizes.max(axis=0)
    starts = minimizers[np.r_[True, np.diff(minimizers) > 1]]
    stops = minimizers[np.r_[np.diff(minimizers) > 1, True]]
    return {
        "n_total": len(records), "n_v1": n,
        "n_unmatched_v1": sum(int(item["correct_cluster_idx"]) < 0 for item in items),
        "support_failure_rate_v1": float(np.mean([
            int(item["correct_cluster_idx"]) < 0 for item in items])),
        "v1_indices_sha256": hashlib.sha256(v1_indices.astype('<i8').tobytes()).hexdigest(),
        "numpy_quantile_level": quantile_level,
        "n_movable": int(movement.sum()), "f_move": float(movement.mean()),
        "apss_min": minimum/n, "apss_max": maximum/n,
        "apss_range": (maximum-minimum)/n,
        "objective_flat": minimum == maximum,
        "n_minimizing_grid_points": len(minimizers),
        "argmin_grid_fraction": len(minimizers)/len(grid),
        "unique_grid_minimizer": len(minimizers) == 1,
        "first_minimizer": float(grid[minimizers[0]]),
        "last_minimizer": float(grid[minimizers[-1]]),
        "minimizing_grid_runs": [[float(grid[start]), float(grid[stop]), int(stop-start+1)]
                                 for start, stop in zip(starts, stops)],
    }, size_sums


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generations_dir", type=Path, default=GENERATIONS_DIR)
    parser.add_argument("--stage3", type=Path, default=RESULTS_DIR / "stage3_results.json",
                        help="Main-table results whose selected lambda* is checked.")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "validation_weight_audit.json")
    args = parser.parse_args()

    stage3 = json.loads(args.stage3.read_text(encoding="utf-8"))
    grid = np.r_[np.linspace(0, .09, 10), np.logspace(-1, 2, 4001)]
    rows = {}
    sources = [{"path": repo_relative(args.stage3), "sha256": content_sha256(args.stage3)}]
    for model in MODELS:
        for dataset in DATASETS:
            key = f"{model}__{dataset}"
            headline = stage3[key]
            path = args.generations_dir / f"unc_{dataset}_{model}_vne_{headline['n_samples']}.json"
            row, sums = audit_cell(path, grid)
            row.update(model=model, dataset=dataset)
            selected = headline["methods"]["CAELUM"]["lambda_star"]
            nearest = int(np.argmin(abs(grid-selected)))
            assert np.isclose(grid[nearest], selected, rtol=1e-10, atol=1e-12)
            row["selected_lambda"] = selected
            row["selected_lambda_is_v1_minimizer"] = bool(sums[nearest] == sums.min())
            row["selected_lambda_is_first_minimizer"] = bool(selected == row["first_minimizer"])
            row["selected_lambda_apss_excess"] = float((sums[nearest]-sums.min())/row["n_v1"])
            rows[key] = row
            sources.append({"path": repo_relative(path), "sha256": content_sha256(path)})
    summary = {
        "n_cells": len(rows),
        "n_zero_movement_cells": sum(r["n_movable"] == 0 for r in rows.values()),
        "n_flat_objective_cells": sum(r["objective_flat"] for r in rows.values()),
        "n_unique_grid_minimizer_cells": sum(r["unique_grid_minimizer"] for r in rows.values()),
        "f_move_min": min(r["f_move"] for r in rows.values()),
        "f_move_max": max(r["f_move"] for r in rows.values()),
        "f_move_cell_mean": float(np.mean([r["f_move"] for r in rows.values()])),
        "argmin_fraction_min": min(r["argmin_grid_fraction"] for r in rows.values()),
        "argmin_fraction_max": max(r["argmin_grid_fraction"] for r in rows.values()),
        "n_selected_lambdas_minimizing": sum(r["selected_lambda_is_v1_minimizer"] for r in rows.values()),
        "n_selected_lambdas_equal_first_minimizer": sum(r["selected_lambda_is_first_minimizer"]
                                                        for r in rows.values()),
    }
    report = {
        "analysis": "Weight-movement audit on V1 of the 18 main cells under deterministic APS.",
        "protocol": {"seed": 42, "alpha": .1, "split": "40/30/30; V1 only",
                     "n_grid": len(grid), "grid": grid.tolist(),
                     "quantile": "ceil(.9*(n+1))/n clipped at 1; numpy method=higher",
                     "set_rule": "cumulative mass before cluster <= tau_lambda*(1+U_ale+lambda*U_epi)",
                     "uncertainty_and_matching": "Cached VNE records and heuristic match labels; "
                                                 "no reclustering or rematching",
                     "tie_rule": "Exact equality of integer sums of V1 set sizes; first minimum on ascending grid",
                     "interpretation": "A unique grid minimum is not statistical or mechanistic identifiability; "
                                       "movement is necessary but insufficient for uniqueness."},
        "sources": sources, "summary": summary, "cells": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Selected lambda* values that do not minimize this audit:",
          [k for k, r in rows.items() if not r["selected_lambda_is_v1_minimizer"]])
    print("Wrote", repo_relative(args.output))


if __name__ == "__main__":
    main()
