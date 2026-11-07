#!/usr/bin/env python3
"""Summarize the selected lambda* over split seeds 42-45 for two protocols.

Seed 42 is the reported run of each protocol. Seeds 43-45 are the same
protocols on other splits, stored in results/split_seeds/seed<N>/ and written by::

    python src/aggregate_with_baselines.py --seed 43 --output_dir results/split_seeds/seed43
    python src/run_stratified_with_lambdastar.py --seed 43 --output_dir results/split_seeds/seed43

(and likewise for 44 and 45). This script writes the per-cell lambda* values,
their mean and population SD to

- results/lambda_deterministic_multiseed.json for the main-table protocol
  (deterministic APS, 40/30/30 split; seed 42 is results/stage3_results.json);
- results/lambda_integrated_multiseed.json for the integrated stratified variant
  (randomized APS, four 25% splits; seed 42 is
  results/stratified_lambdastar_results.json).

The randomized-APS summary in results/headline_multiseed.json is separate.

Usage::

    python src/summarize_lambda_seeds.py
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from caelum.paths import REPO_ROOT, RESULTS_DIR, content_sha256  # noqa: E402

SEEDS = [42, 43, 44, 45]
MODELS = ["qwen3_8b", "ministral_3_8b_instruct_2512", "gemma_4_e4b_it"]
DATASETS = ["triviaqa", "ambigqa", "opennq", "ambiginst", "simpleqa", "popqa"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path,
                        default=RESULTS_DIR / "lambda_deterministic_multiseed.json")
    parser.add_argument("--integrated_output", type=Path,
                        default=RESULTS_DIR / "lambda_integrated_multiseed.json")
    args = parser.parse_args()

    sources, runs = [], []
    for seed in SEEDS:
        relative = ("results/stage3_results.json" if seed == 42 else
                    f"results/split_seeds/seed{seed}/stage3_results.json")
        path = REPO_ROOT / relative
        runs.append(json.loads(path.read_text(encoding="utf-8")))
        sources.append({"seed": seed, "path": relative, "sha256_lf": content_sha256(path)})
    cells = {}
    for model in MODELS:
        for dataset in DATASETS:
            key = f"{model}__{dataset}"
            values = []
            for run in runs:
                cell = run[key]
                assert cell["model"] == model and cell["dataset"] == dataset
                assert cell["n_samples"] == runs[0][key]["n_samples"]
                value = float(cell["methods"]["CAELUM"]["lambda_star"])
                assert math.isfinite(value) and 0 <= value <= 100
                values.append(value)
            cells[key] = {"model": model, "dataset": dataset,
                          "methods": {"CAELUM": {
                              "lam_per_seed": values,
                              "lam_mean": statistics.mean(values),
                              "lam_std": statistics.pstdev(values)}}}
    report = {
        "alpha": 0.1, "seeds": SEEDS,
        "protocol": {"randomize": False, "weight_fn": "additive",
                     "selection_method": "argmin_apss", "split": [0.4, 0.3, 0.3],
                     "labels": "frozen heuristic", "sd": "population (ddof=0)",
                     "variation": "data splits; generated answers are reused"},
        "sources": sources, "cells": cells,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(cells)} cells to {args.output}")
    summarize_integrated(args.integrated_output)


def summarize_integrated(output: Path) -> None:
    """lambda* of the integrated stratified variant over split seeds 42-45."""
    sources, runs = [], []
    for seed in SEEDS:
        relative = ("results/stratified_lambdastar_results.json" if seed == 42 else
                    f"results/split_seeds/seed{seed}/stratified_lambdastar_results.json")
        path = REPO_ROOT / relative
        run = json.loads(path.read_text(encoding="utf-8"))
        assert run["seed"] == seed and run["alpha"] == 0.1, relative
        runs.append({r["cell_key"]: r for r in run["results"]})
        sources.append({"seed": seed, "path": relative, "sha256_lf": content_sha256(path)})
    assert all(set(r) == set(runs[0]) for r in runs), "cell sets differ across seeds"
    cells = {}
    for model in MODELS:
        for dataset in DATASETS:
            key = f"{model}__{dataset}"
            values = [float(r[key]["lambda_star"]) for r in runs]
            assert all(math.isfinite(v) and v >= 0 for v in values), key
            cells[key] = {"model": model, "dataset": dataset, "lam_per_seed": values,
                          "lam_mean": statistics.mean(values),
                          "lam_std": statistics.pstdev(values)}
    report = {
        "alpha": 0.1, "seeds": SEEDS,
        "protocol": {"randomize": True, "weight_fn": "additive",
                     "selection_method": "argmin_apss",
                     "split": "four 25% splits (V0, V1, V2, test)",
                     "labels": "frozen heuristic", "sd": "population (ddof=0)",
                     "variation": "data splits; generated answers are reused"},
        "sources": sources, "cells": cells,
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(cells)} cells to {output}")


if __name__ == "__main__":
    main()
