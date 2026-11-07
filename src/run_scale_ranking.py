#!/usr/bin/env python3
"""Support-failure AUROC of U_epi and U_total on the Qwen3-14B/32B judged caches.

Regenerates the content of ``paper/neurips/data/scale_ranking_verification.json``,
which the table exporter (``paper/neurips/scripts/make_camera_ready_tables.py``,
``historical_scale``) reads for the support, U_epi, U_total and Delta_epi
columns of the historical larger-model table.

The 12 cells are Qwen3-14B and Qwen3-32B on the six datasets. Each cell reads
its judged cache in ``results/llm_judged_unc_full/``; the positive class is
support failure, i.e. no sampled cluster matches the reference after the judge
audit of heuristic negatives (``correct_cluster_idx < 0``). For seeds 42-45 the
test subset is the last 30% of ``caelum.utils.split_data`` (40/30/30), and the
AUROC of the cached U_epi and U_total is computed on it with no model fitting.
Each cell reports the four folds and their mean and population standard
deviation; the folds overlap, so the standard deviation is not a confidence
interval.

Writes ``results/scale_ranking_verification.json``. Run from the repository
root::

    python src/run_scale_ranking.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from caelum.paths import JUDGED_DIR, RESULTS_DIR, content_sha256, repo_relative  # noqa: E402
from caelum.utils import split_data  # noqa: E402

MODELS = ["qwen3_14b", "qwen3_32b"]
DATASETS = {"triviaqa": 300, "ambigqa": 200, "opennq": 300,
            "ambiginst": 304, "simpleqa": 500, "popqa": 500}
SEEDS = [42, 43, 44, 45]

PROTOCOL = {
    "cache_family": "historical llm_judged_unc_full, not Nov3 fresh run",
    "labels": "reference absent from any cluster after one-sided judge audit of heuristic negatives",
    "seeds": SEEDS,
    "test_selection": "np.random.RandomState(seed).permutation(n)[int(.4*n)+int(.3*n):], identical to utils.split_data",
    "signals": "cached U_epi and U_total, no model fitting needed",
    "aggregation": "per-cell mean and population std across four overlapping split test subsets; std is not a confidence interval",
    "source_caches_overwritten": False,
}


def cell_row(path: Path, model: str, dataset: str) -> dict:
    """AUROC of U_epi and U_total for support failure on four test subsets."""
    raw = path.read_bytes()
    items = json.loads(raw.decode("utf-8"))
    failure = np.array([int(item["correct_cluster_idx"] < 0) for item in items])
    folds = []
    for seed in SEEDS:
        _, _, test = split_data(items, val1_frac=0.4, val2_frac=0.3, seed=seed)
        labels = [int(item["correct_cluster_idx"] < 0) for item in test]
        epi = float(roc_auc_score(labels, [item["U_epi"] for item in test]))
        total = float(roc_auc_score(labels, [item["U_total"] for item in test]))
        folds.append({"seed": seed, "n_test": len(test), "epi_auc": epi,
                      "total_auc": total, "delta": epi - total})
    epi = np.array([fold["epi_auc"] for fold in folds])
    total = np.array([fold["total_auc"] for fold in folds])
    delta = np.array([fold["delta"] for fold in folds])
    return {
        "model": model,
        "dataset": dataset,
        "path": repo_relative(path),
        "sha256": content_sha256(path),
        "n": len(items),
        "match_rate": float(1 - failure.mean()),
        "folds": folds,
        "epi_mean": float(epi.mean()),
        "epi_std": float(epi.std()),
        "total_mean": float(total.mean()),
        "total_std": float(total.std()),
        "delta_mean": float(delta.mean()),
        "delta_std": float(delta.std()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judged_dir", type=Path, default=JUDGED_DIR,
                        help="Directory with the judged unc_*_vne_*.json caches.")
    parser.add_argument("--output", type=Path,
                        default=RESULTS_DIR / "scale_ranking_verification.json",
                        help="Output JSON path.")
    args = parser.parse_args()

    cells = []
    for model in MODELS:
        for dataset, n in DATASETS.items():
            path = args.judged_dir / f"unc_{dataset}_{model}_vne_{n}.json"
            row = cell_row(path, model, dataset)
            assert row["n"] == n, (path, row["n"])
            cells.append(row)
            print(f"{model:10s} {dataset:10s} match={row['match_rate']:.3f} "
                  f"epi={row['epi_mean']:.3f} total={row['total_mean']:.3f} "
                  f"delta={row['delta_mean']:+.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"protocol": PROTOCOL, "cells": cells}, indent=2),
                           encoding="utf-8")
    print(f"Wrote {repo_relative(args.output)}")


if __name__ == "__main__":
    main()
