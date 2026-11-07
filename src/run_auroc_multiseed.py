#!/usr/bin/env python3
"""Multi-seed per-cell hallucination AUROC for U_total / U_epi / U_ale / R(x).

Reads cached unc_*_vne_*.json files, splits each cell into Val1/Val2/Test
under each seed, computes lambda* via the standard CAELUM calibrator, then
evaluates per-cell AUROC of each signal as a hallucination predictor on Test.
Aggregates mean +/- std across seeds.

Outputs:
  - results/auroc_multiseed.json
  - results/auroc_multiseed.md (optional; --write-markdown)

The hallucination label is correct_cluster_idx == -1 (ground truth not found
in any generated cluster). Only the items in the test split are scored.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
import warnings as _warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

_warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from caelum.paths import GENERATIONS_DIR, RESULTS_DIR  # noqa: E402

from caelum.calibration import CAELUMCalibrator  # noqa: E402
from caelum.utils import split_data  # noqa: E402

ALPHA = 0.1
SEEDS = [42, 43, 44, 45]
EPS = 1e-6

UNC_RE = re.compile(r"unc_(.+?)_(.+?)_vne_(\d+)\.json")
SKIP_DATASETS = {"condambigqa"}
MODEL_PRETTY = {
    "qwen3_8b": "Qwen3-8B",
    "ministral_3_8b_instruct_2512": "Ministral-3-8B",
    "gemma_4_e4b_it": "Gemma-4-E4B-it",
}
DATASET_PRETTY = {
    "triviaqa": "TriviaQA", "ambigqa": "AmbigQA", "opennq": "OpenNQ",
    "ambiginst": "AmbigInst", "simpleqa": "SimpleQA", "popqa": "PopQA",
}
MODEL_ORDER = ["qwen3_8b", "ministral_3_8b_instruct_2512", "gemma_4_e4b_it"]
DATASET_ORDER = ["triviaqa", "ambigqa", "opennq", "ambiginst", "simpleqa", "popqa"]


def _setup_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S", force=True)
    for n in ("httpx", "httpcore", "sentence_transformers", "transformers",
              "huggingface_hub", "filelock", "urllib3", "fsspec", "matplotlib"):
        logging.getLogger(n).setLevel(logging.WARNING)
    return logging.getLogger("auroc_multiseed")


def _ensure_types(data: list[dict]) -> list[dict]:
    for item in data:
        if "cluster_props" in item:
            item["cluster_props"] = np.asarray(item["cluster_props"], dtype=float)
        for k in ("U_ale", "U_epi", "U_total"):
            if k in item:
                item[k] = float(item[k])
        if "correct_cluster_idx" in item:
            item["correct_cluster_idx"] = int(item["correct_cluster_idx"])
    return data


def discover_cells() -> list[dict]:
    cells = []
    for f in sorted(GENERATIONS_DIR.glob("unc_*_vne_*.json")):
        m = UNC_RE.match(f.name)
        if not m:
            continue
        ds, model, n = m.group(1), m.group(2), int(m.group(3))
        if ds in SKIP_DATASETS or model not in MODEL_PRETTY or ds not in DATASET_PRETTY:
            continue
        cells.append({
            "model": model, "dataset": ds, "n_samples": n,
            "path": str(f),
            "cell_key": f"{model}__{ds}",
        })
    cells.sort(key=lambda c: (
        MODEL_ORDER.index(c["model"]),
        DATASET_ORDER.index(c["dataset"]),
    ))
    best: dict[str, dict] = {}
    for c in cells:
        if c["cell_key"] not in best or c["n_samples"] > best[c["cell_key"]]["n_samples"]:
            best[c["cell_key"]] = c
    return list(best.values())


def per_cell_aurocs(cell: dict, alpha: float, seeds: list[int]) -> dict:
    with open(cell["path"], encoding="utf-8") as f:
        data = _ensure_types(json.load(f))

    seed_aurocs = {"U_total": [], "U_epi": [], "U_ale": [], "R": [],
                   "lambda_star": []}
    for seed in seeds:
        val1, val2, test = split_data(
            data, val1_frac=0.4, val2_frac=0.3, seed=seed,
        )
        cal = CAELUMCalibrator(alpha=alpha, seed=seed, randomize=True)
        cal.calibrate(val1, val2, verbose=False)
        lam = float(cal.optimal_lambda)
        seed_aurocs["lambda_star"].append(lam)

        u_tot = np.array([d["U_total"] for d in test])
        u_epi = np.array([d["U_epi"] for d in test])
        u_ale = np.array([d["U_ale"] for d in test])
        r_x = np.log(lam * u_epi + EPS) - np.log(u_ale + EPS)

        # Hallucination label: 1 if reference not in any cluster
        y = np.array([
            1 if int(d["correct_cluster_idx"]) < 0 else 0 for d in test
        ])

        if len(np.unique(y)) < 2:
            for k in ("U_total", "U_epi", "U_ale", "R"):
                seed_aurocs[k].append(float("nan"))
            continue

        seed_aurocs["U_total"].append(float(roc_auc_score(y, u_tot)))
        seed_aurocs["U_epi"].append(float(roc_auc_score(y, u_epi)))
        seed_aurocs["U_ale"].append(float(roc_auc_score(y, u_ale)))
        seed_aurocs["R"].append(float(roc_auc_score(y, r_x)))

    summary = {}
    for k, v in seed_aurocs.items():
        a = np.array(v)
        a_clean = a[~np.isnan(a)]
        if len(a_clean):
            summary[k] = {
                "mean": float(a_clean.mean()),
                "std": float(a_clean.std()),
                "per_seed": v,
            }
        else:
            summary[k] = {"mean": float("nan"), "std": float("nan"),
                          "per_seed": v}

    return {
        "model": cell["model"], "dataset": cell["dataset"],
        "match_rate": float(np.mean([
            int(d["correct_cluster_idx"]) >= 0 for d in data
        ])),
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-markdown", action="store_true",
                        help="Write an optional Markdown summary.")
    args = parser.parse_args()

    logger = _setup_logging()
    t0 = time.time()
    cells = discover_cells()
    logger.info(
        "Multi-seed AUROC: %d cells, alpha=%.2f, seeds=%s",
        len(cells), ALPHA, SEEDS,
    )

    results: dict[str, dict] = {}
    for i, cell in enumerate(cells, start=1):
        logger.info("[%d/%d] %s", i, len(cells), cell["cell_key"])
        rec = per_cell_aurocs(cell, ALPHA, SEEDS)
        results[cell["cell_key"]] = rec
        s = rec["summary"]
        logger.info(
            "    U_tot=%.3f+/-%.3f  U_epi=%.3f+/-%.3f  U_ale=%.3f+/-%.3f  R=%.3f+/-%.3f  lam*=%.1f+/-%.1f",
            s["U_total"]["mean"], s["U_total"]["std"],
            s["U_epi"]["mean"], s["U_epi"]["std"],
            s["U_ale"]["mean"], s["U_ale"]["std"],
            s["R"]["mean"], s["R"]["std"],
            s["lambda_star"]["mean"], s["lambda_star"]["std"],
        )

    out_json = RESULTS_DIR / "auroc_multiseed.json"
    out_json.write_text(json.dumps({
        "alpha": ALPHA, "seeds": SEEDS, "cells": results,
    }, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_json)

    # Markdown
    lines = [
        "# Hallucination AUROC, multi-seed (seeds 42-45)",
        "",
        "Mean +/- std AUROC across seeds. Hallucination = correct_cluster_idx < 0.",
        "",
        "| Model | Dataset | Match% | U_total | U_epi | U_ale | R(x) | lambda* |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for ck, r in results.items():
        s = r["summary"]
        lines.append(
            f"| {MODEL_PRETTY[r['model']]} | {DATASET_PRETTY[r['dataset']]} | "
            f"{r['match_rate']*100:.1f} | "
            f"{s['U_total']['mean']:.3f}+/-{s['U_total']['std']:.3f} | "
            f"{s['U_epi']['mean']:.3f}+/-{s['U_epi']['std']:.3f} | "
            f"{s['U_ale']['mean']:.3f}+/-{s['U_ale']['std']:.3f} | "
            f"{s['R']['mean']:.3f}+/-{s['R']['std']:.3f} | "
            f"{s['lambda_star']['mean']:.1f}+/-{s['lambda_star']['std']:.1f} |"
        )

    out_md = RESULTS_DIR / "auroc_multiseed.md"
    if args.write_markdown:
        out_md.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Wrote %s", out_md)
    logger.info("Total elapsed: %.1f s", time.time() - t0)


if __name__ == "__main__":
    main()
