#!/usr/bin/env python3
"""Multi-seed headline numbers for Tables 1 and 2.

Runs CAELUM and the conformal-prediction baselines at nominal alpha=0.1 across
4 seeds (42, 43, 44, 45) on the cached unc_*_vne_*.json files, and reports
mean +/- std APSS / coverage / lambda* per (model, dataset) cell.

The output extends ``headline_with_baselines.md`` with seed CI columns and
gives the multi-seed JSON record needed to regenerate Table 1 with error bars.

Outputs:
  - ``results/headline_multiseed.json``
  - ``results/headline_multiseed_summary.md`` (optional; --write-markdown)

Usage:
    python -u src/run_headline_multiseed.py
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
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from caelum.paths import GENERATIONS_DIR, RESULTS_DIR  # noqa: E402

from caelum.calibration import CAELUMCalibrator  # noqa: E402
from caelum.utils import split_data  # noqa: E402

OUTPUT_DIR = RESULTS_DIR

ALPHA = 0.1
SEEDS = [42, 43, 44, 45]
METHODS = ["CAELUM", "Vanilla", "SE-CP", "Equal", "EpiOnly", "TrueAPS"]

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


def _setup_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S", force=True)
    for n in ("httpx", "httpcore", "sentence_transformers", "transformers",
              "huggingface_hub", "filelock", "urllib3", "fsspec", "matplotlib"):
        logging.getLogger(n).setLevel(logging.WARNING)
    return logging.getLogger("headline_multiseed")


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
    # Dedupe (keep largest n)
    best: dict[str, dict] = {}
    for c in cells:
        k = c["cell_key"]
        if k not in best or c["n_samples"] > best[k]["n_samples"]:
            best[k] = c
    return list(best.values())


def make_calibrator(method: str, alpha: float, seed: int = 42):
    if method == "CAELUM":
        return CAELUMCalibrator(alpha=alpha, seed=seed, randomize=True)
    if method == "Vanilla":
        return CAELUMCalibrator.vanilla_conformal(alpha=alpha, randomize=True)
    if method == "Equal":
        return CAELUMCalibrator.equal_weight(alpha=alpha, randomize=True)
    if method == "EpiOnly":
        return CAELUMCalibrator.epistemic_only(alpha=alpha, randomize=True)
    if method == "TrueAPS":
        return CAELUMCalibrator.true_aps_conformal(alpha=alpha, randomize=True)
    if method == "SE-CP":
        from caelum.baselines import SemanticEntropyCalibrator
        return SemanticEntropyCalibrator(alpha=alpha, seed=seed)
    raise ValueError(f"Unknown method: {method!r}")


def evaluate_cell_seeds(cell: dict, alpha: float, seeds: list[int]) -> dict:
    with open(cell["path"], encoding="utf-8") as f:
        data = _ensure_types(json.load(f))
    n = len(data)
    match_rate_overall = float(np.mean([
        int(d.get("correct_cluster_idx", -1)) >= 0 for d in data
    ]))

    methods_seeds: dict[str, dict] = {m: {"apss": [], "cov": [], "lam": []}
                                      for m in METHODS}

    for seed in seeds:
        val1, val2, test = split_data(
            data, val1_frac=0.4, val2_frac=0.3, seed=seed,
        )
        for m in METHODS:
            cal = make_calibrator(m, alpha, seed)
            cal.calibrate(val1, val2, verbose=False)
            sizes = [cal.predict_set_size(it) for it in test]
            cov = float(np.mean([cal.is_covered(it) for it in test]))
            apss = float(np.mean(sizes))
            lam = float(getattr(cal, "optimal_lambda", 0.0))
            methods_seeds[m]["apss"].append(apss)
            methods_seeds[m]["cov"].append(cov)
            methods_seeds[m]["lam"].append(lam)

    methods_summary: dict[str, dict] = {}
    for m, sd in methods_seeds.items():
        a = np.array(sd["apss"])
        c = np.array(sd["cov"])
        lam = np.array(sd["lam"])
        methods_summary[m] = {
            "apss_mean": float(a.mean()), "apss_std": float(a.std()),
            "apss_per_seed": a.tolist(),
            "cov_mean": float(c.mean()), "cov_std": float(c.std()),
            "cov_per_seed": c.tolist(),
            "lam_mean": float(lam.mean()), "lam_std": float(lam.std()),
            "lam_per_seed": lam.tolist(),
        }
    return {
        "model": cell["model"], "dataset": cell["dataset"],
        "n_samples": n, "match_rate": match_rate_overall,
        "alpha": alpha, "seeds": list(seeds),
        "methods": methods_summary,
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
        "Headline multi-seed: %d cells, alpha=%.2f, seeds=%s",
        len(cells), ALPHA, SEEDS,
    )

    results: dict[str, dict] = {}
    for i, cell in enumerate(cells, start=1):
        logger.info("[%d/%d] %s", i, len(cells), cell["cell_key"])
        rec = evaluate_cell_seeds(cell, ALPHA, SEEDS)
        results[cell["cell_key"]] = rec
        c = rec["methods"]["CAELUM"]
        v = rec["methods"]["Vanilla"]
        s = rec["methods"]["SE-CP"]
        t = rec["methods"]["TrueAPS"]
        logger.info(
            "    CAELUM APSS=%.2f+/-%.2f cov=%.2f+/-%.2f lam*=%.1f+/-%.1f | "
            "Vanilla APSS=%.2f+/-%.2f | SE-CP APSS=%.2f+/-%.2f | "
            "TrueAPS APSS=%.2f+/-%.2f",
            c["apss_mean"], c["apss_std"], c["cov_mean"], c["cov_std"],
            c["lam_mean"], c["lam_std"],
            v["apss_mean"], v["apss_std"],
            s["apss_mean"], s["apss_std"],
            t["apss_mean"], t["apss_std"],
        )

    out_json = OUTPUT_DIR / "headline_multiseed.json"
    out_json.write_text(json.dumps({
        "alpha": ALPHA, "seeds": SEEDS,
        "method_order": METHODS,
        "cells": results,
    }, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_json)

    # Markdown with mean +/- std per cell
    lines = [
        "# Headline multi-seed (alpha=0.1, seeds 42-45)",
        "",
        "Mean +/- std across 4 random splits for the 18 (model, dataset) cells.",
        "Coverage and APSS are computed on the test split per seed.",
        "",
        "## Per-cell APSS (mean +/- std)",
        "",
        "| Model | Dataset | Match | CAELUM | Vanilla | SE-CP | TrueAPS | Equal | EpiOnly | lambda* |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for ck, r in results.items():
        ms = r["methods"]
        lines.append(
            f"| {MODEL_PRETTY[r['model']]} | {DATASET_PRETTY[r['dataset']]} | "
            f"{r['match_rate']:.2f} | "
            f"{ms['CAELUM']['apss_mean']:.2f}±{ms['CAELUM']['apss_std']:.2f} | "
            f"{ms['Vanilla']['apss_mean']:.2f}±{ms['Vanilla']['apss_std']:.2f} | "
            f"{ms['SE-CP']['apss_mean']:.2f}±{ms['SE-CP']['apss_std']:.2f} | "
            f"{ms['TrueAPS']['apss_mean']:.2f}±{ms['TrueAPS']['apss_std']:.2f} | "
            f"{ms['Equal']['apss_mean']:.2f}±{ms['Equal']['apss_std']:.2f} | "
            f"{ms['EpiOnly']['apss_mean']:.2f}±{ms['EpiOnly']['apss_std']:.2f} | "
            f"{ms['CAELUM']['lam_mean']:.1f}±{ms['CAELUM']['lam_std']:.1f} |"
        )

    lines += [
        "",
        "## Per-cell coverage (mean +/- std)",
        "",
        "| Model | Dataset | CAELUM | Vanilla | SE-CP | TrueAPS |",
        "|---|---|---|---|---|---|",
    ]
    for ck, r in results.items():
        ms = r["methods"]
        lines.append(
            f"| {MODEL_PRETTY[r['model']]} | {DATASET_PRETTY[r['dataset']]} | "
            f"{ms['CAELUM']['cov_mean']:.3f}±{ms['CAELUM']['cov_std']:.3f} | "
            f"{ms['Vanilla']['cov_mean']:.3f}±{ms['Vanilla']['cov_std']:.3f} | "
            f"{ms['SE-CP']['cov_mean']:.3f}±{ms['SE-CP']['cov_std']:.3f} | "
            f"{ms['TrueAPS']['cov_mean']:.3f}±{ms['TrueAPS']['cov_std']:.3f} |"
        )

    out_md = OUTPUT_DIR / "headline_multiseed_summary.md"
    if args.write_markdown:
        out_md.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Wrote %s", out_md)
    logger.info("Total elapsed: %.1f s", time.time() - t0)


if __name__ == "__main__":
    main()
