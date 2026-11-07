"""Predictive Kernel Entropy (PKE) baseline -- hallucination AUROC.

Implements the missing baseline flagged in
``paper/neurips/AUDIT_adversarial.md`` Attack 6 and ``AUDIT_synthesis.md
Sec. 3.4``. PKE (Gruber & Buettner 2023) is the SOTA total-uncertainty
baseline used in Walha et al.'s Spectral Uncertainty paper. We re-evaluate
hallucination AUROC against the existing CAELUM signals (U_total, U_epi,
U_ale, R(x)) on the SAME Test split (seed=42, val1/val2/test = 0.4/0.3/0.3).

For each (model, dataset) cell with cached ``unc_*_vne_*.json``:
  1. Pool the n_clarifications x m_answers answer strings.
  2. Embed via ``SentenceEncoder`` (all-mpnet-base-v2, normalize=True).
  3. Compute PKE with median-heuristic gamma (per-query bandwidth).
  4. Compute hallucination AUROC: PKE as predictor of
     ``correct_cluster_idx == -1`` on the Test split.
  5. Compare against the cached ``U_total`` AUROC on the same Test split.

Outputs ``results/pke_aurocs.json`` and prints a per-cell + aggregate
breakdown.

Reference:
    Sebastian G. Gruber and Florian Buettner. Better Uncertainty
    Quantification for Deep Neural Networks via Kernel Density Estimation.
    arXiv:2306.12189, 2023.

Usage (from the repository root):
    python -u src/run_pke_baseline.py
"""

from __future__ import annotations

import argparse

# Handle help and invalid arguments before loading analysis dependencies.
if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

import warnings as _warnings
_warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from caelum.paths import GENERATIONS_DIR, RESULTS_DIR  # noqa: E402

from caelum.evaluation import compute_auroc
from caelum.predictive_kernel_entropy import compute_pke
from caelum.utils import SentenceEncoder, split_data

OUTPUT_DIR = RESULTS_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
for noisy in (
    "httpx", "httpcore", "sentence_transformers", "transformers",
    "huggingface_hub", "filelock", "urllib3", "fsspec", "safetensors",
    "accelerate", "torch", "tqdm",
):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("pke_baseline")


# ---------------------------------------------------------------------------
# Cell discovery
# ---------------------------------------------------------------------------

def discover_cells() -> list[dict[str, Any]]:
    """Find all cached unc_*_vne_*.json files (one per model x dataset cell).

    Returns:
        List of dicts with keys ``model``, ``dataset``, ``path``, ``n``.
    """
    cells = []
    for path in sorted(GENERATIONS_DIR.glob("unc_*_vne_*.json")):
        # Filename pattern: unc_{dataset}_{model_tag}_vne_{n}.json
        # Some old files use a different pattern; we parse defensively.
        stem = path.stem  # e.g., unc_triviaqa_qwen3_8b_vne_300
        parts = stem.split("_")
        if parts[0] != "unc" or "vne" not in parts:
            continue
        # Find vne marker
        vne_idx = parts.index("vne")
        n_str = parts[vne_idx + 1]
        try:
            n = int(n_str)
        except ValueError:
            continue
        dataset = parts[1]
        model_tag = "_".join(parts[2:vne_idx])
        cells.append({
            "model": model_tag,
            "dataset": dataset,
            "path": path,
            "n": n,
        })
    return cells


# ---------------------------------------------------------------------------
# PKE computation per item (with caching)
# ---------------------------------------------------------------------------

def compute_pke_for_items(
    items: list[dict[str, Any]],
    encoder: SentenceEncoder,
) -> np.ndarray:
    """Compute PKE for each item by pooling its answers and embedding them.

    To minimize encoder calls we batch all answers across items into a single
    encode() call, then split back per item.

    Args:
        items: List of unc-file items, each with an ``answers`` field that is
            a list[n_clarifications] of list[m_answers] of strings.
        encoder: Sentence encoder.

    Returns:
        (n_items,) array of PKE values.
    """
    # 1. Flatten answers across all items, remember per-item slice bounds.
    flat_answers: list[str] = []
    bounds: list[tuple[int, int]] = []  # (start, end) per item

    for item in items:
        ans_grid = item["answers"]  # list[n_clar] of list[m_ans] of str
        flat = [s for group in ans_grid for s in group]
        start = len(flat_answers)
        flat_answers.extend(flat)
        bounds.append((start, start + len(flat)))

    # 2. Encode in batches via sentence-transformers (fast on CPU/GPU).
    if not flat_answers:
        return np.zeros(len(items))

    embeddings = encoder.encode(flat_answers, normalize=True)

    # 3. Compute PKE per item.
    pke = np.zeros(len(items))
    for i, (s, e) in enumerate(bounds):
        if e - s < 2:
            pke[i] = 0.0
            continue
        # Embeddings are already normalized; pass normalize_embeddings=False.
        pke[i] = compute_pke(embeddings[s:e], normalize_embeddings=False)
    return pke


# ---------------------------------------------------------------------------
# Per-cell evaluation
# ---------------------------------------------------------------------------

def _safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Return NaN if AUROC is undefined (all same label)."""
    n = len(labels)
    n_pos = int(labels.sum())
    if n == 0 or n_pos == 0 or n_pos == n:
        return float("nan")
    return float(compute_auroc(labels, scores))


def evaluate_cell(
    cell: dict[str, Any],
    encoder: SentenceEncoder,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute PKE on Test split and report PKE-vs-U_total AUROC."""
    path = cell["path"]
    logger.info("Loading %s", path.name)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        logger.warning("  empty file, skipping")
        return None

    # Split into 40% weight selection, 30% calibration, and 30% test.
    val1, val2, test = split_data(
        data, val1_frac=0.4, val2_frac=0.3, seed=seed
    )

    # Compute PKE on Test items only (matches evaluation domain).
    t0 = time.time()
    pke_test = compute_pke_for_items(test, encoder)
    elapsed = time.time() - t0

    # Build labels and existing signals.
    labels_hall = np.array(
        [1 if d["correct_cluster_idx"] < 0 else 0 for d in test]
    )
    u_total = np.array([float(d["U_total"]) for d in test])

    auroc_pke = _safe_auroc(labels_hall, pke_test)
    auroc_u_total = _safe_auroc(labels_hall, u_total)

    n_match = int((labels_hall == 0).sum())
    n_pos = int(labels_hall.sum())

    result = {
        "model": cell["model"],
        "dataset": cell["dataset"],
        "n_total": len(data),
        "n_test": len(test),
        "n_test_matched": n_match,
        "n_test_unmatched": n_pos,
        "auroc_pke": auroc_pke,
        "auroc_u_total": auroc_u_total,
        "auroc_diff": (
            float(auroc_pke - auroc_u_total)
            if not (np.isnan(auroc_pke) or np.isnan(auroc_u_total))
            else float("nan")
        ),
        "pke_mean": float(np.mean(pke_test)),
        "pke_std": float(np.std(pke_test)),
        "u_total_mean": float(np.mean(u_total)),
        "u_total_std": float(np.std(u_total)),
        "elapsed_sec": elapsed,
    }

    logger.info(
        "  Test: %d (matched=%d, hall=%d) | PKE=%.3f vs U_tot=%.3f (diff=%+.3f) | %.1fs",
        len(test), n_match, n_pos,
        auroc_pke, auroc_u_total, result["auroc_diff"], elapsed,
    )
    return result


# ---------------------------------------------------------------------------
# Aggregation + reporting
# ---------------------------------------------------------------------------

def aggregate_and_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Print per-cell table, per-dataset breakdown, and grand mean."""
    # Per-cell table
    print("")
    print("=" * 88)
    print("PKE vs U_total AUROC (hallucination = correct_cluster_idx == -1)")
    print("=" * 88)
    print(f"{'Model':<22} {'Dataset':<14} {'n_test':>6} {'hall%':>6} "
          f"{'PKE':>7} {'U_tot':>7} {'diff':>7}")
    print("-" * 88)
    for r in results:
        hall_pct = 100.0 * r["n_test_unmatched"] / max(r["n_test"], 1)
        diff_str = (
            f"{r['auroc_diff']:+.3f}" if not np.isnan(r["auroc_diff"])
            else "  nan "
        )
        pke_str = f"{r['auroc_pke']:.3f}" if not np.isnan(r["auroc_pke"]) else "  nan"
        ut_str = f"{r['auroc_u_total']:.3f}" if not np.isnan(r["auroc_u_total"]) else "  nan"
        print(f"{r['model']:<22} {r['dataset']:<14} {r['n_test']:>6} "
              f"{hall_pct:>5.1f}% {pke_str:>7} {ut_str:>7} {diff_str:>7}")
    print("-" * 88)

    # Identify non-degenerate cells: AUROC defined for both signals.
    valid = [
        r for r in results
        if not (np.isnan(r["auroc_pke"]) or np.isnan(r["auroc_u_total"]))
    ]
    n_total_cells = len(results)
    n_valid = len(valid)
    print(f"Non-degenerate cells: {n_valid}/{n_total_cells}")
    if n_valid == 0:
        return {"summary": {"n_valid": 0}, "per_cell": results}

    pke_arr = np.array([r["auroc_pke"] for r in valid])
    ut_arr = np.array([r["auroc_u_total"] for r in valid])
    diff_arr = pke_arr - ut_arr

    print("")
    print(f"  Mean PKE AUROC:     {pke_arr.mean():.4f}  (std {pke_arr.std():.4f})")
    print(f"  Mean U_total AUROC: {ut_arr.mean():.4f}  (std {ut_arr.std():.4f})")
    print(f"  Mean diff (PKE-U_total): {diff_arr.mean():+.4f} "
          f"(std {diff_arr.std():.4f}, min {diff_arr.min():+.4f}, "
          f"max {diff_arr.max():+.4f})")

    # Per-dataset breakdown (mean across models)
    print("")
    print("Per-dataset breakdown (mean across models):")
    print(f"{'Dataset':<14} {'#cells':>6} {'PKE':>8} {'U_tot':>8} {'diff':>8}")
    print("-" * 50)
    by_dataset: dict[str, list[dict]] = {}
    for r in valid:
        by_dataset.setdefault(r["dataset"], []).append(r)
    per_dataset_summary = {}
    for ds in sorted(by_dataset):
        rs = by_dataset[ds]
        pke_m = float(np.mean([r["auroc_pke"] for r in rs]))
        ut_m = float(np.mean([r["auroc_u_total"] for r in rs]))
        diff_m = pke_m - ut_m
        print(f"{ds:<14} {len(rs):>6} {pke_m:>8.3f} {ut_m:>8.3f} {diff_m:>+8.3f}")
        per_dataset_summary[ds] = {
            "n_cells": len(rs),
            "mean_auroc_pke": pke_m,
            "mean_auroc_u_total": ut_m,
            "mean_diff": diff_m,
        }
    print("-" * 50)

    # Per-model breakdown
    print("")
    print("Per-model breakdown (mean across datasets):")
    print(f"{'Model':<22} {'#cells':>6} {'PKE':>8} {'U_tot':>8} {'diff':>8}")
    print("-" * 60)
    by_model: dict[str, list[dict]] = {}
    for r in valid:
        by_model.setdefault(r["model"], []).append(r)
    per_model_summary = {}
    for m in sorted(by_model):
        rs = by_model[m]
        pke_m = float(np.mean([r["auroc_pke"] for r in rs]))
        ut_m = float(np.mean([r["auroc_u_total"] for r in rs]))
        diff_m = pke_m - ut_m
        print(f"{m:<22} {len(rs):>6} {pke_m:>8.3f} {ut_m:>8.3f} {diff_m:>+8.3f}")
        per_model_summary[m] = {
            "n_cells": len(rs),
            "mean_auroc_pke": pke_m,
            "mean_auroc_u_total": ut_m,
            "mean_diff": diff_m,
        }
    print("-" * 60)

    # Pairwise wins
    n_pke_wins = int((diff_arr > 0).sum())
    n_ut_wins = int((diff_arr < 0).sum())
    n_ties = int((diff_arr == 0).sum())
    print("")
    print(f"Wins (PKE > U_total): {n_pke_wins}/{n_valid}")
    print(f"Wins (U_total > PKE): {n_ut_wins}/{n_valid}")
    print(f"Ties:                 {n_ties}/{n_valid}")

    summary = {
        "n_total_cells": n_total_cells,
        "n_valid_cells": n_valid,
        "mean_auroc_pke": float(pke_arr.mean()),
        "mean_auroc_u_total": float(ut_arr.mean()),
        "mean_diff": float(diff_arr.mean()),
        "std_diff": float(diff_arr.std()),
        "min_diff": float(diff_arr.min()),
        "max_diff": float(diff_arr.max()),
        "n_pke_wins": n_pke_wins,
        "n_u_total_wins": n_ut_wins,
        "n_ties": n_ties,
        "per_dataset": per_dataset_summary,
        "per_model": per_model_summary,
    }
    return {"summary": summary, "per_cell": results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("PKE baseline (Gruber & Buettner 2023) -- hallucination AUROC")
    logger.info("Test split: seed=42, val1/val2/test = 0.4/0.3/0.3")

    cells = discover_cells()
    logger.info("Found %d (model x dataset) cells", len(cells))
    for c in cells:
        logger.info("  %s | %s | n=%d", c["model"], c["dataset"], c["n"])

    if not cells:
        logger.error("No unc_*_vne_*.json files found. Aborting.")
        return

    # Load encoder once (CPU is fine; 18 cells x ~250 answers each = ~5k strings).
    logger.info("Loading SentenceEncoder (all-mpnet-base-v2)...")
    encoder = SentenceEncoder(model_name="all-mpnet-base-v2")
    logger.info("  device = %s", encoder.device)

    results: list[dict[str, Any]] = []
    t_global = time.time()
    for cell in cells:
        try:
            r = evaluate_cell(cell, encoder)
            if r is not None:
                results.append(r)
        except Exception as exc:
            logger.exception("Failed cell %s | %s: %s",
                             cell["model"], cell["dataset"], exc)

    elapsed_total = time.time() - t_global
    logger.info("Total time: %.1fs (%d cells)", elapsed_total, len(results))

    aggregate = aggregate_and_report(results)

    # Save JSON.
    out_path = OUTPUT_DIR / "pke_aurocs.json"
    payload = {
        "method": "Predictive Kernel Entropy (Gruber & Buettner 2023)",
        "kernel": "RBF, gamma = 1 / (2 * median(squared distances))",
        "embedding_model": "all-mpnet-base-v2",
        "normalize_embeddings": True,
        "split": {"val1": 0.4, "val2": 0.3, "test": 0.3, "seed": 42},
        "label": "correct_cluster_idx == -1 (hallucination)",
        "elapsed_sec": elapsed_total,
        **aggregate,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
