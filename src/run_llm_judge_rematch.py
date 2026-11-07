#!/usr/bin/env python3
"""Apply LLM-judge tier 8 to all cached unc files.

This is a post-hoc rematch driver: for every cached
``unc_{dataset}_{model}_vne_{n}.json`` file in ``results/``, it
identifies judge-eligible items and sends them to Gemini Flash Lite for
a single question-aware judge call. If the judge returns a valid cluster
index, the item's ``correct_cluster_idx`` is updated.

Two modes (``--mode``):

* ``borderline`` (the original default): only heuristic-unmatched items whose
  maximum cluster-rep <-> GT cosine similarity falls in the borderline
  zone ``[0.3, 0.4)`` are judged. Outputs to ``llm_judged_unc/``.
* ``full``: ALL heuristic-unmatched items (``correct_cluster_idx=-1``)
  are judged regardless of similarity. This is the headline correctness
  protocol: the heuristic matcher keeps its positive matches, and the
  judge adjudicates every remaining negative. Outputs to
  ``llm_judged_unc_full/``.

Outputs
-------
* ``results/llm_judged_unc/<original_filename>.json`` -- one per
  input file, with the same schema and an additional
  ``llm_judge_audit`` field per item that was sent to the judge.
* ``results/llm_judge_match_rates.json`` -- aggregate per-cell
  metrics: original vs judged match rates, borderline counts, flips,
  total API calls, total estimated cost in USD.

Cost control
------------
* Hard cap on total API calls (``--max_calls``, default 10000).
* Hard cap on total estimated cost (``--max_cost_usd``, default 20.0);
  the run stops gracefully once this is reached.
* Running cost is printed every 100 calls.

Usage
-----
    python -u src/run_llm_judge_rematch.py
    python -u src/run_llm_judge_rematch.py \
        --max_calls 5000 --max_cost_usd 10.0

If ``GEMINI_API_KEY`` is not set, the script aborts cleanly without
making any API calls.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Setup paths and silence noisy libs before importing project modules.
# ---------------------------------------------------------------------------

import warnings as _warnings
_warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from caelum.paths import GENERATIONS_DIR, RESULTS_DIR, repo_relative  # noqa: E402
JUDGED_DIR = RESULTS_DIR / "llm_judged_unc"
JUDGED_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = RESULTS_DIR / "llm_judge_rematch.txt"


# ---------------------------------------------------------------------------
# Tee logger for terminal and file output
# ---------------------------------------------------------------------------

class Tee:
    """Redirect writes to both terminal and a log file."""

    def __init__(self, filepath: str | Path) -> None:
        # Reconfigure stdout to UTF-8 to handle non-ASCII content in
        # questions/answers (e.g., diacritics) on Windows console.
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass
        self.terminal = sys.stdout
        self.log = open(str(filepath), "w", encoding="utf-8", buffering=1)

    def write(self, msg: str) -> None:
        self.terminal.write(msg)
        try:
            self.terminal.flush()
        except Exception:
            pass
        self.log.write(msg)
        self.log.flush()

    def flush(self) -> None:
        try:
            self.terminal.flush()
        except Exception:
            pass
        self.log.flush()


def _setup_logging(log_path: Path = LOG_PATH) -> logging.Logger:
    """Configure stdout teeing + library log levels."""
    sys.stdout = Tee(log_path)
    sys.stderr = sys.stdout
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    for noisy in (
        "httpx", "httpcore", "openai", "google",
        "sentence_transformers", "transformers",
        "huggingface_hub", "filelock", "urllib3", "fsspec",
        "safetensors", "accelerate", "torch", "tqdm",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("llm_judge_rematch")


# ---------------------------------------------------------------------------
# Project imports (after sys.path insert)
# ---------------------------------------------------------------------------

from caelum.utils import SentenceEncoder  # noqa: E402
from llm_judge_matcher import (  # noqa: E402
    BORDERLINE_HIGH,
    BORDERLINE_LOW,
    DEFAULT_JUDGE_MODEL,
    estimate_cost_usd,
    is_in_borderline_zone,
    llm_judge_match,
    load_gemini_api_key,
    make_gemini_client,
)


# ---------------------------------------------------------------------------
# Discovery of VNE uncertainty caches
# ---------------------------------------------------------------------------

def discover_vne_cells(results_dir: Path) -> list[dict[str, Any]]:
    """Find all cached VNE uncertainty files keyed by (model, dataset).

    Files follow the pattern ``unc_{dataset}_{model_tag}_vne_{n}.json``.
    If multiple sample counts exist for the same (dataset, model), the
    largest is preferred.

    Args:
        results_dir: Directory to scan.

    Returns:
        List of dicts with keys: path, dataset, model_tag, n_samples,
        cell_key.
    """
    candidates: list[dict[str, Any]] = []
    for f in results_dir.glob("unc_*_vne_*.json"):
        name = f.stem
        parts = name.split("_")
        if len(parts) < 4 or parts[0] != "unc":
            continue
        try:
            n_samples = int(parts[-1])
        except ValueError:
            continue
        if parts[-2] != "vne":
            continue
        dataset = parts[1]
        model_tag = "_".join(parts[2:-2])
        candidates.append({
            "path": str(f),
            "filename": f.name,
            "dataset": dataset,
            "model_tag": model_tag,
            "n_samples": n_samples,
        })
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for c in candidates:
        key = (c["dataset"], c["model_tag"])
        if key not in best or c["n_samples"] > best[key]["n_samples"]:
            best[key] = c
    cells = sorted(best.values(), key=lambda x: (x["dataset"], x["model_tag"]))
    for c in cells:
        c["cell_key"] = f"{c['model_tag']}__{c['dataset']}"
    return cells


# ---------------------------------------------------------------------------
# Borderline detection (vectorised over a whole cell for efficiency)
# ---------------------------------------------------------------------------

def compute_borderline_flags(
    data: list[dict[str, Any]],
    encoder: SentenceEncoder,
    low: float,
    high: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-item max similarity and borderline flags.

    Items already matched (correct_cluster_idx >= 0) are excluded from
    the borderline pool. Empty rep/GT lists yield max_sim = 0.

    Args:
        data: List of cached unc records.
        encoder: SentenceEncoder for embedding the rep + GT strings.
        low: Inclusive lower bound of borderline zone.
        high: Exclusive upper bound of borderline zone.

    Returns:
        (max_sims, borderline_flags) -- both length-n numpy arrays.
        ``borderline_flags[i]`` is True iff item i is unmatched AND
        max_sim[i] in [low, high).
    """
    n = len(data)
    max_sims = np.zeros(n, dtype=float)
    flags = np.zeros(n, dtype=bool)

    # Batch-encode all reps and GTs once for efficiency.
    rep_offsets: list[tuple[int, int]] = []  # (start, count) per item
    gt_offsets: list[tuple[int, int]] = []
    all_reps: list[str] = []
    all_gts: list[str] = []
    for item in data:
        reps = item.get("cluster_representatives") or []
        gts = item.get("gt_answers") or []
        rep_offsets.append((len(all_reps), len(reps)))
        gt_offsets.append((len(all_gts), len(gts)))
        all_reps.extend([str(r) for r in reps])
        all_gts.extend([str(g) for g in gts])

    rep_embs = (
        encoder.encode(all_reps, normalize=True)
        if all_reps else np.zeros((0, 768))
    )
    gt_embs = (
        encoder.encode(all_gts, normalize=True)
        if all_gts else np.zeros((0, 768))
    )

    for i, item in enumerate(data):
        cidx = int(item.get("correct_cluster_idx", -1))
        if cidx >= 0:
            continue  # already matched
        r_start, r_count = rep_offsets[i]
        g_start, g_count = gt_offsets[i]
        if r_count == 0 or g_count == 0:
            continue
        sim = (
            rep_embs[r_start:r_start + r_count]
            @ gt_embs[g_start:g_start + g_count].T
        )
        ms = float(sim.max())
        max_sims[i] = ms
        if is_in_borderline_zone(ms, low=low, high=high):
            flags[i] = True

    return max_sims, flags


# ---------------------------------------------------------------------------
# Cell processing
# ---------------------------------------------------------------------------

def process_cell(
    cell: dict[str, Any],
    encoder: SentenceEncoder,
    client: Any,
    model: str,
    rate_delay: float,
    budget: dict[str, Any],
    low: float,
    high: float,
    logger: logging.Logger,
    mode: str = "borderline",
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Apply LLM-judge tier 8 to judge-eligible items in one cell.

    In ``borderline`` mode, eligible items are unmatched items with
    max cosine similarity in [low, high). In ``full`` mode, ALL
    unmatched items are eligible.

    Mutates ``budget`` in place (calls, prompt/output tokens, cost).

    Args:
        cell: Discovery dict for the cell.
        encoder: SentenceEncoder for borderline detection.
        client: Gemini OpenAI-compatible client.
        model: Judge model identifier.
        rate_delay: Seconds to sleep between successful API calls.
        budget: Mutable dict tracking total_calls, prompt_tokens,
            output_tokens, cost, remaining_calls, max_cost_usd.
        low: Borderline zone lower bound.
        high: Borderline zone upper bound.
        logger: Logger instance.

    Returns:
        Per-cell metrics dict (includes match-rate change, flips,
        skipped counts, judged data path).
    """
    path = Path(cell["path"])
    label = cell["cell_key"]
    logger.info("-" * 70)
    logger.info(
        "Cell %s  (n=%d, file=%s)",
        label, cell["n_samples"], path.name,
    )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    n = len(data)

    # Compute borderline flags from a deep-clean view (do not mutate originals).
    max_sims, borderline_flags = compute_borderline_flags(
        data, encoder, low=low, high=high,
    )
    n_borderline = int(borderline_flags.sum())
    n_unmatched = int(sum(
        1 for it in data if int(it.get("correct_cluster_idx", -1)) < 0
    ))
    n_matched_orig = n - n_unmatched
    orig_match_rate = n_matched_orig / n if n > 0 else 0.0
    logger.info(
        "  Original match rate: %.3f  (%d/%d matched, %d unmatched, "
        "%d in borderline zone [%.2f, %.2f))",
        orig_match_rate, n_matched_orig, n, n_unmatched,
        n_borderline, low, high,
    )

    judged = deepcopy(data)
    n_calls_cell = 0
    n_flipped = 0
    n_skipped_budget = 0
    cell_prompt_tokens = 0
    cell_output_tokens = 0
    cell_errors: dict[str, int] = {}
    if mode == "full":
        judge_target_indices = [
            i for i, it in enumerate(data)
            if int(it.get("correct_cluster_idx", -1)) < 0
        ]
    else:
        judge_target_indices = list(np.where(borderline_flags)[0])
    n_targets = len(judge_target_indices)
    logger.info(
        "  Judge targets (mode=%s): %d items", mode, n_targets,
    )

    t_cell0 = time.time()
    for idx in judge_target_indices:
        # Budget gates.
        if budget["remaining_calls"] <= 0:
            n_skipped_budget = len(judge_target_indices) - judge_target_indices.index(idx)
            logger.warning(
                "  Cell %s: hit max_calls cap with %d items left in this cell.",
                label, n_skipped_budget,
            )
            break
        if budget["cost"] >= budget["max_cost_usd"]:
            n_skipped_budget = len(judge_target_indices) - judge_target_indices.index(idx)
            logger.warning(
                "  Cell %s: hit cost cap (~$%.4f >= $%.2f) with %d items left.",
                label, budget["cost"], budget["max_cost_usd"], n_skipped_budget,
            )
            break

        item = judged[idx]
        question = str(item.get("question", ""))
        gt_answers = list(item.get("gt_answers") or [])
        cluster_reps = list(item.get("cluster_representatives") or [])

        cidx, audit = llm_judge_match(
            question=question,
            gt_answers=gt_answers,
            cluster_reps=cluster_reps,
            cluster_members=None,
            client=client,
            model=model,
            skip_trivial=True,
        )
        # Annotate this item with the judge audit + max-sim diagnostic.
        audit["max_sim_to_gt"] = float(max_sims[idx])
        audit["pre_correct_cluster_idx"] = int(item.get("correct_cluster_idx", -1))
        audit["post_correct_cluster_idx"] = int(cidx)
        item["llm_judge_audit"] = audit

        if audit["called"]:
            n_calls_cell += 1
            budget["total_calls"] += 1
            budget["remaining_calls"] -= 1
            cell_prompt_tokens += int(audit.get("prompt_tokens", 0) or 0)
            cell_output_tokens += int(audit.get("output_tokens", 0) or 0)
            budget["prompt_tokens"] += int(audit.get("prompt_tokens", 0) or 0)
            budget["output_tokens"] += int(audit.get("output_tokens", 0) or 0)
            budget["cost"] = estimate_cost_usd(
                budget["prompt_tokens"], budget["output_tokens"],
            )

            # Apply the flip if judge returned a valid index.
            if cidx >= 0:
                item["correct_cluster_idx"] = int(cidx)
                n_flipped += 1

            # Periodic running-cost prints (every 100 global calls).
            if budget["total_calls"] % 100 == 0:
                logger.info(
                    "    [global] %d calls, %d prompt-tok, %d out-tok, "
                    "~$%.4f spent",
                    budget["total_calls"],
                    budget["prompt_tokens"],
                    budget["output_tokens"],
                    budget["cost"],
                )

            # Polite spacing between successful API calls.
            if rate_delay > 0:
                time.sleep(rate_delay)
        else:
            err_key = audit.get("error") or "unknown"
            cell_errors[err_key] = cell_errors.get(err_key, 0) + 1

    elapsed_cell = time.time() - t_cell0
    n_matched_judged = sum(
        1 for it in judged if int(it.get("correct_cluster_idx", -1)) >= 0
    )
    judged_match_rate = n_matched_judged / n if n > 0 else 0.0

    # Sanity: judged match rate must be >= original (we only flip from -1 to >=0).
    if judged_match_rate + 1e-9 < orig_match_rate:
        logger.error(
            "  Cell %s: judged match rate %.3f < original %.3f -- "
            "this should be impossible.",
            label, judged_match_rate, orig_match_rate,
        )

    # Persist judged data (preserves originals).
    out_path = (out_dir or JUDGED_DIR) / path.name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(judged, f, ensure_ascii=False, default=float)

    cell_cost = estimate_cost_usd(cell_prompt_tokens, cell_output_tokens)
    logger.info(
        "  Calls=%d  Flipped=%d/%d  judged_match_rate=%.3f  "
        "(delta=+%.3f)  cell_cost~$%.4f  elapsed=%.1fs",
        n_calls_cell, n_flipped, max(n_borderline, 1),
        judged_match_rate, judged_match_rate - orig_match_rate,
        cell_cost, elapsed_cell,
    )
    if cell_errors:
        logger.info("  Cell errors: %s", dict(cell_errors))
    if n_skipped_budget > 0:
        logger.info(
            "  Cell %s: %d borderline items skipped due to budget.",
            label, n_skipped_budget,
        )

    return {
        "cell_key": cell["cell_key"],
        "dataset": cell["dataset"],
        "model_tag": cell["model_tag"],
        "n_samples": n,
        "n_unmatched": n_unmatched,
        "n_borderline": n_borderline,
        "n_targets": n_targets,
        "mode": mode,
        "n_calls": n_calls_cell,
        "n_flipped": n_flipped,
        "n_skipped_budget": n_skipped_budget,
        "original_match_rate": orig_match_rate,
        "judged_match_rate": judged_match_rate,
        "match_rate_delta": judged_match_rate - orig_match_rate,
        "cell_prompt_tokens": cell_prompt_tokens,
        "cell_output_tokens": cell_output_tokens,
        "cell_cost_usd": cell_cost,
        "errors": cell_errors,
        "judged_path": repo_relative(out_path),
        "elapsed_s": elapsed_cell,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Entry point: discover cells, apply LLM-judge tier 8, save artefacts.

    Returns:
        Exit code (0 on success, 1 on missing API key / no cells).
    """
    parser = argparse.ArgumentParser(
        description="Apply LLM-judge tier 8 to cached VNE uncertainty files.",
    )
    parser.add_argument(
        "--mode", type=str, default="borderline",
        choices=("borderline", "full"),
        help="'borderline': judge only unmatched items with max cosine in "
             "[low, high) (the original mode). 'full': judge ALL unmatched items — the "
             "headline correctness protocol.",
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Directory for judged output files. Defaults to "
             "llm_judged_unc/ (borderline) or llm_judged_unc_full/ (full).",
    )
    parser.add_argument(
        "--max_calls", type=int, default=10000,
        help="Hard cap on total API calls across all cells (default 10000).",
    )
    parser.add_argument(
        "--max_cost_usd", type=float, default=20.0,
        help="Hard cap on estimated cost in USD (default 20.0).",
    )
    parser.add_argument(
        "--rate_delay", type=float, default=0.4,
        help="Seconds to sleep between successful API calls (default 0.4 -> "
             "~150 RPM well under Gemini Flash Lite free-tier limits).",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_JUDGE_MODEL,
        help=f"Judge model identifier (default {DEFAULT_JUDGE_MODEL}).",
    )
    parser.add_argument(
        "--low", type=float, default=BORDERLINE_LOW,
        help=f"Borderline-zone lower bound (default {BORDERLINE_LOW}).",
    )
    parser.add_argument(
        "--high", type=float, default=BORDERLINE_HIGH,
        help=f"Borderline-zone upper bound (default {BORDERLINE_HIGH}).",
    )
    parser.add_argument(
        "--generations_dir", type=str, default=str(GENERATIONS_DIR),
        help="Directory containing cached unc_*_vne_*.json files.",
    )
    parser.add_argument(
        "--out_summary", type=str,
        default=str(RESULTS_DIR / "llm_judge_match_rates.json"),
        help="Output path for the aggregate per-cell summary JSON.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (
        RESULTS_DIR / ("llm_judged_unc_full" if args.mode == "full"
                       else "llm_judged_unc")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / f"llm_judge_rematch_{args.mode}.txt"

    logger = _setup_logging(log_path)
    t_start = time.time()

    logger.info("=" * 70)
    logger.info("CAELUM LLM-judge tier 8 rematch (mode=%s)", args.mode)
    logger.info("=" * 70)
    logger.info("model=%s", args.model)
    logger.info("output dir: %s", out_dir)
    if args.mode == "borderline":
        logger.info("borderline zone: [%.2f, %.2f)", args.low, args.high)
    logger.info("budget: max_calls=%d  max_cost_usd=$%.2f  rate_delay=%.2fs",
                args.max_calls, args.max_cost_usd, args.rate_delay)

    # --- API key gate ------------------------------------------------------
    api_key = load_gemini_api_key()
    if not api_key:
        logger.error(
            "GEMINI_API_KEY not set. Aborting cleanly without making any "
            "API calls.",
        )
        return 1

    # --- Discover cells ----------------------------------------------------
    cells = discover_vne_cells(Path(args.generations_dir))
    logger.info("Discovered %d (model, dataset) cells:", len(cells))
    for cell in cells:
        logger.info(
            "  %-32s %-12s n=%d", cell["model_tag"], cell["dataset"],
            cell["n_samples"],
        )
    if not cells:
        logger.error("No VNE unc_*_vne_*.json cells found in %s.",
                     args.generations_dir)
        return 1

    # --- Init client + encoder --------------------------------------------
    logger.info("Initialising Gemini client and sentence encoder...")
    client = make_gemini_client(api_key)
    encoder = SentenceEncoder(model_name="all-mpnet-base-v2")
    logger.info("Encoder + client ready.")

    # --- Process cells -----------------------------------------------------
    budget: dict[str, Any] = {
        "total_calls": 0,
        "prompt_tokens": 0,
        "output_tokens": 0,
        "cost": 0.0,
        "remaining_calls": args.max_calls,
        "max_cost_usd": args.max_cost_usd,
    }
    cell_results: list[dict[str, Any]] = []
    for cell in cells:
        try:
            result = process_cell(
                cell, encoder=encoder, client=client, model=args.model,
                rate_delay=args.rate_delay, budget=budget,
                low=args.low, high=args.high, logger=logger,
                mode=args.mode, out_dir=out_dir,
            )
        except Exception as exc:  # noqa: BLE001 - top-level safety net
            logger.exception("  Cell %s failed: %s", cell["cell_key"], exc)
            continue
        cell_results.append(result)

    elapsed = time.time() - t_start

    # --- Aggregate summary -------------------------------------------------
    summary = {
        "model": args.model,
        "mode": args.mode,
        "out_dir": repo_relative(out_dir),
        "borderline_low": args.low,
        "borderline_high": args.high,
        "max_calls": args.max_calls,
        "max_cost_usd": args.max_cost_usd,
        "rate_delay_s": args.rate_delay,
        "total_api_calls": budget["total_calls"],
        "total_prompt_tokens": budget["prompt_tokens"],
        "total_output_tokens": budget["output_tokens"],
        "total_cost_estimate_usd": budget["cost"],
        "elapsed_s": elapsed,
        "n_cells_processed": len(cell_results),
        "cells": {r["cell_key"]: r for r in cell_results},
        "cells_list": cell_results,
    }
    out_path = Path(args.out_summary)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
    logger.info("Saved per-cell summary to %s", out_path)

    logger.info("=" * 70)
    logger.info("AGGREGATE SUMMARY")
    logger.info("=" * 70)
    logger.info("Cells processed:      %d / %d", len(cell_results), len(cells))
    logger.info("Total API calls:      %d", budget["total_calls"])
    logger.info("Total prompt tokens:  %d", budget["prompt_tokens"])
    logger.info("Total output tokens:  %d", budget["output_tokens"])
    logger.info("Estimated total cost: ~$%.4f", budget["cost"])
    logger.info("Elapsed:              %.1fs (%.1f min)", elapsed, elapsed / 60)
    logger.info("Per-cell match-rate change:")
    logger.info(
        "  %-36s %5s %5s %6s %6s %8s %8s %8s",
        "cell", "n", "bord", "calls", "flip", "orig", "judged", "delta",
    )
    for r in cell_results:
        logger.info(
            "  %-36s %5d %5d %6d %6d %8.3f %8.3f %+8.3f",
            r["cell_key"][:36], r["n_samples"], r["n_borderline"],
            r["n_calls"], r["n_flipped"],
            r["original_match_rate"], r["judged_match_rate"],
            r["match_rate_delta"],
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
