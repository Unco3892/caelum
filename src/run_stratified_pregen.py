#!/usr/bin/env python3
"""Stratified coverage with a *pre-generation* matchability classifier.

The classifier uses question features available before generation, rather
than response entropies or cluster shapes from the sampled answers.

This script trains m_hat(x) using only **pre-generation** features:
    1. Sentence embedding of the question (all-mpnet-base-v2, frozen)
    2. Question length (chars + tokens)
    3. Question first-token (one-hot for {what, who, when, where, why, how, which, name, ...})
    4. Optional: count of capitalized tokens (named-entity proxy)
    5. Optional: presence of a number / year token

The matchability label is the same: 1 if correct_cluster_idx >= 0 else 0.
The classifier is a logistic regression on the concatenated features; with
~768 + small features and Val0 ~ 60-200 items, an L2-regularised logistic
is the right complexity.

Writes ``results/stratified_pregen_seed<N>.json`` for each split seed and, with
several seeds, ``results/stratified_pregen_summary.json`` pooling the cell-seed
rows. ``--all_models`` adds the Qwen3-14B and Qwen3-32B generators (30
model/dataset combinations) and writes ``stratified_pregen_all_models_*`` files.
Each cell uses its largest question cache.

Usage::

    python -u src/run_stratified_pregen.py --seeds 42 43 44 45
    python -u src/run_stratified_pregen.py --seeds 42 43 44 45 --all_models
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
from typing import Any

import numpy as np

_warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from caelum.paths import GENERATIONS_DIR, RESULTS_DIR  # noqa: E402

from caelum.abstention import (  # noqa: E402
    QUESTION_PREFIXES,
    build_pregen_features,
)
from caelum.calibration import (  # noqa: E402
    compute_aps_score,
    compute_prediction_set_size,
    compute_randomized_aps_score,
)
from caelum.utils import SentenceEncoder  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

OUTPUT_DIR = RESULTS_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SEED = 42
DEFAULT_ALPHA = 0.1
WEIGHT_FN = "additive"
LAMBDA_FIXED = 0.0
CLASSIFIER_AUC_THRESHOLD = 0.55

# ``QUESTION_PREFIXES`` and ``build_pregen_features`` now live in
# ``caelum.abstention`` (imported above) so the abstention utilities and this
# script share a single feature implementation.


def _setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S", force=True,
    )
    for noisy in (
        "httpx", "httpcore", "sentence_transformers", "transformers",
        "huggingface_hub", "filelock", "urllib3", "fsspec", "matplotlib",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("stratified_pregen")


# ---------------------------------------------------------------------------
# Splits, conformal, evaluation (mirrors v1)
# ---------------------------------------------------------------------------

def split_three_way(
    n: int, frac_val0: float, frac_val2: float, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val0 = int(round(n * frac_val0))
    n_val2 = int(round(n * frac_val2))
    return perm[:n_val0], perm[n_val0:n_val0 + n_val2], perm[n_val0 + n_val2:]


def _ensure_types(data: list[dict]) -> list[dict]:
    for item in data:
        if "cluster_props" in item:
            item["cluster_props"] = np.asarray(item["cluster_props"], dtype=float)
        for key in ("U_ale", "U_epi", "U_total"):
            if key in item:
                item[key] = float(item[key])
        if "correct_cluster_idx" in item:
            item["correct_cluster_idx"] = int(item["correct_cluster_idx"])
    return data


def compute_normalised_scores(
    items: list[dict], randomize: bool, seed: int,
    lambda_val: float = LAMBDA_FIXED,
) -> np.ndarray:
    n = len(items)
    if n == 0:
        return np.array([], dtype=float)
    rng = np.random.RandomState(seed)
    raw = np.zeros(n, dtype=float)
    for i, item in enumerate(items):
        props = np.asarray(item["cluster_props"], dtype=float)
        cidx = int(item["correct_cluster_idx"])
        if randomize:
            u = float(rng.uniform())
            raw[i] = compute_randomized_aps_score(props, cidx, u)
        else:
            raw[i] = compute_aps_score(
                props, cidx, item["U_ale"], item["U_epi"], lambda_val,
            )
    ale = np.asarray([item["U_ale"] for item in items], dtype=float)
    epi = np.asarray([item["U_epi"] for item in items], dtype=float)
    weights = np.maximum(1.0 + ale + lambda_val * epi, 1e-12)
    return raw / weights


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Conformal quantile at rank ceil(target * (n + 1)); +inf if that rank exceeds n."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    n = len(scores)
    if n == 0:
        return float("inf")
    target = 1.0 - alpha
    rank = math.ceil(target * (n + 1))
    if rank > n:
        return float("inf")
    conf_level = rank / n
    return float(np.quantile(scores, conf_level, method="higher"))


def evaluate_on_items(
    items: list[dict], tau: float, lambda_val: float = LAMBDA_FIXED,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(items)
    covered = np.zeros(n, dtype=bool)
    sizes = np.zeros(n, dtype=int)
    for i, item in enumerate(items):
        props = np.asarray(item["cluster_props"], dtype=float)
        size = compute_prediction_set_size(
            props, item["U_ale"], item["U_epi"],
            lambda_val, tau, WEIGHT_FN,
        )
        sizes[i] = size
        cidx = int(item["correct_cluster_idx"])
        covered[i] = (cidx >= 0) and (cidx < size)
    return covered, sizes


# ---------------------------------------------------------------------------
# Per-cell evaluation
# ---------------------------------------------------------------------------

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
# The 14B and 32B generators extend the analysis to 30 model/dataset combinations.
EXTENDED_MODEL_PRETTY = {"qwen3_14b": "Qwen3-14B", "qwen3_32b": "Qwen3-32B"}
MODEL_ORDER = ["qwen3_8b", "ministral_3_8b_instruct_2512", "gemma_4_e4b_it",
               "qwen3_14b", "qwen3_32b"]
DATASET_ORDER = ["triviaqa", "ambigqa", "opennq", "ambiginst", "simpleqa", "popqa"]


def discover_cells(all_models: bool = False) -> list[dict]:
    """Every main-generator cache (and the 14B/32B ones with ``all_models``)."""
    models = {**MODEL_PRETTY, **EXTENDED_MODEL_PRETTY} if all_models else MODEL_PRETTY
    cells = []
    for f in sorted(GENERATIONS_DIR.glob("unc_*_vne_*.json")):
        m = UNC_RE.match(f.name)
        if not m:
            continue
        ds, model, n = m.group(1), m.group(2), int(m.group(3))
        if ds in SKIP_DATASETS:
            continue
        if model not in models or ds not in DATASET_PRETTY:
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
    return cells


def evaluate_cell(
    cell: dict, encoder: SentenceEncoder, alpha: float = 0.1, seed: int = 42,
) -> dict:
    with open(cell["path"], encoding="utf-8") as f:
        data = _ensure_types(json.load(f))
    n = len(data)

    # Pre-generation features for the whole cell
    X, y, _ = build_pregen_features(data, encoder)

    idx_val0, idx_val2, idx_test = split_three_way(
        n, frac_val0=0.30, frac_val2=0.30, seed=seed,
    )

    items_val0 = [data[i] for i in idx_val0]
    items_val2 = [data[i] for i in idx_val2]
    items_test = [data[i] for i in idx_test]

    X_val0, y_val0 = X[idx_val0], y[idx_val0]
    X_val2, y_val2 = X[idx_val2], y[idx_val2]
    X_test, y_test = X[idx_test], y[idx_test]

    n_val0 = len(idx_val0)
    n_val2 = len(idx_val2)
    n_test = len(idx_test)

    overall_match_rate = float(y.mean()) if n else 0.0

    # Train logistic with L2 (small Val0, high-dim embedding)
    classifier_mode = "learned"
    classifier_auc = float("nan")
    if len(np.unique(y_val0)) < 2:
        # Single-class Val0 -- fall back to oracle
        m_hat_val2 = y_val2.copy()
        m_hat_test = y_test.copy()
        classifier_mode = "oracle_single_class"
    else:
        clf = LogisticRegression(
            C=0.1, max_iter=5000, class_weight="balanced",
            random_state=seed,
        )
        clf.fit(X_val0, y_val0)
        # Sanity-check AUC on Val2 (not on Test)
        prob_val2 = clf.predict_proba(X_val2)[:, 1]
        if len(np.unique(y_val2)) >= 2:
            classifier_auc = float(roc_auc_score(y_val2, prob_val2))
        if classifier_auc < CLASSIFIER_AUC_THRESHOLD or math.isnan(classifier_auc):
            m_hat_val2 = y_val2.copy()
            m_hat_test = y_test.copy()
            classifier_mode = "oracle_low_auc"
        else:
            thresh = 0.5
            m_hat_val2 = (prob_val2 >= thresh).astype(int)
            prob_test = clf.predict_proba(X_test)[:, 1]
            m_hat_test = (prob_test >= thresh).astype(int)

    pi_hat = float(m_hat_test.mean()) if n_test else 0.0

    # Stratified conformal calibration on matched subset of Val2
    val2_matched_idx = np.where(m_hat_val2 == 1)[0]
    n_val2_matched = int(len(val2_matched_idx))
    if n_val2_matched < 10:
        tau_strat = float("inf")
    else:
        items_val2_matched = [items_val2[i] for i in val2_matched_idx]
        scores_val2_matched = compute_normalised_scores(
            items_val2_matched, randomize=True, seed=seed,
        )
        tau_strat = conformal_quantile(scores_val2_matched, alpha=alpha)

    covered, sizes = evaluate_on_items(items_test, tau=tau_strat)

    matched_mask = m_hat_test == 1
    unmatched_mask = ~matched_mask
    n_matched = int(matched_mask.sum())
    n_unmatched = int(unmatched_mask.sum())

    cov_overall = float(covered.mean()) if n_test > 0 else float("nan")
    cov_matched = (
        float(covered[matched_mask].mean()) if n_matched > 0 else float("nan")
    )
    cov_unmatched = (
        float(covered[unmatched_mask].mean()) if n_unmatched > 0 else float("nan")
    )
    apss_overall = float(sizes.mean()) if n_test > 0 else float("nan")
    apss_matched = (
        float(sizes[matched_mask].mean()) if n_matched > 0 else float("nan")
    )

    # Theorem check (1/sqrt(N) slack absorbing binomial fluctuation)
    if n_val2_matched > 0 and math.isfinite(cov_matched):
        slack = 1.0 / math.sqrt(max(n_val2_matched, 1))
        target_with_slack = (1.0 - alpha) - slack
        theorem_holds = bool(cov_matched >= target_with_slack)
    else:
        slack = float("nan")
        target_with_slack = float("nan")
        theorem_holds = False

    return {
        "cell_key": cell["cell_key"],
        "model": cell["model"], "dataset": cell["dataset"],
        "n_total": n, "n_val0": n_val0, "n_val2": n_val2, "n_test": n_test,
        "n_val2_matched": n_val2_matched,
        "match_rate_overall": overall_match_rate,
        "classifier_mode": classifier_mode,
        "classifier_auc": classifier_auc,
        "tau_strat": (None if not math.isfinite(tau_strat) else tau_strat),
        "pi_hat": pi_hat,
        "cov_overall": cov_overall, "cov_matched": cov_matched,
        "cov_unmatched": cov_unmatched,
        "apss_overall": apss_overall, "apss_matched": apss_matched,
        "n_matched_test": n_matched, "n_unmatched_test": n_unmatched,
        "slack": slack, "target_with_slack": target_with_slack,
        "theorem_holds": theorem_holds,
    }


def largest_cache_per_cell(cells: list[dict]) -> tuple[list[dict], list[str]]:
    """Keep the largest question cache for each model/dataset cell."""
    best: dict[str, dict] = {}
    for cell in cells:
        kept = best.get(cell["cell_key"])
        if kept is None or cell["n_samples"] > kept["n_samples"]:
            best[cell["cell_key"]] = cell
    kept_paths = {c["path"] for c in best.values()}
    excluded = [Path(c["path"]).name for c in cells if c["path"] not in kept_paths]
    return [c for c in cells if c["path"] in kept_paths], excluded


def stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)), "n": int(len(arr))}


def pooled_summary(rows: list[dict]) -> dict:
    """Aggregates over cell-seed rows; learned rows exclude oracle fallback."""
    learned = [r for r in rows if r["classifier_mode"] == "learned"]
    return {
        "n_evaluations": len(rows),
        "n_learned": len(learned),
        "n_cells_with_learned": len({r["cell_key"] for r in learned}),
        "cov_matched_learned": stats([r["cov_matched"] for r in learned]),
        "cov_overall_learned": stats([r["cov_overall"] for r in learned]),
        "classifier_auc_learned": stats([r["classifier_auc"] for r in learned]),
        "cov_matched_all": stats([r["cov_matched"] for r in rows]),
        "cov_overall_all": stats([r["cov_overall"] for r in rows]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[DEFAULT_SEED],
                        help="Split seeds. The paper uses 42 43 44 45; with more than "
                             "one seed a pooled summary is also written.")
    parser.add_argument("--all_models", action="store_true",
                        help="Also include the Qwen3-14B and Qwen3-32B generators "
                             "(30 model/dataset combinations instead of 18).")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--write-markdown", action="store_true",
                        help="Also write the Markdown report.")
    args = parser.parse_args()
    prefix = "stratified_pregen_all_models" if args.all_models else "stratified_pregen"

    logger = _setup_logging()
    t0 = time.time()

    cells, excluded = largest_cache_per_cell(discover_cells(args.all_models))
    for name in excluded:
        logger.info("Skipping %s: a larger cache exists for the same cell.", name)
    logger.info(
        "Pre-generation matchability classifier: %d cells, alpha=%.2f, seeds=%s",
        len(cells), args.alpha, args.seeds,
    )

    encoder = SentenceEncoder()
    logger.info("Loaded sentence encoder all-mpnet-base-v2")

    all_rows: list[dict] = []
    for seed in args.seeds:
        results: list[dict] = []
        for i, cell in enumerate(cells, start=1):
            logger.info("[seed %d, %d/%d] %s", seed, i, len(cells), cell["cell_key"])
            rec = evaluate_cell(cell, encoder, alpha=args.alpha, seed=seed)
            results.append(rec)
            logger.info(
                "    classifier=%s AUC=%.3f cov_match=%.3f cov_overall=%.3f thm=%s",
                rec["classifier_mode"], rec["classifier_auc"],
                rec["cov_matched"], rec["cov_overall"],
                "OK" if rec["theorem_holds"] else "FAIL",
            )
        all_rows += results

        # Per-seed aggregate
        auc_arr = np.array([r["classifier_auc"] for r in results
                            if r["classifier_mode"] == "learned"
                            and not math.isnan(r["classifier_auc"])])
        cov_match_arr = np.array([
            r["cov_matched"] for r in results if not math.isnan(r["cov_matched"])
        ])
        cov_overall_arr = np.array([
            r["cov_overall"] for r in results if not math.isnan(r["cov_overall"])
        ])
        n_holds = sum(1 for r in results if r["theorem_holds"])
        n_total = len(results)
        n_oracle = sum(1 for r in results if r["classifier_mode"] != "learned")
        summary = {
            "n_cells": n_total,
            "n_theorem_holds": n_holds,
            "n_oracle_fallback": n_oracle,
            "classifier_auc_mean": float(auc_arr.mean()) if len(auc_arr) else float("nan"),
            "classifier_auc_std": float(auc_arr.std()) if len(auc_arr) else float("nan"),
            "cov_matched_mean": float(cov_match_arr.mean()),
            "cov_matched_std": float(cov_match_arr.std()),
            "cov_overall_mean": float(cov_overall_arr.mean()),
            "cov_overall_std": float(cov_overall_arr.std()),
        }
        payload = {
            "alpha": args.alpha, "seed": seed, "excluded_duplicate_caches": excluded,
            "results": results, "summary": summary,
        }
        out_json = OUTPUT_DIR / f"{prefix}_seed{seed}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Wrote %s", out_json)

        if args.write_markdown:
            lines = [
                "# Pre-generation matchability classifier",
                "",
                f"Seed {seed}, alpha {args.alpha}, {n_total} cells.",
                "",
                f"- Theorem holds: **{n_holds}/{n_total}** (oracle fallback: {n_oracle})",
                f"- Mean classifier AUC (learned cells): "
                f"**{summary['classifier_auc_mean']:.3f} ± {summary['classifier_auc_std']:.3f}**",
                f"- Matched-stratum coverage: "
                f"**{summary['cov_matched_mean']:.3f} ± {summary['cov_matched_std']:.3f}**",
                f"- Overall coverage: "
                f"**{summary['cov_overall_mean']:.3f} ± {summary['cov_overall_std']:.3f}**",
                "",
                "## Per-cell",
                "",
                "| Cell | mode | AUC | cov match | cov overall | theorem |",
                "|---|---|---|---|---|---|",
            ]
            for r in results:
                lines.append(
                    f"| {r['cell_key']} | {r['classifier_mode']} | "
                    f"{r['classifier_auc']:.3f} | {r['cov_matched']:.3f} | "
                    f"{r['cov_overall']:.3f} | "
                    f"{'OK' if r['theorem_holds'] else 'FAIL'} |"
                )
            out_md = OUTPUT_DIR / f"{prefix}_seed{seed}.md"
            out_md.write_text("\n".join(lines), encoding="utf-8")
            logger.info("Wrote %s", out_md)

    if len(args.seeds) > 1:
        models = sorted({r["model"] for r in all_rows}, key=MODEL_ORDER.index)
        pooled = {
            "alpha": args.alpha, "seeds": args.seeds, "n_cells": len(cells),
            "excluded_duplicate_caches": excluded,
            "summary": pooled_summary(all_rows),
            "by_model": {m: pooled_summary([r for r in all_rows if r["model"] == m])
                         for m in models},
        }
        out_summary = OUTPUT_DIR / f"{prefix}_summary.json"
        out_summary.write_text(json.dumps(pooled, indent=2), encoding="utf-8")
        s = pooled["summary"]
        logger.info(
            "Pooled over %d seeds: %d evaluations, %d learned (%d cells); learned-stratum "
            "coverage %.3f +/- %.3f, overall %.3f +/- %.3f, AUC %.3f +/- %.3f",
            len(args.seeds), s["n_evaluations"], s["n_learned"], s["n_cells_with_learned"],
            s["cov_matched_learned"]["mean"], s["cov_matched_learned"]["sd"],
            s["cov_overall_learned"]["mean"], s["cov_overall_learned"]["sd"],
            s["classifier_auc_learned"]["mean"], s["classifier_auc_learned"]["sd"],
        )
        logger.info("Wrote %s", out_summary)
    logger.info("Total elapsed: %.1f s", time.time() - t0)

if __name__ == "__main__":
    main()
