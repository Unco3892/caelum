#!/usr/bin/env python3
"""Stratified coverage with the lambda*-weighted CAELUM score.

Mirrors ``run_stratified_pregen.py`` but replaces the Vanilla-CP score
(``LAMBDA_FIXED = 0.0``) with the lambda*-weighted CAELUM score from the
two-parameter calibrator. lambda* is selected on Val_1 (a held-out split
disjoint from Val_0 used for the matchability classifier and Val_2 used for
tau_strat). This is the App M extension promised in the paper body.

Splits per cell:
    Val_0 (25%)  - matchability classifier training
    Val_1 (25%)  - lambda* selection
    Val_2 (25%)  - tau_strat on the matched stratum
    Test  (25%)  - evaluation

The 25/25/25/25 split is more conservative than the 30/30/40 of v1 because
we now need a 4-way split. lambda* is selected via argmin_apss.

Outputs:
  - ``results/stratified_lambdastar_results.json``
  - Optional ``results/stratified_lambdastar_summary.md`` with --write-markdown.

If several caches describe the same model/dataset cell, use the one with
the largest question count and record the omitted caches in the output JSON.

Usage:
    python -u src/run_stratified_with_lambdastar.py
    python -u src/run_stratified_with_lambdastar.py --seed 43 --output_dir results/split_seeds/seed43
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

_warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from caelum.paths import RESULTS_DIR  # noqa: E402

from caelum.calibration import (  # noqa: E402
    CAELUMCalibrator,
    compute_aps_score,
    compute_prediction_set_size,
    compute_randomized_aps_score,
)
from caelum.utils import SentenceEncoder  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

# Reuse pre-gen feature builder and helpers
from run_stratified_pregen import (  # noqa: E402
    build_pregen_features,
    discover_cells,
    UNC_RE, MODEL_PRETTY, DATASET_PRETTY,
)

OUTPUT_DIR = RESULTS_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SEED = 42
DEFAULT_ALPHA = 0.1
WEIGHT_FN = "additive"
CLASSIFIER_AUC_THRESHOLD = 0.55


def _setup_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S", force=True)
    for n in ("httpx", "httpcore", "sentence_transformers", "transformers",
              "huggingface_hub", "filelock", "urllib3", "fsspec", "matplotlib"):
        logging.getLogger(n).setLevel(logging.WARNING)
    return logging.getLogger("stratified_lambdastar")


def split_four_way(
    n: int, fracs: tuple[float, float, float], seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """4-way split Val0 / Val1 / Val2 / Test."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    f0, f1, f2 = fracs
    n0 = int(round(n * f0))
    n1 = int(round(n * f1))
    n2 = int(round(n * f2))
    return (
        perm[:n0],
        perm[n0:n0 + n1],
        perm[n0 + n1:n0 + n1 + n2],
        perm[n0 + n1 + n2:],
    )


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


def compute_normalised_scores(
    items: list[dict], lambda_val: float, randomize: bool, seed: int,
    weight_fn: str = WEIGHT_FN,
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
    items: list[dict], tau: float, lambda_val: float,
    weight_fn: str = WEIGHT_FN,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(items)
    covered = np.zeros(n, dtype=bool)
    sizes = np.zeros(n, dtype=int)
    for i, item in enumerate(items):
        props = np.asarray(item["cluster_props"], dtype=float)
        size = compute_prediction_set_size(
            props, item["U_ale"], item["U_epi"],
            lambda_val, tau, weight_fn,
        )
        sizes[i] = size
        cidx = int(item["correct_cluster_idx"])
        covered[i] = (cidx >= 0) and (cidx < size)
    return covered, sizes


def evaluate_cell(
    cell: dict, encoder: SentenceEncoder, alpha: float, seed: int,
) -> dict:
    with open(cell["path"], encoding="utf-8") as f:
        data = _ensure_types(json.load(f))
    n = len(data)

    X, y, _ = build_pregen_features(data, encoder)
    idx_val0, idx_val1, idx_val2, idx_test = split_four_way(
        n, fracs=(0.25, 0.25, 0.25), seed=seed,
    )
    items_val0 = [data[i] for i in idx_val0]
    items_val1 = [data[i] for i in idx_val1]
    items_val2 = [data[i] for i in idx_val2]
    items_test = [data[i] for i in idx_test]
    X_val0, y_val0 = X[idx_val0], y[idx_val0]
    X_val2, y_val2 = X[idx_val2], y[idx_val2]
    X_test, y_test = X[idx_test], y[idx_test]

    n_test = len(idx_test)

    # Step 1: lambda* on Val_1 via the CAELUM calibrator (lambda_grid argmin).
    cal = CAELUMCalibrator(alpha=alpha, seed=seed, randomize=True)
    cal.calibrate(items_val1, items_val2, verbose=False)
    lambda_star = float(cal.optimal_lambda)

    # Step 2: matchability classifier on Val_0 (pre-gen features).
    classifier_mode = "learned"
    classifier_auc = float("nan")
    if len(np.unique(y_val0)) < 2:
        m_hat_val2 = y_val2.copy()
        m_hat_test = y_test.copy()
        classifier_mode = "oracle_single_class"
    else:
        clf = LogisticRegression(
            C=0.1, max_iter=5000, class_weight="balanced",
            random_state=seed,
        )
        clf.fit(X_val0, y_val0)
        prob_val2 = clf.predict_proba(X_val2)[:, 1]
        if len(np.unique(y_val2)) >= 2:
            classifier_auc = float(roc_auc_score(y_val2, prob_val2))
        if classifier_auc < CLASSIFIER_AUC_THRESHOLD or math.isnan(classifier_auc):
            m_hat_val2 = y_val2.copy()
            m_hat_test = y_test.copy()
            classifier_mode = "oracle_low_auc"
        else:
            m_hat_val2 = (prob_val2 >= 0.5).astype(int)
            prob_test = clf.predict_proba(X_test)[:, 1]
            m_hat_test = (prob_test >= 0.5).astype(int)

    pi_hat = float(m_hat_test.mean()) if n_test else 0.0

    # Step 3: tau_strat on the matched subset of Val_2 with lambda*-weighted score.
    val2_matched_idx = np.where(m_hat_val2 == 1)[0]
    n_val2_matched = int(len(val2_matched_idx))
    if n_val2_matched < 10:
        tau_strat = float("inf")
    else:
        items_val2_matched = [items_val2[i] for i in val2_matched_idx]
        scores = compute_normalised_scores(
            items_val2_matched, lambda_val=lambda_star,
            randomize=True, seed=seed,
        )
        tau_strat = conformal_quantile(scores, alpha=alpha)

    covered, sizes = evaluate_on_items(items_test, tau_strat, lambda_star)

    matched_mask = m_hat_test == 1
    n_matched = int(matched_mask.sum())
    n_unmatched = int((~matched_mask).sum())
    cov_overall = float(covered.mean()) if n_test else float("nan")
    cov_matched = (
        float(covered[matched_mask].mean()) if n_matched > 0 else float("nan")
    )
    apss_overall = float(sizes.mean()) if n_test else float("nan")
    apss_matched = (
        float(sizes[matched_mask].mean()) if n_matched > 0 else float("nan")
    )

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
        "n_total": n, "n_test": n_test, "n_val2_matched": n_val2_matched,
        "match_rate_overall": float(y.mean()),
        "lambda_star": lambda_star,
        "classifier_mode": classifier_mode,
        "classifier_auc": classifier_auc,
        "tau_strat": (None if not math.isfinite(tau_strat) else tau_strat),
        "pi_hat": pi_hat,
        "cov_overall": cov_overall, "cov_matched": cov_matched,
        "apss_overall": apss_overall, "apss_matched": apss_matched,
        "n_matched_test": n_matched, "n_unmatched_test": n_unmatched,
        "slack": slack, "target_with_slack": target_with_slack,
        "theorem_holds": theorem_holds,
    }


def select_unique_cells(cells: list[dict]) -> tuple[list[dict], list[dict]]:
    """Keep the largest question cache for each model/dataset cell."""
    selected = {}
    for cell in cells:
        key = cell["cell_key"]
        previous = selected.get(key)
        if previous is None or cell["n_samples"] > previous["n_samples"]:
            selected[key] = cell
        elif cell["n_samples"] == previous["n_samples"]:
            raise ValueError(f"Ambiguous caches with equal question counts for {key}")
    kept = list(selected.values())
    excluded = [cell for cell in cells if cell is not selected[cell["cell_key"]]]
    return kept, excluded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR,
                        help="Folder for the JSON output (default: results/).")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--write-markdown", action="store_true",
                        help="Also export a local Markdown report.")
    args = parser.parse_args()

    logger = _setup_logging()
    t0 = time.time()
    cells, excluded_caches = select_unique_cells(discover_cells())
    for cell in excluded_caches:
        logger.info("Omitting smaller cache for %s: %s", cell["cell_key"], cell["path"])
    logger.info(
        "Stratified + lambda*: %d cells, alpha=%.2f, seed=%d",
        len(cells), args.alpha, args.seed,
    )

    encoder = SentenceEncoder()

    results = []
    for i, cell in enumerate(cells, start=1):
        logger.info("[%d/%d] %s", i, len(cells), cell["cell_key"])
        rec = evaluate_cell(cell, encoder, args.alpha, args.seed)
        results.append(rec)
        logger.info(
            "    lam*=%.2f mode=%s AUC=%.3f cov_match=%.3f cov_overall=%.3f thm=%s",
            rec["lambda_star"], rec["classifier_mode"], rec["classifier_auc"],
            rec["cov_matched"], rec["cov_overall"],
            "OK" if rec["theorem_holds"] else "FAIL",
        )

    n_holds = sum(1 for r in results if r["theorem_holds"])
    n_total = len(results)
    n_oracle = sum(1 for r in results if r["classifier_mode"] != "learned")
    cov_match_arr = np.array([r["cov_matched"] for r in results
                              if not math.isnan(r["cov_matched"])])
    cov_overall_arr = np.array([r["cov_overall"] for r in results
                                if not math.isnan(r["cov_overall"])])
    auc_arr = np.array([r["classifier_auc"] for r in results
                        if r["classifier_mode"] == "learned"
                        and not math.isnan(r["classifier_auc"])])
    summary = {
        "n_cells": n_total, "n_theorem_holds": n_holds,
        "n_oracle_fallback": n_oracle,
        "classifier_auc_mean": float(auc_arr.mean()) if len(auc_arr) else float("nan"),
        "classifier_auc_std": float(auc_arr.std()) if len(auc_arr) else float("nan"),
        "cov_matched_mean": float(cov_match_arr.mean()),
        "cov_matched_std": float(cov_match_arr.std()),
        "cov_overall_mean": float(cov_overall_arr.mean()),
        "cov_overall_std": float(cov_overall_arr.std()),
    }

    learned = [r for r in results if r["classifier_mode"] == "learned"]
    learned_summary = {
        "n_cells": len(learned),
        "n_diagnostic_passes": sum(r["theorem_holds"] for r in learned),
    }
    for key in ("cov_matched", "cov_overall", "classifier_auc"):
        values = np.array([r[key] for r in learned if math.isfinite(r[key])])
        learned_summary[f"{key}_mean"] = float(values.mean()) if len(values) else float("nan")
        learned_summary[f"{key}_std"] = float(values.std()) if len(values) else float("nan")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / "stratified_lambdastar_results.json"
    out_json.write_text(json.dumps(
        {"alpha": args.alpha, "seed": args.seed,
         "results": results, "summary": summary,
         "learned_only_summary": learned_summary,
         "excluded_duplicate_caches": excluded_caches}, indent=2,
    ), encoding="utf-8")
    logger.info("Wrote %s", out_json)

    if args.write_markdown:
        lines = [
            "# Stratified coverage with lambda*-weighted CAELUM score",
            "",
            ("The Mondrian construction is instantiated "
             "with the full CAELUM score s/w_lambda*(x), as opposed to the "
             "Vanilla-CP score (lambda=0) used in the body."),
            "",
            f"Seed {args.seed}, alpha {args.alpha}, {n_total} cells "
            f"(25/25/25/25 split for Val_0/Val_1/Val_2/Test).",
            "",
            f"- Empirical diagnostic passes: **{n_holds}/{n_total}** "
            f"(oracle fallback: {n_oracle})",
            f"- Mean validation classifier AUC (learned rows only): "
            f"**{summary['classifier_auc_mean']:.3f} +/- "
            f"{summary['classifier_auc_std']:.3f}**",
            f"- Matched-stratum coverage (including oracle fallbacks): "
            f"**{summary['cov_matched_mean']:.3f} +/- "
            f"{summary['cov_matched_std']:.3f}**",
            f"- Overall coverage: "
            f"**{summary['cov_overall_mean']:.3f} +/- "
            f"{summary['cov_overall_std']:.3f}**",
            "",
            f"- Learned-only matched-stratum coverage ({len(learned)} cells): "
            f"**{learned_summary['cov_matched_mean']:.3f} +/- "
            f"{learned_summary['cov_matched_std']:.3f}**",
            "",
            "Means weight cells equally; standard deviations use ddof=0. "
            "The empirical diagnostic is not a test of the theorem. "
            "Smaller duplicate caches are listed in the JSON output.",
            "",
            "## Per-cell",
            "",
            "| Cell | lambda* | mode | AUC | cov match | cov overall | diagnostic |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in results:
            lines.append(
                f"| {r['cell_key']} | {r['lambda_star']:.2f} | "
                f"{r['classifier_mode']} | {r['classifier_auc']:.3f} | "
                f"{r['cov_matched']:.3f} | {r['cov_overall']:.3f} | "
                f"{'OK' if r['theorem_holds'] else 'FAIL'} |"
            )
        out_md = args.output_dir / "stratified_lambdastar_summary.md"
        out_md.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Wrote %s", out_md)
    logger.info("Total elapsed: %.1f s", time.time() - t0)


if __name__ == "__main__":
    main()
