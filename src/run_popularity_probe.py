#!/usr/bin/env python3
"""Exploratory PopQA popularity probe (appendix: classifier operating points).

Compares two logistic matchability classifiers on the five judged PopQA caches
(500 questions each) over seeds 42-45:

    * ``question_metadata``: non-embedding question features (length, token
      counts, capitalised and numeric tokens, a year pattern, lead word).
    * ``question_metadata_plus_popularity``: the same plus log(1 + s_pop),
      PopQA's supplied subject-popularity value.

Each seed permutes the questions; the first 30% train the classifier, the
middle 30% are unused, and the last 40% are the test fold. Features are scaled
on the training fold, the classifier is an L2 logistic regression (C=0.1,
class-balanced) and the selection threshold is 0.5. There is no oracle
fallback. Results are macro averages over the 20 model-seed folds.

Writes ``results/popqa_popularity_probe.json``. Run from the repository root:

    python src/run_popularity_probe.py
"""
from __future__ import annotations

import argparse

# Handle help and invalid arguments before loading analysis dependencies.
if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    _parser.add_argument("--overwrite", action="store_true",
                         help="Replace an existing results/popqa_popularity_probe.json.")
    _ARGS = _parser.parse_args()

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
JUDGED_DIR = RESULTS_DIR / "llm_judged_unc_full"
OUT_JSON = RESULTS_DIR / "popqa_popularity_probe.json"

MODELS = ["qwen3_8b", "ministral_3_8b_instruct_2512", "gemma_4_e4b_it", "qwen3_14b", "qwen3_32b"]
SEEDS = [42, 43, 44, 45]
VARIANTS = ["question_metadata", "question_metadata_plus_popularity"]
PREFIXES = "what who when where why how which is are was were did do does can name list describe".split()
METRICS = ["test_auc", "selection_fraction", "precision", "recall", "rho"]


def question_features(question: str) -> list[float]:
    """Non-embedding features of one question."""
    tokens = question.split()
    lead = tokens[0].lower() if tokens else ""
    return ([len(question), len(tokens), sum(t[0].isupper() for t in tokens),
             sum(any(c.isdigit() for c in t) for t in tokens),
             bool(re.search(r"\b(?:19|20)\d{2}\b", question))]
            + [lead == w for w in PREFIXES] + [lead not in PREFIXES])


def evaluate_fold(x: np.ndarray, y: np.ndarray, seed: int) -> dict[str, float]:
    """Train on the first 30% of a seeded permutation and test on the last 40%."""
    idx = np.random.RandomState(seed).permutation(len(y))
    n = len(y)
    train, test = idx[:int(0.3 * n)], idx[int(0.6 * n):]
    classifier = make_pipeline(StandardScaler(), LogisticRegression(
        C=0.1, class_weight="balanced", max_iter=5000, random_state=seed))
    classifier.fit(x[train], y[train])
    prob = classifier.predict_proba(x[test])[:, 1]
    selected = prob >= 0.5
    precision = float(precision_score(y[test], selected, zero_division=0))
    return {"test_auc": float(roc_auc_score(y[test], prob)),
            "selection_fraction": float(selected.mean()),
            "precision": precision,
            "recall": float(recall_score(y[test], selected)),
            "rho": 1.0 - precision}


def main() -> None:
    if OUT_JSON.exists() and not _ARGS.overwrite:
        raise SystemExit(f"{OUT_JSON} already exists; pass --overwrite to replace it.")
    cells, folds_by_variant = [], {v: [] for v in VARIANTS}
    for model in MODELS:
        path = JUDGED_DIR / f"unc_popqa_{model}_vne_500.json"
        raw = path.read_bytes()
        items = json.loads(raw.decode("utf-8"))
        y = np.array([int(item["correct_cluster_idx"] >= 0) for item in items])
        base = np.asarray([question_features(item["question"]) for item in items], dtype=float)
        log_pop = np.log1p([float(item["s_pop"]) for item in items])
        variants = {"question_metadata": base,
                    "question_metadata_plus_popularity": np.column_stack([base, log_pop])}
        folds = []
        for seed in SEEDS:
            for variant in VARIANTS:
                fold = {"seed": seed, "variant": variant, **evaluate_fold(variants[variant], y, seed)}
                folds.append(fold)
                folds_by_variant[variant].append(fold)
        cells.append({"model": model, "sha256": hashlib.sha256(raw).hexdigest(), "n": len(items),
                      "log_popularity_auc_all_queries": float(roc_auc_score(y, log_pop)),
                      "folds": folds})
    aggregate = {v: {m: float(np.mean([f[m] for f in folds_by_variant[v]])) for m in METRICS}
                 for v in VARIANTS}
    OUT_JSON.write_text(json.dumps({"protocol": __doc__.split("\n\n")[1].replace("\n", " "),
                                    "aggregate": aggregate, "cells": cells}, indent=2) + "\n",
                        encoding="utf-8")
    for variant in VARIANTS:
        print(variant, " ".join(f"{m}={aggregate[variant][m]:.3f}" for m in METRICS))
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
