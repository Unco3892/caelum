#!/usr/bin/env python3
"""Cross-model transfer of the pre-generation matchability classifier.

Hypothesis: matchability is a property of the *question*, not the *model*.
If true, a classifier trained on (question, Qwen3-matchable) labels should
predict Ministral-matchable above chance — i.e., questions hard for Qwen3
are hard for Ministral too.

For each (source_model, target_model) pair with source != target, we:
  1. Pool all questions from all 6 datasets under source_model into a
     train set with matchability labels under source_model.
  2. Train a logistic regression on pre-generation features (mpnet
     embedding + question prefix one-hot + length features).
  3. Evaluate AUC on target_model's matchability labels for the same pool
     of questions.

If transfer AUC is comparable to within-model AUC (~0.69), matchability is
substantially a property of the question. If transfer AUC drops to ~0.55,
matchability is model-specific.

Outputs ``results/matchability_transfer.json``. Add --write-markdown for a summary.

Usage::

    python -u src/run_matchability_transfer.py
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

from caelum.utils import SentenceEncoder  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from run_stratified_pregen import build_pregen_features  # noqa: E402
from caelum.transfer_alignment import (  # noqa: E402
    TransferRecord, align_transfer_records, load_transfer_records,
)

OUTPUT_DIR = RESULTS_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UNC_RE = re.compile(r"unc_(.+?)_(.+?)_vne_(\d+)\.json")
SKIP_DATASETS = {"condambigqa"}
MODELS = ["qwen3_8b", "ministral_3_8b_instruct_2512", "gemma_4_e4b_it",
          "qwen3_14b", "qwen3_32b"]


def _setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S", force=True,
    )
    for noisy in ("httpx", "httpcore", "sentence_transformers", "transformers",
                  "huggingface_hub", "filelock", "urllib3", "fsspec",
                  "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("transfer")


def discover_files() -> dict[str, dict[str, str]]:
    """Map model -> dataset -> path (largest sample count wins)."""
    out: dict[str, dict[str, tuple[int, str]]] = {}
    for f in sorted(GENERATIONS_DIR.glob("unc_*_vne_*.json")):
        m = UNC_RE.match(f.name)
        if not m:
            continue
        ds, model, n = m.group(1), m.group(2), int(m.group(3))
        if ds in SKIP_DATASETS or model not in MODELS:
            continue
        out.setdefault(model, {})
        if ds not in out[model] or out[model][ds][0] < n:
            out[model][ds] = (n, str(f))
    return {m: {ds: t[1] for ds, t in inner.items()}
            for m, inner in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-markdown", action="store_true",
                        help="Write an optional Markdown summary.")
    args = parser.parse_args()

    logger = _setup_logging()
    t0 = time.time()
    encoder = SentenceEncoder()

    files = discover_files()
    datasets = sorted({ds for inner in files.values() for ds in inner})
    logger.info("Models: %s | datasets: %s", list(files.keys()), datasets)

    # Retain dataset-scoped record IDs: repeated text can have different
    # references or different generation outcomes, hence different labels.
    model_corpus: dict[str, list[TransferRecord]] = {}
    for model, ds_paths in files.items():
        records = []
        for ds, path in sorted(ds_paths.items()):
            records.extend(load_transfer_records(path, ds))
        model_corpus[model] = records
        logger.info(
            "  %s pool: %d questions, match rate %.3f",
            model, len(records), np.mean([r.label for r in records]) if records else 0.0,
        )

    # Build features ONCE for the union of questions (efficient)
    all_questions = sorted({r.question for records in model_corpus.values() for r in records})
    logger.info("Building pre-gen features for %d unique questions", len(all_questions))

    # Make a fake "data" list of question-only items
    fake_data = [{"question": q, "correct_cluster_idx": 0} for q in all_questions]
    X_all, _, _ = build_pregen_features(fake_data, encoder)
    q_to_idx = {q: i for i, q in enumerate(all_questions)}

    transfer_results = {}

    for src in MODELS:
        for tgt in MODELS:
            if src == tgt:
                continue
            aligned_q, src_labels, tgt_labels = align_transfer_records(
                model_corpus[src], model_corpus[tgt],
            )
            src_y_al = np.asarray(src_labels, dtype=int)
            tgt_y_al = np.asarray(tgt_labels, dtype=int)
            if len(aligned_q) < 50:
                logger.warning(
                    "  %s -> %s: only %d aligned questions, skipping",
                    src, tgt, len(aligned_q),
                )
                continue

            # Pull features for the aligned questions
            idx = [q_to_idx[q] for q in aligned_q]
            X = X_all[idx]
            # Train on src labels, evaluate on tgt labels
            if len(np.unique(src_y_al)) < 2 or len(np.unique(tgt_y_al)) < 2:
                logger.warning(
                    "  %s -> %s: degenerate labels, skipping", src, tgt,
                )
                continue

            clf = LogisticRegression(
                C=0.1, max_iter=5000, class_weight="balanced",
                random_state=42,
            )
            clf.fit(X, src_y_al)

            # Predict probabilities, score against target labels
            prob = clf.predict_proba(X)[:, 1]
            auc_in = roc_auc_score(src_y_al, prob)
            auc_transfer = roc_auc_score(tgt_y_al, prob)
            agreement = float((src_y_al == tgt_y_al).mean())

            key = f"{src}__to__{tgt}"
            transfer_results[key] = {
                "n_aligned": len(aligned_q),
                "src_match_rate": float(src_y_al.mean()),
                "tgt_match_rate": float(tgt_y_al.mean()),
                "agreement_rate": agreement,
                "auc_in_sample_on_src": float(auc_in),
                "auc_transfer_to_tgt": float(auc_transfer),
                "delta": float(auc_transfer - auc_in),
            }
            logger.info(
                "  %s -> %s: n=%d  AUC_src=%.3f  AUC_tgt=%.3f  Δ=%+.3f  "
                "label_agree=%.3f",
                src, tgt, len(aligned_q),
                auc_in, auc_transfer, auc_transfer - auc_in, agreement,
            )

    out_json = OUTPUT_DIR / "matchability_transfer.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(transfer_results, f, indent=2)
    logger.info("Wrote %s", out_json)

    # Markdown summary
    lines = [
        "# Cross-model matchability transfer",
        "",
        ("Hypothesis: matchability is a property of the question, not the "
         "model. We train a logistic regression on pre-generation features "
         "(mpnet embedding + question prefix + length) using one model's "
         "matchability labels, and evaluate AUC against another model's "
         "labels for the same questions."),
        "",
        "## Transfer matrix",
        "",
        "| source -> target | n_aligned | src match | tgt match | "
        "AUC in-sample (src) | AUC transfer (tgt) | Δ | label agree |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k, r in transfer_results.items():
        lines.append(
            f"| {k} | {r['n_aligned']} | {r['src_match_rate']:.3f} | "
            f"{r['tgt_match_rate']:.3f} | {r['auc_in_sample_on_src']:.3f} | "
            f"{r['auc_transfer_to_tgt']:.3f} | {r['delta']:+.3f} | "
            f"{r['agreement_rate']:.3f} |"
        )

    auc_in = np.array([r['auc_in_sample_on_src'] for r in transfer_results.values()])
    auc_t = np.array([r['auc_transfer_to_tgt'] for r in transfer_results.values()])
    lines += [
        "",
        f"Mean in-sample AUC: **{auc_in.mean():.3f} ± {auc_in.std():.3f}**",
        f"Mean transfer AUC: **{auc_t.mean():.3f} ± {auc_t.std():.3f}**",
        f"Mean Δ (transfer − in-sample): **{(auc_t - auc_in).mean():+.3f}**",
        "",
        ("Interpretation: if mean transfer AUC ≥ 0.65 it supports the claim "
         "that matchability is substantially a property of the question. "
         "If transfer AUC ≈ 0.5 it indicates the matchability classifier "
         "captured model-specific signal."),
    ]
    out_md = OUTPUT_DIR / "matchability_transfer.md"
    if args.write_markdown:
        out_md.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Wrote %s", out_md)
    logger.info("Total elapsed: %.1f s", time.time() - t0)


if __name__ == "__main__":
    main()
