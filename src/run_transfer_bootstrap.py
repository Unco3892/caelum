#!/usr/bin/env python3
"""Bootstrap CIs on the cross-model matchability transfer AUC.

The marquee finding -- that a logistic regression trained on one LLM's
matchability labels transfers to another LLM at AUC ~0.78 vs in-sample
~0.80 -- currently reports `0.778 +/- 0.010` (std across the 6 transfer
pairs). For a NeurIPS-grade headline number we want a proper bootstrap
CI on each individual transfer AUC and on the aggregate.

Method: For each (source, target) pair, the classifier is trained on the
full source corpus once. Then we bootstrap-resample the (question, target
label) pairs B times, computing the transfer AUC each time. The empirical
2.5/97.5 quantiles give a 95% CI.

Usage:
    python -u src/run_transfer_bootstrap.py \\
        --bootstrap 1000

Outputs ``results/matchability_transfer_bootstrap.json``.
Add --write-markdown for a Markdown summary.
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
from caelum.paths import RESULTS_DIR  # noqa: E402

from caelum.utils import SentenceEncoder  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from run_stratified_pregen import build_pregen_features  # noqa: E402
from run_matchability_transfer import discover_files, MODELS  # noqa: E402
from caelum.transfer_alignment import (  # noqa: E402
    TransferRecord, align_transfer_records, load_transfer_records,
)

OUTPUT_DIR = RESULTS_DIR


def _setup_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S", force=True)
    for n in ("httpx", "httpcore", "sentence_transformers", "transformers",
              "huggingface_hub", "filelock", "urllib3", "fsspec",
              "matplotlib"):
        logging.getLogger(n).setLevel(logging.WARNING)
    return logging.getLogger("transfer_bootstrap")


def bootstrap_auc(
    y_true: np.ndarray, scores: np.ndarray, B: int = 1000, seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap (mean, 2.5%, 97.5%) AUC by resampling (y, score) pairs."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aucs = []
    for _ in range(B):
        idx = rng.randint(0, n, size=n)
        ys = y_true[idx]
        ss = scores[idx]
        if len(np.unique(ys)) < 2:
            continue
        aucs.append(roc_auc_score(ys, ss))
    a = np.array(aucs)
    return float(a.mean()), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-markdown", action="store_true",
                        help="Write an optional Markdown summary.")
    args = parser.parse_args()

    logger = _setup_logging()
    t0 = time.time()
    encoder = SentenceEncoder()

    files = discover_files()
    model_corpus: dict[str, list[TransferRecord]] = {}
    for model, ds_paths in files.items():
        records = []
        for ds, path in sorted(ds_paths.items()):
            records.extend(load_transfer_records(path, ds))
        model_corpus[model] = records

    all_questions = sorted({r.question for records in model_corpus.values() for r in records})
    fake_data = [{"question": q, "correct_cluster_idx": 0} for q in all_questions]
    X_all, _, _ = build_pregen_features(fake_data, encoder)
    q_to_idx = {q: i for i, q in enumerate(all_questions)}

    logger.info("Bootstrap AUCs with B=%d", args.bootstrap)
    transfer_results: dict[str, dict] = {}

    for src in MODELS:
        for tgt in MODELS:
            if src == tgt:
                continue
            aligned_q, src_labels, tgt_labels = align_transfer_records(
                model_corpus[src], model_corpus[tgt],
            )
            src_y_al = np.asarray(src_labels, dtype=int)
            tgt_y_al = np.asarray(tgt_labels, dtype=int)
            if len(aligned_q) < 50 or len(np.unique(src_y_al)) < 2 or len(np.unique(tgt_y_al)) < 2:
                continue

            idx = [q_to_idx[q] for q in aligned_q]
            X = X_all[idx]
            clf = LogisticRegression(
                C=0.1, max_iter=5000, class_weight="balanced",
                random_state=args.seed,
            )
            clf.fit(X, src_y_al)
            prob = clf.predict_proba(X)[:, 1]

            auc_in_pt = roc_auc_score(src_y_al, prob)
            auc_tg_pt = roc_auc_score(tgt_y_al, prob)

            auc_in_mean, auc_in_lo, auc_in_hi = bootstrap_auc(
                src_y_al, prob, B=args.bootstrap, seed=args.seed,
            )
            auc_tg_mean, auc_tg_lo, auc_tg_hi = bootstrap_auc(
                tgt_y_al, prob, B=args.bootstrap, seed=args.seed + 1,
            )

            key = f"{src}__to__{tgt}"
            transfer_results[key] = {
                "n_aligned": len(aligned_q),
                "auc_in_point": auc_in_pt,
                "auc_in_mean": auc_in_mean,
                "auc_in_ci_low": auc_in_lo,
                "auc_in_ci_high": auc_in_hi,
                "auc_transfer_point": auc_tg_pt,
                "auc_transfer_mean": auc_tg_mean,
                "auc_transfer_ci_low": auc_tg_lo,
                "auc_transfer_ci_high": auc_tg_hi,
                "delta_point": auc_tg_pt - auc_in_pt,
            }
            logger.info(
                "  %s -> %s: AUC_src=%.3f [%.3f, %.3f]  "
                "AUC_tgt=%.3f [%.3f, %.3f]  delta=%+.3f",
                src.split("_")[0], tgt.split("_")[0],
                auc_in_pt, auc_in_lo, auc_in_hi,
                auc_tg_pt, auc_tg_lo, auc_tg_hi,
                auc_tg_pt - auc_in_pt,
            )

    out_json = OUTPUT_DIR / "matchability_transfer_bootstrap.json"
    out_json.write_text(json.dumps(transfer_results, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_json)

    # Aggregate
    auc_in_pts = np.array([r["auc_in_point"] for r in transfer_results.values()])
    auc_tg_pts = np.array([r["auc_transfer_point"] for r in transfer_results.values()])

    lines = [
        "# Cross-model matchability transfer (bootstrap CIs)",
        "",
        f"Bootstrap B = {args.bootstrap}; resample (y, score) pairs.",
        "",
        "## Per-pair transfer with 95% bootstrap CIs",
        "",
        "| source -> target | n | AUC in-sample (95% CI) | AUC transfer (95% CI) | Δ |",
        "|---|---|---|---|---|",
    ]
    for k, r in transfer_results.items():
        lines.append(
            f"| {k} | {r['n_aligned']} | "
            f"{r['auc_in_point']:.3f} [{r['auc_in_ci_low']:.3f}, {r['auc_in_ci_high']:.3f}] | "
            f"{r['auc_transfer_point']:.3f} [{r['auc_transfer_ci_low']:.3f}, {r['auc_transfer_ci_high']:.3f}] | "
            f"{r['delta_point']:+.3f} |"
        )

    n_above_chance = sum(r["auc_transfer_ci_low"] > 0.5
                         for r in transfer_results.values())
    lines += [
        "",
        f"Mean in-sample AUC across pairs: **{auc_in_pts.mean():.3f}**",
        f"Mean transfer AUC across pairs: **{auc_tg_pts.mean():.3f}**",
        f"Mean Δ: **{(auc_tg_pts - auc_in_pts).mean():+.3f}**",
        "",
        (f"{n_above_chance} of {len(transfer_results)} transfer pairs have a "
         "95% bootstrap CI lower bound on transfer AUC strictly above 0.5 "
         "(the chance baseline)."),
        ("These evaluations reuse the same questions for fitting and scoring. "
         "They measure transfer across model labels, not generalization to unseen questions."),
    ]
    out_md = OUTPUT_DIR / "matchability_transfer_bootstrap.md"
    if args.write_markdown:
        out_md.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Wrote %s", out_md)
    logger.info("Total elapsed: %.1f s", time.time() - t0)


if __name__ == "__main__":
    main()
