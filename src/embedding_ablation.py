#!/usr/bin/env python3
"""Embedding-sensitivity ablation for CAELUM.

Re-clusters cached gen_*.json with several sentence encoders and reports how
λ*, Coverage, APSS, and U_epi AUROC vary across embedders.

Embedders compared:
    - sentence-transformers/all-mpnet-base-v2   (default; matches Spectral)
    - sentence-transformers/all-MiniLM-L12-v2   (smaller/faster, 384-dim)
    - BAAI/bge-large-en-v1.5                    (modern, 1024-dim)
    - intfloat/e5-large-v2                      (modern, 1024-dim)

For each (gen_file, embedder), the script:
  1. Re-encodes all answer strings with that embedder
  2. Re-clusters at threshold 0.7
  3. Recomputes VNE uncertainty
  4. Re-runs CAELUM calibration on identical 40/30/30 splits
  5. Records λ*, Cov, APSS, U_total/U_epi AUROC

Outputs:
  results/embedding_ablation.md (optional; --write-markdown or --output_md)
  results/embedding_ablation.json

Pure CPU. No new generations. Each (file, embedder) pair takes ~2-5 min.

Usage (default: 3 representative cells):
    python src/embedding_ablation.py
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from caelum.paths import GENERATIONS_DIR, RESULTS_DIR  # noqa: E402

from caelum.calibration import CAELUMCalibrator
from caelum.uncertainty import compute_uncertainty_for_dataset
from caelum.utils import SentenceEncoder, split_data


EMBEDDERS = [
    "sentence-transformers/all-mpnet-base-v2",     # default
    "sentence-transformers/all-MiniLM-L12-v2",     # smaller
    "BAAI/bge-large-en-v1.5",                       # modern, large
    "intfloat/e5-large-v2",                         # modern alt
]

# Representative cells: 1 factual, 1 ambiguous, 1 mixed
DEFAULT_CELLS = [
    str(GENERATIONS_DIR / "gen_triviaqa_qwen3_8b_300.json"),
    str(GENERATIONS_DIR / "gen_ambigqa_qwen3_8b_200.json"),
    str(GENERATIONS_DIR / "gen_popqa_qwen3_8b_500.json"),
]


def manual_auroc(scores, labels):
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    ranks = rankdata(s)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def evaluate_one(gen_path, embedder_name, alpha=0.1, seed=42, cluster_threshold=0.7):
    log = logging.getLogger("ablation")
    log.info("  loading %s", gen_path)
    with open(gen_path, encoding="utf-8") as f:
        gens = json.load(f)
    log.info("  loading encoder: %s", embedder_name)
    encoder = SentenceEncoder(embedder_name)
    log.info("  computing uncertainty (VNE)...")
    items = compute_uncertainty_for_dataset(
        gens,
        encoder=encoder,
        cluster_threshold=cluster_threshold,
        uncertainty_method="vne",
    )
    n_total = len(items)
    n_matched = sum(1 for it in items if it.get("correct_cluster_idx", -1) >= 0)
    match_rate = n_matched / n_total if n_total else 0.0

    val1, val2, test = split_data(items, val1_frac=0.4, val2_frac=0.3, seed=seed)
    cal = CAELUMCalibrator(alpha=alpha, seed=seed)
    cal.calibrate(val1, val2, verbose=False)
    coverage = float(np.mean([cal.is_covered(it) for it in test]))
    apss = float(np.mean([cal.predict_set_size(it) for it in test]))

    incorrect = [int(it["correct_cluster_idx"] < 0) for it in test]
    u_total = [it.get("U_total", it["U_ale"] + it["U_epi"]) for it in test]
    u_epi = [it["U_epi"] for it in test]
    auroc_total = manual_auroc(u_total, incorrect)
    auroc_epi = manual_auroc(u_epi, incorrect)
    r_vals = [cal.diagnostic_ratio(it) for it in test]
    auroc_r = manual_auroc(r_vals, incorrect)

    return {
        "n_samples": n_total,
        "match_rate": match_rate,
        "lambda_star": cal.optimal_lambda,
        "coverage": coverage,
        "apss": apss,
        "auroc_total": auroc_total,
        "auroc_epi": auroc_epi,
        "auroc_r": auroc_r,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_files", nargs="*", default=DEFAULT_CELLS)
    ap.add_argument("--embedders", nargs="*", default=EMBEDDERS)
    ap.add_argument("--output_md", default=None,
                    help="Write Markdown to this path (also enables the summary).")
    ap.add_argument("--output_json", default=str(RESULTS_DIR / "embedding_ablation.json"))
    ap.add_argument("--write-markdown", action="store_true",
                    help="Write an optional Markdown summary.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    log = logging.getLogger("ablation")
    for noisy in ("sentence_transformers", "transformers", "huggingface_hub", "filelock", "urllib3", "torch", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    results = {}  # gen_file -> embedder -> metrics
    for gen_path in args.gen_files:
        gp = Path(gen_path)
        if not gp.exists():
            log.warning("Missing: %s", gp)
            continue
        results[gp.name] = {}
        for emb in args.embedders:
            log.info("--- %s   x   %s ---", gp.name, emb.split("/")[-1])
            try:
                metrics = evaluate_one(gp, emb)
                results[gp.name][emb] = metrics
                log.info(
                    "  match=%.1f%%  lambda*=%.3g  cov=%.3f  APSS=%.2f  "
                    "U_epi AUROC=%.3f  R(x) AUROC=%.3f",
                    100*metrics["match_rate"], metrics["lambda_star"],
                    metrics["coverage"], metrics["apss"],
                    metrics["auroc_epi"], metrics["auroc_r"],
                )
            except Exception as e:
                log.warning("FAILED: %s", e)
                results[gp.name][emb] = {"error": str(e)}

    Path(args.output_json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote: %s", args.output_json)

    # Markdown table
    lines = ["# Embedding-sensitivity ablation\n"]
    lines.append(
        "How CAELUM's outputs change when answer clustering uses a different sentence encoder. "
        "Pipeline is otherwise identical (cluster_threshold=0.7, VNE, additive weight, hard "
        "non-randomized APS, seed=42, alpha=0.1).\n"
    )
    headers = ["Gen file", "Embedder", "N", "Match%", "λ*", "Coverage", "APSS",
               "U_total AUROC", "U_epi AUROC", "R(x) AUROC"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for gen_name, by_emb in results.items():
        for emb, m in by_emb.items():
            if "error" in m:
                continue
            short_gen = gen_name.replace("gen_", "").replace(".json", "")
            short_emb = emb.split("/")[-1]
            lines.append(
                f"| {short_gen} | {short_emb} | {m['n_samples']} | "
                f"{100*m['match_rate']:.1f} | {m['lambda_star']:.3g} | "
                f"{m['coverage']:.3f} | {m['apss']:.2f} | "
                f"{m['auroc_total']:.3f} | {m['auroc_epi']:.3f} | {m['auroc_r']:.3f} |"
            )
    if args.write_markdown or args.output_md is not None:
        output_md = Path(args.output_md or RESULTS_DIR / "embedding_ablation.md")
        output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        log.info("Wrote: %s", output_md)


if __name__ == "__main__":
    main()
