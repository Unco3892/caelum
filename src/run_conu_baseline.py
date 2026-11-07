#!/usr/bin/env python3
"""ConU baseline (Wang et al., EMNLP Findings 2024) on the frozen Table 1 cells.

Two stages, each reading the frozen ``results/generations/unc_*_vne_*.json`` caches
read-only:

``similarities`` (GPU recommended)
    For one cell, recover the project cluster of every sampled answer and
    compute the authors' semantic-similarity inputs.

    * Cluster labels: the pooled answers are re-clustered with the project's
      own greedy all-mpnet-base-v2 rule at 0.7 and mapped to the stored sorted
      cluster index through the stored representative strings. A question is
      accepted only if the recovered cluster shares equal the stored
      ``cluster_props`` and each representative maps to a distinct cluster.
      GPU embeddings are tried first, CPU embeddings second.
    * STS: ``cross-encoder/stsb-roberta-large`` (pinned revision) on the
      authors' question-prefixed pairs, upper triangle predicted and mirrored,
      unit diagonal, over answers cleaned with the authors' text filter. Also
      STS(question + reference, question + answer) for every reference.

``table1``
    Apply the authors' released scoring and set rule
    (https://github.com/Zhiyuan-GG/Conformal-Uncertainty-Criterion, commit
    c0c450e8dbef489cfed28f585f1a4ddf0e66078b) on the Table 1 protocol:
    the seed-42 40/30/30 split, calibration on V2 questions whose reference
    matches a sampled cluster (ConU requires support), all test questions in
    the evaluation, the project's matching labels, and set size counted in
    project clusters (sampled-answer counts are also reported). The paper
    Eq. 1 variant (lambda = 0.5 with semantic diversity) is reported
    separately. Seeds 43-45 are included for stability. For a like-for-like
    comparison, CAELUM and Vanilla are also calibrated on the same supported
    V2 questions (``support_conditional=True``) with Table 1's own code path
    and seed-42 split; only the calibration population differs from Table 1.

Usage (see slurm/conu_table1.sh for the cluster array)::

    python src/run_conu_baseline.py similarities --input results/generations/unc_triviaqa_qwen3_8b_vne_300.json \
        --out results/conu_table1/stage1/unc_triviaqa_qwen3_8b_vne_300 --cache <hf_cache>
    python src/run_conu_baseline.py table1 --root results/conu_table1
        (writes results/conu_table1/conu_table1_protocol.json)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from caelum.paths import GENERATIONS_DIR, content_sha256  # noqa: E402

STS_MODEL = "cross-encoder/stsb-roberta-large"
STS_REVISION = "2b12c2c0088918e76151fd5937b7bba986ef1f98"
UPSTREAM_COMMIT = "c0c450e8dbef489cfed28f585f1a4ddf0e66078b"
THRESHOLD = 0.7
MODELS = ["qwen3_8b", "ministral_3_8b_instruct_2512", "gemma_4_e4b_it"]
DATASETS = {"triviaqa": 300, "ambigqa": 200, "opennq": 300,
            "ambiginst": 304, "simpleqa": 500, "popqa": 500}


def sha256(path: Path) -> str:
    return content_sha256(path)


def released_clean(text: str) -> str:
    """Behaviour of the authors' ``filter()`` text cleaner (verified by tests)."""
    text = text.strip()
    while "." in text:
        i = text.index(".")
        if i != len(text) - 1 and text[i - 1].isdigit() and text[i + 1].isdigit():
            text = text.replace(".", "(dp)", 1)
        else:
            break
    for sep in ["\n", ".", "#"]:
        if sep in text:
            text = text.split(sep)[0]
    text = text.replace("(dp)", ".")
    return "".join(ch for ch in text if ch.isascii()) or "error"


def references(row: dict) -> list[str]:
    first = row.get("gt_answer")
    refs = ([str(first)] if first else []) + [str(g) for g in row.get("gt_answers", []) if g]
    return list(dict.fromkeys(refs))


def sample_labels(row: dict, cids: list[int]) -> list[int] | None:
    """Map pooled answers to the stored sorted cluster index, or None."""
    pooled = [a for g in row["answers"] for a in g]
    n = len(pooled)
    counts = np.bincount(cids)
    stored = np.asarray(row["cluster_props"])
    reps = row["cluster_representatives"]
    if len(counts) != len(stored) or not np.allclose(np.sort(counts / n)[::-1], stored):
        return None
    mapping: dict[int, int] = {}
    for k, rep in enumerate(reps):
        cands = {cids[j] for j, a in enumerate(pooled) if a == rep}
        if len(cands) != 1:
            return None
        c = cands.pop()
        if not np.isclose(counts[c] / n, stored[k]) or c in mapping.values():
            return None
        mapping[k] = c
    inverse = {c: k for k, c in mapping.items()}
    return [inverse[c] for c in cids]


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

class _STS:
    def __init__(self, cache: Path, device: str):
        from huggingface_hub import snapshot_download
        from sentence_transformers import CrossEncoder
        location = snapshot_download(STS_MODEL, revision=STS_REVISION, cache_dir=str(cache),
                                     allow_patterns=["*.json", "*.txt", "merges.txt",
                                                     "vocab.json", "model.safetensors"])
        self.model = CrossEncoder(location, num_labels=1, device=device)
        self.pairs = 0

    def predict(self, pairs):
        unique = list(dict.fromkeys(pairs))
        out = self.model.predict(unique, batch_size=256, show_progress_bar=False)
        self.pairs += len(unique)
        lookup = dict(zip(unique, map(float, np.asarray(out).reshape(-1))))
        return np.array([lookup[p] for p in pairs])


def stage_similarities(args) -> None:
    from caelum.utils import SentenceEncoder
    from caelum.uncertainty import cluster_by_embedding

    started = time.time()
    args.out.mkdir(parents=True, exist_ok=False)
    digest = sha256(args.input)
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    device = args.device
    enc_main = SentenceEncoder(device=device)
    enc_cpu = None
    labels, how = [], {device: 0, "cpu_fallback": 0, "failed": 0}
    for row in rows:
        pooled = [x for g in row["answers"] for x in g]
        lab = sample_labels(row, cluster_by_embedding(pooled, enc_main, threshold=THRESHOLD))
        if lab is not None:
            how[device] += 1
        else:
            enc_cpu = enc_cpu or SentenceEncoder(device="cpu")
            lab = sample_labels(row, cluster_by_embedding(pooled, enc_cpu, threshold=THRESHOLD))
            if lab is not None:
                how["cpu_fallback"] += 1
            else:
                how["failed"] += 1
                lab = [-1] * len(pooled)
        labels.append(lab)
    print("labels", how, f"{time.time() - started:.0f}s", flush=True)

    sts = _STS(args.cache, device)
    sims, gts = [], []
    for qi, row in enumerate(rows):
        q = row["question"]
        answers = [released_clean(x) for g in row["answers"] for x in g]
        idx = [(i, j) for i in range(len(answers)) for j in range(i + 1, len(answers))]
        values = sts.predict([(q + " " + answers[i], q + " " + answers[j]) for i, j in idx])
        s = np.eye(len(answers), dtype=np.float64)
        for (i, j), v in zip(idx, values):
            s[i, j] = s[j, i] = v
        sims.append(s.astype(np.float32))
        refs = references(row)
        if not refs:
            raise SystemExit(f"no reference for row {qi}")
        gts.append(np.stack([sts.predict([(q + " " + g, q + " " + x) for x in answers])
                             for g in refs]).astype(np.float32))
        if (qi + 1) % 50 == 0:
            print("sts", qi + 1, "/", len(rows), f"pairs={sts.pairs} {time.time() - started:.0f}s", flush=True)
    np.savez_compressed(args.out / "sims.npz", sim=np.stack(sims),
                        labels=np.asarray(labels, dtype=np.int16),
                        **{f"gt_{i}": g for i, g in enumerate(gts)})
    if sha256(args.input) != digest:
        raise SystemExit("input changed during the run")
    meta = {"input": args.input.name, "input_sha256": digest, "n": len(rows),
            "label_resolution": how, "sts_unique_pairs": sts.pairs,
            "seconds": time.time() - started, "model": STS_MODEL, "revision": STS_REVISION,
            "job": os.environ.get("SLURM_JOB_ID")}
    (args.out / "SUCCESS.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("done", meta, flush=True)


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------

def _split(n: int, seed: int):
    order = np.random.RandomState(seed).permutation(n)
    a, b = int(.4 * n), int(.3 * n)
    return order[:a], order[a:a + b], order[a + b:]


def _scores(sim: np.ndarray, variant: str) -> np.ndarray:
    freq = (sim >= THRESHOLD).mean(axis=1)
    p = freq / freq.sum()
    if variant == "released":
        return 1 - p
    return 1 - .5 * p - .5 * (sim @ p) / len(p)


def evaluate_cell(rows: list[dict], z, seeds=(42, 43, 44, 45)) -> dict:
    sim, labels = z["sim"].astype(float), z["labels"]
    best_ref = [z[f"gt_{i}"].max(axis=0) for i in range(len(rows))]
    if (labels < 0).any():
        raise ValueError("unresolved answer-to-cluster labels")
    out = {}
    for variant in ("released", "paper_eq1"):
        scores = [_scores(sim[i], variant) for i in range(len(rows))]
        per_seed = []
        for seed in seeds:
            _, v2, test = _split(len(rows), seed)
            cal = []
            for i in v2:
                k = rows[i]["correct_cluster_idx"]
                if k < 0:
                    continue
                members = np.flatnonzero(labels[i] == k)
                cal.append(scores[i][members[np.argmax(best_ref[i][members])]])
            n = len(cal)
            q_hat = float(np.quantile(cal, min(math.ceil((n + 1) * .9) / n, 1.0), method="higher"))
            cov, size_c, size_a, empty, cov_sup = [], [], [], [], []
            for i in test:
                chosen = np.flatnonzero(scores[i] <= q_hat)
                k = rows[i]["correct_cluster_idx"]
                hit = k >= 0 and bool((labels[i][chosen] == k).any())
                cov.append(hit)
                size_c.append(len(set(labels[i][chosen].tolist())))
                size_a.append(len(chosen))
                empty.append(len(chosen) == 0)
                if k >= 0:
                    cov_sup.append(hit)
            per_seed.append({"seed": seed, "q_hat": q_hat, "n_cal_supported": n,
                             "coverage": float(np.mean(cov)),
                             "coverage_supported": float(np.mean(cov_sup)),
                             "apss_clusters": float(np.mean(size_c)),
                             "apss_answers": float(np.mean(size_a)),
                             "empty_rate": float(np.mean(empty)),
                             "pi_test": len(cov_sup) / len(test)})
        out[variant] = per_seed
    return out


def support_calibrated(rows: list[dict]) -> dict:
    """CAELUM and Vanilla calibrated, like ConU, on supported V2 questions only.

    Uses Table 1's own code path (aggregate_with_baselines, seed-42 40/30/30
    split, alpha 0.1); only the calibration population changes.
    """
    from aggregate_with_baselines import evaluate_one_calibrator, split_data
    from caelum.calibration import CAELUMCalibrator

    val1, val2, test = split_data(rows, val1_frac=0.4, val2_frac=0.3, seed=42)
    out = {}
    for name, cal in (("caelum", CAELUMCalibrator(alpha=0.1, seed=42)),
                      ("vanilla", CAELUMCalibrator.vanilla_conformal(alpha=0.1))):
        cal.support_conditional = True
        r = evaluate_one_calibrator(cal, val1, val2, test, name)
        out[name] = {"coverage": r["coverage"], "apss": r["apss"],
                     "lambda_star": float(r["lambda_star"])}
    return out


def stage_table1(args) -> None:
    summary = {"protocol": "authors' released ConU scoring/set rule on the Table 1 split",
               "threshold": THRESHOLD, "sts_model": STS_MODEL, "sts_revision": STS_REVISION,
               "upstream_commit": UPSTREAM_COMMIT, "cells": {}}
    for m in MODELS:
        for d, n in DATASETS.items():
            cell = f"unc_{d}_{m}_vne_{n}"
            path = args.generations_dir / f"{cell}.json"
            rows = json.loads(path.read_text(encoding="utf-8"))
            z = np.load(args.root / "stage1" / cell / "sims.npz")
            entry = {"input_sha256": sha256(path), **evaluate_cell(rows, z),
                     "support_calibrated": support_calibrated(rows)}
            summary["cells"][f"{m}__{d}"] = entry
            s, c = entry["released"][0], entry["support_calibrated"]["caelum"]
            print(f"{m}__{d}: cov={s['coverage']:.3f} sup={s['coverage_supported']:.3f} "
                  f"size={s['apss_clusters']:.2f} answers={s['apss_answers']:.1f} empty={s['empty_rate']:.3f} | "
                  f"CAELUM support-calibrated cov={c['coverage']:.3f} size={c['apss']:.2f}")
    output = args.output or args.root / "conu_table1_protocol.json"
    output.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("wrote", output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("similarities")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p = sub.add_parser("table1")
    p.add_argument("--root", type=Path, required=True, help="directory with stage1/")
    p.add_argument("--generations_dir", type=Path, default=GENERATIONS_DIR)
    p.add_argument("--output", type=Path, default=None,
                   help="default: <root>/conu_table1_protocol.json")
    args = parser.parse_args()
    {"similarities": stage_similarities, "table1": stage_table1}[args.stage](args)


if __name__ == "__main__":
    main()
