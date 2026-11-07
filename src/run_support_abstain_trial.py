#!/usr/bin/env python3
"""Cache-only support-aware CAELUM trial.

This script does not call any LLM API and does not regenerate answers. It uses
the cached ``unc_*_vne_*.json`` files and tests the minimal theory repair:

  prediction labels = generated semantic clusters + ABSTAIN_NOT_IN_SUPPORT.

For unmatched examples, the augmented label is covered when the calibrated APS
threshold is large enough to include a final "not generated" label after all
generated clusters. This makes the empirical coverage event match the actual
prediction object instead of silently treating unmatched answers as uncovered.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from caelum.paths import GENERATIONS_DIR, RESULTS_DIR  # noqa: E402

from caelum.calibration import CAELUMCalibrator, _get_score_for_item, _standalone_weight
from caelum.utils import split_data


UNC_RE = re.compile(r"^unc_([a-z0-9]+)_(.+)_vne(?:_(.+))?_(\d+)\.json$")
SKIP_DATASETS = {"condambigqa"}

MODEL_PRETTY = {
    "qwen3_8b": "Qwen3-8B",
    "qwen3_14b": "Qwen3-14B",
    "qwen3_32b": "Qwen3-32B",
    "phi_4": "Phi-4",
    "ministral_3_8b_instruct_2512": "Ministral-3-8B",
    "gemma_4_e4b_it": "Gemma-4-E4B-it",
}

DATASET_PRETTY = {
    "triviaqa": "TriviaQA",
    "ambigqa": "AmbigQA",
    "opennq": "OpenNQ",
    "ambiginst": "AmbigInst",
    "simpleqa": "SimpleQA",
    "popqa": "PopQA",
}

MODEL_ORDER = [
    "qwen3_8b",
    "qwen3_14b",
    "qwen3_32b",
    "phi_4",
    "ministral_3_8b_instruct_2512",
    "gemma_4_e4b_it",
]
DATASET_ORDER = ["triviaqa", "ambigqa", "opennq", "ambiginst", "simpleqa", "popqa"]


def parse_unc_path(path: Path) -> dict[str, Any] | None:
    match = UNC_RE.match(path.name)
    if not match:
        return None
    dataset, model, suffix, n_s = match.groups()
    if dataset in SKIP_DATASETS:
        return None
    n = int(n_s)
    if n <= 50:
        return None
    if suffix:
        model = f"{model}__{suffix}"
    return {"dataset": dataset, "model": model, "n": n, "path": path}


def model_sort_key(model: str) -> tuple[int, str]:
    base = model.split("__", 1)[0]
    return (
        MODEL_ORDER.index(base) if base in MODEL_ORDER else 99,
        model,
    )


def pretty_model(model: str) -> str:
    base, sep, suffix = model.partition("__")
    label = MODEL_PRETTY.get(base, base)
    if sep:
        label = f"{label}+{suffix.replace('_', '-')}"
    return label


def select_unc_files(results_dir: Path) -> list[dict[str, Any]]:
    """Select one cached uncertainty file per model/dataset cell."""
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for path in results_dir.glob("unc_*_vne_*.json"):
        parsed = parse_unc_path(path)
        if parsed is None:
            continue
        key = (parsed["dataset"], parsed["model"])
        if key not in best or parsed["n"] > best[key]["n"]:
            best[key] = parsed
    return sorted(
        best.values(),
        key=lambda row: (
            model_sort_key(row["model"]),
            DATASET_ORDER.index(row["dataset"]) if row["dataset"] in DATASET_ORDER else 99,
        ),
    )


def manual_auroc(scores: list[float], labels: list[int]) -> float:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    if len(y) == 0 or int(y.sum()) == 0 or int(y.sum()) == len(y):
        return float("nan")
    ranks = rankdata(s)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if math.isfinite(precision) and math.isfinite(recall) and precision + recall
        else float("nan")
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def topk_detector_metrics(scores: list[float], labels: list[int], k: int) -> dict[str, float]:
    if k <= 0 or not scores:
        return {"precision": float("nan"), "recall": 0.0, "f1": float("nan")}
    # Stable sort: tied scores keep their input order on every NumPy version.
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    chosen = set(int(i) for i in order[: min(k, len(order))])
    tp = sum(1 for i in chosen if labels[i] == 1)
    fp = sum(1 for i in chosen if labels[i] == 0)
    fn = sum(1 for i, lab in enumerate(labels) if lab == 1 and i not in chosen)
    return prf(tp, fp, fn)


def item_weight(cal: CAELUMCalibrator, item: dict[str, Any]) -> float:
    return _standalone_weight(
        float(item["U_ale"]),
        float(item["U_epi"]),
        cal.optimal_lambda,
        cal.weight_fn,
    )


def include_abstain(cal: CAELUMCalibrator, item: dict[str, Any]) -> bool:
    props = np.asarray(item["cluster_props"], dtype=float)
    other_score = min(float(np.sum(props)), 1.0)
    return other_score <= cal.tau_hat * item_weight(cal, item) + 1e-12


def evaluate_cell(path: Path, alpha: float, seed: int) -> dict[str, Any]:
    items = json.loads(path.read_text(encoding="utf-8"))
    val1, val2, test = split_data(items, val1_frac=0.4, val2_frac=0.3, seed=seed)

    cal = CAELUMCalibrator(alpha=alpha, seed=seed, randomize=False)
    cal.calibrate(val1, val2, verbose=False)

    n = len(test)
    labels_unmatched: list[int] = []
    abstain_scores: list[float] = []
    u_total_scores: list[float] = []
    u_epi_scores: list[float] = []
    u_ale_scores: list[float] = []
    r_scores: list[float] = []
    inverse_max_prop_scores: list[float] = []

    n_matched = n_unmatched = 0
    semantic_covered = augmented_covered = 0
    apss_semantic = 0
    apss_augmented = 0
    abstain_count = 0
    abstain_tp = abstain_fp = abstain_fn = 0
    equivalence_mismatches = 0

    for item in test:
        idx = int(item.get("correct_cluster_idx", -1))
        matched = idx >= 0
        if matched:
            n_matched += 1
        else:
            n_unmatched += 1

        set_size = int(cal.predict_set_size(item))
        abstain = include_abstain(cal, item)
        cluster_covered = matched and idx < set_size
        aug_covered = cluster_covered or ((not matched) and abstain)

        raw_score = _get_score_for_item(
            item, cal.score_mode, cal.optimal_lambda, randomize=cal.randomize
        )
        score_event = raw_score / item_weight(cal, item) <= cal.tau_hat + 1e-12
        if score_event != aug_covered:
            equivalence_mismatches += 1

        semantic_covered += int(cluster_covered)
        augmented_covered += int(aug_covered)
        apss_semantic += set_size
        apss_augmented += set_size + int(abstain)
        abstain_count += int(abstain)

        if abstain and not matched:
            abstain_tp += 1
        elif abstain and matched:
            abstain_fp += 1
        elif (not abstain) and (not matched):
            abstain_fn += 1

        labels_unmatched.append(int(not matched))
        # Higher score means "more likely to abstain / hallucinate".
        abstain_scores.append(cal.tau_hat * item_weight(cal, item))
        u_total_scores.append(float(item.get("U_total", item["U_ale"] + item["U_epi"])))
        u_epi_scores.append(float(item["U_epi"]))
        u_ale_scores.append(float(item["U_ale"]))
        r_scores.append(float(cal.diagnostic_ratio(item)))
        props = np.asarray(item["cluster_props"], dtype=float)
        inverse_max_prop_scores.append(1.0 - float(np.max(props)) if len(props) else 0.0)

    abstain_metrics = prf(abstain_tp, abstain_fp, abstain_fn)
    k = abstain_count
    same_budget = {
        "U_total": topk_detector_metrics(u_total_scores, labels_unmatched, k),
        "U_epi": topk_detector_metrics(u_epi_scores, labels_unmatched, k),
        "U_ale": topk_detector_metrics(u_ale_scores, labels_unmatched, k),
        "R(x)": topk_detector_metrics(r_scores, labels_unmatched, k),
        "1-max_prop": topk_detector_metrics(inverse_max_prop_scores, labels_unmatched, k),
    }

    return {
        "n_test": n,
        "n_matched_test": n_matched,
        "n_unmatched_test": n_unmatched,
        "match_rate_test": n_matched / n if n else float("nan"),
        "lambda_star": cal.optimal_lambda,
        "tau_hat": cal.tau_hat,
        "semantic_coverage": semantic_covered / n if n else float("nan"),
        "augmented_coverage": augmented_covered / n if n else float("nan"),
        "matched_semantic_coverage": semantic_covered / n_matched if n_matched else float("nan"),
        "apss_semantic": apss_semantic / n if n else float("nan"),
        "apss_augmented": apss_augmented / n if n else float("nan"),
        "abstain_rate": abstain_count / n if n else float("nan"),
        "abstain_count": abstain_count,
        "abstain_precision_for_unmatched": abstain_metrics["precision"],
        "abstain_recall_for_unmatched": abstain_metrics["recall"],
        "abstain_f1_for_unmatched": abstain_metrics["f1"],
        "unmatched_leak_rate": abstain_fn / n_unmatched if n_unmatched else float("nan"),
        "matched_false_abstain_rate": abstain_fp / n_matched if n_matched else float("nan"),
        "equivalence_mismatches": equivalence_mismatches,
        "hallucination_auroc": {
            "abstain_effective_threshold": manual_auroc(abstain_scores, labels_unmatched),
            "U_total": manual_auroc(u_total_scores, labels_unmatched),
            "U_epi": manual_auroc(u_epi_scores, labels_unmatched),
            "U_ale": manual_auroc(u_ale_scores, labels_unmatched),
            "R(x)": manual_auroc(r_scores, labels_unmatched),
            "1-max_prop": manual_auroc(inverse_max_prop_scores, labels_unmatched),
        },
        "same_abstain_budget_detector": same_budget,
    }


def fmt_float(x: float, digits: int = 3) -> str:
    if not math.isfinite(float(x)):
        return "NA"
    return f"{x:.{digits}f}"


def write_markdown(out_path: Path, results: dict[str, Any]) -> None:
    rows = list(results["cells"].values())
    summary = results["summary"]
    lines = [
        "# Support-aware ABSTAIN trial",
        "",
        "Cache-only run. No LLM calls, no Gemini, single split seed.",
        "",
        "Augmented label space: generated semantic clusters plus `ABSTAIN_NOT_IN_SUPPORT`.",
        "",
        "## Summary",
        "",
        f"- Cells: {summary['n_cells']}",
        f"- Mean test match rate: {fmt_float(summary['mean_match_rate_test'])}",
        f"- Mean semantic coverage: {fmt_float(summary['mean_semantic_coverage'])}",
        f"- Mean augmented coverage: {fmt_float(summary['mean_augmented_coverage'])}",
        f"- Cells with augmented coverage >= target: {summary['n_cells_augmented_ge_target']}/{summary['n_cells']}",
        f"- Mean semantic APSS: {fmt_float(summary['mean_apss_semantic'])}",
        f"- Mean augmented APSS: {fmt_float(summary['mean_apss_augmented'])}",
        f"- Mean abstain rate: {fmt_float(summary['mean_abstain_rate'])}",
        f"- Mean abstain precision for unmatched: {fmt_float(summary['mean_abstain_precision_for_unmatched'])}",
        f"- Mean abstain recall for unmatched: {fmt_float(summary['mean_abstain_recall_for_unmatched'])}",
        f"- Total score/prediction equivalence mismatches: {summary['total_equivalence_mismatches']}",
        "",
        "## Per-cell results",
        "",
        "| Model | Dataset | Match(test) | Cov semantic | Cov augmented | APSS sem | APSS aug | Abstain | Prec(unmatched) | Recall(unmatched) | False abstain matched | Leak unmatched | lambda |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {dataset} | {match} | {cov_s} | {cov_a} | {apss_s} | "
            "{apss_a} | {abs_rate} | {prec} | {rec} | {fa} | {leak} | {lam} |".format(
                model=pretty_model(row["model"]),
                dataset=DATASET_PRETTY.get(row["dataset"], row["dataset"]),
                match=fmt_float(row["match_rate_test"]),
                cov_s=fmt_float(row["semantic_coverage"]),
                cov_a=fmt_float(row["augmented_coverage"]),
                apss_s=fmt_float(row["apss_semantic"], 2),
                apss_a=fmt_float(row["apss_augmented"], 2),
                abs_rate=fmt_float(row["abstain_rate"]),
                prec=fmt_float(row["abstain_precision_for_unmatched"]),
                rec=fmt_float(row["abstain_recall_for_unmatched"]),
                fa=fmt_float(row["matched_false_abstain_rate"]),
                leak=fmt_float(row["unmatched_leak_rate"]),
                lam=fmt_float(row["lambda_star"], 2),
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(cells: dict[str, Any], alpha: float) -> dict[str, Any]:
    rows = list(cells.values())

    def mean_key(key: str) -> float:
        vals = [float(r[key]) for r in rows if math.isfinite(float(r[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "n_cells": len(rows),
        "mean_match_rate_test": mean_key("match_rate_test"),
        "mean_semantic_coverage": mean_key("semantic_coverage"),
        "mean_augmented_coverage": mean_key("augmented_coverage"),
        "n_cells_augmented_ge_target": sum(
            1 for r in rows if float(r["augmented_coverage"]) >= 1.0 - alpha
        ),
        "mean_matched_semantic_coverage": mean_key("matched_semantic_coverage"),
        "mean_apss_semantic": mean_key("apss_semantic"),
        "mean_apss_augmented": mean_key("apss_augmented"),
        "mean_abstain_rate": mean_key("abstain_rate"),
        "mean_abstain_precision_for_unmatched": mean_key("abstain_precision_for_unmatched"),
        "mean_abstain_recall_for_unmatched": mean_key("abstain_recall_for_unmatched"),
        "mean_unmatched_leak_rate": mean_key("unmatched_leak_rate"),
        "mean_matched_false_abstain_rate": mean_key("matched_false_abstain_rate"),
        "total_equivalence_mismatches": sum(int(r["equivalence_mismatches"]) for r in rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations_dir", type=Path, default=GENERATIONS_DIR,
                    help="Folder with the unc_* caches.")
    ap.add_argument("--output_dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--write-markdown", action="store_true",
                        help="Also write Markdown reports (off by default).")
    args = ap.parse_args()

    cells: dict[str, Any] = {}
    for row in select_unc_files(args.generations_dir):
        key = f"{row['model']}__{row['dataset']}"
        metrics = evaluate_cell(row["path"], alpha=args.alpha, seed=args.seed)
        metrics.update(
            {
                "model": row["model"],
                "dataset": row["dataset"],
                "n_samples": row["n"],
                "source_file": row["path"].name,
            }
        )
        cells[key] = metrics

    payload = {
        "alpha": args.alpha,
        "seed": args.seed,
        "definition": "generated semantic clusters + ABSTAIN_NOT_IN_SUPPORT",
        "cells": cells,
        "summary": summarize(cells, args.alpha),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "support_abstain_trial.json"
    md_path = args.output_dir / "support_abstain_trial.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.write_markdown:
        write_markdown(md_path, payload)

    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote {json_path}")
    if args.write_markdown:
        print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
