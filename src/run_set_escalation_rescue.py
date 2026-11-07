#!/usr/bin/env python3
"""Set-valued model escalation from cached generations.

The top-1 escalation pilot asks whether a stronger fallback model's best answer
rescues base-model mistakes.  This script asks the conformal version:

  answer with the base model's calibrated semantic set unless a base-only risk
  score triggers escalation; then answer with the fallback model's calibrated
  semantic set.

No test labels are used by any policy score.  Test labels are used only for
evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from caelum.paths import GENERATIONS_DIR, RESULTS_DIR  # noqa: E402

from caelum.calibration import CAELUMCalibrator  # noqa: E402
from run_model_escalation_rescue import (  # noqa: E402
    DATASET_ORDER,
    MODEL_ORDER,
    discover,
    finite_mean,
    learned_escalation_scores,
    load_by_qid,
    manual_auroc,
    model_key,
    oracle_topk,
    pretty_dataset,
    pretty_model,
    risk_scores,
    split_three_way,
    support_match,
    top1_correct,
    topk,
)


def covered(cal: CAELUMCalibrator, item: dict[str, Any]) -> bool:
    idx = int(item.get("correct_cluster_idx", -1))
    return idx >= 0 and idx < int(cal.predict_set_size(item))


def set_size(cal: CAELUMCalibrator, item: dict[str, Any]) -> int:
    return int(cal.predict_set_size(item))


def eval_set_policy(
    escalate: np.ndarray,
    base_cov: np.ndarray,
    fallback_cov: np.ndarray,
    base_size: np.ndarray,
    fallback_size: np.ndarray,
    base_support: np.ndarray,
    fallback_support: np.ndarray,
    base_top1: np.ndarray,
    fallback_top1: np.ndarray,
) -> dict[str, float]:
    final_cov = np.where(escalate, fallback_cov, base_cov)
    final_size = np.where(escalate, fallback_size, base_size)
    final_support = np.where(escalate, fallback_support, base_support)
    final_top1 = np.where(escalate, fallback_top1, base_top1)
    helpful_cov = escalate & (~base_cov) & fallback_cov
    harmful_cov = escalate & base_cov & (~fallback_cov)
    return {
        "escalate_rate": float(np.mean(escalate)),
        "coverage": float(np.mean(final_cov)),
        "delta_coverage_vs_base": float(np.mean(final_cov) - np.mean(base_cov)),
        "apss": float(np.mean(final_size)),
        "delta_apss_vs_base": float(np.mean(final_size) - np.mean(base_size)),
        "support": float(np.mean(final_support)),
        "delta_support_vs_base": float(np.mean(final_support) - np.mean(base_support)),
        "top1_acc": float(np.mean(final_top1)),
        "delta_top1_vs_base": float(np.mean(final_top1) - np.mean(base_top1)),
        "helpful_coverage_rate": float(np.mean(helpful_cov)),
        "harmful_coverage_rate": float(np.mean(harmful_cov)),
        "precision_helpful_coverage": (
            float(np.sum(helpful_cov) / np.sum(escalate)) if np.sum(escalate) else float("nan")
        ),
        "effective_gpu_cost_units": float(1.0 + np.mean(escalate)),
    }


def learned_set_scores(
    base_train: list[dict[str, Any]],
    fallback_train: list[dict[str, Any]],
    base_test: list[dict[str, Any]],
    base_cal: CAELUMCalibrator,
    fallback_cal: CAELUMCalibrator,
    seed: int,
) -> dict[str, np.ndarray]:
    # Re-use the top-1 learned score machinery as a conservative baseline.
    # It is trained only on calibration items and evaluated on held-out test.
    # The score is not a coverage guarantee; it is an action policy.
    return learned_escalation_scores(
        base_train,
        fallback_train,
        base_test,
        [fallback_train[0]] * len(base_test),  # overwritten below by caller-free labels are not used for test
        seed,
    )


def eval_pair_seed(
    dataset: str,
    base_model: str,
    fallback_model: str,
    base_path: str,
    fallback_path: str,
    seed: int,
    alpha: float,
    rates: tuple[float, ...],
    lambda_grid: np.ndarray | None,
) -> dict[str, Any] | None:
    base_by_id = load_by_qid(base_path)
    fallback_by_id = load_by_qid(fallback_path)
    shared_ids = sorted(set(base_by_id) & set(fallback_by_id))
    if len(shared_ids) < 50:
        return None

    idx1, idx2, idxt = split_three_way(len(shared_ids), seed=seed)
    val1_ids = [shared_ids[i] for i in idx1]
    val2_ids = [shared_ids[i] for i in idx2]
    train_ids = [shared_ids[i] for i in np.concatenate([idx1, idx2])]
    test_ids = [shared_ids[i] for i in idxt]

    base_val1 = [base_by_id[qid] for qid in val1_ids]
    base_val2 = [base_by_id[qid] for qid in val2_ids]
    fallback_val1 = [fallback_by_id[qid] for qid in val1_ids]
    fallback_val2 = [fallback_by_id[qid] for qid in val2_ids]
    base_train = [base_by_id[qid] for qid in train_ids]
    fallback_train = [fallback_by_id[qid] for qid in train_ids]
    base_test = [base_by_id[qid] for qid in test_ids]
    fallback_test = [fallback_by_id[qid] for qid in test_ids]

    base_cal = CAELUMCalibrator(alpha=alpha, seed=seed, randomize=False, lambda_grid=lambda_grid)
    fallback_cal = CAELUMCalibrator(alpha=alpha, seed=seed + 1009, randomize=False, lambda_grid=lambda_grid)
    base_cal.calibrate(base_val1, base_val2, verbose=False)
    fallback_cal.calibrate(fallback_val1, fallback_val2, verbose=False)

    base_cov = np.asarray([covered(base_cal, it) for it in base_test], dtype=bool)
    fallback_cov = np.asarray([covered(fallback_cal, it) for it in fallback_test], dtype=bool)
    base_size = np.asarray([set_size(base_cal, it) for it in base_test], dtype=float)
    fallback_size = np.asarray([set_size(fallback_cal, it) for it in fallback_test], dtype=float)
    base_support = np.asarray([support_match(it) for it in base_test], dtype=bool)
    fallback_support = np.asarray([support_match(it) for it in fallback_test], dtype=bool)
    base_top1 = np.asarray([top1_correct(it) for it in base_test], dtype=bool)
    fallback_top1 = np.asarray([top1_correct(it) for it in fallback_test], dtype=bool)
    helpful_cov = (~base_cov) & fallback_cov

    scores = risk_scores(base_test, base_cal)
    # Top-1 learned policies are included as calibration-only action baselines.
    # They do not use test labels, but their target is top-1 rescue, not set
    # coverage, so they are intentionally not emphasized in the paper text.
    try:
        scores.update(
            learned_escalation_scores(
                base_train,
                fallback_train,
                base_test,
                fallback_test,
                seed=seed,
            )
        )
    except Exception:
        pass
    rng = np.random.RandomState(seed + 1301)
    scores["Random"] = rng.uniform(size=len(base_test))

    rate_rows: dict[str, dict[str, dict[str, float]]] = {}
    for rate in rates:
        rk = f"{rate:.2f}"
        rate_rows[rk] = {}
        for name, score in scores.items():
            rate_rows[rk][name] = eval_set_policy(
                topk(score, rate),
                base_cov,
                fallback_cov,
                base_size,
                fallback_size,
                base_support,
                fallback_support,
                base_top1,
                fallback_top1,
            )
        rate_rows[rk]["OracleCoverageUpper"] = eval_set_policy(
            oracle_topk(base_cov, fallback_cov, rate),
            base_cov,
            fallback_cov,
            base_size,
            fallback_size,
            base_support,
            fallback_support,
            base_top1,
            fallback_top1,
        )

    always_base = eval_set_policy(
        np.zeros(len(base_test), dtype=bool),
        base_cov,
        fallback_cov,
        base_size,
        fallback_size,
        base_support,
        fallback_support,
        base_top1,
        fallback_top1,
    )
    always_fallback = eval_set_policy(
        np.ones(len(base_test), dtype=bool),
        base_cov,
        fallback_cov,
        base_size,
        fallback_size,
        base_support,
        fallback_support,
        base_top1,
        fallback_top1,
    )

    return {
        "dataset": dataset,
        "base_model": base_model,
        "fallback_model": fallback_model,
        "seed": seed,
        "n_shared": len(shared_ids),
        "n_test": len(test_ids),
        "alpha": alpha,
        "base_lambda_star": float(base_cal.optimal_lambda),
        "fallback_lambda_star": float(fallback_cal.optimal_lambda),
        "base_coverage": float(np.mean(base_cov)),
        "fallback_coverage": float(np.mean(fallback_cov)),
        "base_apss": float(np.mean(base_size)),
        "fallback_apss": float(np.mean(fallback_size)),
        "base_support": float(np.mean(base_support)),
        "fallback_support": float(np.mean(fallback_support)),
        "union_support": float(np.mean(base_support | fallback_support)),
        "base_top1_acc": float(np.mean(base_top1)),
        "fallback_top1_acc": float(np.mean(fallback_top1)),
        "helpful_coverage_pool": float(np.mean(helpful_cov)),
        "harmful_coverage_pool": float(np.mean(base_cov & ~fallback_cov)),
        "auroc_helpful_coverage": {
            name: manual_auroc(score, helpful_cov.astype(int))
            for name, score in scores.items()
        },
        "always_base": always_base,
        "always_fallback": always_fallback,
        "rates": rate_rows,
    }


def summarize(rows: list[dict[str, Any]], rates: tuple[float, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n_pair_seed_evals": len(rows),
        "mean_base_coverage": finite_mean([r["base_coverage"] for r in rows]),
        "mean_fallback_coverage": finite_mean([r["fallback_coverage"] for r in rows]),
        "mean_base_apss": finite_mean([r["base_apss"] for r in rows]),
        "mean_fallback_apss": finite_mean([r["fallback_apss"] for r in rows]),
        "mean_base_support": finite_mean([r["base_support"] for r in rows]),
        "mean_fallback_support": finite_mean([r["fallback_support"] for r in rows]),
        "mean_union_support": finite_mean([r["union_support"] for r in rows]),
        "mean_helpful_coverage_pool": finite_mean([r["helpful_coverage_pool"] for r in rows]),
        "mean_harmful_coverage_pool": finite_mean([r["harmful_coverage_pool"] for r in rows]),
        "auroc_helpful_coverage": {},
        "rates": {},
    }
    score_names = sorted({name for r in rows for name in r["auroc_helpful_coverage"]})
    for name in score_names:
        out["auroc_helpful_coverage"][name] = finite_mean([
            r["auroc_helpful_coverage"].get(name, float("nan")) for r in rows
        ])
    for rate in rates:
        rk = f"{rate:.2f}"
        out["rates"][rk] = {}
        rules = sorted({name for r in rows for name in r["rates"][rk]})
        for rule in rules:
            mets = [r["rates"][rk][rule] for r in rows if rule in r["rates"][rk]]
            out["rates"][rk][rule] = {
                key: finite_mean([m[key] for m in mets])
                for key in (
                    "coverage",
                    "delta_coverage_vs_base",
                    "apss",
                    "delta_apss_vs_base",
                    "support",
                    "delta_support_vs_base",
                    "top1_acc",
                    "delta_top1_vs_base",
                    "helpful_coverage_rate",
                    "harmful_coverage_rate",
                    "precision_helpful_coverage",
                    "effective_gpu_cost_units",
                )
            }
    return out


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# Set-valued model-escalation rescue",
        "",
        "Policy: use the base calibrated semantic set unless a base-only risk score triggers escalation; then use the fallback calibrated semantic set.",
        "",
        "No test labels are used by policy scores. OracleCoverageUpper is an upper bound at the same escalation budget.",
        "",
        "## Summary",
        "",
        f"- Pair-seed evaluations: {s['n_pair_seed_evals']}",
        f"- Mean base coverage: {s['mean_base_coverage']:.3f}",
        f"- Mean fallback coverage: {s['mean_fallback_coverage']:.3f}",
        f"- Mean base APSS: {s['mean_base_apss']:.2f}",
        f"- Mean fallback APSS: {s['mean_fallback_apss']:.2f}",
        f"- Mean base support: {s['mean_base_support']:.3f}",
        f"- Mean fallback support: {s['mean_fallback_support']:.3f}",
        f"- Mean union support: {s['mean_union_support']:.3f}",
        f"- Helpful coverage pool: {s['mean_helpful_coverage_pool']:.3f}",
        f"- Harmful coverage pool: {s['mean_harmful_coverage_pool']:.3f}",
        "",
        "## Helpful-coverage ranking",
        "",
        "| Rule | AUROC helpful coverage escalation |",
        "|---|---:|",
    ]
    for name, val in sorted(
        s["auroc_helpful_coverage"].items(),
        key=lambda kv: (-1 if math.isnan(kv[1]) else -kv[1], kv[0]),
    ):
        lines.append(f"| {name} | {'NA' if math.isnan(val) else f'{val:.3f}'} |")

    for rk, rows in s["rates"].items():
        lines.extend([
            "",
            f"## Fixed escalation rate {int(round(100 * float(rk)))}%",
            "",
            "| Rule | Cov | Delta cov | APSS | Delta APSS | Support | Delta support | Top1 | Delta top1 | Helpful/all | Harmful/all | Prec helpful | Cost units |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for name, met in sorted(
            rows.items(),
            key=lambda kv: (kv[1]["delta_coverage_vs_base"], -kv[1]["apss"]),
            reverse=True,
        ):
            lines.append(
                f"| {name} | {met['coverage']:.3f} | {met['delta_coverage_vs_base']:.3f} | "
                f"{met['apss']:.2f} | {met['delta_apss_vs_base']:.2f} | "
                f"{met['support']:.3f} | {met['delta_support_vs_base']:.3f} | "
                f"{met['top1_acc']:.3f} | {met['delta_top1_vs_base']:.3f} | "
                f"{met['helpful_coverage_rate']:.3f} | {met['harmful_coverage_rate']:.3f} | "
                f"{met['precision_helpful_coverage']:.3f} | {met['effective_gpu_cost_units']:.2f} |"
            )

    lines.extend([
        "",
        "## Pair rows",
        "",
        "| Dataset | Base | Fallback | Seed | n | Base cov | Fallback cov | Base APSS | Fallback APSS | Base support | Fallback support | Union support | Helpful cov pool | Harmful cov pool |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in payload["rows"]:
        lines.append(
            f"| {pretty_dataset(r['dataset'])} | {pretty_model(r['base_model'])} | "
            f"{pretty_model(r['fallback_model'])} | {r['seed']} | {r['n_test']} | "
            f"{r['base_coverage']:.3f} | {r['fallback_coverage']:.3f} | "
            f"{r['base_apss']:.2f} | {r['fallback_apss']:.2f} | "
            f"{r['base_support']:.3f} | {r['fallback_support']:.3f} | "
            f"{r['union_support']:.3f} | {r['helpful_coverage_pool']:.3f} | "
            f"{r['harmful_coverage_pool']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations_dir", type=Path, default=GENERATIONS_DIR,
                    help="Folder with the unc_* caches.")
    ap.add_argument("--output_dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--out_prefix", default="set_escalation_all_to_phi4_multiseed")
    ap.add_argument("--fallback_model", default="phi_4")
    ap.add_argument("--base_models", nargs="*", default=["qwen3_8b", "gemma_4_e4b_it", "ministral_3_8b_instruct_2512", "qwen3_14b"])
    ap.add_argument("--datasets", nargs="*", default=DATASET_ORDER)
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44, 45])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--rates", nargs="*", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--fast_lambda_grid", action="store_true")
    ap.add_argument(
        "--include_suffix",
        action="store_true",
        help="Include cache files with clarifier/judge suffixes in the model key.",
    )
    ap.add_argument("--write-markdown", action="store_true",
                        help="Also write Markdown reports (off by default).")
    args = ap.parse_args()

    lambda_grid = None
    if args.fast_lambda_grid:
        lambda_grid = np.asarray([0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])

    discovered = discover(args.generations_dir, include_suffix=args.include_suffix)
    rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        fb = discovered.get((dataset, args.fallback_model))
        if not fb:
            continue
        for base_model in sorted(args.base_models, key=model_key):
            if base_model == args.fallback_model:
                continue
            base = discovered.get((dataset, base_model))
            if not base:
                continue
            for seed in args.seeds:
                row = eval_pair_seed(
                    dataset=dataset,
                    base_model=base_model,
                    fallback_model=args.fallback_model,
                    base_path=base["path"],
                    fallback_path=fb["path"],
                    seed=seed,
                    alpha=args.alpha,
                    rates=tuple(args.rates),
                    lambda_grid=lambda_grid,
                )
                if row is not None:
                    rows.append(row)

    if not rows:
        raise SystemExit("No shared model pairs found.")

    rows = sorted(
        rows,
        key=lambda r: (
            DATASET_ORDER.index(r["dataset"]) if r["dataset"] in DATASET_ORDER else 99,
            model_key(r["base_model"]),
            r["seed"],
        ),
    )
    payload = {
        "alpha": args.alpha,
        "fallback_model": args.fallback_model,
        "rates": args.rates,
        "rows": rows,
        "summary": summarize(rows, tuple(args.rates)),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / f"{args.out_prefix}.json"
    out_md = args.output_dir / f"{args.out_prefix}.md"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.write_markdown:
        write_markdown(out_md, payload)
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote {out_json}")
    if args.write_markdown:
        print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
