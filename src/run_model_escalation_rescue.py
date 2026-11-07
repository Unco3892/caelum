#!/usr/bin/env python3
"""Model-escalation rescue experiment from cached generations.

Policy setting:
  For a base LLM on a query, either answer with its top semantic cluster or
  escalate to a fallback LLM and answer with the fallback top cluster.

The policy score is computed only from the base model cache. Test labels are
used only for evaluation. This is meant to test a real action story without
API calls or oracle labels in the policy.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from caelum.paths import GENERATIONS_DIR, RESULTS_DIR, repo_relative  # noqa: E402

from caelum.calibration import CAELUMCalibrator, _standalone_weight  # noqa: E402


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
    "gemma_4_e4b_it",
    "ministral_3_8b_instruct_2512",
    "qwen3_14b",
    "phi_4",
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
    return {"dataset": dataset, "model": model, "n": n, "path": str(path)}


def discover(results_dir: Path, include_suffix: bool) -> dict[tuple[str, str], dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for path in results_dir.glob("unc_*_vne_*.json"):
        parsed = parse_unc_path(path)
        if parsed is None:
            continue
        if "__" in parsed["model"] and not include_suffix:
            continue
        key = (parsed["dataset"], parsed["model"])
        if key not in best or parsed["n"] > best[key]["n"]:
            best[key] = parsed
    return best


def pretty_model(model: str) -> str:
    base, sep, suffix = model.partition("__")
    out = MODEL_PRETTY.get(base, base)
    if sep:
        out += "+" + suffix.replace("_", "-")
    return out


def pretty_dataset(dataset: str) -> str:
    return DATASET_PRETTY.get(dataset, dataset)


def model_key(model: str) -> tuple[int, str]:
    base = model.split("__", 1)[0]
    return (MODEL_ORDER.index(base) if base in MODEL_ORDER else 99, model)


def split_three_way(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n1 = int(round(0.35 * n))
    n2 = int(round(0.30 * n))
    return perm[:n1], perm[n1:n1 + n2], perm[n1 + n2:]


def item_key(item: dict[str, Any]) -> str:
    qid = item.get("question_id", item.get("datapoint_id"))
    return str(qid)


def top1_correct(item: dict[str, Any]) -> bool:
    return int(item.get("correct_cluster_idx", -1)) == 0


def support_match(item: dict[str, Any]) -> bool:
    return int(item.get("correct_cluster_idx", -1)) >= 0


def load_by_qid(path: str) -> dict[str, dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {item_key(item): item for item in data}


def finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else float("nan")


def manual_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    ok = np.isfinite(scores)
    scores = scores[ok]
    labels = labels[ok]
    if len(labels) == 0 or int(labels.sum()) == 0 or int(labels.sum()) == len(labels):
        return float("nan")
    ranks = rankdata(scores)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def topk(scores: np.ndarray, rate: float) -> np.ndarray:
    n = len(scores)
    k = int(round(rate * n))
    out = np.zeros(n, dtype=bool)
    if k <= 0:
        return out
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    out[order[: min(k, n)]] = True
    return out


def oracle_topk(base_ok: np.ndarray, fallback_ok: np.ndarray, rate: float) -> np.ndarray:
    """Oracle upper bound: choose helpful escalations before harmful ones."""
    gain = fallback_ok.astype(int) - base_ok.astype(int)
    # helpful (+1), neutral (0), harmful (-1)
    scores = gain.astype(float)
    return topk(scores, rate)


def eval_policy(escalate: np.ndarray, base_ok: np.ndarray, fallback_ok: np.ndarray) -> dict[str, float]:
    final_ok = np.where(escalate, fallback_ok, base_ok)
    harmful = escalate & base_ok & ~fallback_ok
    helpful = escalate & ~base_ok & fallback_ok
    return {
        "escalate_rate": float(np.mean(escalate)) if len(escalate) else float("nan"),
        "final_acc": float(np.mean(final_ok)) if len(final_ok) else float("nan"),
        "delta_vs_base": float(np.mean(final_ok) - np.mean(base_ok)) if len(final_ok) else float("nan"),
        "helpful_rate": float(np.mean(helpful)) if len(helpful) else float("nan"),
        "harmful_rate": float(np.mean(harmful)) if len(harmful) else float("nan"),
        "precision_helpful": (
            float(np.sum(helpful) / np.sum(escalate)) if np.sum(escalate) else float("nan")
        ),
    }


def weight(cal: CAELUMCalibrator, item: dict[str, Any]) -> float:
    return _standalone_weight(
        float(item["U_ale"]),
        float(item["U_epi"]),
        cal.optimal_lambda,
        cal.weight_fn,
    )


def risk_scores(base_items: list[dict[str, Any]], cal: CAELUMCalibrator) -> dict[str, np.ndarray]:
    scores: dict[str, np.ndarray] = {}
    scores["U_total"] = np.asarray([
        float(it.get("U_total", it["U_ale"] + it["U_epi"])) for it in base_items
    ])
    scores["U_epi"] = np.asarray([float(it["U_epi"]) for it in base_items])
    scores["U_ale"] = np.asarray([float(it["U_ale"]) for it in base_items])
    scores["1-max_prop"] = np.asarray([
        1.0 - float(np.max(np.asarray(it.get("cluster_props", []), dtype=float)))
        if len(it.get("cluster_props", [])) else 0.0
        for it in base_items
    ])
    scores["CAELUM_margin"] = np.asarray([
        float(cal.tau_hat * weight(cal, it) - 1.0) for it in base_items
    ])
    scores["R(x)"] = np.asarray([float(cal.diagnostic_ratio(it)) for it in base_items])
    return scores


def base_feature_matrix(items: list[dict[str, Any]]) -> np.ndarray:
    rows: list[list[float]] = []
    for it in items:
        props = np.asarray(it.get("cluster_props", []), dtype=float)
        max_prop = float(np.max(props)) if len(props) else 0.0
        p2 = float(props[1]) if len(props) > 1 else 0.0
        p3 = float(props[2]) if len(props) > 2 else 0.0
        gap = max_prop - p2
        u_total = float(it.get("U_total", it["U_ale"] + it["U_epi"]))
        u_epi = float(it["U_epi"])
        u_ale = float(it["U_ale"])
        n_clusters = float(it.get("n_clusters", len(props)))
        entropy = float(-(np.clip(props, 1e-12, 1.0) * np.log(np.clip(props, 1e-12, 1.0))).sum()) if len(props) else 0.0
        rows.append([
            u_total,
            u_epi,
            u_ale,
            u_epi / max(u_ale, 1e-6),
            u_ale / max(u_total, 1e-6),
            max_prop,
            p2,
            p3,
            gap,
            1.0 - max_prop,
            n_clusters,
            math.log1p(n_clusters),
            entropy,
            float(it.get("U_ale_jsd", 0.0) or 0.0),
        ])
    return np.nan_to_num(np.asarray(rows, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)


def fit_binary_score(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    seed: int,
    kind: str,
) -> np.ndarray:
    if len(np.unique(y_train)) < 2:
        return np.full(len(X_test), float(np.mean(y_train)) if len(y_train) else 0.0)
    if kind == "logistic":
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        )
    elif kind == "rf":
        clf = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )
    elif kind == "gbm":
        clf = GradientBoostingClassifier(
            n_estimators=120,
            max_depth=2,
            learning_rate=0.04,
            random_state=seed,
        )
    else:
        raise ValueError(kind)
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1]


def learned_escalation_scores(
    base_train: list[dict[str, Any]],
    fallback_train: list[dict[str, Any]],
    base_test: list[dict[str, Any]],
    fallback_test: list[dict[str, Any]],
    seed: int,
) -> dict[str, np.ndarray]:
    X_train = base_feature_matrix(base_train)
    X_test = base_feature_matrix(base_test)
    base_ok_train = np.asarray([top1_correct(it) for it in base_train], dtype=bool)
    fall_ok_train = np.asarray([top1_correct(it) for it in fallback_train], dtype=bool)
    helpful = ((~base_ok_train) & fall_ok_train).astype(int)
    harmful = (base_ok_train & ~fall_ok_train).astype(int)

    out: dict[str, np.ndarray] = {}
    for kind in ("logistic", "gbm", "rf"):
        p_help = fit_binary_score(X_train, helpful, X_test, seed=seed, kind=kind)
        p_harm = fit_binary_score(X_train, harmful, X_test, seed=seed + 17, kind=kind)
        out[f"LearnedNet_{kind}"] = p_help - p_harm
        out[f"LearnedHelp_{kind}"] = p_help
    return out


def evaluate_pair_seed(
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
    val1 = [base_by_id[shared_ids[i]] for i in idx1]
    val2 = [base_by_id[shared_ids[i]] for i in idx2]
    fallback_train = [fallback_by_id[shared_ids[i]] for i in np.concatenate([idx1, idx2])]
    base_train = [base_by_id[shared_ids[i]] for i in np.concatenate([idx1, idx2])]
    test_ids = [shared_ids[i] for i in idxt]
    base_test = [base_by_id[qid] for qid in test_ids]
    fallback_test = [fallback_by_id[qid] for qid in test_ids]

    cal = CAELUMCalibrator(alpha=alpha, seed=seed, randomize=False, lambda_grid=lambda_grid)
    cal.calibrate(val1, val2, verbose=False)

    base_top1 = np.asarray([top1_correct(it) for it in base_test], dtype=bool)
    fallback_top1 = np.asarray([top1_correct(it) for it in fallback_test], dtype=bool)
    base_support = np.asarray([support_match(it) for it in base_test], dtype=bool)
    fallback_support = np.asarray([support_match(it) for it in fallback_test], dtype=bool)
    helpful = (~base_top1) & fallback_top1

    scores = risk_scores(base_test, cal)
    scores.update(
        learned_escalation_scores(
            base_train,
            fallback_train,
            base_test,
            fallback_test,
            seed=seed,
        )
    )
    rng = np.random.RandomState(seed + 991)
    scores["Random"] = rng.uniform(size=len(base_test))

    rate_rows: dict[str, dict[str, dict[str, float]]] = {}
    for rate in rates:
        rk = f"{rate:.2f}"
        rate_rows[rk] = {}
        for name, score in scores.items():
            rate_rows[rk][name] = eval_policy(topk(score, rate), base_top1, fallback_top1)
        rate_rows[rk]["OracleUpper"] = eval_policy(
            oracle_topk(base_top1, fallback_top1, rate), base_top1, fallback_top1
        )

    return {
        "dataset": dataset,
        "base_model": base_model,
        "fallback_model": fallback_model,
        "seed": seed,
        "n_shared": len(shared_ids),
        "n_test": len(test_ids),
        "alpha": alpha,
        "lambda_star": float(cal.optimal_lambda),
        "base_top1_acc": float(np.mean(base_top1)),
        "fallback_top1_acc": float(np.mean(fallback_top1)),
        "base_support": float(np.mean(base_support)),
        "fallback_support": float(np.mean(fallback_support)),
        "helpful_pool": float(np.mean(helpful)),
        "harmful_pool": float(np.mean(base_top1 & ~fallback_top1)),
        "auroc_helpful": {
            name: manual_auroc(score, helpful.astype(int))
            for name, score in scores.items()
        },
        "rates": rate_rows,
    }


def aggregate(rows: list[dict[str, Any]], rates: tuple[float, ...]) -> dict[str, Any]:
    rule_names = sorted({name for r in rows for rk in r["rates"] for name in r["rates"][rk]})
    by_rule = {
        name: {
            "auroc_helpful": finite_mean([
                r["auroc_helpful"].get(name, float("nan")) for r in rows
            ]),
            "rates": {},
        }
        for name in rule_names
    }
    for name in rule_names:
        for rate in rates:
            rk = f"{rate:.2f}"
            by_rule[name]["rates"][rk] = {
                key: finite_mean([
                    r["rates"][rk][name][key]
                    for r in rows
                    if name in r["rates"][rk]
                ])
                for key in (
                    "final_acc",
                    "delta_vs_base",
                    "helpful_rate",
                    "harmful_rate",
                    "precision_helpful",
                )
            }

    wins = {f"{rate:.2f}": defaultdict(int) for rate in rates}
    for r in rows:
        for rate in rates:
            rk = f"{rate:.2f}"
            candidates = {
                k: v for k, v in r["rates"][rk].items() if k != "OracleUpper"
            }
            best = max(candidates, key=lambda k: candidates[k]["final_acc"])
            wins[rk][best] += 1

    return {
        "n_pair_seeds": len(rows),
        "mean_base_top1_acc": finite_mean([r["base_top1_acc"] for r in rows]),
        "mean_fallback_top1_acc": finite_mean([r["fallback_top1_acc"] for r in rows]),
        "mean_base_support": finite_mean([r["base_support"] for r in rows]),
        "mean_fallback_support": finite_mean([r["fallback_support"] for r in rows]),
        "mean_helpful_pool": finite_mean([r["helpful_pool"] for r in rows]),
        "mean_harmful_pool": finite_mean([r["harmful_pool"] for r in rows]),
        "rules": by_rule,
        "wins_by_final_acc": {
            rk: dict(v) for rk, v in wins.items()
        },
    }


def fmt(x: float, digits: int = 3) -> str:
    return "NA" if not math.isfinite(float(x)) else f"{x:.{digits}f}"


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    rates = payload["rates"]
    lines = [
        "# Model-escalation rescue",
        "",
        "Policy: answer with the base model unless the base risk score triggers escalation; "
        "then answer with the fallback model's top cluster.",
        "",
        "No policy uses test labels. `OracleUpper` is an upper bound at the same budget.",
        "",
        "## Summary",
        "",
        f"- Pair-seed evaluations: {summary['n_pair_seeds']}",
        f"- Mean base top-1 accuracy: {summary['mean_base_top1_acc']:.3f}",
        f"- Mean fallback top-1 accuracy: {summary['mean_fallback_top1_acc']:.3f}",
        f"- Mean base support: {summary['mean_base_support']:.3f}",
        f"- Mean fallback support: {summary['mean_fallback_support']:.3f}",
        f"- Helpful escalation pool: {summary['mean_helpful_pool']:.3f}",
        f"- Harmful escalation pool: {summary['mean_harmful_pool']:.3f}",
        "",
        "## Helpful-escalation ranking",
        "",
        "| Rule | AUROC helpful escalation |",
        "|---|---:|",
    ]
    for name, vals in sorted(
        summary["rules"].items(),
        key=lambda kv: kv[1]["auroc_helpful"],
        reverse=True,
    ):
        lines.append(f"| {name} | {fmt(vals['auroc_helpful'])} |")

    for rate in rates:
        rk = f"{rate:.2f}"
        lines.extend([
            "",
            f"## Fixed escalation rate {int(round(100 * rate))}%",
            "",
            "| Rule | Final acc | Delta vs base | Helpful/all | Harmful/all | "
            "Precision helpful | Wins |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        ordered = sorted(
            summary["rules"].items(),
            key=lambda kv: kv[1]["rates"][rk]["final_acc"],
            reverse=True,
        )
        for name, vals in ordered:
            met = vals["rates"][rk]
            lines.append(
                f"| {name} | {fmt(met['final_acc'])} | "
                f"{fmt(met['delta_vs_base'])} | "
                f"{fmt(met['helpful_rate'])} | "
                f"{fmt(met['harmful_rate'])} | "
                f"{fmt(met['precision_helpful'])} | "
                f"{summary['wins_by_final_acc'].get(rk, {}).get(name, 0)} |"
            )

    lines.extend([
        "",
        "## Pair rows",
        "",
        "| Dataset | Base | Fallback | Seed | n | Base acc | Fallback acc | "
        "Base support | Fallback support | Helpful pool | Harmful pool |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in payload["rows"]:
        lines.append(
            f"| {pretty_dataset(row['dataset'])} | {pretty_model(row['base_model'])} | "
            f"{pretty_model(row['fallback_model'])} | {row['seed']} | {row['n_test']} | "
            f"{fmt(row['base_top1_acc'])} | {fmt(row['fallback_top1_acc'])} | "
            f"{fmt(row['base_support'])} | {fmt(row['fallback_support'])} | "
            f"{fmt(row['helpful_pool'])} | {fmt(row['harmful_pool'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations_dir", type=Path, default=GENERATIONS_DIR,
                        help="Folder with the unc_* caches.")
    parser.add_argument("--output_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out_prefix", default="model_escalation_rescue")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--rates", nargs="+", type=float, default=[0.10, 0.20, 0.30, 0.40, 0.50])
    parser.add_argument("--include_suffix", action="store_true")
    parser.add_argument("--base_models", nargs="*", default=None)
    parser.add_argument("--fallback_models", nargs="*", default=None)
    parser.add_argument("--fallback_model", default=None, help="Shortcut for one fallback model.")
    parser.add_argument("--quick_lambda_grid", action="store_true")
    parser.add_argument("--write-markdown", action="store_true",
                        help="Also write Markdown reports (off by default).")
    args = parser.parse_args()

    cells = discover(args.generations_dir, include_suffix=args.include_suffix)
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cells.values():
        by_dataset[row["dataset"]].append(row)

    rates = tuple(float(r) for r in args.rates)
    lambda_grid = (
        np.asarray([0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
        if args.quick_lambda_grid else None
    )
    fallback_filter = set(args.fallback_models or [])
    if args.fallback_model:
        fallback_filter.add(args.fallback_model)
    base_filter = set(args.base_models or [])

    rows: list[dict[str, Any]] = []
    for dataset in sorted(
        by_dataset,
        key=lambda d: DATASET_ORDER.index(d) if d in DATASET_ORDER else 99,
    ):
        ds_rows = sorted(by_dataset[dataset], key=lambda r: model_key(r["model"]))
        for base in ds_rows:
            if base_filter and base["model"] not in base_filter:
                continue
            for fallback in ds_rows:
                if fallback["model"] == base["model"]:
                    continue
                if fallback_filter and fallback["model"] not in fallback_filter:
                    continue
                for seed in args.seeds:
                    result = evaluate_pair_seed(
                        dataset,
                        base["model"],
                        fallback["model"],
                        base["path"],
                        fallback["path"],
                        seed=seed,
                        alpha=args.alpha,
                        rates=rates,
                        lambda_grid=lambda_grid,
                    )
                    if result is not None:
                        rows.append(result)

    if not rows:
        raise SystemExit("No shared model pairs found.")

    payload = {
        "generations_dir": repo_relative(args.generations_dir),
        "alpha": args.alpha,
        "seeds": args.seeds,
        "rates": list(rates),
        "quick_lambda_grid": bool(args.quick_lambda_grid),
        "summary": aggregate(rows, rates),
        "rows": rows,
    }
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.out_prefix}.json"
    md_path = out_dir / f"{args.out_prefix}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.write_markdown:
        write_markdown(md_path, payload)
    print(f"Wrote {json_path}")
    if args.write_markdown:
        print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
