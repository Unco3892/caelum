#!/usr/bin/env python3
"""CAELUM-E: escalate-then-abstain with a composed CRC coverage guarantee.

CAELUM-A repairs the support ceiling by abstaining. This script asks whether a
stronger fallback model can convert part of that abstention budget back into
covered answers, without losing the unconditional guarantee:

  1. Score each query with the base model's learned support-failure score
     a(x) (PostgenScorer trained on V0 u V1 of the BASE cache only).
  2. Escalate when a(x) >= t_esc: serve the FALLBACK model's calibrated
     semantic set instead of the base model's. t_esc is a quantile of the
     training-pool scores at an escalation budget selected on V1, so neither
     V2b nor Test influences the escalation rule.
  3. Abstain (append the element ``\\perp``) when a(x) >= t, with t chosen by
     the same conformal-risk-control rule as CAELUM-A, applied on V2b to the
     COMPOSED outcome: an item is covered when the served set contains the
     reference, or when it abstains and the reference is absent from the
     served model's sampled support.

Validity: the escalation rule, both calibrated thresholds tau (base on V2a of
the base cache, fallback on V2a of the fallback cache) and the score a() are
all fixed before V2b is touched, and composed augmented miscoverage is
monotone non-increasing as t decreases, so the CRC selection on V2b retains
the unconditional 1 - alpha guarantee over the augmented output space. No
per-arm claim is made for the score-selected escalated stratum.

Writes ``results/<out_prefix>.json`` (and ``.md``) with per-(base, seed) rows,
a budget frontier, and an explicit gate evaluation for the pilot criteria.

Inputs are the heuristic caches in ``results/generations`` and, where present,
their judged copies in ``results/llm_judged_unc_full``.

Usage (the four runs reported for escalation to Qwen3-14B and Qwen3-32B)::

    D="triviaqa opennq ambiginst simpleqa popqa ambigqa"
    python -u src/run_caelum_escalate.py --fallback_model qwen3_14b --datasets $D \
        --policy switch --out_prefix caelum_escalate_full_switch
    python -u src/run_caelum_escalate.py --fallback_model qwen3_14b --datasets $D \
        --policy union --out_prefix caelum_escalate_full_union
    python -u src/run_caelum_escalate.py --fallback_model qwen3_32b --datasets $D \
        --base_models qwen3_8b ministral_3_8b_instruct_2512 gemma_4_e4b_it qwen3_14b \
        --policy switch --out_prefix caelum_escalate_32b_switch
    python -u src/run_caelum_escalate.py --fallback_model qwen3_32b --datasets $D \
        --base_models qwen3_8b ministral_3_8b_instruct_2512 gemma_4_e4b_it qwen3_14b \
        --policy union --out_prefix caelum_escalate_32b_union
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from caelum.abstention import (  # noqa: E402
    attach_refusal_fractions,
    build_pregen_features,
    train_matchability_classifier,
)
from caelum.calibration import CAELUMCalibrator  # noqa: E402
from caelum.utils import SentenceEncoder  # noqa: E402

from run_caelum_abstain import (  # noqa: E402
    PostgenScorer,
    _failure_labels,
    _oof_pregen_probs,
    augmented_eval,
    crc_abstain_threshold,
)
from run_judged_rerun import _ensure_types, _setup_logging  # noqa: E402
from run_model_escalation_rescue import (  # noqa: E402
    discover,
    item_key,
    pretty_dataset,
    pretty_model,
)

from caelum.paths import GENERATIONS_DIR, JUDGED_DIR, RESULTS_DIR  # noqa: E402

DEFAULT_UNC_DIR = JUDGED_DIR

ALPHA = 0.1
SEEDS = [42, 43, 44, 45]
BUDGET_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


# ---------------------------------------------------------------------------
# Data loading and alignment
# ---------------------------------------------------------------------------

def resolve_judged(parsed: dict[str, Any], unc_dir: Path) -> tuple[str, str]:
    """Prefer the judged cache for a discovered cell, else the heuristic one."""
    judged = unc_dir / Path(parsed["path"]).name
    if judged.exists():
        return str(judged), "judged"
    return parsed["path"], "heuristic"


def load_cell(path: str, generations_dir: Path) -> list[dict]:
    """Load one unc cache with type coercion and refusal fractions attached."""
    with open(path, "r", encoding="utf-8") as f:
        data = _ensure_types(json.load(f))
    gen_name = Path(path).name.replace("unc_", "gen_").replace("_vne", "")
    attach_refusal_fractions(data, generations_dir / gen_name)
    return data


def align_pairs(
    base_items: list[dict], fb_items: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Return (base, fallback) lists restricted to shared qids, base order."""
    fb_by_qid = {item_key(it): it for it in fb_items}
    base_out, fb_out = [], []
    for it in base_items:
        match = fb_by_qid.get(item_key(it))
        if match is not None:
            base_out.append(it)
            fb_out.append(match)
    return base_out, fb_out


# ---------------------------------------------------------------------------
# Composed policy evaluation
# ---------------------------------------------------------------------------

def set_arrays(
    cal: CAELUMCalibrator, items: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(covered, sizes, unmatched) for one calibrated model on its items."""
    covered = np.array([bool(cal.is_covered(it)) for it in items])
    sizes = np.array([float(cal.predict_set_size(it)) for it in items])
    unmatched = _failure_labels(items).astype(bool)
    return covered, sizes, unmatched


def compose(
    escalated: np.ndarray,
    base: tuple[np.ndarray, np.ndarray, np.ndarray],
    fb: tuple[np.ndarray, np.ndarray, np.ndarray],
    policy: str = "switch",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Served arrays under an escalation mask.

    ``switch`` serves the fallback set INSTEAD of the base set on escalated
    items; an abstention is then correct when the reference is absent from
    the served model's support. ``union`` serves the MERGED base+fallback
    set on escalated items, so coverage can only improve, at the price of a
    larger set; an abstention on an escalated item is correct only when the
    reference is absent from BOTH supports (strict). Returns
    (covered_served, sizes_served, unmatched_served, unmatched_union).
    """
    unm_union = base[2] & fb[2]
    if policy == "union":
        cov = np.where(escalated, base[0] | fb[0], base[0])
        siz = np.where(escalated, base[1] + fb[1], base[1])
        unm = np.where(escalated, unm_union, base[2])
    else:
        cov = np.where(escalated, fb[0], base[0])
        siz = np.where(escalated, fb[1], base[1])
        unm = np.where(escalated, fb[2], base[2])
    return cov, siz, unm, unm_union


def eval_budget(
    b: float,
    t_esc: float,
    scores_cal: np.ndarray,
    scores_test: np.ndarray,
    arrays_cal: tuple,
    arrays_test: tuple,
    alpha: float,
    policy: str = "switch",
) -> dict[str, Any]:
    """CRC-calibrate on the composed cal split, evaluate on the test split."""
    esc_cal = scores_cal >= t_esc
    esc_test = scores_test >= t_esc
    cov_c, siz_c, unm_c, _ = compose(
        esc_cal, arrays_cal[0], arrays_cal[1], policy
    )
    cov_t, siz_t, unm_t, unm_t_union = compose(
        esc_test, arrays_test[0], arrays_test[1], policy
    )
    t = crc_abstain_threshold(scores_cal, cov_c, unm_c, alpha)
    m = augmented_eval(scores_test, t, cov_t, siz_t, unm_t)
    abstained = scores_test >= t
    covered_aug_union = cov_t | (abstained & unm_t_union)
    n_esc = int(esc_test.sum())
    answered_covered = esc_test & ~abstained & cov_t
    m.update({
        "budget": float(b),
        "t_esc": float(t_esc),
        "escalation_rate": float(esc_test.mean()),
        "coverage_aug_union": float(covered_aug_union.mean()),
        "escalated_answered_covered_frac": (
            float(answered_covered.sum() / n_esc) if n_esc else float("nan")
        ),
    })
    return m


def evaluate_pair(
    base_items: list[dict],
    fb_items: list[dict],
    alpha: float,
    seeds: list[int],
    budgets: list[float],
    encoder: SentenceEncoder,
    logger,
    policy: str = "switch",
) -> dict[str, Any]:
    """Composed escalate-then-abstain over seeds for one (base, fallback) pair."""
    per_seed: list[dict[str, Any]] = []
    n = len(base_items)

    for seed in seeds:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        q = n // 4
        idx_v0 = perm[:q]
        idx_v1 = perm[q:2 * q]
        idx_v2 = perm[2 * q:3 * q]
        idx_test = perm[3 * q:]
        half = len(idx_v2) // 2
        idx_v2a, idx_v2b = idx_v2[:half], idx_v2[half:]

        def take(items: list[dict], idx: np.ndarray) -> list[dict]:
            return [items[i] for i in idx]

        b_v0, b_v1 = take(base_items, idx_v0), take(base_items, idx_v1)
        b_v2a, b_v2b = take(base_items, idx_v2a), take(base_items, idx_v2b)
        b_test = take(base_items, idx_test)
        f_v1 = take(fb_items, idx_v1)
        f_v2a, f_v2b = take(fb_items, idx_v2a), take(fb_items, idx_v2b)
        f_test = take(fb_items, idx_test)

        # --- base-only support-failure score, as in run_caelum_abstain ---
        v01 = b_v0 + b_v1
        pregen_clf = train_matchability_classifier(v01, encoder=encoder, seed=seed)

        def pregen_probs(items: list[dict]) -> np.ndarray:
            if not items:
                return np.array([], dtype=float)
            X, _, _ = build_pregen_features(items, encoder)
            return np.asarray(pregen_clf.predict_proba(X)[:, 1], dtype=float)

        pg_v01 = _oof_pregen_probs(v01, encoder, seed)
        scorer = PostgenScorer(seed=seed).fit(v01, pg_v01)
        a_v01 = scorer.score(v01, pg_v01)
        a_v1 = scorer.score(b_v1, pregen_probs(b_v1))
        a_v2b = scorer.score(b_v2b, pregen_probs(b_v2b))
        a_test = scorer.score(b_test, pregen_probs(b_test))

        # --- calibrated sets: base tau on base V2a, fallback tau on its V2a ---
        cal_base = CAELUMCalibrator(
            alpha=alpha, seed=seed, randomize=True, weight_fn="standardized",
            selection_method="set_score", support_conditional=True,
        )
        cal_base.calibrate(b_v1, b_v2a, verbose=False)
        cal_fb = CAELUMCalibrator(
            alpha=alpha, seed=seed, randomize=True, weight_fn="standardized",
            selection_method="set_score", support_conditional=True,
        )
        cal_fb.calibrate(f_v1, f_v2a, verbose=False)

        arrays_v2b = (set_arrays(cal_base, b_v2b), set_arrays(cal_fb, f_v2b))
        arrays_test = (set_arrays(cal_base, b_test), set_arrays(cal_fb, f_test))
        arrays_v1 = (set_arrays(cal_base, b_v1), set_arrays(cal_fb, f_v1))

        # --- escalation thresholds from training-pool score quantiles ---
        def budget_threshold(b: float) -> float:
            if b <= 0.0:
                return float(a_v01.max()) + 1.0  # never escalate
            return float(np.quantile(a_v01, 1.0 - b))

        # Budget selection on V1 only (validity: V2b/Test untouched).
        sel_rows = []
        for b in budgets:
            t_esc = budget_threshold(b)
            row = eval_budget(
                b, t_esc, a_v1, a_v1, (arrays_v1[0], arrays_v1[1]),
                (arrays_v1[0], arrays_v1[1]), alpha, policy,
            )
            sel_rows.append(row)
        feasible = [
            r for r in sel_rows
            if r["coverage_aug"] >= 1.0 - alpha and r["escalation_rate"] <= 0.6
        ]
        pick_from = feasible if feasible else sel_rows
        best_sel = min(pick_from, key=lambda r: r["abstain_rate"])
        b_star = float(best_sel["budget"])

        # --- final: CRC t on V2b, evaluate on Test, full frontier ---
        frontier = []
        for b in budgets:
            t_esc = budget_threshold(b)
            row = eval_budget(
                b, t_esc, a_v2b, a_test, arrays_v2b, arrays_test, alpha,
                policy,
            )
            frontier.append(row)
        headline = next(r for r in frontier if r["budget"] == b_star)
        baseline = next(r for r in frontier if r["budget"] == 0.0)

        # Conversion: items abstained under b=0 that are now answered+covered.
        t0 = baseline["threshold"]
        abst0 = a_test >= t0
        t_esc_star = budget_threshold(b_star)
        esc_star = a_test >= t_esc_star
        cov_star, _, _, _ = compose(
            esc_star, arrays_test[0], arrays_test[1], policy
        )
        answered_now = a_test < headline["threshold"]
        n_abst0 = int(abst0.sum())
        converted = int((abst0 & answered_now & cov_star).sum())

        per_seed.append({
            "seed": seed,
            "b_star": b_star,
            "headline": headline,
            "baseline_no_escalation": baseline,
            "frontier": frontier,
            "conversion_rate": converted / n_abst0 if n_abst0 else float("nan"),
        })
        logger.info(
            "    seed %d: b*=%.1f  cov_aug %.3f->%.3f  abstain %.3f->%.3f  "
            "esc_rate=%.3f  conv=%.3f",
            seed, b_star,
            baseline["coverage_aug"], headline["coverage_aug"],
            baseline["abstain_rate"], headline["abstain_rate"],
            headline["escalation_rate"],
            per_seed[-1]["conversion_rate"],
        )

    def seed_mean(getter) -> float:
        vals = [getter(r) for r in per_seed]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    agg = {
        "coverage_aug": seed_mean(lambda r: r["headline"]["coverage_aug"]),
        "coverage_aug_union": seed_mean(
            lambda r: r["headline"]["coverage_aug_union"]),
        "abstain_rate": seed_mean(lambda r: r["headline"]["abstain_rate"]),
        "abstain_precision": seed_mean(
            lambda r: r["headline"]["abstain_precision"]),
        "escalation_rate": seed_mean(lambda r: r["headline"]["escalation_rate"]),
        "escalated_answered_covered_frac": seed_mean(
            lambda r: r["headline"]["escalated_answered_covered_frac"]),
        "conversion_rate": seed_mean(lambda r: r["conversion_rate"]),
        "baseline_coverage_aug": seed_mean(
            lambda r: r["baseline_no_escalation"]["coverage_aug"]),
        "baseline_abstain_rate": seed_mean(
            lambda r: r["baseline_no_escalation"]["abstain_rate"]),
        "mean_b_star": seed_mean(lambda r: r["b_star"]),
    }
    return {"aggregate": agg, "per_seed": per_seed, "n_aligned": n}


# ---------------------------------------------------------------------------
# Self-test: composed monotonicity + CRC bound on synthetic data
# ---------------------------------------------------------------------------

def selftest() -> int:
    """Verify composed augmented miscoverage is monotone and CRC holds.

    Synthetic data respects the real pipeline's structure: a set can only
    cover the reference when the item is matched (covered => not unmatched),
    so abstaining on everything achieves full augmented coverage and the CRC
    bound is feasible. The guarantee checked is the marginal one: mean test
    coverage over many calibration/test splits >= 1 - alpha.
    """
    rng = np.random.RandomState(0)
    n = 400
    # Support-failure score correlated with true unmatched status.
    unm_base = rng.rand(n) < 0.35
    unm_fb = np.where(unm_base, rng.rand(n) < 0.6, rng.rand(n) < 0.15)
    scores = np.clip(
        0.35 * unm_base + 0.25 * rng.rand(n) + 0.2 * unm_fb, 0.0, 1.0
    )
    esc = scores >= 0.7
    base = (~unm_base & (rng.rand(n) < 0.85),
            rng.randint(1, 4, n).astype(float), unm_base)
    fb = (~unm_fb & (rng.rand(n) < 0.9),
          rng.randint(1, 4, n).astype(float), unm_fb)
    cov, siz, unm, _ = compose(esc, base, fb)
    ts = np.linspace(1.0, 0.0, 101)
    covs = [(cov | ((scores >= t) & unm)).mean() for t in ts]
    assert all(b >= a - 1e-12 for a, b in zip(covs, covs[1:])), (
        "composed augmented coverage must be monotone as t decreases")
    # Marginal CRC check over resampled calibration/test splits.
    alpha = 0.2
    trials = 500
    test_covs = []
    for k in range(trials):
        r = np.random.RandomState(k)
        idx = r.permutation(n)
        c_idx, t_idx = idx[:200], idx[200:]
        t_star = crc_abstain_threshold(
            scores[c_idx], cov[c_idx], unm[c_idx], alpha)
        test_covs.append(
            (cov[t_idx] | ((scores[t_idx] >= t_star) & unm[t_idx])).mean()
        )
    mean_cov = float(np.mean(test_covs))
    print(f"selftest: monotonicity OK; mean test coverage over {trials} "
          f"splits = {mean_cov:.3f} (target >= {1 - alpha:.2f})")
    assert mean_cov >= 1 - alpha - 0.01, "marginal CRC guarantee violated"
    return 0


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

GATE_ABSTAIN_MAX = 0.35
GATE_COVERAGE_MIN = 0.90
GATE_CONVERSION_MIN = 0.40
GATE_ESCALATION_MAX = 0.60


def gate_row(base_model: str, agg: dict[str, float]) -> dict[str, Any]:
    """Evaluate the pilot gate criteria for one base model."""
    if base_model == "gemma_4_e4b_it":
        crit = (f"escalated answered+covered >= {GATE_CONVERSION_MIN:.2f}")
        ok = (
            agg["escalated_answered_covered_frac"] >= GATE_CONVERSION_MIN
            and agg["coverage_aug"] >= GATE_COVERAGE_MIN
        )
    else:
        crit = (f"abstain <= {GATE_ABSTAIN_MAX:.2f} @ cov >= "
                f"{GATE_COVERAGE_MIN:.2f}")
        ok = (
            agg["abstain_rate"] <= GATE_ABSTAIN_MAX
            and agg["coverage_aug"] >= GATE_COVERAGE_MIN
        )
    ok = ok and agg["escalation_rate"] <= GATE_ESCALATION_MAX
    return {"criterion": crit, "pass": bool(ok)}


def write_reports(results: dict[str, dict], out_prefix: Path) -> None:
    """Write JSON and a markdown summary with the gate table."""
    with open(f"{out_prefix}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    lines = [
        "# CAELUM-E: escalate-then-abstain pilot",
        "",
        "Composed policy: serve the fallback model's calibrated set when the",
        "base support-failure score exceeds a budgeted threshold (budget",
        "selected on V1), abstain via the CRC threshold on V2b over the",
        "composed outcome. Coverage counts an abstention as correct only when",
        "the reference is absent from the served model's support;",
        "`cov_union` is the strict both-models variant.",
        "",
        "## Headline (per base model, mean over seeds)",
        "",
        "| Base | Fallback | n | cov_aug (base->comp) | abstain (base->comp) "
        "| esc rate | conv | esc ans+cov | prec | cov_union | gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key, res in results["pairs"].items():
        agg = res["aggregate"]
        g = res["gate"]
        base, fb, dataset = key.split("|")
        lines.append(
            f"| {pretty_model(base)} | {pretty_model(fb)} "
            f"({pretty_dataset(dataset)}) | {res['n_aligned']} "
            f"| {agg['baseline_coverage_aug']:.3f}->{agg['coverage_aug']:.3f} "
            f"| {agg['baseline_abstain_rate']:.3f}->{agg['abstain_rate']:.3f} "
            f"| {agg['escalation_rate']:.3f} | {agg['conversion_rate']:.3f} "
            f"| {agg['escalated_answered_covered_frac']:.3f} "
            f"| {agg['abstain_precision']:.3f} "
            f"| {agg['coverage_aug_union']:.3f} "
            f"| {'PASS' if g['pass'] else 'FAIL'} ({g['criterion']}) |"
        )
    n_pass = sum(1 for r in results["pairs"].values() if r["gate"]["pass"])
    n_all = len(results["pairs"])
    if n_all <= 3:
        verdict = (f"pilot {'PASSES' if n_pass >= 2 else 'FAILS'} "
                   f"(need >= 2 of 3)")
    else:
        verdict = ("full-grid summary; the 2-of-3 pilot criterion applies "
                   "only to the single-dataset pilot")
    lines += [
        "",
        f"**GATE: {n_pass}/{n_all} (base, dataset) pairs pass; {verdict}.**",
        "",
        "## Budget frontier (mean over seeds)",
        "",
    ]
    for key, res in results["pairs"].items():
        base, fb, dataset = key.split("|")
        lines += [
            f"### {pretty_model(base)} -> {pretty_model(fb)} "
            f"({pretty_dataset(dataset)})",
            "",
            "| budget | esc rate | cov_aug | abstain | prec | cov_union |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        budgets = [r["budget"] for r in res["per_seed"][0]["frontier"]]
        for i, b in enumerate(budgets):
            def m(field: str, i=i) -> float:
                vals = [
                    r["frontier"][i][field] for r in res["per_seed"]
                    if np.isfinite(r["frontier"][i][field])
                ]
                return float(np.mean(vals)) if vals else float("nan")
            lines.append(
                f"| {b:.1f} | {m('escalation_rate'):.3f} "
                f"| {m('coverage_aug'):.3f} | {m('abstain_rate'):.3f} "
                f"| {m('abstain_precision'):.3f} "
                f"| {m('coverage_aug_union'):.3f} |"
            )
        lines.append("")
    with open(f"{out_prefix}.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="CAELUM-E composed escalate-then-abstain pilot.")
    ap.add_argument("--generations_dir", type=Path, default=GENERATIONS_DIR)
    ap.add_argument("--unc_dir", type=Path, default=DEFAULT_UNC_DIR)
    ap.add_argument("--fallback_model", default="qwen3_14b")
    ap.add_argument(
        "--base_models", nargs="*",
        default=["qwen3_8b", "ministral_3_8b_instruct_2512", "gemma_4_e4b_it"],
    )
    ap.add_argument("--datasets", nargs="*", default=["ambigqa"])
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--budgets", nargs="*", type=float, default=BUDGET_GRID)
    ap.add_argument("--out_prefix", default="caelum_escalate_pilot")
    ap.add_argument(
        "--policy", choices=("switch", "union"), default="switch",
        help="'switch': serve the fallback set instead of the base set on "
             "escalated items. 'union': serve the merged base+fallback set "
             "(coverage can only improve; abstention credit is strict).",
    )
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    logger = _setup_logging(RESULTS_DIR / f"{args.out_prefix}.txt")
    t0 = time.time()
    cells = discover(args.generations_dir, include_suffix=False)
    encoder = SentenceEncoder()

    results: dict[str, Any] = {
        "alpha": args.alpha, "seeds": args.seeds, "budgets": args.budgets,
        "fallback_model": args.fallback_model, "policy": args.policy,
        "pairs": {},
    }
    for dataset in args.datasets:
        fb_parsed = cells.get((dataset, args.fallback_model))
        if fb_parsed is None:
            logger.warning("no fallback cache for %s on %s; skipping",
                           args.fallback_model, dataset)
            continue
        fb_path, fb_src = resolve_judged(fb_parsed, args.unc_dir)
        fb_items = load_cell(fb_path, args.generations_dir)
        for base_model in args.base_models:
            parsed = cells.get((dataset, base_model))
            if parsed is None:
                logger.warning("no base cache for %s on %s; skipping",
                               base_model, dataset)
                continue
            base_path, base_src = resolve_judged(parsed, args.unc_dir)
            base_items = load_cell(base_path, args.generations_dir)
            b_al, f_al = align_pairs(base_items, fb_items)
            logger.info(
                "%s -> %s on %s [%s/%s]: %d aligned of %d/%d",
                base_model, args.fallback_model, dataset, base_src, fb_src,
                len(b_al), len(base_items), len(fb_items),
            )
            res = evaluate_pair(
                b_al, f_al, args.alpha, args.seeds, args.budgets, encoder,
                logger, args.policy,
            )
            res["gate"] = gate_row(base_model, res["aggregate"])
            key = f"{base_model}|{args.fallback_model}|{dataset}"
            results["pairs"][key] = res

    out_prefix = RESULTS_DIR / args.out_prefix
    write_reports(results, out_prefix)
    logger.info("Wrote %s.{json,md} in %.1f min", out_prefix,
                (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
