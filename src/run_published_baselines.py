#!/usr/bin/env python3
"""Published baselines on the judged caches: CPQ (Noorani et al.) and SAFER.

(1) CPQ / missing-mass EE sets (Noorani et al., arXiv 2506.05497), faithful
    calibration-module implementation at fixed sampling budget t=50 (their
    query module adaptively chooses T(x); our caches are fixed-budget, which
    their decoupled analysis permits — the one documented deviation).
    Construction, following their Section 4 verbatim:

      labels        = semantic clusters; count r_k = round(p_k * t)
      N_r           = #clusters occurring exactly r times
      theta_hat     = N_1 / t                       (Good-Turing missing mass)
      omega_hat_k   = ((r_k+1)/t) * N_{r_k+1}/N_{r_k}   (smoothed frequency)
      S(x, y_k)     = 1 - omega_hat_k               (sampled label)
      S(x, EE)      = 2 - theta_hat                 (Everything-Else label)
      calibration   s_i = S(x_i, matching cluster) if matched else 2 - theta_i
      q*            = Quantile_{ceil((1-a)(n+1))/n}(s_1..s_n, +inf)
      C(x)          = {y_k : S(x,y_k) <= q*} u ({EE} if 2 - theta_hat <= q*)

    Coverage over the augmented space (Thm 4.1): covered iff the matching
    cluster is in C(x), or the item is unmatched and EE in C(x).

(2) SAFER-style Stage-1 budget gate (Wang et al., arXiv 2510.10193):
    per cell, a Clopper-Pearson upper confidence bound (delta=0.05) on
    P(no admissible answer in the t=50 samples), estimated on V2. If the
    UCB exceeds alpha the cell abstains wholesale (their "abstain from
    making a prediction"); otherwise a standard split-conformal cluster-APS
    set is produced. Stage-2 candidate filtering is not replicated (it
    prunes within-set candidates and does not change the coverage
    accounting that matters here); documented as a simplification.

Protocol: judged caches, seeds 42-45, 25/25/25/25 splits matching
run_caelum_abstain.py; CPQ calibrates on the full V2 (more calibration data
than CAELUM-A's V2b uses). Outputs results/published_baselines.json;
add --write-markdown for the Markdown report.
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

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from caelum.paths import GENERATIONS_DIR, JUDGED_DIR, RESULTS_DIR  # noqa: E402

from scipy.stats import beta as beta_dist  # noqa: E402

from caelum.calibration import CAELUMCalibrator  # noqa: E402
from run_judged_rerun import (  # noqa: E402
    _ensure_types, _setup_logging, discover_cells, resolve_source,
    MODEL_PRETTY, DATASET_PRETTY,
)

DEFAULT_UNC_DIR = JUDGED_DIR
ALPHA = 0.1
SEEDS = [42, 43, 44, 45]
T_BUDGET = 50


# ---------------------------------------------------------------------------
# CPQ (Noorani et al.) at fixed budget
# ---------------------------------------------------------------------------

def _cluster_counts(item: dict, t: int = T_BUDGET) -> np.ndarray:
    props = np.asarray(item["cluster_props"], dtype=float)
    if len(props) == 0:
        return np.zeros(0, dtype=int)
    return np.maximum(np.rint(props * t).astype(int), 1)


def cpq_scores_item(item: dict, t: int = T_BUDGET) -> tuple[np.ndarray, float]:
    """Per-cluster CPQ conformity scores and the EE score for one item.

    Returns (scores_clusters, score_EE) with S = 1 - omega_hat for sampled
    clusters and S_EE = 2 - theta_hat.
    """
    counts = _cluster_counts(item, t)
    if len(counts) == 0:
        return np.zeros(0), 2.0 - 1.0  # everything unseen: theta = 1
    n_r: dict[int, int] = {}
    for r in counts:
        n_r[int(r)] = n_r.get(int(r), 0) + 1
    theta = n_r.get(1, 0) / t
    scores = np.empty(len(counts), dtype=float)
    for k, r in enumerate(counts):
        r = int(r)
        omega = ((r + 1) / t) * (n_r.get(r + 1, 0) / n_r[r])
        scores[k] = 1.0 - omega
    return scores, 2.0 - theta


def run_cpq_cell(data: list[dict], alpha: float, seeds: list[int]) -> dict:
    """CPQ metrics (mean/std over seeds) for one cell."""
    acc: dict[str, list[float]] = {}
    pre = [cpq_scores_item(it) for it in data]
    cidx_all = np.array([int(it.get("correct_cluster_idx", -1)) for it in data])

    for seed in seeds:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(data))
        q = len(data) // 4
        v2_idx = perm[2 * q:3 * q]          # full V2 as D_cal2
        test_idx = perm[3 * q:]

        cal_scores = []
        for i in v2_idx:
            sc, s_ee = pre[i]
            k = cidx_all[i]
            cal_scores.append(float(sc[k]) if 0 <= k < len(sc) else s_ee)
        cal_scores.append(math.inf)  # the +inf debiasing adjustment
        n = len(cal_scores)
        level = min(math.ceil((1 - alpha) * n) / n, 1.0)
        finite = np.array(cal_scores[:-1], dtype=float)
        # Quantile over scores u {inf}: if the level index exceeds the finite
        # part, q* = inf (include everything incl. EE).
        idx = math.ceil(level * n) - 1
        srt = np.sort(finite)
        qstar = math.inf if idx >= len(srt) else float(srt[idx])

        covered, ee_in, sizes, unmatched = [], [], [], []
        for i in test_idx:
            sc, s_ee = pre[i]
            in_set = sc <= qstar
            has_ee = s_ee <= qstar
            k = cidx_all[i]
            unm = k < 0
            cov = (0 <= k < len(sc) and bool(in_set[k])) or (unm and has_ee)
            covered.append(cov)
            ee_in.append(has_ee)
            sizes.append(int(in_set.sum()) + int(has_ee))
            unmatched.append(unm)
        covered = np.array(covered)
        ee_in = np.array(ee_in)
        unmatched = np.array(unmatched)
        n_ee = int(ee_in.sum())
        acc.setdefault("coverage_aug", []).append(float(covered.mean()))
        acc.setdefault("ee_rate", []).append(float(ee_in.mean()))
        acc.setdefault("ee_precision", []).append(
            float((ee_in & unmatched).sum()) / n_ee if n_ee else float("nan")
        )
        acc.setdefault("apss_aug", []).append(float(np.mean(sizes)))
    return {k: [float(np.nanmean(v)), float(np.nanstd(v))] for k, v in acc.items()}


# ---------------------------------------------------------------------------
# SAFER-style Stage-1 gate + conformal set
# ---------------------------------------------------------------------------

def clopper_pearson_ucb(k: int, n: int, delta: float = 0.05) -> float:
    """Upper Clopper-Pearson bound on a binomial proportion."""
    if n == 0:
        return 1.0
    if k == n:
        return 1.0
    return float(beta_dist.ppf(1 - delta, k + 1, n - k))


def run_safer_cell(data: list[dict], alpha: float, seeds: list[int]) -> dict:
    """SAFER-style metrics: budget gate on V2, TrueAPS set when open."""
    acc: dict[str, list[float]] = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(data))
        q = len(data) // 4
        v1 = [data[i] for i in perm[q:2 * q]]
        v2 = [data[i] for i in perm[2 * q:3 * q]]
        test = [data[i] for i in perm[3 * q:]]

        n_unm = sum(1 for it in v2 if int(it.get("correct_cluster_idx", -1)) < 0)
        ucb = clopper_pearson_ucb(n_unm, len(v2))
        gate_open = ucb <= alpha
        acc.setdefault("gate_open", []).append(float(gate_open))
        acc.setdefault("risk_ucb", []).append(ucb)
        if not gate_open:
            # Whole-instance abstention across the cell: no predictions.
            acc.setdefault("coverage_answered", []).append(float("nan"))
            acc.setdefault("answer_rate", []).append(0.0)
            acc.setdefault("apss", []).append(float("nan"))
            continue
        cal = CAELUMCalibrator.true_aps_conformal(alpha=alpha, randomize=True)
        cal.seed = seed
        cal.calibrate(v1, v2, verbose=False)
        cov = np.array([bool(cal.is_covered(it)) for it in test])
        siz = np.array([float(cal.predict_set_size(it)) for it in test])
        acc.setdefault("coverage_answered", []).append(float(cov.mean()))
        acc.setdefault("answer_rate", []).append(1.0)
        acc.setdefault("apss", []).append(float(siz.mean()))
    return {k: [float(np.nanmean(v)), float(np.nanstd(v))] for k, v in acc.items()}


# ---------------------------------------------------------------------------
# ConU-style self-consistency conformal (Wang et al., EMNLP 2024)
# ---------------------------------------------------------------------------

def run_conu_cell(data: list[dict], alpha: float, seeds: list[int]) -> dict:
    """ConU-style baseline: self-consistency nonconformity on clusters.

    Calibration score = 1 - p_{k*} (frequency of the cluster containing the
    reference; 1.0 when unmatched). Prediction set = clusters whose
    self-consistency ratio satisfies 1 - p_k <= q_hat. No abstention
    element: coverage is capped by matchability like all sampled-support
    threshold methods.
    """
    acc: dict[str, list[float]] = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(data))
        q = len(data) // 4
        v2 = [data[i] for i in perm[2 * q:3 * q]]
        test = [data[i] for i in perm[3 * q:]]
        scores = []
        for it in v2:
            props = np.asarray(it["cluster_props"], dtype=float)
            k = int(it.get("correct_cluster_idx", -1))
            scores.append(1.0 - float(props[k]) if 0 <= k < len(props) else 1.0)
        n = len(scores)
        level = min(math.ceil((1 - alpha) * (n + 1)) / n, 1.0)
        qhat = float(np.quantile(np.asarray(scores), level, method="higher"))
        cov, siz = [], []
        for it in test:
            props = np.asarray(it["cluster_props"], dtype=float)
            in_set = (1.0 - props) <= qhat
            k = int(it.get("correct_cluster_idx", -1))
            cov.append(0 <= k < len(props) and bool(in_set[k]))
            siz.append(int(in_set.sum()))
        acc.setdefault("coverage", []).append(float(np.mean(cov)))
        acc.setdefault("apss", []).append(float(np.mean(siz)))
    return {k: [float(np.nanmean(v)), float(np.nanstd(v))] for k, v in acc.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unc_dir", type=str, default=str(DEFAULT_UNC_DIR))
    parser.add_argument("--generations_dir", type=str, default=str(GENERATIONS_DIR),
                        help="Folder with the heuristic unc_* caches.")
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    parser.add_argument("--write-markdown", action="store_true",
                        help="Also write the Markdown report.")
    args = parser.parse_args()

    logger = _setup_logging(RESULTS_DIR / "published_baselines.txt")
    t0 = time.time()
    cells = discover_cells(Path(args.generations_dir))
    logger.info("Published baselines: %d cells, seeds=%s", len(cells), args.seeds)

    out: dict[str, dict] = {}
    for cell in cells:
        path, source = resolve_source(cell, Path(args.unc_dir), Path(args.generations_dir))
        with open(path, "r", encoding="utf-8") as f:
            data = _ensure_types(json.load(f))
        cpq = run_cpq_cell(data, args.alpha, args.seeds)
        safer = run_safer_cell(data, args.alpha, args.seeds)
        conu = run_conu_cell(data, args.alpha, args.seeds)
        out[cell["cell_key"]] = {"cpq": cpq, "safer": safer, "conu": conu,
                                 "source": source}
        logger.info(
            "  %s [%s]: CPQ cov=%.3f EE-rate=%.3f prec=%.3f size=%.2f | "
            "SAFER gate-open=%.2f cov|ans=%.3f",
            cell["cell_key"], source,
            cpq["coverage_aug"][0], cpq["ee_rate"][0], cpq["ee_precision"][0],
            cpq["apss_aug"][0],
            safer["gate_open"][0], safer["coverage_answered"][0],
        )

    with open(RESULTS_DIR / "published_baselines.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)

    # Markdown comparison
    lines = [
        "# Published baselines on judged caches (CPQ, SAFER-style)", "",
        "CPQ = Noorani et al. 2506.05497 calibration module at fixed t=50;",
        "SAFER = Wang et al. 2510.10193 Stage-1 Clopper-Pearson gate +",
        "TrueAPS set (Stage-2 filter not replicated). CAELUM-A pooled",
        "aggregates for reference: cov_aug 0.927 / abstain 0.493 / prec 0.528.",
        "",
        "| Model | Dataset | CPQ cov | CPQ EE-rate | CPQ prec | CPQ size | "
        "SAFER open | SAFER cov|ans |",
        "|---|---|---|---|---|---|---|---|",
    ]
    aggs: dict[str, list[float]] = {}
    for key, res in out.items():
        model, ds = key.split("__")
        c, s = res["cpq"], res["safer"]
        lines.append(
            f"| {MODEL_PRETTY.get(model, model)} | {DATASET_PRETTY.get(ds, ds)}"
            f" | {c['coverage_aug'][0]:.3f} | {c['ee_rate'][0]:.3f}"
            f" | {c['ee_precision'][0]:.3f} | {c['apss_aug'][0]:.2f}"
            f" | {s['gate_open'][0]:.2f} | {s['coverage_answered'][0]:.3f} |"
        )
        u = res["conu"]
        for name, val in (
            ("cpq_cov", c["coverage_aug"][0]), ("cpq_ee", c["ee_rate"][0]),
            ("cpq_prec", c["ee_precision"][0]), ("cpq_size", c["apss_aug"][0]),
            ("safer_open", s["gate_open"][0]),
            ("conu_cov", u["coverage"][0]), ("conu_apss", u["apss"][0]),
        ):
            aggs.setdefault(name, []).append(val)
    lines += ["", "**Aggregates:** " + ", ".join(
        f"{k}={np.nanmean(v):.3f}" for k, v in aggs.items()
    )]
    if args.write_markdown:
        with open(RESULTS_DIR / "published_baselines.md", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    logger.info("Done in %.1f min", (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
