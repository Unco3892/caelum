#!/usr/bin/env python3
"""CAELUM-A: post-generation abstention scoring + two-threshold calibration.

Implements the abstention-centric upgrade (task #7):

(A) POST-generation support-failure scoring. A lightweight classifier is
    trained on V0 to predict support failure (``correct_cluster_idx < 0``)
    from post-generation signals [U_epi, U_ale, U_total, top-cluster mass,
    cluster-count/entropy statistics] plus the pre-generation matchability
    probability and simple question statistics. Reported as support-failure
    AUROC on Test against single-signal baselines (U_epi, U_total, U_ale,
    1 - p1 self-consistency, pre-generation probability).

(B) Two-threshold CAELUM-A. A base CAELUM calibrator (standardized weight,
    set_score selection, support-conditional tau on V2a) is paired with an
    abstention trigger: abstain when the support-failure score A(x) >= t.
    The threshold t is chosen on V2b via a conformal-risk-control-style rule:
    the LARGEST t (fewest abstentions) such that

        (n/(n+1)) * miscoverage_aug(t) + 1/(n+1)  <=  alpha,

    where augmented coverage counts covered-by-set OR (abstained AND truly
    unmatched). Since miscoverage_aug is monotone non-increasing as t
    decreases, this is the standard monotone-risk threshold selection.
    Both the post-generation trigger and the pre-generation trigger
    (1 - P(matchable)) are calibrated and compared; a full threshold sweep
    on Test provides the abstention-rate / coverage Pareto frontier.

(C) Top-1 selective-answering baseline (conformal-abstention style, no sets):
    answer with the top cluster representative iff A(x) < t, with t chosen on
    V2b as the largest threshold whose selective risk (error among answered)
    satisfies the same CRC-corrected bound. Reports answer rate and selective
    accuracy, plus the head-to-head "reference delivered" comparison at
    matched abstention rate against CAELUM-A sets.

(D) Sampling-budget curves (``--budget``): from the cached generations,
    subsample m in {1,2,4,6,8,10} answers per clarification, re-cluster with
    the pipeline's greedy cosine algorithm on cached embeddings, re-match
    heuristically, and report match rate as a function of the sampling
    budget. (Heuristic matching only; the LLM-judge lift is +0-4pp per the
    full rematch, so curve shapes are unaffected.)

Splits per seed: 25% V0 (classifiers) / 25% V1 (lambda) / 25% V2 (split into
V2a for tau, V2b for t) / 25% Test. Seeds 42-45, randomized APS, alpha=0.1.

Usage::

    python -u src/run_caelum_abstain.py \
        --unc_dir results/llm_judged_unc_full
    python -u src/run_caelum_abstain.py --budget
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings as _warnings
from pathlib import Path

import numpy as np

_warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from caelum.paths import GENERATIONS_DIR, JUDGED_DIR, PLOTS_DIR, RESULTS_DIR  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_curve  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from caelum.abstention import (  # noqa: E402
    attach_refusal_fractions,
    build_pregen_features,
    train_matchability_classifier,
)
from caelum.calibration import CAELUMCalibrator  # noqa: E402
from caelum.evaluation import compute_auroc  # noqa: E402
from caelum.utils import SentenceEncoder, find_correct_cluster  # noqa: E402
from caelum.uncertainty import cluster_proportions_and_representatives  # noqa: E402

from run_judged_rerun import (  # noqa: E402
    Tee, _ensure_types, _setup_logging, discover_cells, resolve_source,
    MODEL_PRETTY, DATASET_PRETTY,
)

DEFAULT_UNC_DIR = JUDGED_DIR

ALPHA = 0.1
SEEDS = [42, 43, 44, 45]
BUDGETS = [1, 2, 4, 6, 8, 10]  # answers per clarification (x5 clarifications)


# ---------------------------------------------------------------------------
# (A) Post-generation support-failure scoring
# ---------------------------------------------------------------------------

def _postgen_features(
    items: list[dict], pregen_probs: np.ndarray, feature_set: str = "v1",
) -> np.ndarray:
    """Assemble the post-generation feature matrix for support-failure scoring.

    Features: U_epi, U_ale, U_total, p1, p2, margin p1-p2, n_clusters,
    1/n_clusters, Shannon entropy of cluster proportions, plus the
    pre-generation matchability probability and simple question statistics.
    ``feature_set="v1_pke"`` appends the predictive kernel entropy, which
    must have been attached to each item as ``item["pke"]`` (see
    ``attach_pke``). The ``FEATURE_SUBSETS`` names select column subsets of
    the v1 matrix for the decomposition ablation; "v1" itself is unchanged.
    """
    rows = []
    for it, pg in zip(items, pregen_probs):
        props = np.asarray(it["cluster_props"], dtype=float)
        p1 = float(props[0]) if len(props) else 0.0
        p2 = float(props[1]) if len(props) > 1 else 0.0
        pos = props[props > 0]
        h_props = float(-(pos * np.log(pos)).sum()) if len(pos) else 0.0
        k = max(len(props), 1)
        q = str(it.get("question", "") or "")
        toks = q.split()
        per_clar = np.asarray(
            it.get("per_clarification_entropy") or [0.0], dtype=float
        )
        row = [
            float(it["U_epi"]), float(it["U_ale"]), float(it["U_total"]),
            p1, p2, p1 - p2, float(k), 1.0 / k, h_props,
            float(per_clar.std()), float(per_clar.max()),
            float(pg),
            float(len(q)), float(len(toks)),
            float(it.get("refusal_frac", 0.0)),
        ]
        if feature_set == "v1_pke":
            row.append(float(it.get("pke", 0.0)))
        rows.append(row)
    X = np.asarray(rows, dtype=float)
    cols = FEATURE_SUBSETS.get(feature_set)
    if cols is not None:
        X = X[:, cols]
    return X


FEATURE_NAMES = [
    "U_epi", "U_ale", "U_total", "p1", "p2", "margin", "K", "inv_K",
    "H_props", "std_H_clar", "max_H_clar", "pregen_prob", "q_chars",
    "q_tokens", "refusal_frac",
]

# Column subsets of the 15 v1 features (indices into FEATURE_NAMES) for the
# decomposition ablation. "decomp" = the features that exist only because of
# the aleatoric/epistemic decomposition over clarification rephrasings:
# U_epi, U_ale, U_total and the per-clarification entropy statistics.
_DECOMP_COLS = [0, 1, 2, 9, 10]
_ALL_COLS = list(range(len(FEATURE_NAMES)))
FEATURE_SUBSETS = {
    "v1_no_uale": [c for c in _ALL_COLS if c != 1],
    "v1_no_uepi": [c for c in _ALL_COLS if c != 0],
    "v1_no_ae": [c for c in _ALL_COLS if c not in (0, 1)],
    "v1_no_decomp": [c for c in _ALL_COLS if c not in _DECOMP_COLS],
    "decomp_only": _DECOMP_COLS,
}


def attach_pke(data: list[dict], encoder: SentenceEncoder) -> None:
    """Attach per-item predictive kernel entropy as ``item["pke"]``.

    PKE (Gruber & Buettner 2023) over the pooled answer embeddings; its
    solo-AUROC ablation (``results/pke_aurocs.json``) shows the largest
    gains on exactly the weak SimpleQA cells, motivating its inclusion as a
    scorer feature. Computed once per cell at load time (the encoder pass
    over all answers is the slow part).
    """
    from run_pke_baseline import compute_pke_for_items
    pke = compute_pke_for_items(data, encoder)
    for it, v in zip(data, pke):
        it["pke"] = float(v)


def _failure_labels(items: list[dict]) -> np.ndarray:
    """Binary labels: 1 = support failure (reference not in any cluster)."""
    return np.array(
        [1 if int(it.get("correct_cluster_idx", -1)) < 0 else 0 for it in items],
        dtype=int,
    )


class PostgenScorer:
    """Support-failure scorer combining post-generation and pre-gen signals.

    Trains a standardized L2 logistic (primary) and a small gradient-boosted
    classifier (secondary) on V0. ``score(items, pregen_probs)`` returns
    P(support failure) in [0, 1]; higher = more likely unmatched.
    """

    LR_C_GRID = [0.1, 1.0, 10.0]
    GB_GRID = [(100, 2, 0.05), (200, 3, 0.05)]

    def __init__(self, seed: int = 42, feature_set: str = "v1",
                 tune: bool = False) -> None:
        self.seed = seed
        self.feature_set = feature_set
        self.tune = tune
        self.scaler: StandardScaler | None = None
        self.lr: LogisticRegression | None = None
        self.gb: GradientBoostingClassifier | None = None
        self.base_rate = 0.5

    def _select_by_cv(self, X: np.ndarray, y: np.ndarray) -> tuple[float, tuple]:
        """Pick LR C and GB params by 5-fold CV AUROC on the training pool.

        Selection touches only the data already used for fitting (V0 u V1 or
        the LOCO pool), so downstream CRC validity is unaffected.
        """
        n_minority = int(min((y == 0).sum(), (y == 1).sum()))
        n_splits = int(min(5, max(2, n_minority)))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                             random_state=self.seed)

        def cv_auroc(make_clf, Xf: np.ndarray) -> float:
            proba = cross_val_predict(
                make_clf(), Xf, y, cv=cv, method="predict_proba")
            return compute_auroc(y, proba[:, 1])

        Xs = StandardScaler().fit_transform(X)
        best_c = max(self.LR_C_GRID, key=lambda c: cv_auroc(
            lambda c=c: LogisticRegression(
                C=c, max_iter=5000, class_weight="balanced",
                random_state=self.seed), Xs))
        best_gb = max(self.GB_GRID, key=lambda g: cv_auroc(
            lambda g=g: GradientBoostingClassifier(
                n_estimators=g[0], max_depth=g[1], learning_rate=g[2],
                random_state=self.seed), X))
        return best_c, best_gb

    def fit(self, items_v0: list[dict], pregen_probs_v0: np.ndarray) -> "PostgenScorer":
        X = _postgen_features(items_v0, pregen_probs_v0, self.feature_set)
        y = _failure_labels(items_v0)
        if len(np.unique(y)) < 2:
            self.base_rate = float(y.mean()) if len(y) else 0.5
            return self
        lr_c, gb_params = (1.0, (100, 2, 0.05))
        if self.tune:
            lr_c, gb_params = self._select_by_cv(X, y)
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.lr = LogisticRegression(
            C=lr_c, max_iter=5000, class_weight="balanced",
            random_state=self.seed,
        ).fit(Xs, y)
        self.gb = GradientBoostingClassifier(
            n_estimators=gb_params[0], max_depth=gb_params[1],
            learning_rate=gb_params[2], random_state=self.seed,
        ).fit(X, y)
        return self

    def score(self, items: list[dict], pregen_probs: np.ndarray,
              kind: str = "lr") -> np.ndarray:
        if self.lr is None:
            return np.full(len(items), self.base_rate, dtype=float)
        X = _postgen_features(items, pregen_probs, self.feature_set)
        if kind == "gb":
            return np.asarray(self.gb.predict_proba(X)[:, 1], dtype=float)
        return np.asarray(
            self.lr.predict_proba(self.scaler.transform(X))[:, 1], dtype=float
        )


def _oof_pregen_probs(
    items_v0: list[dict], encoder: SentenceEncoder, seed: int,
) -> np.ndarray:
    """Out-of-fold P(matchable) on V0 for honest stacking.

    Mirrors train_matchability_classifier's model (L2 logistic, C=0.1,
    balanced) but produces cross-validated predictions so the stacked
    postgen scorer sees pregen probabilities with test-time-like calibration.
    Falls back to the base rate when V0 is single-class or too small.
    """
    X, y, _ = build_pregen_features(items_v0, encoder)
    n_minority = int(min((y == 0).sum(), (y == 1).sum()))
    if n_minority < 2:
        return np.full(len(items_v0), float(y.mean()), dtype=float)
    n_splits = int(min(5, n_minority))
    clf = LogisticRegression(
        C=0.1, max_iter=5000, class_weight="balanced", random_state=seed,
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")
    return np.asarray(proba[:, 1], dtype=float)


# ---------------------------------------------------------------------------
# (B) CRC-style threshold selection and augmented evaluation
# ---------------------------------------------------------------------------

def crc_abstain_threshold(
    scores: np.ndarray,
    covered_semantic: np.ndarray,
    unmatched: np.ndarray,
    alpha: float,
) -> float:
    """Largest abstain threshold meeting the CRC-corrected coverage bound.

    Abstain when score >= t. Augmented miscoverage(t) is monotone
    non-increasing as t decreases (abstention only adds coverage on unmatched
    items and never removes set coverage), so choosing

        t* = max{ t : (n/(n+1)) * R_hat(t) + 1/(n+1) <= alpha }

    is the standard monotone conformal-risk-control selection with loss bound
    B = 1. Falls back to the smallest candidate (abstain on everything) when
    no candidate satisfies the bound.
    """
    n = len(scores)
    if n == 0:
        return 0.5
    candidates = np.unique(np.concatenate([
        scores, [0.0, 1.0 + 1e-9],
    ]))[::-1]  # descending: fewest abstentions first
    for t in candidates:
        abstained = scores >= t
        cov = (covered_semantic | (abstained & unmatched)).mean()
        r_hat = 1.0 - cov
        if (n / (n + 1)) * r_hat + 1.0 / (n + 1) <= alpha:
            return float(t)
    return float(candidates[-1])


def augmented_eval(
    scores: np.ndarray,
    t: float,
    covered_semantic: np.ndarray,
    sizes: np.ndarray,
    unmatched: np.ndarray,
) -> dict[str, float]:
    """Augmented-coverage metrics for abstain-if-score>=t on precomputed arrays."""
    n = len(scores)
    abstained = scores >= t
    covered_aug = covered_semantic | (abstained & unmatched)
    n_abst = int(abstained.sum())
    n_unm = int(unmatched.sum())
    hit = int((abstained & unmatched).sum())
    return {
        "coverage_aug": float(covered_aug.mean()) if n else float("nan"),
        "coverage_semantic": float(covered_semantic.mean()) if n else float("nan"),
        "apss_semantic": float(sizes.mean()) if n else float("nan"),
        "apss_aug": float((sizes + abstained).mean()) if n else float("nan"),
        "abstain_rate": float(abstained.mean()) if n else float("nan"),
        "abstain_precision": hit / n_abst if n_abst else float("nan"),
        "abstain_recall": hit / n_unm if n_unm else float("nan"),
        "threshold": float(t),
        "n": n,
    }


def selective_top1_threshold(
    scores: np.ndarray, top1_wrong: np.ndarray, alpha: float,
) -> float:
    """Largest t with CRC-corrected selective risk (error among answered) <= alpha.

    Answer iff score < t. Selective risk is not globally monotone in t, so we
    scan candidates descending and keep the largest satisfying t (matching the
    conformal-abstention style of risk control on the answered subset).
    Falls back to the smallest candidate (answer nothing) when unsatisfiable.
    """
    n = len(scores)
    if n == 0:
        return 0.5
    candidates = np.unique(np.concatenate([scores, [0.0, 1.0 + 1e-9]]))[::-1]
    for t in candidates:
        answered = scores < t
        n_ans = int(answered.sum())
        if n_ans == 0:
            continue
        r_hat = float((top1_wrong & answered).sum()) / n_ans
        if (n_ans / (n_ans + 1)) * r_hat + 1.0 / (n_ans + 1) <= alpha:
            return float(t)
    return float(candidates[-1])


# ---------------------------------------------------------------------------
# Per-cell evaluation
# ---------------------------------------------------------------------------

def evaluate_cell(
    data: list[dict], alpha: float, seeds: list[int], encoder: SentenceEncoder,
    logger, feature_set: str = "v1", tune: bool = False,
    trigger_kind: str = "lr",
) -> dict:
    """Run blocks (A)-(C) for one cell; returns per-seed-aggregated metrics."""
    auroc_acc: dict[str, list[float]] = {}
    auprc_acc: dict[str, list[float]] = {}
    fpr90_acc: dict[str, list[float]] = {}
    caelum_a_acc: dict[str, list[float]] = {}
    pregen_trig_acc: dict[str, list[float]] = {}
    top1_acc: dict[str, list[float]] = {}
    pareto_rows: list[dict] = []

    for seed in seeds:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(data))
        q = len(data) // 4
        v0 = [data[i] for i in perm[:q]]
        v1 = [data[i] for i in perm[q:2 * q]]
        v2 = [data[i] for i in perm[2 * q:3 * q]]
        test = [data[i] for i in perm[3 * q:]]
        # V2 halves: V2a fixes tau (support-conditional), V2b fixes t.
        half = len(v2) // 2
        v2a, v2b = v2[:half], v2[half:]

        # --- classifiers on V0 u V1 ---
        # Learned components (pregen clf, postgen scorer) may share V1 with
        # lambda selection: all of them are fixed before V2a (tau), V2b (t)
        # and Test are touched, which is what the split-conformal / CRC
        # arguments require. Pooling doubles the training set on small cells.
        v01 = v0 + v1
        pregen_clf = train_matchability_classifier(v01, encoder=encoder, seed=seed)

        def pregen_probs(items: list[dict]) -> np.ndarray:
            if len(items) == 0:
                return np.array([], dtype=float)
            X, _, _ = build_pregen_features(items, encoder)
            return np.asarray(pregen_clf.predict_proba(X)[:, 1], dtype=float)

        # Out-of-fold pregen probabilities on the training pool: the pregen
        # logistic is near-perfectly separable in-sample (768-dim on ~150
        # items), so feeding in-sample probs to the stacked postgen scorer
        # teaches it to over-rely on a feature whose distribution collapses
        # at test time.
        pg_v01 = _oof_pregen_probs(v01, encoder, seed)
        scorer = PostgenScorer(
            seed=seed, feature_set=feature_set, tune=tune,
        ).fit(v01, pg_v01)

        pg_v2b = pregen_probs(v2b)
        pg_test = pregen_probs(test)
        a_v2b = scorer.score(v2b, pg_v2b, kind=trigger_kind)
        a_test = scorer.score(test, pg_test, kind=trigger_kind)
        a_test_lr = (a_test if trigger_kind == "lr"
                     else scorer.score(test, pg_test, kind="lr"))
        a_test_gb = scorer.score(test, pg_test, kind="gb")

        # --- (A) support-failure AUROC on Test ---
        y_test = _failure_labels(test)
        p1_test = np.array([
            float(np.asarray(it["cluster_props"])[0])
            if len(it["cluster_props"]) else 0.0 for it in test
        ])
        signals = {
            "postgen_lr": a_test_lr,
            "postgen_gb": a_test_gb,
            "U_epi": np.array([float(it["U_epi"]) for it in test]),
            "U_total": np.array([float(it["U_total"]) for it in test]),
            "U_ale": np.array([float(it["U_ale"]) for it in test]),
            "one_minus_p1": 1.0 - p1_test,
            "pregen": 1.0 - pg_test,
        }
        for name, s in signals.items():
            auroc_acc.setdefault(name, []).append(compute_auroc(y_test, s))
            auprc_acc.setdefault(name, []).append(compute_auprc(y_test, s))
            fpr90_acc.setdefault(name, []).append(
                compute_fpr_at_tpr(y_test, s)
            )

        # --- base calibrator: tau on V2a (support-conditional) ---
        cal = CAELUMCalibrator(
            alpha=alpha, seed=seed, randomize=True, weight_fn="standardized",
            selection_method="set_score", support_conditional=True,
        )
        cal.calibrate(v1, v2a, verbose=False)

        def set_arrays(items: list[dict]):
            covered = np.array([bool(cal.is_covered(it)) for it in items])
            sizes = np.array(
                [float(cal.predict_set_size(it)) for it in items]
            )
            unmatched = _failure_labels(items).astype(bool)
            return covered, sizes, unmatched

        cov_v2b, siz_v2b, unm_v2b = set_arrays(v2b)
        cov_test, siz_test, unm_test = set_arrays(test)

        # --- (B) CAELUM-A: postgen trigger, CRC threshold on V2b ---
        t_post = crc_abstain_threshold(a_v2b, cov_v2b, unm_v2b, alpha)
        m_post = augmented_eval(a_test, t_post, cov_test, siz_test, unm_test)
        for k, v in m_post.items():
            caelum_a_acc.setdefault(k, []).append(float(v))

        # Pre-generation trigger for comparison (same CRC rule).
        pre_v2b = 1.0 - pg_v2b
        pre_test = 1.0 - pg_test
        t_pre = crc_abstain_threshold(pre_v2b, cov_v2b, unm_v2b, alpha)
        m_pre = augmented_eval(pre_test, t_pre, cov_test, siz_test, unm_test)
        for k, v in m_pre.items():
            pregen_trig_acc.setdefault(k, []).append(float(v))

        # Pareto sweep on Test (postgen trigger), for the frontier figure.
        for t in np.linspace(0.0, 1.0, 21):
            m = augmented_eval(a_test, t, cov_test, siz_test, unm_test)
            pareto_rows.append({
                "seed": seed, "threshold": float(t),
                "abstain_rate": m["abstain_rate"],
                "coverage_aug": m["coverage_aug"],
                "apss_aug": m["apss_aug"],
            })

        # --- (C) top-1 selective answering baseline ---
        cidx_test = np.array(
            [int(it.get("correct_cluster_idx", -1)) for it in test]
        )
        cidx_v2b = np.array(
            [int(it.get("correct_cluster_idx", -1)) for it in v2b]
        )
        t1 = selective_top1_threshold(a_v2b, cidx_v2b != 0, alpha)
        answered = a_test < t1
        n_ans = int(answered.sum())
        top1_correct = cidx_test == 0
        top1_acc.setdefault("answer_rate", []).append(float(answered.mean()))
        top1_acc.setdefault("selective_acc", []).append(
            float(top1_correct[answered].mean()) if n_ans else float("nan")
        )
        # Head-to-head at CAELUM-A's abstain rate: fraction of items where the
        # reference is delivered (in set vs as the single top-1 answer).
        k_keep = int(round((1.0 - m_post["abstain_rate"]) * len(test)))
        order = np.argsort(a_test)  # most-confident first
        keep = order[:k_keep]
        top1_acc.setdefault("ref_delivered_top1", []).append(
            float(top1_correct[keep].sum()) / len(test) if len(test) else float("nan")
        )
        top1_acc.setdefault("ref_delivered_set", []).append(
            float(cov_test[keep].sum()) / len(test) if len(test) else float("nan")
        )

    def agg(d: dict[str, list[float]]) -> dict[str, list[float]]:
        return {
            k: [float(np.nanmean(v)), float(np.nanstd(v))] for k, v in d.items()
        }

    return {
        "auroc": agg(auroc_acc),
        "auprc": agg(auprc_acc),
        "fpr90": agg(fpr90_acc),
        "caelum_a": agg(caelum_a_acc),
        "pregen_trigger": agg(pregen_trig_acc),
        "top1": agg(top1_acc),
        "pareto": pareto_rows,
        "failure_rate": float(_failure_labels(data).mean()),
    }


def compute_auprc(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision, nan when only one class is present.

    AUROC is prevalence-insensitive, so on near-saturated cells it can mask
    poor precision at any usable threshold; AP is the standard complement
    under class imbalance.
    """
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, np.asarray(s, dtype=float)))


def compute_fpr_at_tpr(
    y: np.ndarray, s: np.ndarray, tpr_target: float = 0.9,
) -> float:
    """FPR at the lowest threshold reaching ``tpr_target`` recall.

    Reads as: to catch ``tpr_target`` of support failures, what fraction of
    answerable items would the trigger needlessly flag.
    """
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, np.asarray(s, dtype=float))
    ok = tpr >= tpr_target
    return float(fpr[ok][0]) if ok.any() else float("nan")


def _model_family(model_tag: str) -> str:
    """Collapse a model tag to its family, e.g. qwen3_32b -> qwen3.

    Family = the leading vendor token, so all Qwen3 checkpoints (8b/14b/32b)
    share one family while Ministral and Gemma are their own.
    """
    return model_tag.split("_", 1)[0]


def _degenerate(res: dict) -> bool:
    """True when the cell has no class variation in support failure.

    A cell with failure rate 0 (e.g. Qwen3-8B on AmbigInst, 100% matchable)
    or 1 forces AUROC to 0.5 and abstention to a constant, so it carries no
    information about the detector and is excluded from aggregate means and
    reported separately.
    """
    fr = res.get("failure_rate")
    return fr is not None and (fr <= 0.0 or fr >= 1.0)


# ---------------------------------------------------------------------------
# (E) Pooled leave-one-cell-out (LOCO) support-failure scorer
# ---------------------------------------------------------------------------

def good_turing_missing_mass(items: list[dict], n_answers: int = 50) -> np.ndarray:
    """Good-Turing missing-mass estimate per item: theta = N1 / t.

    N1 is the number of singleton semantic clusters (count == 1) among the
    t sampled answers; cluster counts are recovered from the cached
    proportions as round(p * t). This is the frequency signal that drives
    the EE-label inclusion of Noorani et al. (arXiv 2506.05497), computed
    on our semantic clusters for a like-for-like trigger comparison.
    """
    out = np.zeros(len(items), dtype=float)
    for i, it in enumerate(items):
        props = np.asarray(it["cluster_props"], dtype=float)
        if len(props) == 0:
            out[i] = 1.0
            continue
        counts = np.rint(props * n_answers)
        n1 = int((counts == 1).sum())
        out[i] = n1 / n_answers
    return out


def run_pooled(
    cells: list[dict], unc_dir: Path, base_dir: Path, alpha: float,
    seeds: list[int], encoder: SentenceEncoder, logger,
    loco_scope: str = "cell",
    target_filter: list[str] | None = None,
    feature_set: str = "v1", tune: bool = False, trigger_kind: str = "lr",
) -> dict[str, dict]:
    """Pooled LOCO evaluation: train the support-failure scorer on the OTHER
    cells' full labeled data, deploy on the target cell with a per-cell
    CRC abstain threshold (V2b) and the standard support-conditional base
    calibrator (V1 lambda, V2a tau).

    The target cell's labels are never used to train the scorer, so this
    simultaneously tests cross-cell transfer and the value of pooled
    training data (~6k items vs ~150 per-cell). The pregen logistic used as
    a stacking feature is likewise LOCO-trained; its in-pool probabilities
    are in-sample, which is acceptable at pool scale (C=0.1, n~6k) and
    honest on the target cell by construction.
    """
    # Cache per-cell arrays once (question embeddings are the slow part).
    cache: dict[str, dict] = {}
    for cell in cells:
        path, source = resolve_source(cell, unc_dir, base_dir)
        with open(path, "r", encoding="utf-8") as f:
            data = _ensure_types(json.load(f))
        gen_name = cell["filename"].replace("unc_", "gen_").replace("_vne", "")
        attach_refusal_fractions(data, base_dir / gen_name)
        if feature_set == "v1_pke":
            attach_pke(data, encoder)
        X_pre, y_match, _ = build_pregen_features(data, encoder)
        cache[cell["cell_key"]] = {
            "data": data, "X_pre": X_pre,
            "y_match": np.asarray(y_match, dtype=int), "source": source,
        }
        logger.info("  cached %s [%s] (n=%d)", cell["cell_key"], source, len(data))

    results: dict[str, dict] = {}
    for cell in cells:
        key = cell["cell_key"]
        if target_filter and not any(sub in key for sub in target_filter):
            continue
        target = cache[key]
        # loco_scope="cell": pool = all other cells (default).
        # loco_scope="model": pool = all cells from OTHER models only — the
        # strict cross-model/cross-scale transfer test (the scorer never sees
        # any data generated by the target's LLM).
        # loco_scope="family": pool = all cells from OTHER model FAMILIES —
        # for a Qwen3-32B target the pool excludes Qwen3-8B and Qwen3-14B
        # too, so the claim "the target's entire model family is held out"
        # is literal.
        if loco_scope == "model":
            others = [
                cache[c["cell_key"]] for c in cells
                if c["model"] != cell["model"]
            ]
        elif loco_scope == "family":
            others = [
                cache[c["cell_key"]] for c in cells
                if _model_family(c["model"]) != _model_family(cell["model"])
            ]
        else:
            others = [
                cache[c["cell_key"]] for c in cells if c["cell_key"] != key
            ]

        # --- LOCO pool ---
        X_pool_pre = np.vstack([o["X_pre"] for o in others])
        y_pool_match = np.concatenate([o["y_match"] for o in others])
        items_pool = [it for o in others for it in o["data"]]

        pre_clf = LogisticRegression(
            C=0.1, max_iter=5000, class_weight="balanced", random_state=42,
        ).fit(X_pool_pre, y_pool_match)
        pg_pool = np.asarray(pre_clf.predict_proba(X_pool_pre)[:, 1])
        scorer = PostgenScorer(
            seed=42, feature_set=feature_set, tune=tune,
        ).fit(items_pool, pg_pool)

        pg_target = np.asarray(pre_clf.predict_proba(target["X_pre"])[:, 1])
        a_target = scorer.score(target["data"], pg_target, kind=trigger_kind)
        gt_target = good_turing_missing_mass(target["data"])
        uepi_target = np.array(
            [float(it["U_epi"]) for it in target["data"]]
        )

        data = target["data"]
        auroc_acc: dict[str, list[float]] = {}
        auprc_acc: dict[str, list[float]] = {}
        fpr90_acc: dict[str, list[float]] = {}
        caelum_acc: dict[str, list[float]] = {}
        trigger_acc: dict[str, dict[str, list[float]]] = {}
        sweep_rows: dict[str, list[dict]] = {}
        quart_acc: dict[str, list[float]] = {}
        for seed in seeds:
            rng = np.random.RandomState(seed)
            perm = rng.permutation(len(data))
            q = len(data) // 4
            v1 = [data[i] for i in perm[q:2 * q]]
            v2 = perm[2 * q:3 * q]
            half = len(v2) // 2
            v2a = [data[i] for i in v2[:half]]
            v2b_idx = v2[half:]
            test_idx = perm[3 * q:]
            v2b = [data[i] for i in v2b_idx]
            test = [data[i] for i in test_idx]

            cal = CAELUMCalibrator(
                alpha=alpha, seed=seed, randomize=True,
                weight_fn="standardized", selection_method="set_score",
                support_conditional=True,
            )
            cal.calibrate(v1, v2a, verbose=False)

            cov_test = np.array([bool(cal.is_covered(it)) for it in test])
            siz_test = np.array([float(cal.predict_set_size(it)) for it in test])
            unm_test = _failure_labels(test).astype(bool)
            cov_v2b = np.array([bool(cal.is_covered(it)) for it in v2b])
            unm_v2b = _failure_labels(v2b).astype(bool)

            # loco_scope="all": upper-bound control — the scorer additionally
            # sees the TARGET cell's V0 u V1 labels (never V2/Test). Used to
            # quantify what LOCO gives up by not having target labels.
            if loco_scope == "all":
                v01_idx = np.concatenate([perm[:q], perm[q:2 * q]])
                scorer_s = PostgenScorer(
                    seed=seed, feature_set=feature_set, tune=tune,
                ).fit(
                    items_pool + [data[i] for i in v01_idx],
                    np.concatenate([pg_pool, pg_target[v01_idx]]),
                )
                a_seed = scorer_s.score(data, pg_target, kind=trigger_kind)
            else:
                a_seed = a_target
            a_v2b = a_seed[v2b_idx]
            a_test = a_seed[test_idx]

            y_pool = unm_test.astype(int)
            uepi_test = np.array([float(it["U_epi"]) for it in test])
            for name, sig in (
                ("pooled_lr", a_test),
                ("U_epi", uepi_test),
                ("gt_missing_mass", gt_target[test_idx]),
            ):
                auroc_acc.setdefault(name, []).append(
                    compute_auroc(y_pool, sig)
                )
                auprc_acc.setdefault(name, []).append(
                    compute_auprc(y_pool, sig)
                )
                fpr90_acc.setdefault(name, []).append(
                    compute_fpr_at_tpr(y_pool, sig)
                )

            t = crc_abstain_threshold(a_v2b, cov_v2b, unm_v2b, alpha)
            m = augmented_eval(a_test, t, cov_test, siz_test, unm_test)
            for k, v in m.items():
                caelum_acc.setdefault(k, []).append(float(v))

            # Unrepaired vs repaired coverage by U_epi quartile: the direct
            # evidence that abstention concentrates where plain sets fail.
            u_test = uepi_target[test_idx]
            cov_rep = cov_test | ((a_test >= t) & unm_test)
            edges = np.quantile(u_test, [0.0, 0.25, 0.5, 0.75, 1.0])
            edges[-1] += 1e-12
            b_idx = np.clip(np.digitize(u_test, edges[1:-1]), 0, 3)
            for b in range(4):
                mask = b_idx == b
                if mask.any():
                    quart_acc.setdefault(f"unrepaired_b{b}", []).append(
                        float(cov_test[mask].mean())
                    )
                    quart_acc.setdefault(f"repaired_b{b}", []).append(
                        float(cov_rep[mask].mean())
                    )
                    quart_acc.setdefault(f"abstain_b{b}", []).append(
                        float((a_test >= t)[mask].mean())
                    )

            # Head-to-head trigger comparison (identical CRC rule): the
            # learned LOCO score vs the Good-Turing missing-mass signal of
            # Noorani et al. vs raw U_epi.
            for name, sig in (
                ("gt_missing_mass", gt_target),
                ("U_epi_trigger", uepi_target / max(uepi_target.max(), 1e-9)),
            ):
                t_b = crc_abstain_threshold(
                    sig[v2b_idx], cov_v2b, unm_v2b, alpha,
                )
                m_b = augmented_eval(
                    sig[test_idx], t_b, cov_test, siz_test, unm_test,
                )
                for k, v in m_b.items():
                    trigger_acc.setdefault(name, {}).setdefault(k, []).append(
                        float(v)
                    )

            # Full frontier sweeps for every trigger signal (for the paper's
            # aggregate frontier figure). Thresholds are taken at each
            # signal's own test-quantiles so scales are comparable.
            for name, sig in (
                ("learned", a_seed),
                ("pregen", 1.0 - pg_target),
                ("U_epi", uepi_target / max(uepi_target.max(), 1e-9)),
                ("gt_missing_mass", gt_target),
            ):
                s_test = sig[test_idx]
                qs = np.unique(np.concatenate([
                    np.quantile(s_test, np.linspace(0.0, 1.0, 21)),
                    [s_test.max() + 1e-9],
                ]))
                for t in qs:
                    m_s = augmented_eval(
                        s_test, float(t), cov_test, siz_test, unm_test,
                    )
                    sweep_rows.setdefault(name, []).append({
                        "seed": seed,
                        "abstain_rate": m_s["abstain_rate"],
                        "coverage_aug": m_s["coverage_aug"],
                        "apss_aug": m_s["apss_aug"],
                    })

        agg = lambda d: {k: [float(np.nanmean(v)), float(np.nanstd(v))]
                         for k, v in d.items()}
        results[key] = {
            "auroc": agg(auroc_acc), "auprc": agg(auprc_acc),
            "fpr90": agg(fpr90_acc), "caelum_a": agg(caelum_acc),
            "triggers": {n: agg(d) for n, d in trigger_acc.items()},
            "sweeps": sweep_rows,
            "quartile_repair": agg(quart_acc),
            "source": target["source"],
            "failure_rate": float(_failure_labels(data).mean()),
        }
        logger.info(
            "  %s: pooled AUROC=%.3f (U_epi=%.3f) | cov_aug=%.3f abst=%.3f "
            "prec=%.3f",
            key, results[key]["auroc"]["pooled_lr"][0],
            results[key]["auroc"]["U_epi"][0],
            results[key]["caelum_a"]["coverage_aug"][0],
            results[key]["caelum_a"]["abstain_rate"][0],
            results[key]["caelum_a"]["abstain_precision"][0],
        )
    return results


def write_pooled_report(
    results: dict[str, dict], out_prefix: Path, *, write_markdown: bool = False,
) -> None:
    """Write JSON and optionally Markdown for the pooled LOCO block."""
    with open(f"{out_prefix}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, default=float)
    # Side-by-side with the per-cell run if available.
    percell_path = RESULTS_DIR / "caelum_abstain_main.json"
    percell = {}
    if percell_path.exists():
        with open(percell_path, "r", encoding="utf-8") as f:
            percell = json.load(f)
    lines = [
        "# CAELUM-A pooled LOCO scorer (trained on the other cells)", "",
        "| Model | Dataset | AUROC pooled | AUROC per-cell | AUROC U_epi | "
        "cov_aug | abstain | precision | prec per-cell |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    aggs: dict[str, list[float]] = {}
    excluded = [k for k, r in results.items() if _degenerate(r)]
    for key, res in results.items():
        model, ds = key.split("__")
        pc = percell.get(key, {})
        pc_auroc = pc.get("auroc", {}).get("postgen_lr", [float("nan")])[0]
        pc_prec = pc.get("caelum_a", {}).get(
            "abstain_precision", [float("nan")])[0]
        row = [
            MODEL_PRETTY.get(model, model), DATASET_PRETTY.get(ds, ds),
            f"{res['auroc']['pooled_lr'][0]:.3f}", f"{pc_auroc:.3f}",
            f"{res['auroc']['U_epi'][0]:.3f}",
            f"{res['caelum_a']['coverage_aug'][0]:.3f}",
            f"{res['caelum_a']['abstain_rate'][0]:.3f}",
            f"{res['caelum_a']['abstain_precision'][0]:.3f}",
            f"{pc_prec:.3f}",
        ]
        lines.append("| " + " | ".join(row) + " |")
        if _degenerate(res):
            continue
        for name, val in (
            ("pooled", res["auroc"]["pooled_lr"][0]),
            ("percell", pc_auroc),
            ("uepi", res["auroc"]["U_epi"][0]),
            ("cov_aug", res["caelum_a"]["coverage_aug"][0]),
            ("abstain", res["caelum_a"]["abstain_rate"][0]),
            ("prec", res["caelum_a"]["abstain_precision"][0]),
            ("prec_percell", pc_prec),
        ):
            aggs.setdefault(name, []).append(val)
    note = (
        f" (excluding {len(excluded)} degenerate cell(s): "
        f"{', '.join(excluded)})" if excluded else ""
    )
    n_agg = len(aggs.get("pooled", []))
    lines += ["", f"**Aggregates (mean over {n_agg} cells{note}):** "
              + ", ".join(f"{k}={np.nanmean(v):.3f}" for k, v in aggs.items())]
    if write_markdown:
        with open(f"{out_prefix}.md", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# (D) Sampling-budget curves
# ---------------------------------------------------------------------------

def _greedy_cluster_from_emb(emb: np.ndarray, threshold: float = 0.7) -> list[int]:
    """Greedy cosine clustering on precomputed normalized embeddings."""
    n = len(emb)
    cids = [-1] * n
    nxt = 0
    for i in range(n):
        if cids[i] == -1:
            cids[i] = nxt
            for j in range(i + 1, n):
                if cids[j] == -1 and float(np.dot(emb[i], emb[j])) >= threshold:
                    cids[j] = nxt
            nxt += 1
    return cids


def budget_curve_cell(
    gen_path: Path, encoder: SentenceEncoder, budgets: list[int], logger,
) -> dict[str, list[float]]:
    """Match rate as a function of answers-per-clarification budget."""
    with open(gen_path, "r", encoding="utf-8") as f:
        gen = json.load(f)
    rates: dict[int, list[int]] = {m: [] for m in budgets}
    for item in gen:
        answers_per_clar = item.get("answers") or []
        gts = item.get("gt_answers") or []
        if not answers_per_clar or not gts:
            continue
        flat = [str(a) for cl in answers_per_clar for a in cl]
        if not flat:
            continue
        try:
            emb = encoder.encode(flat, normalize=True)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("    budget: encode failed on one item (%s); skipped", exc)
            continue
        # Index map: (clar, j) -> flat index
        offsets = np.cumsum([0] + [len(cl) for cl in answers_per_clar])
        for m in budgets:
            idx = [
                offsets[c] + j
                for c, cl in enumerate(answers_per_clar)
                for j in range(min(m, len(cl)))
            ]
            sub_answers = [flat[i] for i in idx]
            sub_emb = emb[idx]
            cids = _greedy_cluster_from_emb(sub_emb)
            props, reps = cluster_proportions_and_representatives(
                sub_answers, cids
            )
            # cluster members in sorted order
            counts = np.bincount(cids)
            order = np.argsort(-counts / counts.sum())
            old_to_new = {int(o): k for k, o in enumerate(order)}
            members: list[list[str]] = [[] for _ in range(len(reps))]
            for a, cid in zip(sub_answers, cids):
                members[old_to_new[cid]].append(a)
            k = find_correct_cluster(
                reps, gts, encoder,
                embed_threshold=0.4, rouge_threshold=0.15,
                jaccard_threshold=0.25, cluster_members=members,
            )
            rates[m].append(1 if k >= 0 else 0)
    return {
        "budgets": budgets,
        "match_rate": [
            float(np.mean(rates[m])) if rates[m] else float("nan")
            for m in budgets
        ],
        "n": len(rates[budgets[0]]) if budgets else 0,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(ms: list[float]) -> str:
    return f"{ms[0]:.3f}±{ms[1]:.3f}"


def write_reports(
    cell_results: dict[str, dict], out_prefix: Path, *, write_markdown: bool = False,
) -> None:
    """Write JSON and optionally Markdown for blocks (A)-(C)."""
    with open(f"{out_prefix}.json", "w", encoding="utf-8") as f:
        json.dump(cell_results, f, indent=1, default=float)

    lines = ["# CAELUM-A: abstention-augmented sets (judged matching)", ""]
    lines += ["## (A) Support-failure AUROC (mean±std over seeds)", ""]
    sig_names = ["postgen_lr", "postgen_gb", "U_epi", "U_total", "U_ale",
                 "one_minus_p1", "pregen"]
    lines.append("| Model | Dataset | " + " | ".join(sig_names) + " |")
    lines.append("|" + "---|" * (len(sig_names) + 2))
    for key, res in cell_results.items():
        model, ds = key.split("__")
        row = [MODEL_PRETTY.get(model, model), DATASET_PRETTY.get(ds, ds)]
        row += [_fmt(res["auroc"].get(s, [float("nan")] * 2)) for s in sig_names]
        lines.append("| " + " | ".join(row) + " |")
    # aggregate over non-degenerate cells only (17+1 reporting)
    normal = {k: r for k, r in cell_results.items() if not _degenerate(r)}
    excluded = [k for k, r in cell_results.items() if _degenerate(r)]
    note = (
        f" (excluding {len(excluded)} degenerate cell(s): "
        f"{', '.join(excluded)}; failure rate 0 or 1 forces AUROC=0.5)"
        if excluded else ""
    )
    lines += ["", f"**Aggregate (mean over {len(normal)} cells{note}):**", ""]
    for s in sig_names:
        vals = [res["auroc"][s][0] for res in normal.values()
                if s in res["auroc"] and np.isfinite(res["auroc"][s][0])]
        lines.append(f"- {s}: {np.mean(vals):.3f}")

    lines += ["", "## (B) CAELUM-A two-threshold (postgen trigger, CRC on V2b)", ""]
    cols = ["coverage_aug", "coverage_semantic", "abstain_rate",
            "abstain_precision", "abstain_recall", "apss_aug"]
    lines.append("| Model | Dataset | " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * (len(cols) + 2))
    for key, res in cell_results.items():
        model, ds = key.split("__")
        row = [MODEL_PRETTY.get(model, model), DATASET_PRETTY.get(ds, ds)]
        row += [_fmt(res["caelum_a"].get(c, [float("nan")] * 2)) for c in cols]
        lines.append("| " + " | ".join(row) + " |")
    for label, block in (("postgen", "caelum_a"), ("pregen", "pregen_trigger")):
        vals = {c: np.mean([
            res[block][c][0] for res in normal.values()
            if c in res[block] and np.isfinite(res[block][c][0])
        ]) for c in cols}
        lines.append(
            f"\n**Aggregate ({label} trigger, {len(normal)} cells{note}):** "
            + ", ".join(f"{c}={vals[c]:.3f}" for c in cols)
        )

    lines += ["", "## (C) Top-1 selective answering vs CAELUM-A sets", ""]
    cols_c = ["answer_rate", "selective_acc", "ref_delivered_top1",
              "ref_delivered_set"]
    lines.append("| Model | Dataset | " + " | ".join(cols_c) + " |")
    lines.append("|" + "---|" * (len(cols_c) + 2))
    for key, res in cell_results.items():
        model, ds = key.split("__")
        row = [MODEL_PRETTY.get(model, model), DATASET_PRETTY.get(ds, ds)]
        row += [_fmt(res["top1"].get(c, [float("nan")] * 2)) for c in cols_c]
        lines.append("| " + " | ".join(row) + " |")

    if write_markdown:
        with open(f"{out_prefix}.md", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def plot_pareto(cell_results: dict[str, dict], out_path: Path) -> None:
    """Grid of abstention-rate vs augmented-coverage frontiers per cell."""
    keys = list(cell_results.keys())
    ncols = 6
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.0 * ncols, 2.6 * nrows), squeeze=False,
    )
    for ax in axes.flat:
        ax.set_visible(False)
    for i, key in enumerate(keys):
        ax = axes[i // ncols][i % ncols]
        ax.set_visible(True)
        rows = cell_results[key]["pareto"]
        by_t: dict[float, list[tuple[float, float]]] = {}
        for r in rows:
            by_t.setdefault(r["threshold"], []).append(
                (r["abstain_rate"], r["coverage_aug"])
            )
        pts = sorted(
            (np.mean([p[0] for p in v]), np.mean([p[1] for p in v]))
            for v in by_t.values()
        )
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", ms=2.5, lw=1)
        ax.axhline(1 - ALPHA, ls=":", c="gray", lw=0.8)
        model, ds = key.split("__")
        ax.set_title(
            f"{MODEL_PRETTY.get(model, model)} / {DATASET_PRETTY.get(ds, ds)}",
            fontsize=8,
        )
        ax.set_xlabel("abstain rate", fontsize=7)
        ax.set_ylabel("aug. coverage", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_ylim(0.0, 1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unc_dir", type=str, default=str(DEFAULT_UNC_DIR))
    parser.add_argument("--generations_dir", type=str, default=str(GENERATIONS_DIR),
                        help="Folder with the heuristic unc_* and gen_* caches.")
    parser.add_argument("--cells", type=str, nargs="*", default=None)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    parser.add_argument(
        "--budget", action="store_true",
        help="Run the sampling-budget curves block instead of (A)-(C).",
    )
    parser.add_argument(
        "--pooled", action="store_true",
        help="Run the pooled leave-one-cell-out scorer block.",
    )
    parser.add_argument(
        "--loco_scope", type=str, default="cell",
        choices=("cell", "model", "family", "all"),
        help="Pool scope for --pooled: exclude only the target cell "
             "('cell'), all cells of the target's model ('model', the "
             "strict cross-model transfer test), or all cells of the "
             "target's model family ('family', e.g. every Qwen3 size).",
    )
    parser.add_argument(
        "--exclude_models", type=str, nargs="*", default=None,
        help="Model tags to drop from discovery entirely (targets AND "
             "pools). Use to pin an evaluation to a fixed cell set while "
             "new model caches are landing.",
    )
    parser.add_argument(
        "--feature_set", type=str, default="v1",
        choices=("v1", "v1_pke") + tuple(FEATURE_SUBSETS),
        help="Scorer feature set: 'v1' (15 features, the paper default), "
             "'v1_pke' (adds predictive kernel entropy, computed at load "
             "time from the cached answers), or a v1 column subset for the "
             "decomposition ablation (v1_no_uale, v1_no_uepi, v1_no_ae, "
             "v1_no_decomp, decomp_only).",
    )
    parser.add_argument(
        "--tune_scorer", action="store_true",
        help="Select LR C and GB size by CV AUROC on the training pool "
             "(default off: fixed C=1.0, GB(100,2) as in the paper).",
    )
    parser.add_argument(
        "--trigger_kind", type=str, default="lr", choices=("lr", "gb"),
        help="Which scorer head drives the abstention trigger.",
    )
    parser.add_argument(
        "--out_tag", type=str, default="",
        help="Suffix appended to output file prefixes so variant runs "
             "(e.g. _pke) never overwrite the headline results.",
    )
    parser.add_argument("--write-markdown", action="store_true",
                        help="Also write Markdown reports (off by default).")
    args = parser.parse_args()

    tag = "budget" if args.budget else ("pooled" if args.pooled else "main")
    if args.out_tag:
        tag = f"{tag}_{args.out_tag}"
    logger = _setup_logging(RESULTS_DIR / f"caelum_abstain_{tag}.txt")
    t0 = time.time()

    cells = discover_cells(Path(args.generations_dir))
    if args.exclude_models:
        cells = [c for c in cells if c["model"] not in args.exclude_models]
    if args.cells:
        cells = [
            c for c in cells
            if any(sub in c["cell_key"] for sub in args.cells)
        ]
    logger.info("CAELUM-A (%s block): %d cells, seeds=%s, alpha=%.2f",
                tag, len(cells), args.seeds, args.alpha)

    encoder = SentenceEncoder()

    if args.pooled:
        suffix = "" if args.loco_scope == "cell" else f"_{args.loco_scope}"
        # The LOCO pool must come from the FULL discovery — --cells only
        # narrows which targets are evaluated, never the training pool.
        # --exclude_models, by contrast, removes cells from BOTH.
        all_cells = discover_cells(Path(args.generations_dir))
        if args.exclude_models:
            all_cells = [
                c for c in all_cells if c["model"] not in args.exclude_models
            ]
        results = run_pooled(
            all_cells, Path(args.unc_dir), Path(args.generations_dir), args.alpha,
            args.seeds, encoder, logger,
            loco_scope=args.loco_scope, target_filter=args.cells,
            feature_set=args.feature_set, tune=args.tune_scorer,
            trigger_kind=args.trigger_kind,
        )
        if args.out_tag:
            suffix = f"{suffix}_{args.out_tag}"
        write_pooled_report(
            results, RESULTS_DIR / f"caelum_abstain_pooled{suffix}",
            write_markdown=args.write_markdown,
        )
        logger.info("Pooled block done in %.1f min", (time.time() - t0) / 60)
        return 0

    if args.budget:
        budget_json = RESULTS_DIR / "caelum_abstain_budget.json"
        # Resume: skip cells already present in a partial output file.
        out: dict[str, dict] = {}
        if budget_json.exists():
            with open(budget_json, "r", encoding="utf-8") as f:
                out = json.load(f)
            logger.info("Resuming: %d cells already computed", len(out))
        for cell in cells:
            if cell["cell_key"] in out:
                continue
            gen_name = cell["filename"].replace("unc_", "gen_").replace("_vne", "")
            gen_path = Path(args.generations_dir) / gen_name
            if not gen_path.exists():
                logger.warning("  %s: gen file missing (%s), skipping",
                               cell["cell_key"], gen_name)
                continue
            t_cell = time.time()
            try:
                res = budget_curve_cell(gen_path, encoder, BUDGETS, logger)
            except Exception:
                logger.exception("  %s: budget block failed; continuing",
                                 cell["cell_key"])
                continue
            out[cell["cell_key"]] = res
            # Incremental write so a hard crash never loses computed cells.
            with open(budget_json, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=1, default=float)
            logger.info(
                "  %s: %s  (%.0fs)", cell["cell_key"],
                " ".join(f"m={m}:{r:.3f}" for m, r in
                         zip(res["budgets"], res["match_rate"])),
                time.time() - t_cell,
            )
        with open(budget_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, default=float)
        # plot
        fig, ax = plt.subplots(figsize=(6, 4))
        for key, res in out.items():
            ax.plot(
                [5 * m for m in res["budgets"]], res["match_rate"],
                "-o", ms=3, lw=1, alpha=0.7, label=key,
            )
        ax.set_xlabel("total sampled answers (5 clarifications x m)")
        ax.set_ylabel("match rate (heuristic)")
        ax.axhline(0.9, ls=":", c="gray", lw=0.8)
        ax.legend(fontsize=5, ncols=2)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "caelum_abstain_budget.png", dpi=150)
        plt.close(fig)
        logger.info("Budget block done in %.1f min", (time.time() - t0) / 60)
        return 0

    cell_results: dict[str, dict] = {}
    for cell in cells:
        path, source = resolve_source(cell, Path(args.unc_dir), Path(args.generations_dir))
        with open(path, "r", encoding="utf-8") as f:
            data = _ensure_types(json.load(f))
        gen_name = cell["filename"].replace("unc_", "gen_").replace("_vne", "")
        attach_refusal_fractions(data, Path(args.generations_dir) / gen_name)
        if args.feature_set == "v1_pke":
            attach_pke(data, encoder)
        t_cell = time.time()
        res = evaluate_cell(
            data, args.alpha, args.seeds, encoder, logger,
            feature_set=args.feature_set, tune=args.tune_scorer,
            trigger_kind=args.trigger_kind,
        )
        cell_results[cell["cell_key"]] = res
        logger.info(
            "  %s [%s]: AUROC postgen=%.3f U_epi=%.3f | cov_aug=%.3f "
            "abst=%.3f prec=%.3f  (%.0fs)",
            cell["cell_key"], source,
            res["auroc"]["postgen_lr"][0], res["auroc"]["U_epi"][0],
            res["caelum_a"]["coverage_aug"][0],
            res["caelum_a"]["abstain_rate"][0],
            res["caelum_a"]["abstain_precision"][0],
            time.time() - t_cell,
        )

    write_reports(cell_results, RESULTS_DIR / f"caelum_abstain_{tag}",
                  write_markdown=args.write_markdown)
    plot_pareto(cell_results, PLOTS_DIR / f"caelum_abstain_pareto_{tag}.png"
                if args.out_tag else PLOTS_DIR / "caelum_abstain_pareto.png")
    logger.info("Main block done in %.1f min", (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
