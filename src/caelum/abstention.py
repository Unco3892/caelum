"""Abstention-augmented prediction sets and pre-generation matchability.

This module provides the machinery for the *abstention* extension of CAELUM:
a question-level matchability classifier m_hat(x) predicts whether a question
is answerable on the generated support (i.e., whether the reference answer
lands in some sampled cluster). When m_hat's confidence is low, the prediction
set is augmented with an explicit ABSTAIN symbol. Coverage is then counted over
the augmented output space: an item is covered if either its semantic set
covers the reference OR the model abstained on a genuinely unmatched item.

The matchability classifier uses only **pre-generation** features of the
question (sentence embedding + a handful of hand-crafted signals), so the
stratum is a property of the question rather than of the generation
realisation. These features were originally defined in
``src/run_stratified_pregen.py``; they live here so both that script and the
abstention utilities share a single implementation.

Public API:
    - :func:`build_pregen_features`         -- feature matrix from questions.
    - :func:`train_matchability_classifier` -- fit L2 logistic on Val0.
    - :func:`augmented_metrics`             -- abstention-augmented coverage/APSS.
    - :func:`calibrate_abstain_threshold`   -- choose the abstain threshold on V2.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

from sklearn.linear_model import LogisticRegression

from caelum.utils import SentenceEncoder

logger = logging.getLogger(__name__)

# Symbol appended to a prediction set when the model abstains.
ABSTAIN = "<ABSTAIN>"

# Keyword tier for detecting refusal/hedge answers. Refusals collapse the
# sampled answer distribution (low entropy reads as confidence) while
# explicitly verbalising a knowledge gap, so the fraction of refusing
# samples is an epistemic signal invisible to distributional entropy.
REFUSAL_KEYWORDS = [
    "i don't know", "i do not know", "no information", "not aware",
    "cannot find", "can't find", "couldn't find", "does not exist",
    "doesn't exist", "fictional", "i'm sorry", "i am sorry", "unable to",
    "do not have", "don't have", "no record", "cannot determine",
    "can't determine", "not familiar", "no publicly available",
    "insufficient information", "please provide", "could you clarify",
    "i'm not sure", "i am not sure", "unknown",
]


def refusal_fraction(answers_nested: list[list[str]]) -> float:
    """Share of sampled answers hitting the refusal/hedge keyword tier."""
    flat = [str(a).lower() for cl in answers_nested for a in cl]
    if not flat:
        return 0.0
    hits = sum(1 for a in flat if any(k in a for k in REFUSAL_KEYWORDS))
    return hits / len(flat)


def attach_refusal_fractions(items: list[dict], gen_path) -> int:
    """Attach ``refusal_frac`` to each item by joining the raw generations.

    Items are matched to the cached ``gen_*.json`` records by question
    string. Items without a match get ``refusal_frac = 0.0``.

    Args:
        items: Cached unc records (mutated in place).
        gen_path: Path to the corresponding gen_*.json file.

    Returns:
        Number of items that could not be matched to a generation record.
    """
    import json as _json
    from pathlib import Path as _Path
    gen_path = _Path(gen_path)
    if not gen_path.exists():
        for it in items:
            it.setdefault("refusal_frac", 0.0)
        return len(items)
    with open(gen_path, "r", encoding="utf-8") as f:
        gen = _json.load(f)
    gen_by_q = {str(g.get("question", "")): g for g in gen}
    n_missing = 0
    for it in items:
        g = gen_by_q.get(str(it.get("question", "")))
        if g is None:
            it["refusal_frac"] = 0.0
            n_missing += 1
        else:
            it["refusal_frac"] = refusal_fraction(g.get("answers") or [])
    return n_missing

# Question-prefix one-hot vocabulary. Items not matching any of these get a
# residual "other" indicator.
QUESTION_PREFIXES = [
    "what", "who", "when", "where", "why", "how", "which",
    "is", "are", "was", "were", "did", "do", "does", "can",
    "name", "list", "describe",
]

_TOKEN_RE = re.compile(r"\b\w+\b")
_NUMBER_RE = re.compile(r"\b\d+\b")


# ---------------------------------------------------------------------------
# Pre-generation feature extraction (shared with run_stratified_pregen.py)
# ---------------------------------------------------------------------------

def _question_prefix_one_hot(question: str) -> np.ndarray:
    """One-hot of the first lowercased word against QUESTION_PREFIXES + 'other'."""
    out = np.zeros(len(QUESTION_PREFIXES) + 1, dtype=float)
    if not question:
        out[-1] = 1.0
        return out
    first = question.strip().lower().split(maxsplit=1)
    if not first:
        out[-1] = 1.0
        return out
    head = re.sub(r"[^\w]", "", first[0])
    if head in QUESTION_PREFIXES:
        out[QUESTION_PREFIXES.index(head)] = 1.0
    else:
        out[-1] = 1.0
    return out


def build_pregen_features(
    data: list[dict], encoder: SentenceEncoder,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build (X, y, feature_names) using ONLY pre-generation question features.

    Args:
        data: List of cached unc_* records (questions + matchability labels).
        encoder: A SentenceEncoder for the all-mpnet-base-v2 embedding.

    Returns:
        X: (n, 768 + small) feature matrix.
        y: (n,) binary matchability labels (1 if correct_cluster_idx >= 0).
        feature_names: list of human-readable feature names matching the
            non-embedding columns (the embedding columns are unnamed).
    """
    n = len(data)
    questions = [str(item.get("question", "") or "") for item in data]

    # Sentence embeddings (frozen mpnet)
    embs = encoder.encode(questions, normalize=True)  # (n, 768)
    if not isinstance(embs, np.ndarray):
        embs = np.asarray(embs)

    # Hand-crafted features
    handcrafted_names = [
        "q_chars", "q_tokens", "n_capitalized_tokens",
        "n_numeric_tokens", "has_year",
    ] + [f"prefix_{p}" for p in QUESTION_PREFIXES] + ["prefix_other"]

    handcrafted = np.zeros((n, len(handcrafted_names)), dtype=float)
    for i, q in enumerate(questions):
        toks = _TOKEN_RE.findall(q)
        handcrafted[i, 0] = float(len(q))
        handcrafted[i, 1] = float(len(toks))
        # Capitalised tokens (excluding sentence-initial)
        if len(toks) > 1:
            cap_count = sum(
                1 for t in toks[1:] if t and t[0].isupper()
            )
        else:
            cap_count = 0
        handcrafted[i, 2] = float(cap_count)
        nums = _NUMBER_RE.findall(q)
        handcrafted[i, 3] = float(len(nums))
        # Has a 4-digit token plausibly a year
        handcrafted[i, 4] = float(any(
            len(s) == 4 and 1000 <= int(s) <= 2100 for s in nums
        ))
        # One-hot prefix
        prefix_oh = _question_prefix_one_hot(q)
        handcrafted[i, 5: 5 + len(prefix_oh)] = prefix_oh

    X = np.concatenate([embs, handcrafted], axis=1)
    y = np.array([
        1 if int(item.get("correct_cluster_idx", -1)) >= 0 else 0
        for item in data
    ], dtype=int)

    return X, y, handcrafted_names


# ---------------------------------------------------------------------------
# Degenerate-case fallback classifier
# ---------------------------------------------------------------------------

class _ConstantProbaClassifier:
    """Fallback used when Val0 has a single matchability class.

    Mimics the subset of the scikit-learn classifier API consumed by this
    module (``predict_proba`` returning P(matchable) in column 1) by always
    predicting the constant base rate observed at fit time. This keeps the
    downstream abstention code branch-free when a cell's Val0 is entirely
    matched or entirely unmatched.
    """

    def __init__(self, p_matchable: float) -> None:
        self.p_matchable = float(np.clip(p_matchable, 0.0, 1.0))
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        col1 = np.full(n, self.p_matchable, dtype=float)
        return np.stack([1.0 - col1, col1], axis=1)


# ---------------------------------------------------------------------------
# Matchability classifier
# ---------------------------------------------------------------------------

def train_matchability_classifier(
    items_v0: list[dict],
    encoder: SentenceEncoder | None = None,
    *,
    seed: int = 42,
    C: float = 0.1,
) -> Any:
    """Fit an L2 logistic regression predicting question matchability.

    The classifier is trained on pre-generation features (see
    :func:`build_pregen_features`) with the matchability label
    ``1{correct_cluster_idx >= 0}``. With a ~768-dim embedding and a small
    Val0, an L2-regularised logistic (``C=0.1``) is the right complexity.

    Args:
        items_v0: Val0 records used to fit the classifier.
        encoder: Optional shared SentenceEncoder. A fresh one is constructed
            when omitted (loads all-mpnet-base-v2; deterministic).
        seed: Random state for the logistic solver.
        C: Inverse L2-regularisation strength.

    Returns:
        A fitted ``sklearn.linear_model.LogisticRegression``. When Val0 has a
        single class, a :class:`_ConstantProbaClassifier` predicting the base
        rate is returned instead (same ``predict_proba`` contract).
    """
    if encoder is None:
        encoder = SentenceEncoder()
    X, y, _ = build_pregen_features(items_v0, encoder)

    if len(np.unique(y)) < 2:
        base_rate = float(y.mean()) if len(y) else 0.0
        logger.warning(
            "train_matchability_classifier: single-class Val0 (base rate "
            "%.3f); returning a constant-probability fallback classifier.",
            base_rate,
        )
        return _ConstantProbaClassifier(base_rate)

    clf = LogisticRegression(
        C=C, max_iter=5000, class_weight="balanced", random_state=seed,
    )
    clf.fit(X, y)
    return clf


def _matchable_probs(
    items: list[dict], clf: Any, encoder: SentenceEncoder | None,
) -> np.ndarray:
    """Return P(matchable) for each item under ``clf`` (empty -> empty array)."""
    if len(items) == 0:
        return np.array([], dtype=float)
    if encoder is None:
        encoder = SentenceEncoder()
    X, _, _ = build_pregen_features(items, encoder)
    return np.asarray(clf.predict_proba(X)[:, 1], dtype=float)


# ---------------------------------------------------------------------------
# Abstention-augmented evaluation
# ---------------------------------------------------------------------------

def augmented_metrics(
    test_items: list[dict],
    calibrator: Any,
    clf: Any,
    threshold: float,
    encoder: SentenceEncoder | None = None,
) -> dict[str, float]:
    """Evaluate abstention-augmented prediction sets on a test split.

    For each item the classifier predicts matchability probability p. When
    ``p < threshold`` the model abstains: the semantic set C(x) is augmented
    with an explicit ABSTAIN symbol, C_aug(x) = C(x) u {ABSTAIN}. Coverage is
    counted over the augmented output space:

        covered_aug = is_covered(x)  OR  (abstained AND correct_cluster_idx < 0)

    i.e., abstaining on a genuinely unmatched item counts as a correct
    "I don't know", while abstaining on a matchable item does not rescue a
    missed set.

    Args:
        test_items: Test records with cluster_props / correct_cluster_idx / U_*.
        calibrator: A calibrated CAELUM-style calibrator (is_covered /
            predict_set_size).
        clf: Fitted matchability classifier (predict_proba).
        threshold: Abstain when P(matchable) < threshold.
        encoder: Optional shared SentenceEncoder for feature extraction.

    Returns:
        Dict with:
            - coverage_aug:     augmented coverage.
            - coverage_semantic: coverage of the raw semantic set (no abstain).
            - apss_semantic:    mean |C(x)|.
            - apss_aug:         mean |C(x)| + 1{abstained}.
            - abstain_rate:     fraction of items abstained on.
            - abstain_precision: P(unmatched | abstained).
            - abstain_recall:   P(abstained | unmatched).
            - n:                number of test items.
    """
    n = len(test_items)
    if n == 0:
        return _augmented_core(
            np.array([], dtype=bool), np.array([], dtype=float),
            np.array([], dtype=bool), np.array([], dtype=bool),
        )

    probs = _matchable_probs(test_items, clf, encoder)
    abstained = probs < threshold
    cidx = np.array([int(it.get("correct_cluster_idx", -1)) for it in test_items])
    unmatched = cidx < 0
    sizes = np.array(
        [calibrator.predict_set_size(it) for it in test_items], dtype=float
    )
    covered_semantic = np.array(
        [bool(calibrator.is_covered(it)) for it in test_items], dtype=bool
    )
    return _augmented_core(covered_semantic, sizes, abstained, unmatched)


def _augmented_core(
    covered_semantic: np.ndarray,
    sizes: np.ndarray,
    abstained: np.ndarray,
    unmatched: np.ndarray,
) -> dict[str, float]:
    """Assemble abstention-augmented metrics from precomputed per-item arrays.

    Factored out so :func:`calibrate_abstain_threshold` can sweep thresholds
    without recomputing sentence embeddings or prediction sets for each
    candidate.
    """
    n = len(covered_semantic)
    if n == 0:
        return {
            "coverage_aug": float("nan"), "coverage_semantic": float("nan"),
            "apss_semantic": float("nan"), "apss_aug": float("nan"),
            "abstain_rate": float("nan"), "abstain_precision": float("nan"),
            "abstain_recall": float("nan"), "n": 0,
        }
    covered_aug = covered_semantic | (abstained & unmatched)
    n_abstain = int(abstained.sum())
    n_unmatched = int(unmatched.sum())
    n_abstain_unmatched = int((abstained & unmatched).sum())
    abstain_precision = (
        n_abstain_unmatched / n_abstain if n_abstain > 0 else float("nan")
    )
    abstain_recall = (
        n_abstain_unmatched / n_unmatched if n_unmatched > 0 else float("nan")
    )
    return {
        "coverage_aug": float(covered_aug.mean()),
        "coverage_semantic": float(covered_semantic.mean()),
        "apss_semantic": float(sizes.mean()),
        "apss_aug": float((sizes + abstained.astype(float)).mean()),
        "abstain_rate": float(abstained.mean()),
        "abstain_precision": float(abstain_precision),
        "abstain_recall": float(abstain_recall),
        "n": n,
    }


def calibrate_abstain_threshold(
    v2_items: list[dict],
    calibrator: Any,
    clf: Any,
    alpha: float,
    encoder: SentenceEncoder | None = None,
    n_grid: int = 21,
) -> float:
    """Choose the smallest abstain threshold hitting augmented coverage on V2.

    Sweeps candidate thresholds drawn from quantiles of the classifier's
    predicted matchability probabilities on V2, in ascending order, and returns
    the smallest threshold at which augmented miscoverage on V2 is <= alpha
    (equivalently augmented coverage >= 1 - alpha). A smaller threshold abstains
    on fewer items, so the smallest satisfying threshold abstains as little as
    possible while meeting the coverage target.

    Args:
        v2_items: Calibration split for the abstain threshold.
        calibrator: A calibrated CAELUM-style calibrator.
        clf: Fitted matchability classifier.
        alpha: Target miscoverage level.
        encoder: Optional shared SentenceEncoder.
        n_grid: Number of quantile levels to sweep.

    Returns:
        The chosen abstain threshold. Falls back to 0.5 when V2 is empty or no
        candidate reaches the target coverage.
    """
    if len(v2_items) == 0:
        return 0.5

    # Precompute the per-item quantities once; sweeping thresholds then only
    # re-thresholds ``probs`` (no re-encoding / re-prediction per candidate).
    probs = _matchable_probs(v2_items, clf, encoder)
    cidx = np.array([int(it.get("correct_cluster_idx", -1)) for it in v2_items])
    unmatched = cidx < 0
    covered_semantic = np.array(
        [bool(calibrator.is_covered(it)) for it in v2_items], dtype=bool
    )

    # Candidate thresholds: quantiles of the predicted probabilities, ascending,
    # deduplicated. Extending the range so the "abstain on nobody" and "abstain
    # on everybody" extremes are both representable.
    qs = np.quantile(probs, np.linspace(0.0, 1.0, n_grid))
    candidates = np.unique(np.concatenate([[0.0], qs, [probs.max() + 1e-6]]))

    target = 1.0 - alpha
    best_cov = -1.0
    for thresh in candidates:  # ascending -> smallest satisfying wins
        abstained = probs < thresh
        coverage_aug = float((covered_semantic | (abstained & unmatched)).mean())
        best_cov = max(best_cov, coverage_aug)
        if coverage_aug >= target:
            return float(thresh)

    logger.warning(
        "calibrate_abstain_threshold: no candidate reached coverage %.3f on "
        "V2 (max achievable %.3f); falling back to 0.5.",
        target, best_cov,
    )
    return 0.5
