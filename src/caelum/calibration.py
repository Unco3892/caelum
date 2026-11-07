"""
CAELUM Calibrator: two-parameter conformal calibration for semantic prediction sets.

Adapted from clear/code/src/clear/clear.py  calibrate_split().

Key idea:
    For each question x with clusters sorted by proportion p_1 >= p_2 >= ... >= p_K,
    define APS-style nonconformity score (raw, following JUCAL):

        s(x, y*) = Sigma_{j < k*} p_j

    The raw score is **normalised** by the uncertainty weight before computing
    the conformal quantile, matching the CLEAR regression approach where the
    residual is divided by the predicted interval width:

        normalised_score(x) = s(x, y*) / w_lambda(x)

    where  w_lambda(x) = 1 + U_ale(x) + lambda * U_epi(x).

    The conformal quantile  tau_lambda  is then the (1-alpha)(1+1/n)-quantile
    of normalised scores.  Because the normalisation depends on lambda, different
    lambda values produce different score distributions and therefore different
    tau_lambda values.

    Prediction sets are constructed as before -- include cluster k iff:

        cumulative_before(k)  <=  tau_hat * w_lambda*(x)

    which is equivalent to  normalised_cumulative(k) <= tau_hat.

    Pipeline:
    1. Sweep lambda on Val1: normalise scores -> compute tau_lambda -> measure
       APSS -> pick lambda*.
    2. Conformalize on Val2: with lambda*, normalise scores -> set tau_hat.
       Coverage is support-limited: if correct_cluster_idx=-1, no subset of
       sampled clusters can contain the reference answer.

Score modes:
    - "hard" (default): standard APS score. correct_cluster_idx=-1 yields s=1.0.
    - "soft": embedding-based soft score for ALL samples:
        s_soft = 1 - max_k cosine_similarity(embed(gt), embed(rep_k))
    - "hybrid": use hard APS when correct_cluster_idx >= 0, soft score when -1.
      Eliminates the binary cliff at s=1.0 for unmatched samples.
    - "randomized": shorthand for score_mode="hard" + randomize=True.

Randomized conformal prediction (``randomize=True``):
    Applies the standard randomized APS technique from Vovk et al. (2005) and
    Romano et al. (2020, Appendix A).  Can be combined with any score_mode.

    For "hard" mode:
        s_rand = cumul_before + U * p_correct, where U ~ Uniform(0,1).
      Breaks the bimodal {0, 1} score distribution by injecting continuous
      noise into the correct-cluster contribution.

    For "soft" or "hybrid" modes:
        s_rand = s_base + U * margin, where margin is a small perturbation
      proportional to the cluster proportion of the best-matching cluster.

    Randomization produces approximately continuous score distributions,
    enabling the conformal quantile to be set more precisely on the generated
    support. Each sample receives a fixed U_i during calibration (seeded) so
    scores are reproducible.

    Prediction-set construction remains deterministic (uses hard threshold).
    Randomized calibration therefore gives conservative sets, not exact
    nominal semantic coverage. The paper's set-coverage argument applies to
    hard APS (and hybrid scores on truly matched items), not arbitrary soft
    embedding scores paired with this cumulative-mass prediction rule.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default lambda grid  (matches CLEAR: fine near 0, log-spaced to 100)
# ---------------------------------------------------------------------------

DEFAULT_LAMBDA_GRID = np.concatenate([
    np.linspace(0, 0.09, 10),
    np.logspace(-1, 2, 4001),
])


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal quantile of the calibration scores.

    Uses the conformal rank r = ceil((1 - alpha)(n + 1)) with NumPy's
    ``higher`` quantile rule, which returns the r-th order statistic or the
    next larger one. If the calibration set is empty or r exceeds its size,
    no finite threshold certifies the target, so the threshold is +infinity
    and every sampled cluster is kept.
    """
    n = len(scores)
    if n == 0:
        return float("inf")
    rank = math.ceil((1.0 - alpha) * (n + 1))
    if rank > n:
        return float("inf")
    return float(np.quantile(scores, min(rank / n, 1.0), method="higher"))


# ---------------------------------------------------------------------------
# Core score computation
# ---------------------------------------------------------------------------

def compute_aps_score(
    cluster_props: np.ndarray,
    correct_cluster_idx: int,
    u_ale: float,
    u_epi: float,
    lambda_val: float,
) -> float:
    """Compute APS-style nonconformity score for one sample.

    Following JUCAL's standard APS formulation (Romano et al., 2020):

        score = cumulative probability BEFORE the correct cluster

    The score is a RAW cumulative probability (not normalised by uncertainty).
    The calibrator divides this score by the uncertainty weight before taking
    its quantile, then multiplies the prediction threshold by the same weight.

    Args:
        cluster_props:  sorted descending cluster proportions [p_1, p_2, ...].
        correct_cluster_idx: index of the correct cluster in the sorted array.
            If -1 (no correct cluster found), returns a large score (worst).
        u_ale:  aleatoric uncertainty (unused in scoring; used in prediction).
        u_epi:  epistemic uncertainty (unused in scoring; used in prediction).
        lambda_val: λ weight (unused in scoring; used in prediction).

    Returns:
        Scalar nonconformity score in [0, 1].
    """
    if correct_cluster_idx < 0 or correct_cluster_idx >= len(cluster_props):
        # Ground truth not in any cluster -> worst score (all mass is "before")
        # Clamp to 1.0 to handle floating-point imprecision in props sum
        return min(float(np.sum(cluster_props)), 1.0)

    # Cumulative probability of clusters ranked BEFORE the correct one
    return float(np.sum(cluster_props[:correct_cluster_idx]))


def compute_randomized_aps_score(
    cluster_props: np.ndarray,
    correct_cluster_idx: int,
    u_random: float,
) -> float:
    """Compute randomized APS nonconformity score for one sample.

    Standard APS produces degenerate scores (0 when cidx=0, 1 when cidx=-1).
    Randomized APS injects a Uniform(0,1) random variable to break ties:

        s_rand(x, y*) = cumul_before(correct) + U * p_correct

    where U ~ Uniform(0, 1) is drawn once per sample and held fixed.

    This has the randomized APS score distribution of Romano et al. (2020).
    Since the prediction rule keeps every cluster whose preceding mass is
    below or equal to the threshold, the randomized score event is sufficient
    for inclusion on generated support. Coverage can be conservative: finite
    ranks, score ties, and deterministic sets do not give exact 1-alpha coverage.

    For cidx=0: score goes from 0.0 to U * p_0 in [0, p_0].
    For cidx=-1: score remains sum(props) ~ 1.0 (no cluster to randomize).
    For cidx>0: score = cumul_before + U * p_cidx (slightly perturbed).

    Args:
        cluster_props: Sorted descending cluster proportions [p_1, p_2, ...].
        correct_cluster_idx: Index of the correct cluster in sorted array.
            If -1, returns sum(cluster_props) (worst score, same as hard).
        u_random: Pre-drawn uniform random variable in [0, 1].

    Returns:
        Scalar nonconformity score in [0, 1].
    """
    if correct_cluster_idx < 0 or correct_cluster_idx >= len(cluster_props):
        return min(float(np.sum(cluster_props)), 1.0)

    cumul_before = float(np.sum(cluster_props[:correct_cluster_idx]))
    p_correct = float(cluster_props[correct_cluster_idx])
    return min(cumul_before + u_random * p_correct, 1.0)


def compute_soft_aps_score(
    gt_embedding: np.ndarray,
    cluster_embeddings: np.ndarray,
) -> float:
    """Compute embedding-based soft nonconformity score for one sample.

    Instead of a binary match (0 if correct cluster found, 1 if not), this
    computes a continuous score based on the best cosine similarity between
    the ground-truth answer embedding and any cluster representative embedding:

        s_soft(x, y*) = 1 - max_k cos_sim(embed(y*), embed(rep_k))

    This yields values in [0, 1] where:
      - Perfect match to some cluster representative -> score ~ 0
      - Completely unrelated -> score ~ 1
      - Partial matches -> intermediate scores (no binary cliff)

    Args:
        gt_embedding: Embedding vector of the ground-truth answer.
            Shape (dim,) or (1, dim).
        cluster_embeddings: Embedding vectors of cluster representatives.
            Shape (n_clusters, dim).

    Returns:
        Scalar soft nonconformity score in [0, 1].
    """
    gt_emb = np.asarray(gt_embedding).flatten()
    cl_embs = np.asarray(cluster_embeddings)

    if cl_embs.ndim == 1:
        cl_embs = cl_embs.reshape(1, -1)

    if cl_embs.shape[0] == 0:
        return 1.0

    # Cosine similarity (assumes normalized embeddings; handle unnormalized too)
    gt_norm = np.linalg.norm(gt_emb)
    if gt_norm < 1e-12:
        return 1.0
    gt_unit = gt_emb / gt_norm

    cl_norms = np.linalg.norm(cl_embs, axis=1, keepdims=True)
    cl_norms = np.maximum(cl_norms, 1e-12)
    cl_unit = cl_embs / cl_norms

    similarities = cl_unit @ gt_unit  # (n_clusters,)
    max_sim = float(np.max(similarities))

    # Clamp to [0, 1]: similarity can be negative, but score should stay in [0, 1]
    return float(np.clip(1.0 - max_sim, 0.0, 1.0))


def _get_score_for_item(
    item: Dict,
    score_mode: str,
    lambda_val: float = 0.0,
    u_random: float | None = None,
    randomize: bool = False,
) -> float:
    """Compute the nonconformity score for one item respecting score_mode.

    When ``randomize=True``, the base score is augmented with a uniform
    random perturbation following the randomized conformal prediction
    framework (Vovk et al., 2005; Romano et al., 2020).

    For hard APS scores:
        s_rand = cumul_before + U * p_correct

    For soft/hybrid scores when the correct cluster IS found (hybrid path):
        Uses the same hard randomized formula as above.

    For soft scores or hybrid when the correct cluster is NOT found:
        The soft embedding score already provides continuous values, but
        randomization still helps break ties.  We add U * p_best where
        p_best is the proportion of the cluster with highest cosine
        similarity to the ground truth.

    Args:
        item: Data dict with cluster_props, correct_cluster_idx, U_ale, U_epi,
            and optionally gt_embedding, cluster_embeddings.
        score_mode: One of "hard", "soft", "hybrid", "randomized".
        lambda_val: Lambda weight (unused in scoring, present for API compat).
        u_random: Pre-drawn uniform random variable in [0, 1]. Required when
            randomize=True or score_mode="randomized"; ignored otherwise.
        randomize: Whether to apply randomized scoring. Independent of
            score_mode.

    Returns:
        Scalar nonconformity score.
    """
    props = np.asarray(item["cluster_props"])
    cidx = item["correct_cluster_idx"]
    u_ale = item["U_ale"]
    u_epi = item["U_epi"]

    # score_mode="randomized" is shorthand for hard + randomize
    if score_mode == "randomized":
        if u_random is None:
            raise ValueError("score_mode='randomized' requires u_random to be "
                             "provided (a float in [0, 1]).")
        return compute_randomized_aps_score(props, cidx, u_random)

    if score_mode == "hard":
        if randomize:
            if u_random is None:
                raise ValueError("randomize=True requires u_random to be "
                                 "provided (a float in [0, 1]).")
            return compute_randomized_aps_score(props, cidx, u_random)
        return compute_aps_score(props, cidx, u_ale, u_epi, lambda_val)

    elif score_mode == "soft":
        gt_emb = item.get("gt_embedding")
        cl_embs = item.get("cluster_embeddings")
        if gt_emb is None or cl_embs is None:
            logger.warning("soft score requested but embeddings missing; "
                           "falling back to hard score")
            if randomize and u_random is not None:
                return compute_randomized_aps_score(props, cidx, u_random)
            return compute_aps_score(props, cidx, u_ale, u_epi, lambda_val)
        base_score = compute_soft_aps_score(gt_emb, cl_embs)
        if randomize and u_random is not None:
            # Add small randomized perturbation to break ties in soft scores
            p_best = float(props[0]) if len(props) > 0 else 0.0
            base_score = base_score + u_random * p_best * 0.01
            base_score = min(base_score, 1.0)
        return base_score

    elif score_mode == "hybrid":
        # Use hard APS when we found a correct cluster; soft when not
        if cidx >= 0 and cidx < len(props):
            if randomize and u_random is not None:
                return compute_randomized_aps_score(props, cidx, u_random)
            return compute_aps_score(props, cidx, u_ale, u_epi, lambda_val)
        else:
            gt_emb = item.get("gt_embedding")
            cl_embs = item.get("cluster_embeddings")
            if gt_emb is None or cl_embs is None:
                logger.warning("hybrid score requested but embeddings missing; "
                               "using hard fallback (s=1.0)")
                return compute_aps_score(props, cidx, u_ale, u_epi, lambda_val)
            base_score = compute_soft_aps_score(gt_emb, cl_embs)
            if randomize and u_random is not None:
                p_best = float(props[0]) if len(props) > 0 else 0.0
                base_score = base_score + u_random * p_best * 0.01
                base_score = min(base_score, 1.0)
            return base_score

    else:
        raise ValueError(f"Unknown score_mode: {score_mode!r}. "
                         f"Expected 'hard', 'soft', 'hybrid', or 'randomized'.")


def _weight_core(
    u_ale,
    u_epi,
    lambda_val: float,
    weight_fn: str = "additive",
    mu_ale: float = 1.0,
    mu_epi: float = 1.0,
):
    """Elementwise uncertainty weight w_lambda(x), shared by all callers.

    This is the single source of truth for the weight function used both
    during calibration (vectorised over Val1/Val2) and during prediction-set
    construction (scalar). It accepts scalars or numpy arrays interchangeably
    via ``np.asarray``.

    Supported weight functions:
        - "unit":           w = 1 (textbook unweighted APS).
        - "additive":       w = 1 + u_ale + lambda * u_epi.
        - "multiplicative": w = (1 + u_ale) * (1 + lambda * u_epi).
        - "inverse":        w = 1 / (1 + u_ale + lambda * u_epi).
        - "standardized":   w = 1 + u_ale/mu_ale + lambda * u_epi/mu_epi,
              where mu_ale, mu_epi are the Val1 means of U_ale, U_epi. This
              puts the aleatoric and epistemic terms on a comparable scale so
              that lambda is interpretable as a relative weight rather than an
              absolute one.

    Args:
        u_ale: Aleatoric uncertainty (scalar or array).
        u_epi: Epistemic uncertainty (scalar or array).
        lambda_val: Epistemic weight lambda.
        weight_fn: One of the supported names above.
        mu_ale: Aleatoric normaliser for "standardized" (guarded to >= 1e-8,
            else 1.0). Ignored by other weight functions.
        mu_epi: Epistemic normaliser for "standardized" (same guard).

    Returns:
        The weight as a 0-d array (scalar input) or array (array input).
    """
    ale = np.asarray(u_ale, dtype=float)
    epi = np.asarray(u_epi, dtype=float)
    ma = mu_ale if (mu_ale is not None and mu_ale >= 1e-8) else 1.0
    me = mu_epi if (mu_epi is not None and mu_epi >= 1e-8) else 1.0

    if weight_fn == "unit":
        return np.ones_like(ale)
    if weight_fn == "additive":
        return np.maximum(1.0 + ale + lambda_val * epi, 1e-12)
    if weight_fn == "multiplicative":
        return np.maximum((1.0 + ale) * (1.0 + lambda_val * epi), 1e-12)
    if weight_fn == "inverse":
        return 1.0 / np.maximum(1.0 + ale + lambda_val * epi, 1e-12)
    if weight_fn == "standardized":
        return np.maximum(1.0 + ale / ma + lambda_val * epi / me, 1e-12)
    raise ValueError(f"Unknown weight_fn: {weight_fn!r}")


def _standalone_weight(
    u_ale: float,
    u_epi: float,
    lambda_val: float,
    weight_fn: str = "additive",
    mu_ale: float = 1.0,
    mu_epi: float = 1.0,
) -> float:
    """Scalar weight for standalone functions (thin wrapper over _weight_core).

    The 'unit' weight returns 1 regardless of inputs; this corresponds to
    standard unweighted APS, the textbook baseline (no per-instance threshold
    scaling). ``mu_ale`` / ``mu_epi`` are only consulted by the "standardized"
    weight and default to 1.0 (a no-op) for every other weight function.
    """
    return float(_weight_core(u_ale, u_epi, lambda_val, weight_fn, mu_ale, mu_epi))


def compute_prediction_set_size(
    cluster_props: np.ndarray,
    u_ale: float,
    u_epi: float,
    lambda_val: float,
    tau: float,
    weight_fn: str = "additive",
    mu_ale: float = 1.0,
    mu_epi: float = 1.0,
) -> int:
    """Count clusters in the prediction set for a given threshold τ.

    Following JUCAL's APS construction: include clusters in decreasing p_k
    order until the cumulative preceding mass reaches the (weight-adjusted)
    threshold.  The per-instance weight scales τ so that uncertain items
    get larger (or smaller, for inverse) prediction sets.

    Concretely, include cluster k iff cumulative_before(k) <= τ · w(x).
    Equality includes the next cluster; even at threshold zero the first
    cluster is included. An infinite threshold returns every sampled cluster.

    ``mu_ale`` / ``mu_epi`` are only consulted by the "standardized" weight and
    default to 1.0 (a no-op) for every other weight function.
    """
    weight = _standalone_weight(u_ale, u_epi, lambda_val, weight_fn, mu_ale, mu_epi)
    effective_tau = tau * weight
    cumulative_before = 0.0
    for k, p_k in enumerate(cluster_props):
        # Strict > so that cluster 0 is always included when cumul=0 and tau=0
        # (degenerate-but-valid case where all calibration scores are 0).
        # Convention: include cluster k iff cumulative_before(k) <= effective_tau.
        if cumulative_before > effective_tau:
            return k  # clusters 0..k-1 are included
        cumulative_before += p_k
    return len(cluster_props)  # all clusters included


def _sorted_correct_idx(cluster_props: List[float], correct_cluster_idx: int) -> int:
    """Map the correct cluster index to its rank in descending-sorted proportions.

    cluster_props is already sorted descending. We need the position of the
    correct cluster's proportion in that sorted array.

    Because multiple clusters can share the same proportion, we just trust
    the index as-is (it should already point into the sorted array after
    compute_uncertainty_for_dataset).
    """
    return correct_cluster_idx


# ---------------------------------------------------------------------------
# Calibrator class
# ---------------------------------------------------------------------------

class CAELUMCalibrator:
    """Two-parameter conformal calibrator for semantic prediction sets.

    Args:
        alpha: Miscoverage level. The nominal target is 1 - alpha on generated
            support; items with correct_cluster_idx=-1 are support failures.
        lambda_grid: Array of lambda values to search over. Uses the default
            CLEAR-style grid (fine near 0, log-spaced to 100) if not provided.
        seed: Random seed for reproducibility of randomized scoring.
        score_mode: Base scoring function -- "hard", "soft", "hybrid", or
            "randomized" (shorthand for hard + randomize=True).
        weight_fn: Weight function for uncertainty-adaptive sets -- "additive",
            "multiplicative", "inverse", "unit", or "standardized". The
            "standardized" weight rescales U_ale / U_epi by their Val1 means
            (fit in :meth:`calibrate`) so lambda is a relative weight.
        randomize: If True, apply randomized conformal prediction (Vovk et al.,
            2005; Romano et al., 2020) to break score ties. Injects
            U ~ Uniform(0,1) noise into APS scores during calibration, which
            produces approximately continuous score distributions on the
            generated support. Can be combined with any score_mode. Default is
            False.
        selection_method: How lambda* is chosen on Val1 -- "argmin_apss"
            (tau and APSS both computed in-sample on the full Val1),
            "held_out_apss" (split Val1 into train for tau and eval for APSS),
            or "set_score" (same split as held_out_apss, but the eval objective
            is the interval-score analogue |C(x)| + (2/alpha) * 1{miss} which
            penalises misses instead of blindly minimising size).
        support_conditional: If True, calibrate the conformal threshold using
            only *matched* items (correct_cluster_idx >= 0). On Val2 this yields
            the support-conditional threshold tau_hat_M of Proposition 1; on
            Val1 the lambda quantile and objective likewise use matched items
            only. Prediction-set construction and ``is_covered`` are unchanged
            (an unmatched item is still never covered). If the calibration
            subset is empty or too small for the conformal rank, the
            threshold is +infinity and every sampled cluster is kept.

    Usage::

        cal = CAELUMCalibrator(alpha=0.1, randomize=True)
        cal.calibrate(val1_data, val2_data)
        sets = cal.predict_set(test_item)
    """

    VALID_WEIGHT_FNS = (
        "additive", "multiplicative", "inverse", "unit", "standardized",
    )
    VALID_SELECTION_METHODS = ("argmin_apss", "held_out_apss", "set_score")

    def __init__(
        self,
        alpha: float = 0.1,
        lambda_grid: Optional[np.ndarray] = None,
        seed: int = 42,
        score_mode: str = "hard",
        weight_fn: str = "additive",
        randomize: bool = False,
        selection_method: str = "argmin_apss",
        val1_eval_frac: float = 0.4,
        support_conditional: bool = False,
    ):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1")
        if score_mode not in ("hard", "soft", "hybrid", "randomized"):
            raise ValueError(f"score_mode must be 'hard', 'soft', 'hybrid', "
                             f"or 'randomized', got {score_mode!r}")
        if weight_fn not in self.VALID_WEIGHT_FNS:
            raise ValueError(f"weight_fn must be one of {self.VALID_WEIGHT_FNS}, "
                             f"got {weight_fn!r}")
        if selection_method not in self.VALID_SELECTION_METHODS:
            raise ValueError(
                f"selection_method must be one of {self.VALID_SELECTION_METHODS}; "
                f"got {selection_method!r}"
            )

        # score_mode="randomized" is shorthand for hard + randomize=True.
        if score_mode == "randomized":
            score_mode = "hard"
            randomize = True

        self.alpha = alpha
        self.lambda_grid = lambda_grid if lambda_grid is not None else DEFAULT_LAMBDA_GRID
        self.seed = seed
        self.score_mode = score_mode
        self.weight_fn = weight_fn
        self.randomize = randomize
        self.selection_method = selection_method
        self.val1_eval_frac = val1_eval_frac
        self.support_conditional = support_conditional

        # Calibrated parameters (set after .calibrate())
        self.optimal_lambda: float = 0.0
        self.tau_hat: float = 0.5
        self.lambda_losses: Optional[np.ndarray] = None  # APSS per lambda (for plotting)
        # Val1 means for the "standardized" weight (1.0 = no-op until fit).
        self.mu_ale: float = 1.0
        self.mu_epi: float = 1.0

    # ---- weight computation --------------------------------------------------

    def _compute_weights_vec(
        self, ale: np.ndarray, epi: np.ndarray, lambda_val: float,
    ) -> np.ndarray:
        """Vectorised weight computation for arrays of U_ale, U_epi.

        Delegates to the shared :func:`_weight_core` so calibration and
        prediction use exactly the same formula. 'unit' returns ones-like,
        the textbook unweighted split-conformal APS baseline; 'standardized'
        uses the stored Val1 means ``self.mu_ale`` / ``self.mu_epi``.
        """
        return np.asarray(
            _weight_core(
                ale, epi, lambda_val, self.weight_fn,
                self.mu_ale, self.mu_epi,
            ),
            dtype=float,
        )

    # ---- calibration --------------------------------------------------------

    def calibrate(
        self,
        val1: List[Dict],
        val2: List[Dict],
        verbose: bool = True,
    ) -> "CAELUMCalibrator":
        """Run split-validation calibration with score normalisation.

        Scores are **normalised** by the uncertainty weight before computing
        the conformal quantile, following the CLEAR regression paper:

            normalised_score(x) = raw_score(x) / w_lambda(x)

        where  w_lambda(x) = 1 + U_ale(x) + lambda * U_epi(x).

        This ensures different lambda values produce genuinely different score
        distributions and conformal quantiles, enabling the grid search to
        find a meaningful lambda* > 0 when epistemic weighting helps.

        When ``self.randomize`` is True, scores are computed using randomized
        APS (Vovk et al., 2005) which injects U ~ Uniform(0,1) noise to break
        ties. This produces approximately continuous score distributions and
        gives a more precise conformal quantile on the generated support.

        Step 1 (Val1): for each lambda, normalise scores by w_lambda,
            compute tau_lambda as conformal quantile, evaluate the selection
            objective (mean set size, or the set-score analogue when
            ``selection_method='set_score'``), pick lambda* minimising it.
        Step 2 (Val2): with lambda*, normalise scores by w_lambda* and
            compute conformal quantile tau_hat.

        When ``weight_fn='standardized'``, the Val1 means of U_ale / U_epi are
        fit here and stored on ``self.mu_ale`` / ``self.mu_epi``; prediction
        reuses those stored means. When ``support_conditional=True``, both the
        Val1 lambda quantile and the Val2 threshold are computed over matched
        items only (correct_cluster_idx >= 0). With hard APS and the default
        safe small-sample convention this gives tau_hat_M of Prop. 1. Soft
        scores paired with the hard set rule do not inherit that guarantee.
        Val1 must be nonempty. If no matched Val1 objective items remain, all
        weights tie and the smallest grid weight is selected.

        Each element of *val1* / *val2* must have:
            cluster_props, correct_cluster_idx, U_ale, U_epi

        For score_mode "soft" or "hybrid", items must also have:
            gt_embedding (np.ndarray), cluster_embeddings (np.ndarray)
        """
        n1 = len(val1)
        if n1 == 0:
            raise ValueError("Val1 must be nonempty for lambda selection")
        if len(self.lambda_grid) == 0:
            raise ValueError("lambda_grid must contain at least one weight")

        # ---- Fit standardized-weight normalisers on Val1 ----
        # These are the means of U_ale / U_epi over the full Val1 split. They
        # are a no-op for every weight_fn other than "standardized" (guarded
        # to 1.0 when a mean is degenerate).
        if n1 > 0:
            self.mu_ale = float(np.mean([d["U_ale"] for d in val1]))
            self.mu_epi = float(np.mean([d["U_epi"] for d in val1]))
        if self.mu_ale < 1e-8:
            self.mu_ale = 1.0
        if self.mu_epi < 1e-8:
            self.mu_epi = 1.0

        # ---- Step 1: Lambda selection on Val1 ----
        # Optionally split Val1 into Val1_train (for tau_lambda) and Val1_eval
        # (for the selection objective) to address the selection-bias risk of
        # argmin over a 4011-point grid on a small Val1. "set_score" reuses the
        # same split machinery as "held_out_apss".
        use_split = (
            self.selection_method in ("held_out_apss", "set_score") and n1 >= 20
        )
        if use_split:
            split_rng = np.random.RandomState(self.seed + 7)
            perm = split_rng.permutation(n1)
            n_eval = max(8, int(round(n1 * self.val1_eval_frac)))
            n_train = n1 - n_eval
            val1_train = [val1[i] for i in perm[:n_train]]
            val1_eval = [val1[i] for i in perm[n_train:]]
        else:
            # argmin_apss: tau and objective both on the full Val1 (in-sample)
            val1_train = val1
            val1_eval = val1

        n1_train = len(val1_train)

        props1_train = [np.asarray(d["cluster_props"]) for d in val1_train]
        ale1_train = np.array([d["U_ale"] for d in val1_train])
        epi1_train = np.array([d["U_epi"] for d in val1_train])
        cidx1_train = np.array(
            [int(d["correct_cluster_idx"]) for d in val1_train]
        )
        matched_train = cidx1_train >= 0

        props1_eval = [np.asarray(d["cluster_props"]) for d in val1_eval]
        ale1_eval = np.array([d["U_ale"] for d in val1_eval])
        epi1_eval = np.array([d["U_epi"] for d in val1_eval])
        cidx1_eval = np.array([int(d["correct_cluster_idx"]) for d in val1_eval])
        matched_eval = cidx1_eval >= 0

        # Support-conditional (Prop. 1): the lambda quantile is computed over
        # matched Val1-train items only, and the selection objective over
        # matched Val1-eval items only.
        if self.support_conditional:
            quant_mask_train = matched_train
            obj_mask_eval = matched_eval
        else:
            quant_mask_train = np.ones(n1_train, dtype=bool)
            obj_mask_eval = np.ones(len(val1_eval), dtype=bool)
        n_q_train = int(quant_mask_train.sum())

        # Pre-draw random variables for randomized APS (fixed per sample,
        # shared across all lambda values in grid search).
        u_rand_val1_train: list[float | None] = [None] * n1_train
        if self.randomize:
            rng = np.random.RandomState(self.seed)
            u_rand_val1_train = [float(rng.uniform()) for _ in range(n1_train)]

        lambda_objectives = []
        for lam in self.lambda_grid:
            raw_scores_train = np.array([
                _get_score_for_item(
                    d, self.score_mode, lam,
                    u_random=u_rand_val1_train[i],
                    randomize=self.randomize,
                )
                for i, d in enumerate(val1_train)
            ])
            # Normalise scores by weight (matches CLEAR regression approach)
            weights_train = self._compute_weights_vec(
                ale1_train, epi1_train, lam,
            )
            normalised_scores_train = raw_scores_train / weights_train
            tau_lam = _conformal_quantile(
                normalised_scores_train[quant_mask_train], self.alpha,
            )

            # Evaluate the selection objective on Val1_eval (held out from the
            # tau choice when a split is used; equals Val1_train otherwise).
            sizes_eval = np.array([
                compute_prediction_set_size(
                    p, a, e, lam, tau_lam, self.weight_fn,
                    self.mu_ale, self.mu_epi,
                )
                for p, a, e in zip(props1_eval, ale1_eval, epi1_eval)
            ], dtype=float)
            if not obj_mask_eval.any():
                # No matched validation evidence distinguishes the weights.
                obj = 0.0
            elif self.selection_method == "set_score":
                # Interval-score analogue: penalise misses with 2/alpha per
                # uncovered item instead of blindly minimising set size. A
                # miss is a matched item whose correct cluster is excluded.
                covered_eval = (cidx1_eval >= 0) & (cidx1_eval < sizes_eval)
                per_item = sizes_eval + (2.0 / self.alpha) * (~covered_eval)
                obj = float(np.mean(per_item[obj_mask_eval]))
            else:
                obj = float(np.mean(sizes_eval[obj_mask_eval]))
            lambda_objectives.append(obj)

        lambda_objectives = np.array(lambda_objectives)
        best_indices = np.flatnonzero(lambda_objectives == lambda_objectives.min())
        best_idx = int(best_indices[np.argmin(np.asarray(self.lambda_grid)[best_indices])])
        self.optimal_lambda = float(self.lambda_grid[best_idx])
        self.lambda_losses = lambda_objectives

        if verbose:
            logger.info(
                "Lambda grid search on Val1 (%d samples, score_mode=%s, "
                "weight_fn=%s, randomize=%s, selection=%s, support_cond=%s):",
                n1, self.score_mode, self.weight_fn, self.randomize,
                self.selection_method, self.support_conditional,
            )
            logger.info(
                "  Val1 matched: quantile subset=%d/%d, objective subset=%d/%d",
                n_q_train, n1_train, int(obj_mask_eval.sum()), len(val1_eval),
            )
            logger.info("  optimal lambda* = %.4f  (objective = %.3f)",
                        self.optimal_lambda, lambda_objectives[best_idx])

        # ---- Step 2: Conformal threshold on Val2 ----
        n2 = len(val2)

        cidx2 = np.array([int(d["correct_cluster_idx"]) for d in val2])
        ale2 = np.array([d["U_ale"] for d in val2])
        epi2 = np.array([d["U_epi"] for d in val2])
        props2 = [np.asarray(d["cluster_props"]) for d in val2]
        matched2 = cidx2 >= 0

        # Pre-draw random variables for Val2 (independent from Val1 draws).
        u_rand_val2: list[float | None] = [None] * n2
        if self.randomize:
            rng2 = np.random.RandomState(self.seed + 1)
            u_rand_val2 = [float(rng2.uniform()) for _ in range(n2)]

        raw_scores2 = np.array([
            _get_score_for_item(
                d, self.score_mode, self.optimal_lambda,
                u_random=u_rand_val2[i],
                randomize=self.randomize,
            )
            for i, d in enumerate(val2)
        ])
        # Normalise scores by weight at optimal lambda
        weights2 = self._compute_weights_vec(ale2, epi2, self.optimal_lambda)
        normalised_scores2 = raw_scores2 / weights2

        # Support-conditional threshold tau_hat_M: quantile over matched Val2
        # items only (Prop. 1), even when none or only one is matched.
        quant_mask2 = matched2 if self.support_conditional else np.ones(n2, dtype=bool)
        n_q2 = int(quant_mask2.sum())
        self.tau_hat = _conformal_quantile(normalised_scores2[quant_mask2], self.alpha)

        # Bookkeeping for diagnostics / logging.
        self.n_val1_matched = int(sum(
            int(d["correct_cluster_idx"]) >= 0 for d in val1
        ))
        self.n_val2_matched = int(matched2.sum())

        # Coverage check on Val2 (full population + matched stratum)
        sizes2 = np.array([
            compute_prediction_set_size(
                p, a, e, self.optimal_lambda, self.tau_hat,
                self.weight_fn, self.mu_ale, self.mu_epi,
            )
            for p, a, e in zip(props2, ale2, epi2)
        ], dtype=float)
        covered2 = (cidx2 >= 0) & (cidx2 < sizes2)
        coverage2 = float(np.mean(covered2)) if n2 else float("nan")
        coverage2_matched = (
            float(np.mean(covered2[matched2])) if int(matched2.sum()) > 0
            else float("nan")
        )

        if verbose:
            logger.info("Conformal threshold on Val2 (%d samples, %d matched):",
                        n2, n_q2)
            logger.info("  tau_hat  = %.4f", self.tau_hat)
            logger.info("  Val2 APSS = %.3f", float(np.mean(sizes2)) if n2 else float("nan"))
            logger.info("  Val2 empirical coverage = %.3f (target %.2f)",
                        coverage2, 1.0 - self.alpha)
            logger.info("  Val2 matched-stratum coverage = %.3f",
                        coverage2_matched)

        return self

    # ---- prediction ---------------------------------------------------------

    def predict_set_size(self, item: Dict) -> int:
        """Return the prediction set size for one item."""
        props = np.asarray(item["cluster_props"])
        return compute_prediction_set_size(
            props, item["U_ale"], item["U_epi"],
            self.optimal_lambda, self.tau_hat, self.weight_fn,
            self.mu_ale, self.mu_epi,
        )

    def predict_set(self, item: Dict) -> List[str]:
        """Return the list of cluster representative answers in the prediction set."""
        set_size = self.predict_set_size(item)
        reps = item.get("cluster_representatives", [])
        return list(reps[:set_size])

    def is_covered(self, item: Dict) -> bool:
        """Check if the ground-truth cluster is within the prediction set."""
        idx = item.get("correct_cluster_idx", -1)
        if idx < 0:
            return False
        set_size = self.predict_set_size(item)
        return idx < set_size

    def diagnostic_ratio(self, item: Dict) -> float:
        """Instance-level diagnostic R(x) = log(λ*·U_epi + ε) - log(U_ale + ε).

        The sign compares weighted within-clarification variability with the
        residual component. It does not identify hallucination or ambiguity;
        an unstable lambda also makes this descriptive ratio unstable. The
        epsilon stabilizes the logarithms when a component is zero.
        """
        u_epi = item["U_epi"]
        u_ale = item["U_ale"]
        eps = 1e-6
        return float(np.log(self.optimal_lambda * u_epi + eps) - np.log(u_ale + eps))

    # ---- baselines ----------------------------------------------------------

    @classmethod
    def vanilla_conformal(
        cls,
        alpha: float = 0.1,
        score_mode: str = "hard",
        weight_fn: str = "additive",
        randomize: bool = False,
    ) -> "CAELUMCalibrator":
        """Baseline: lambda = 0 (only aleatoric matters).

        Note: with the default ``additive`` weight, this gives
        ``w(x) = 1 + U_ale(x)``, i.e., aleatoric-weighted APS. For
        the textbook unweighted APS baseline (``w(x) = 1``), use
        :meth:`true_aps_conformal` instead.
        """
        return cls(alpha=alpha, lambda_grid=np.array([0.0]),
                   score_mode=score_mode, weight_fn=weight_fn,
                   randomize=randomize)

    @classmethod
    def true_aps_conformal(
        cls,
        alpha: float = 0.1,
        score_mode: str = "hard",
        randomize: bool = False,
    ) -> "CAELUMCalibrator":
        """Textbook unweighted split-conformal APS baseline (``w(x) = 1``).

        No per-instance threshold scaling: pure APS quantile on Val2.
        This is the baseline most readers picture when they hear "Vanilla CP".
        """
        return cls(
            alpha=alpha, lambda_grid=np.array([0.0]),
            score_mode=score_mode, weight_fn="unit", randomize=randomize,
        )

    @classmethod
    def epistemic_only(
        cls,
        alpha: float = 0.1,
        score_mode: str = "hard",
        weight_fn: str = "additive",
        randomize: bool = False,
    ) -> "CAELUMCalibrator":
        """Baseline: lambda -> inf (only epistemic matters)."""
        return cls(alpha=alpha, lambda_grid=np.array([1e6]),
                   score_mode=score_mode, weight_fn=weight_fn,
                   randomize=randomize)

    @classmethod
    def equal_weight(
        cls,
        alpha: float = 0.1,
        score_mode: str = "hard",
        weight_fn: str = "additive",
        randomize: bool = False,
    ) -> "CAELUMCalibrator":
        """Baseline: lambda = 1.0 (equal weight on aleatoric and epistemic)."""
        return cls(alpha=alpha, lambda_grid=np.array([1.0]),
                   score_mode=score_mode, weight_fn=weight_fn,
                   randomize=randomize)

    # ---- diagnostics --------------------------------------------------------

    def stratified_coverage(
        self, data: List[Dict], n_bins: int = 4, by: str = "U_total",
    ) -> Dict[str, np.ndarray]:
        """Compute coverage in equal-count bins of a chosen uncertainty signal.

        Defends against the CLEAR/JUCAL critique that minimizing average set
        size can sacrifice conditional validity: if coverage is roughly flat
        across uncertainty bins, the marginal coverage is honest; if it drops
        sharply in high-uncertainty bins, lambda* is over-shrinking sets there.

        Args:
            data: list of items with cluster_props, correct_cluster_idx,
                U_ale, U_epi, U_total (or computable as U_ale + U_epi).
            n_bins: number of equal-count quantile bins. Default 4 (quartiles).
            by: which uncertainty signal to bin by; "U_total", "U_ale",
                or "U_epi".

        Returns:
            Dict with arrays of length n_bins:
                - "bin_edges": bin upper edges (length n_bins + 1)
                - "coverage": coverage rate in each bin
                - "apss": mean prediction set size in each bin
                - "n": count per bin
        """
        if by == "U_total":
            signal = np.array([
                d.get("U_total", d["U_ale"] + d["U_epi"]) for d in data
            ])
        elif by == "U_ale":
            signal = np.array([d["U_ale"] for d in data])
        elif by == "U_epi":
            signal = np.array([d["U_epi"] for d in data])
        else:
            raise ValueError(f"by must be 'U_total', 'U_ale', or 'U_epi', got {by!r}")

        edges = np.quantile(signal, np.linspace(0, 1, n_bins + 1))
        # Make rightmost edge slightly larger so np.digitize includes the max
        edges[-1] = edges[-1] + 1e-12

        bin_idx = np.clip(np.digitize(signal, edges[1:-1]), 0, n_bins - 1)
        coverage = np.zeros(n_bins)
        apss = np.zeros(n_bins)
        n = np.zeros(n_bins, dtype=int)
        for b in range(n_bins):
            mask = bin_idx == b
            if not mask.any():
                continue
            items_b = [data[i] for i in np.where(mask)[0]]
            covered = [self.is_covered(it) for it in items_b]
            sizes = [self.predict_set_size(it) for it in items_b]
            coverage[b] = float(np.mean(covered))
            apss[b] = float(np.mean(sizes))
            n[b] = int(mask.sum())
        return {
            "bin_edges": edges,
            "coverage": coverage,
            "apss": apss,
            "n": n,
            "by": by,
        }


# Backward-compatibility alias (temporary)
CLEARLLMCalibrator = CAELUMCalibrator
