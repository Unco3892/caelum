"""Conformal-prediction baselines for CAELUM comparison.

These calibrators implement competitor methods on the same cluster-prop
inputs that CAELUMCalibrator consumes (so they can be benchmarked head-to-head
on identical (val1, val2, test) splits).

Implemented:
    - ConUStyleCalibrator      : Wang et al., EMNLP 2024 (consistency-based)
    - SemanticEntropyCalibrator: Kuhn et al., 2023 (U_total as nonconformity)

API matches CAELUMCalibrator: .calibrate(val1, val2), .predict_set_size(item),
.is_covered(item), .diagnostic_ratio(item).
"""

import logging
import math
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ConU (Wang et al., EMNLP 2024)
# ---------------------------------------------------------------------------

class ConUStyleCalibrator:
    """Conformal predictor using self-consistency as nonconformity.

    Score:
        s(x) = 1 - max_k p_k(x)
    The most-frequent cluster's proportion measures self-consistency; one minus
    that proportion is "inconsistency" and serves as the nonconformity score.
    Scores DO NOT depend on which cluster contains the ground truth — ConU
    treats consistency itself as the confidence signal.

    Calibration: split-conformal quantile of scores on Val2 at level (1-alpha).
    Val1 is unused (no lambda).

    Prediction set: include clusters in descending proportion until cumulative
    proportion >= tau_hat (i.e., enough mass to "cover" the consistency budget).

    Notes vs CAELUM:
        * No epistemic/aleatoric decomposition.
        * No per-instance weight; tau_hat is a fixed scalar.
        * Smaller sets when the model is highly self-consistent — but this can
          MISS cases where the model is confidently wrong (low U_epi but
          incorrect_cluster_idx = -1).
    """

    def __init__(self, alpha: float = 0.1, seed: int = 42):
        self.alpha = alpha
        self.seed = seed
        self.tau_hat: float = 0.5
        self.optimal_lambda: float = 0.0  # API compat; ConU has no lambda

    def calibrate(self, val1: List[Dict], val2: List[Dict], verbose: bool = False):
        n2 = len(val2)
        if n2 == 0:
            if verbose:
                logger.warning("Val2 empty; using default tau_hat=0.5")
            return self
        target = 1.0 - self.alpha
        conf_level = min(math.ceil(target * (n2 + 1)) / n2, 1.0)
        scores = np.array([
            1.0 - float(it["cluster_props"][0]) if len(it["cluster_props"]) > 0 else 1.0
            for it in val2
        ])
        self.tau_hat = float(np.quantile(scores, conf_level, method="higher"))
        if verbose:
            logger.info("ConU calibration on Val2 (%d): tau_hat=%.4f", n2, self.tau_hat)
        return self

    def predict_set_size(self, item: Dict) -> int:
        props = np.asarray(item["cluster_props"])
        if len(props) == 0:
            return 1
        cumul = 0.0
        for k, p_k in enumerate(props):
            if cumul > self.tau_hat:
                return k
            cumul += float(p_k)
        return len(props)

    def predict_set(self, item: Dict) -> List[str]:
        return list(item.get("cluster_representatives", [])[: self.predict_set_size(item)])

    def is_covered(self, item: Dict) -> bool:
        idx = item.get("correct_cluster_idx", -1)
        if idx < 0:
            return False
        return idx < self.predict_set_size(item)

    def diagnostic_ratio(self, item: Dict) -> float:
        return 0.0  # ConU has no decomposition


# ---------------------------------------------------------------------------
# Semantic Entropy CP (Kuhn et al., 2023, with conformal calibration)
# ---------------------------------------------------------------------------

class SemanticEntropyCalibrator:
    """Conformal CP using U_total (no decomposition).

    Equivalent to a CAELUM where the weight is w(x) = 1 + U_total(x). Used as
    "Spectral / Semantic Entropy + conformal" baseline. Standard split-conformal:
    Val1 unused (no lambda); Val2 sets the threshold on normalized scores.
    """

    def __init__(
        self, alpha: float = 0.1, seed: int = 42,
        support_conditional: bool = False,
    ):
        self.alpha = alpha
        self.seed = seed
        self.support_conditional = support_conditional
        self.tau_hat: float = 0.5
        self.optimal_lambda: float = 0.0

    @staticmethod
    def _aps_score(item: Dict) -> float:
        cidx = item.get("correct_cluster_idx", -1)
        props = item["cluster_props"]
        if cidx < 0 or cidx >= len(props):
            return min(float(np.sum(props)), 1.0)
        return float(np.sum(props[:cidx]))

    @staticmethod
    def _weight(item: Dict) -> float:
        u_total = float(item.get("U_total", item["U_ale"] + item["U_epi"]))
        return max(1.0 + u_total, 1e-12)

    def calibrate(self, val1: List[Dict], val2: List[Dict], verbose: bool = False):
        # When support_conditional, the threshold is computed over matched
        # Val2 items only (the tau_hat_M of Prop. 1), falling back to all items
        # when there are too few matches.
        cal_items = val2
        if self.support_conditional:
            matched = [it for it in val2
                       if int(it.get("correct_cluster_idx", -1)) >= 0]
            if len(matched) >= 2:
                cal_items = matched
        n2 = len(cal_items)
        if n2 == 0:
            return self
        target = 1.0 - self.alpha
        conf_level = min(math.ceil(target * (n2 + 1)) / n2, 1.0)
        scores = np.array([self._aps_score(it) / self._weight(it)
                           for it in cal_items])
        self.tau_hat = float(np.quantile(scores, conf_level, method="higher"))
        if verbose:
            logger.info("SemanticEntropyCP on Val2 (%d, support_cond=%s): "
                        "tau_hat=%.4f", n2, self.support_conditional,
                        self.tau_hat)
        return self

    def predict_set_size(self, item: Dict) -> int:
        props = np.asarray(item["cluster_props"])
        if len(props) == 0:
            return 0
        weight = self._weight(item)
        effective_tau = self.tau_hat * weight
        cumul = 0.0
        for k, p_k in enumerate(props):
            if cumul > effective_tau:
                return k
            cumul += float(p_k)
        return len(props)

    def predict_set(self, item: Dict) -> List[str]:
        return list(item.get("cluster_representatives", [])[: self.predict_set_size(item)])

    def is_covered(self, item: Dict) -> bool:
        idx = item.get("correct_cluster_idx", -1)
        if idx < 0:
            return False
        return idx < self.predict_set_size(item)

    def diagnostic_ratio(self, item: Dict) -> float:
        return 0.0
