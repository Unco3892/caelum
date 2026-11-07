"""Predictive Kernel Entropy (Gruber & Buettner 2023) for LLM uncertainty.

Implements the SOTA total-uncertainty baseline from the Spectral Uncertainty
literature: a Von Neumann Entropy on an RBF kernel where the bandwidth gamma
is chosen via the median heuristic on a per-query basis.

The "predictive" aspect distinguishes this from the standard VNE used inside
CAELUM (which uses a fixed gamma = 1.0): each query gets its own data-driven
bandwidth, so the kernel adapts to the local geometry of the answer cloud.

Reference:
    Sebastian G. Gruber and Florian Buettner. Better Uncertainty Quantification
    for Deep Neural Networks via Kernel Density Estimation. arXiv:2306.12189,
    2023.

This baseline is used in Walha et al. (Spectral Uncertainty Decomposition,
AAAI 2026) under the name `predictive_kernel_entropy`. Their reference
implementation lives at::

    paper/literature/methods/fine_grained_conformal/code/
        spectral_uncertainty_decomposition-main/src/uncertainty_metrics/common.py

Their `compute_predictive_kernel_entropy_no_model` is a kernel-density
estimator (off-diagonal sum), and `compute_von_neuman_entropy_no_model` is the
eigenvalue entropy used here. Both are kernel-based total-uncertainty signals;
this file implements the eigenvalue variant with median-heuristic gamma, which
is the formulation explicitly named in Gruber & Buettner 2023.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg


_EPS = 1e-12


def _median_heuristic_gamma(distances_sq: np.ndarray) -> float:
    """Compute RBF bandwidth gamma via the median heuristic.

    Given pairwise *squared* distances between points, returns

        gamma = 1 / (2 * median(distances_sq))

    so that the RBF kernel exp(-gamma * d^2) has scale matched to the data.

    Only off-diagonal entries are used (the diagonal of squared distances is
    always zero and would bias the median toward zero).

    Args:
        distances_sq: Symmetric (n, n) array of squared pairwise distances.

    Returns:
        Scalar bandwidth gamma > 0. Returns 1.0 as a fallback when the
        median is zero (e.g., all answers identical, n == 1).
    """
    n = distances_sq.shape[0]
    if n < 2:
        return 1.0
    # Take strictly upper triangular (off-diagonal) entries
    iu, ju = np.triu_indices(n, k=1)
    off_diag = distances_sq[iu, ju]
    median = float(np.median(off_diag))
    if median <= _EPS:
        return 1.0
    return 1.0 / (2.0 * median)


def compute_pke(
    embeddings: np.ndarray,
    normalize_embeddings: bool = True,
) -> float:
    """Predictive Kernel Entropy with median-heuristic bandwidth.

    Given embeddings of generated answers ``{phi(y_i)}``:
      1. Optionally L2-normalize each embedding (default; matches the
         normalize=True convention used in CAELUM and the Spectral repo).
      2. Compute pairwise squared distances ``D_ij = ||phi(y_i) - phi(y_j)||^2``.
      3. Pick ``gamma = 1 / (2 * median(D_ij))`` over off-diagonal entries
         (Gruber & Buettner 2023 median heuristic).
      4. Build RBF kernel ``K_ij = exp(-gamma * D_ij)``.
      5. Trace-normalize so ``Tr(K) = 1`` (i.e., divide by n since K_ii = 1).
      6. Compute eigenvalues ``{lambda_k}`` and return the von-Neumann entropy
         ``-sum_k lambda_k * log(lambda_k)`` (clipped to non-negative
         eigenvalues, with eigenvalues smaller than ``EPS`` ignored).

    Reference:
        Sebastian G. Gruber and Florian Buettner. Better Uncertainty
        Quantification for Deep Neural Networks via Kernel Density
        Estimation. arXiv:2306.12189, 2023.

    Args:
        embeddings: (n, d) array of answer embeddings.
        normalize_embeddings: If True, L2-normalize each embedding to unit
            norm before computing distances. Default True (matches the
            sentence-transformers `normalize=True` convention used elsewhere
            in CAELUM and in the Spectral Uncertainty reference codebase).

    Returns:
        PKE in [0, log(n)]. Returns 0.0 for n < 2 (no diversity to measure).
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be 2D (n, d); got shape {embeddings.shape}"
        )
    n = embeddings.shape[0]
    if n < 2:
        return 0.0

    # 1. L2-normalize if requested.
    if normalize_embeddings:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms < _EPS, 1.0, norms)
        embeddings = embeddings / norms

    # 2. Pairwise squared distances. ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b.
    #    For unit-norm embeddings, this reduces to 2 * (1 - a.b), but we use
    #    the general form so the function is correct for un-normalized inputs.
    sq_norms = np.sum(embeddings ** 2, axis=1)
    distances_sq = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (embeddings @ embeddings.T)
    # Numerical floor: distances must be non-negative.
    distances_sq = np.maximum(distances_sq, 0.0)

    # 3. Median heuristic gamma.
    gamma = _median_heuristic_gamma(distances_sq)

    # 4. RBF kernel.
    K = np.exp(-gamma * distances_sq)

    # 5. Trace-normalize. Since K_ii = exp(0) = 1 and there are n diagonal
    #    entries, Tr(K) = n. Dividing by n is exactly trace-normalization.
    K = K / float(n)

    # 6. Eigenvalues + von-Neumann entropy.
    try:
        eigvals = scipy.linalg.eigvalsh(K)
    except np.linalg.LinAlgError:
        # Add a tiny diagonal jitter and retry.
        K = K + 1e-9 * np.eye(n)
        eigvals = scipy.linalg.eigvalsh(K)

    # Clip negative eigenvalues from numerical noise (kernel is PSD).
    eigvals = np.clip(eigvals, 0.0, None)
    pos = eigvals[eigvals > _EPS]
    if pos.size == 0:
        return 0.0
    entropy = float(-np.sum(pos * np.log(pos)))
    # Numerical floor (entropy is non-negative).
    return max(entropy, 0.0)


def compute_pke_from_answer_groups(
    answer_groups: list[list[str]],
    encoder,
    normalize_embeddings: bool = True,
) -> float:
    """Convenience wrapper: pool answers across clarifications and compute PKE.

    Mirrors the CAELUM "total" uncertainty path: flatten the n_clarifications x
    m_answers grid into a single pool of n*m strings, embed them, and compute
    PKE. This matches how U_total is computed inside CAELUM (pooled VNE) so
    that PKE is directly comparable to U_total.

    Args:
        answer_groups: List of length n_clarifications, each entry a list of
            sampled answer strings.
        encoder: An object with an ``.encode(texts: list[str]) -> np.ndarray``
            method (e.g., ``caelum.utils.SentenceEncoder``).
        normalize_embeddings: Forwarded to ``compute_pke``.

    Returns:
        Scalar PKE.
    """
    pooled = [s for group in answer_groups for s in group]
    if len(pooled) < 2:
        return 0.0
    embeddings = encoder.encode(pooled, normalize=normalize_embeddings)
    return compute_pke(embeddings, normalize_embeddings=False)
