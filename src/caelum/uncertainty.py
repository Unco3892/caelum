"""
Semantic clustering and ICE-style uncertainty decomposition for CAELUM.

Implements:
  - Embedding-based semantic clustering (cosine similarity threshold)
  - NLI-based semantic clustering (bidirectional entailment via DeBERTa)
  - ICE decomposition (two methods):
    * Cluster Entropy (CE): Shannon entropy of cluster assignment proportions
    * Von Neumann Entropy (VNE): Shannon entropy of RBF kernel eigenvalues
  - Both use the same ICE decomposition structure:
    * Epistemic  = (1/n) Σ H_i   (avg within-clarification entropy)
    * Total      = H(pooled)       (entropy over all answers)
    * Aleatoric  = Total - Epistemic
    With equal answer counts, CE uses shared pooled clusters and VNE uses a
    shared encoder/kernel feature space, so Total >= Epistemic follows from
    entropy concavity for both. Computing Gram blocks separately does not
    change that identity. Unequal counts require size-weighted averaging.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.linalg
import scipy.stats
import sklearn.metrics.pairwise
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .utils import SentenceEncoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding-based clustering
# ---------------------------------------------------------------------------

def cluster_by_embedding(
    texts: List[str],
    encoder: SentenceEncoder,
    threshold: float = 0.7,
) -> List[int]:
    """Cluster texts by embedding cosine similarity (greedy, O(n^2)).

    Texts are scanned in input order. An unassigned text opens a new cluster,
    and later unassigned texts join that cluster if their cosine similarity to
    the cluster seed is at least ``threshold``. This is a seed-order greedy
    approximation, not an order-invariant equivalence relation.

    Returns:
        List of integer cluster IDs (same length as *texts*).
    """
    if not texts:
        return []

    embeddings = encoder.encode(texts, normalize=True)
    n = len(texts)
    cluster_ids = [-1] * n
    next_id = 0

    for i in range(n):
        if cluster_ids[i] == -1:
            cluster_ids[i] = next_id
            for j in range(i + 1, n):
                if cluster_ids[j] == -1:
                    sim = float(np.dot(embeddings[i], embeddings[j]))
                    if sim >= threshold:
                        cluster_ids[j] = next_id
            next_id += 1

    return cluster_ids


# ---------------------------------------------------------------------------
# NLI-based clustering (optional, heavier)
# ---------------------------------------------------------------------------

def _nli_label_ids(config) -> Dict[str, int]:
    """Resolve semantic labels from a model config instead of assuming IDs.

    NLI checkpoints use different output orders. In particular, the default
    DeBERTa checkpoint uses contradiction=0, entailment=1, neutral=2.
    Opaque LABEL_0-style mappings cannot safely determine the entailment ID.
    """
    required = {"contradiction", "neutral", "entailment"}
    resolved: Dict[str, int] = {}
    pairs = list((getattr(config, "label2id", None) or {}).items())
    pairs += [(name, idx) for idx, name in
              (getattr(config, "id2label", None) or {}).items()]
    for name, idx in pairs:
        name = str(name).strip().lower()
        if name not in required:
            continue
        idx = int(idx)
        if name in resolved and resolved[name] != idx:
            raise ValueError(f"Conflicting NLI label IDs for {name!r} in model config")
        resolved[name] = idx
    if set(resolved) != required or len(set(resolved.values())) != 3:
        raise ValueError(
            "NLI model config must identify distinct contradiction, neutral, "
            "and entailment output IDs in id2label or label2id."
        )
    return resolved


class NLIEntailment:
    """Bidirectional entailment checker using a DeBERTa NLI model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        logger.info("Loading NLI model %s …", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(device)
        self.model.eval()
        self.label_ids = _nli_label_ids(self.model.config)

    @torch.no_grad()
    def check_implication(self, text1: str, text2: str) -> int:
        """Return a model-specific output ID; meanings are in ``label_ids``."""
        inputs = self.tokenizer(
            text1, text2, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        logits = self.model(**inputs).logits
        return int(torch.argmax(logits, dim=-1).item())

    def are_equivalent(self, text1: str, text2: str, strict: bool = True) -> bool:
        """Check bidirectional entailment, or one-way entailment if relaxed.

        The relaxed rule permits an entailment/neutral pair, but excludes
        contradiction and a neutral/neutral pair. Neither learned relation
        is guaranteed transitive, so greedy clustering remains order-dependent.
        """
        fwd = self.check_implication(text1, text2)
        bwd = self.check_implication(text2, text1)
        entailment = self.label_ids["entailment"]
        if strict:
            return fwd == entailment and bwd == entailment
        else:
            contradiction = self.label_ids["contradiction"]
            return contradiction not in (fwd, bwd) and entailment in (fwd, bwd)

    def unload(self):
        """Free GPU memory."""
        import gc
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def cluster_by_nli(
    texts: List[str],
    nli_model: NLIEntailment,
    strict: bool = True,
) -> List[int]:
    """Cluster texts via pairwise NLI (like get_semantic_ids)."""
    n = len(texts)
    cluster_ids = [-1] * n
    next_id = 0
    for i in range(n):
        if cluster_ids[i] == -1:
            cluster_ids[i] = next_id
            for j in range(i + 1, n):
                if cluster_ids[j] == -1:
                    if nli_model.are_equivalent(texts[i], texts[j], strict=strict):
                        cluster_ids[j] = next_id
            next_id += 1
    return cluster_ids


# ---------------------------------------------------------------------------
# Entropy helpers
# ---------------------------------------------------------------------------

def jsd_across_clarifications(
    pooled_cluster_ids: List[int],
    clar_sizes: List[int],
) -> float:
    """Jensen-Shannon Divergence across per-clarification cluster distributions.

    Directly measures whether different clarifications produce different answer
    distributions over semantic clusters. High JSD = different clarifications
    elicit different clusters (ambiguity signal). Low JSD = all clarifications
    produce the same cluster distribution (no ambiguity).

    Args:
        pooled_cluster_ids: Cluster IDs for all pooled answers (flat list).
        clar_sizes: Number of answers per clarification (to split pooled_cluster_ids).

    Returns:
        JSD value in nats. 0 = identical distributions, log(n_clar) = maximally different.
    """
    if not pooled_cluster_ids or not clar_sizes:
        return 0.0

    n_clusters = max(pooled_cluster_ids) + 1
    n_clar = len(clar_sizes)
    if n_clar < 2:
        return 0.0

    # Build per-clarification probability vectors over the shared cluster space
    distributions = []
    offset = 0
    for size in clar_sizes:
        if size == 0:
            distributions.append(np.ones(n_clusters) / n_clusters)
        else:
            cids = pooled_cluster_ids[offset:offset + size]
            counts = np.bincount(cids, minlength=n_clusters).astype(float)
            counts += 1e-12  # avoid log(0)
            distributions.append(counts / counts.sum())
        offset += size

    # JSD = H(mixture) - mean(H(components))
    mixture = np.mean(distributions, axis=0)
    h_mixture = float(scipy.stats.entropy(mixture))
    h_components = float(np.mean([scipy.stats.entropy(d) for d in distributions]))

    return max(h_mixture - h_components, 0.0)


def cluster_assignment_entropy(cluster_ids: List[int]) -> float:
    """Entropy over the empirical cluster proportion distribution.

    H = -Σ p_k log(p_k)
    """
    if not cluster_ids:
        return 0.0
    counts = np.bincount(cluster_ids)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return max(float(-np.sum(probs * np.log(probs))), 0.0)  # clamp -0.0 → 0.0


def cluster_proportions_and_representatives(
    texts: List[str],
    cluster_ids: List[int],
) -> Tuple[np.ndarray, List[str]]:
    """Return cluster proportions and representative texts, ALIGNED and sorted descending.

    Both arrays are sorted so that index 0 = largest cluster, index 1 = next, etc.
    This alignment is critical for APS score computation.

    Returns:
        (props, reps): props is ndarray of proportions sorted desc;
                        reps is list of representative texts in same order.
    """
    if not cluster_ids:
        return np.array([1.0]), []

    counts = np.bincount(cluster_ids)
    n_clusters = len(counts)
    probs = counts / counts.sum()

    # Pick the most-frequent (modal) text in each cluster as representative.
    # Ties broken by first-occurrence (Counter preserves insertion order).
    from collections import Counter
    texts_by_id: Dict[int, List[str]] = {}
    for text, cid in zip(texts, cluster_ids):
        texts_by_id.setdefault(int(cid), []).append(text)
    reps_by_id: Dict[int, str] = {
        cid: Counter(ts).most_common(1)[0][0] for cid, ts in texts_by_id.items()
    }

    # Sort by descending proportion — BOTH props and reps
    order = np.argsort(-probs)  # indices of clusters sorted by desc proportion
    sorted_props = probs[order]
    sorted_reps = [reps_by_id.get(int(cid), "") for cid in order]

    return sorted_props, sorted_reps


def cluster_proportions(cluster_ids: List[int]) -> np.ndarray:
    """Return sorted (descending) cluster proportions."""
    if not cluster_ids:
        return np.array([1.0])
    counts = np.bincount(cluster_ids)
    probs = counts / counts.sum()
    return np.sort(probs)[::-1]


def cluster_representatives(
    texts: List[str],
    cluster_ids: List[int],
) -> Dict[int, str]:
    """Pick the most-frequent (modal) text per cluster as representative."""
    from collections import Counter
    texts_by_id: Dict[int, List[str]] = {}
    for text, cid in zip(texts, cluster_ids):
        texts_by_id.setdefault(int(cid), []).append(text)
    return {cid: Counter(ts).most_common(1)[0][0] for cid, ts in texts_by_id.items()}


# ---------------------------------------------------------------------------
# Von Neumann Entropy (VNE) — kernel-based uncertainty
# ---------------------------------------------------------------------------

def compute_vne(
    embeddings: np.ndarray,
    kernel: str = "rbf",
    gamma: float = 1.0,
    scale: bool = True,
) -> float:
    """Von Neumann Entropy from embeddings via RBF kernel eigenvalues.

    Matches the spectral paper (Walha et al., AAAI 2026) implementation:
      1. Build kernel matrix K from the supplied embeddings
      2. Normalize: K = K / n
      3. Eigendecompose
      4. Clip negative eigenvalues to 0
      5. Compute entropy via scipy.stats.entropy (Shannon entropy of eigenvalues)
      6. Optionally scale by 1/log(n) to normalize to [0, 1]

    Note: For RBF kernel, K_ii = 1 so Tr(K/n) = 1, meaning eigenvalues
    sum to ~1 and scipy.stats.entropy's renormalization is negligible.

    Args:
        embeddings: (n_samples, dim) array. CAELUM's decomposition passes
            L2-normalized sentence embeddings.
        kernel: Kernel type ('rbf' or 'cosine').
        gamma: RBF kernel bandwidth parameter.
        scale: If True (default), divide by log(n) to normalize to [0, 1].
            CAELUM's decomposition explicitly passes scale=False for both
            pooled and per-clarification entropy.

    Returns:
        VNE value. If scale=True, in [0, 1]. Higher = more diverse answers.
    """
    n = len(embeddings)
    if n < 2:
        return 0.0

    if kernel == "rbf":
        K = sklearn.metrics.pairwise.rbf_kernel(embeddings, gamma=gamma)
    elif kernel == "cosine":
        K = sklearn.metrics.pairwise.linear_kernel(embeddings)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    K = K / n

    try:
        eigenvalues, _ = scipy.linalg.eigh(K)
    except np.linalg.LinAlgError:
        try:
            eigenvalues, _ = np.linalg.eigh(K)
        except np.linalg.LinAlgError:
            K += 1e-6 * np.eye(K.shape[0])
            eigenvalues, _ = scipy.linalg.eigh(K)

    # Clip negative eigenvalues to 0 (matching spectral paper — theoretically
    # impossible for PSD kernel matrices, but numerical issues cause them)
    eigenvalues = np.clip(eigenvalues, 0, None)

    # Shannon entropy of eigenvalue distribution (matches spectral paper)
    result = float(scipy.stats.entropy(eigenvalues))

    # Scale to [0, 1] by dividing by max entropy log(n)
    if scale and n > 1:
        result = result / np.log(n)

    return max(float(result), 0.0)


def compute_vne_decomposition(
    answers_per_clarification: List[List[str]],
    encoder: SentenceEncoder,
    gamma: float = 1.0,
) -> Dict[str, float]:
    """ICE decomposition using Von Neumann Entropy instead of cluster entropy.

    Same decomposition structure as CE:
        Total     = VNE(all answers pooled)
        Epistemic = mean(VNE per clarification)
        Aleatoric = Total - Epistemic

    Note: Uses scale=False for both per-clarification and pooled VNE to ensure
    consistent scaling. Previously, per-clarification VNE was scaled by
    1/log(n_answers_per_clar) and pooled by 1/log(n_total), making the
    subtraction-based decomposition meaningless.

    Args:
        answers_per_clarification: list of list of answer strings.
        encoder: sentence encoder for embeddings.
        gamma: RBF kernel bandwidth.

    Returns:
        Dict with 'total', 'epistemic', 'aleatoric', 'per_clarification_entropy'.
    """
    n_clar = len(answers_per_clarification)
    if n_clar == 0:
        return {"total": 0.0, "epistemic": 0.0, "aleatoric": 0.0,
                "per_clarification_entropy": []}

    # Encode all answers once
    all_answers = []
    clar_indices = []
    for i, answers in enumerate(answers_per_clarification):
        for a in answers:
            all_answers.append(a)
            clar_indices.append(i)

    if not all_answers:
        return {"total": 0.0, "epistemic": 0.0, "aleatoric": 0.0,
                "per_clarification_entropy": []}

    embeddings = encoder.encode(all_answers, normalize=True)
    clar_indices = np.array(clar_indices)

    # Per-clarification VNE (epistemic component)
    # scale=False: raw entropy values, no division by log(n_per_clar)
    per_clar_vne = []
    for i in range(n_clar):
        mask = clar_indices == i
        emb_i = embeddings[mask]
        if len(emb_i) >= 2:
            per_clar_vne.append(compute_vne(emb_i, gamma=gamma, scale=False))
        else:
            per_clar_vne.append(0.0)

    epistemic = float(np.mean(per_clar_vne))

    # Total VNE (pooled) — same scale=False for consistent decomposition
    total = compute_vne(embeddings, gamma=gamma, scale=False)

    # Aleatoric = residual
    aleatoric = max(total - epistemic, 0.0)

    return {
        "total": total,
        "epistemic": epistemic,
        "aleatoric": aleatoric,
        "per_clarification_entropy": per_clar_vne,
    }


# ---------------------------------------------------------------------------
# ICE decomposition (Cluster Entropy — original method)
# ---------------------------------------------------------------------------

def compute_ice_decomposition(
    answers_per_clarification: List[List[str]],
    encoder: SentenceEncoder,
    cluster_threshold: float = 0.7,
    use_nli: bool = False,
    nli_model: Optional[NLIEntailment] = None,
) -> Dict[str, float]:
    """Compute the ICE (Input Clarification Ensembling) uncertainty decomposition.

    Given answers grouped by clarification (shape: n_clarifications × n_answers),
    compute:
        Epistemic  = (1/n) Σ H_i          (avg within-clarification entropy)
        Total      = H(pooled answers)      (entropy over all answers pooled)
        Aleatoric  = Total - Epistemic       (between-clarification disagreement)

    This mirrors compute_disagreement_measure[_no_model]() from the spectral
    uncertainty code.

    Args:
        answers_per_clarification: list of list of answer strings.
            answers_per_clarification[i][j] = j-th answer for i-th clarification.
        encoder: sentence encoder for embedding-based clustering.
        cluster_threshold: cosine similarity threshold for clustering.
        use_nli: if True, use NLI model instead of embeddings.
        nli_model: required if use_nli is True.

    Returns:
        Dict with keys 'total', 'epistemic', 'aleatoric',
        'per_clarification_entropy', 'n_clusters_total',
        'cluster_props_total', 'cluster_representatives'.
    """
    n_clar = len(answers_per_clarification)
    if n_clar == 0:
        return {
            "total": 0.0, "epistemic": 0.0, "aleatoric": 0.0,
            "per_clarification_entropy": [],
            "n_clusters_total": 0,
            "cluster_props_total": [],
            "cluster_representatives": [],
            "cluster_members": [],
            "cluster_ids": [],
        }

    # --- Step 1: cluster ALL pooled answers ONCE (shared clustering) ---
    # This guarantees the decomposition Total = Epistemic + Aleatoric holds
    # exactly, since per-clarification entropies use the same cluster IDs.
    clar_sizes = [len(a) for a in answers_per_clarification]
    pooled_answers = [a for clar_answers in answers_per_clarification for a in clar_answers]
    if use_nli and nli_model is not None:
        pooled_cids = cluster_by_nli(pooled_answers, nli_model)
    else:
        pooled_cids = cluster_by_embedding(pooled_answers, encoder, threshold=cluster_threshold)

    total = cluster_assignment_entropy(pooled_cids)

    # --- Step 2: per-clarification entropy using SHARED cluster assignments ---
    per_clar_entropies: List[float] = []
    offset = 0
    for size in clar_sizes:
        if size == 0:
            per_clar_entropies.append(0.0)
        else:
            clar_cids = pooled_cids[offset:offset + size]
            h_i = cluster_assignment_entropy(clar_cids)
            per_clar_entropies.append(h_i)
        offset += size

    epistemic = float(np.mean(per_clar_entropies))

    # With shared clustering, Total >= Epistemic by concavity of entropy.
    # A small numerical tolerance is allowed.
    if total < epistemic - 1e-9:
        logger.warning("Unexpected: total=%.6f < epistemic=%.6f after shared clustering",
                        total, epistemic)
    aleatoric = max(total - epistemic, 0.0)

    # Cluster info for downstream use — ALIGNED sorted descending
    props, reps_list = cluster_proportions_and_representatives(pooled_answers, pooled_cids)
    n_clusters = len(reps_list)

    # Build cluster_members: list of lists of raw answers per cluster, sorted
    # in the same order as props/reps (descending by proportion).
    counts = np.bincount(pooled_cids)
    order = np.argsort(-counts / counts.sum())
    members_by_sorted_idx: List[List[str]] = [[] for _ in range(n_clusters)]
    # Map each pooled answer to its sorted cluster index
    old_to_new = {int(old): new for new, old in enumerate(order)}
    for ans, cid in zip(pooled_answers, pooled_cids):
        sorted_idx = old_to_new.get(cid)
        if sorted_idx is not None and sorted_idx < n_clusters:
            members_by_sorted_idx[sorted_idx].append(ans)

    return {
        "total": total,
        "epistemic": epistemic,
        "aleatoric": aleatoric,
        "per_clarification_entropy": per_clar_entropies,
        "n_clusters_total": n_clusters,
        "cluster_props_total": props.tolist(),
        "cluster_representatives": reps_list,
        "cluster_members": members_by_sorted_idx,
        "cluster_ids": pooled_cids,
    }


# ---------------------------------------------------------------------------
# Batch decomposition over a dataset
# ---------------------------------------------------------------------------

def compute_uncertainty_for_dataset(
    generation_results: List[Dict],
    encoder: SentenceEncoder,
    cluster_threshold: float = 0.7,
    use_nli: bool = False,
    nli_model: Optional[NLIEntailment] = None,
    uncertainty_method: str = "ce",
    vne_gamma: float = 1.0,
    llm_judge_client: object | None = None,
    llm_judge_model: str = "gemini-2.5-flash-lite",
) -> List[Dict]:
    """Compute ICE decomposition for every question in a generated dataset.

    Supports two uncertainty methods:
      - 'ce': Cluster Entropy (original, based on cluster assignment proportions)
      - 'vne': Von Neumann Entropy (kernel-based, uses RBF kernel eigenvalues)

    Both methods still require clustering for conformal prediction sets
    (cluster_props, correct_cluster_idx). When method='vne', clustering is
    computed separately from VNE uncertainty values.

    Expects each item in *generation_results* to have:
        - 'answers': List[List[str]]  (clarifications x answers)
        - 'gt_answers': List[str]

    Args:
        generation_results: List of generation result dicts.
        encoder: Sentence encoder for embeddings.
        cluster_threshold: Cosine similarity threshold for clustering.
        use_nli: Use NLI-based clustering instead of embeddings.
        nli_model: Required if use_nli is True.
        uncertainty_method: 'ce' for cluster entropy, 'vne' for Von Neumann entropy.
        vne_gamma: RBF kernel bandwidth for VNE (default 1.0).
        llm_judge_client: Optional OpenAI-compatible client for LLM-judge
            fallback when heuristic matching fails. If None (default), only
            heuristic matching is used.
        llm_judge_model: Model identifier for the LLM judge (default:
            gemini-2.5-flash-lite).

    Returns the input list augmented with uncertainty fields.
    """
    from .utils import find_correct_cluster, find_correct_cluster_llm

    if uncertainty_method not in ("ce", "vne"):
        raise ValueError(f"Unknown uncertainty_method: {uncertainty_method!r}. Use 'ce' or 'vne'.")

    method_label = "VNE" if uncertainty_method == "vne" else "CE"
    logger.info("Using %s uncertainty decomposition", method_label)
    if llm_judge_client is not None:
        logger.info("LLM-judge fallback enabled (model=%s)", llm_judge_model)

    n_llm_recovered = 0  # count of samples recovered by LLM judge

    for idx, item in enumerate(generation_results):
        answers_per_clar = item["answers"]

        # --- Always compute clustering (needed for conformal prediction sets) ---
        pooled_answers = [a for clar in answers_per_clar for a in clar]
        if use_nli and nli_model is not None:
            pooled_cids = cluster_by_nli(pooled_answers, nli_model)
        else:
            pooled_cids = cluster_by_embedding(
                pooled_answers, encoder, threshold=cluster_threshold
            )

        props, reps_list = cluster_proportions_and_representatives(
            pooled_answers, pooled_cids
        )
        n_clusters = len(reps_list)

        # Build cluster_members
        counts = np.bincount(pooled_cids)
        order = np.argsort(-counts / counts.sum())
        members_by_sorted_idx: List[List[str]] = [[] for _ in range(n_clusters)]
        old_to_new = {int(old): new for new, old in enumerate(order)}
        for ans, cid in zip(pooled_answers, pooled_cids):
            sorted_idx = old_to_new.get(cid)
            if sorted_idx is not None and sorted_idx < n_clusters:
                members_by_sorted_idx[sorted_idx].append(ans)

        # --- Compute uncertainty values ---
        clar_sizes = [len(a) for a in answers_per_clar]

        # CE-based decomposition (used for aleatoric in both methods)
        ce_total = cluster_assignment_entropy(pooled_cids)
        ce_per_clar: List[float] = []
        offset = 0
        for size in clar_sizes:
            if size == 0:
                ce_per_clar.append(0.0)
            else:
                ce_per_clar.append(
                    cluster_assignment_entropy(pooled_cids[offset:offset + size])
                )
            offset += size
        ce_epi = float(np.mean(ce_per_clar))
        u_ale_ce = max(ce_total - ce_epi, 0.0)

        # JSD across clarifications (alternative aleatoric signal)
        u_jsd = jsd_across_clarifications(pooled_cids, clar_sizes)

        if uncertainty_method == "vne":
            vne_decomp = compute_vne_decomposition(
                answers_per_clar, encoder=encoder, gamma=vne_gamma
            )
            u_total = vne_decomp["total"]
            u_epi = vne_decomp["epistemic"]
            per_clar_ent = vne_decomp["per_clarification_entropy"]
            # The same embeddings and RBF kernel define all Gram matrices.
            # With equal clarification sizes, shared-feature entropy
            # concavity guarantees Total >= Epistemic up to numerical error.
            u_ale = vne_decomp["aleatoric"]
        else:
            u_total = ce_total
            per_clar_ent = ce_per_clar
            u_epi = ce_epi
            u_ale = u_ale_ce

        item.update({
            "U_total": u_total,
            "U_epi": u_epi,
            "U_ale": u_ale,
            "U_ale_jsd": u_jsd,
            "per_clarification_entropy": per_clar_ent,
            "n_clusters": n_clusters,
            "cluster_props": props.tolist(),
            "cluster_representatives": reps_list,
            "uncertainty_method": uncertainty_method,
        })

        # --- Determine correct cluster ---
        gt_answers = item.get("gt_answers", [])
        if gt_answers and reps_list:
            if llm_judge_client is not None:
                # Use heuristic + LLM-judge fallback
                correct_k = find_correct_cluster_llm(
                    reps_list, gt_answers, encoder,
                    client=llm_judge_client,
                    model=llm_judge_model,
                    embed_threshold=0.4,
                    rouge_threshold=0.15,
                    jaccard_threshold=0.25,
                    cluster_members=members_by_sorted_idx,
                )
                # Track whether LLM judge was responsible for the match
                heuristic_k = find_correct_cluster(
                    reps_list, gt_answers, encoder,
                    embed_threshold=0.4,
                    rouge_threshold=0.15,
                    jaccard_threshold=0.25,
                    cluster_members=members_by_sorted_idx,
                )
                if correct_k >= 0 and heuristic_k < 0:
                    n_llm_recovered += 1
            else:
                correct_k = find_correct_cluster(
                    reps_list, gt_answers, encoder,
                    embed_threshold=0.4,
                    rouge_threshold=0.15,
                    jaccard_threshold=0.25,
                    cluster_members=members_by_sorted_idx,
                )
        else:
            correct_k = -1
        item["correct_cluster_idx"] = correct_k

        if (idx + 1) % 20 == 0:
            logger.info(
                "[%d/%d] %s: U_total=%.3f  U_epi=%.3f  U_ale=%.3f  K=%d  correct_k=%d",
                idx + 1, len(generation_results), method_label,
                u_total, u_epi, u_ale, n_clusters, correct_k,
            )

    # Summary
    n_total = len(generation_results)
    n_matched = sum(1 for d in generation_results if d.get("correct_cluster_idx", -1) >= 0)
    logger.info("Match rate: %d/%d (%.1f%%)", n_matched, n_total, 100 * n_matched / max(n_total, 1))
    if llm_judge_client is not None:
        logger.info("LLM-judge recovered: %d additional matches", n_llm_recovered)

    return generation_results
