"""
Utility functions for CAELUM: data loading, embeddings, correctness checking.

Includes an LLM-as-judge fallback (via Gemini Flash Lite) for correctness
matching when the heuristic 4-tier strategy fails.
"""

import hashlib
import json
import os
import logging
import time
from pathlib import Path
from typing import Callable, List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_triviaqa(path: str, n_samples: int = 200, seed: int = 42) -> List[Dict[str, Any]]:
    """Load TriviaQA evaluation data.

    JSON schema per entry:
        question_id: int
        datapoint_id: int
        question: str
        gt_answer: str
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rng = np.random.RandomState(seed)
    if len(data) > n_samples:
        indices = rng.choice(len(data), size=n_samples, replace=False)
        data = [data[i] for i in sorted(indices)]

    # Normalise ground-truth answers to a list.
    # TriviaQA entries may include a gt_aliases field with multiple valid
    # answer forms (aliases, normalized_aliases from the original dataset).
    for item in data:
        # Prefer gt_aliases (rich alias list) if available; fall back to gt_answer
        aliases = item.get("gt_aliases")
        if aliases and isinstance(aliases, list) and len(aliases) > 0:
            # Deduplicate while preserving order
            seen = set()
            deduped = []
            for a in aliases:
                al = str(a).strip().lower()
                if al and al not in seen:
                    seen.add(al)
                    deduped.append(str(a).strip())
            item["gt_answers"] = deduped if deduped else [""]
        else:
            gt = item.get("gt_answer", item.get("gold_standard_answer", ""))
            if isinstance(gt, str):
                item["gt_answers"] = [gt] if gt else [""]
            elif isinstance(gt, list):
                item["gt_answers"] = [str(a) for a in gt if a]
            else:
                item["gt_answers"] = [str(gt)]
        item["is_ambig"] = False  # TriviaQA questions are factual

    logger.info("Loaded %d TriviaQA samples from %s", len(data), path)
    return data


def load_ambigqa(path: str, n_samples: int = 200, seed: int = 42) -> List[Dict[str, Any]]:
    """Load AmbigQA evaluation data.

    JSON schema per entry:
        question_id: int
        question: str
        is_ambig: bool
        gold_standard_answer: list[str] | null  (for unambiguous questions)
        gt_clarifications: list[str] | null     (for ambiguous questions)
        gt_answers: list[list[str]] | null      (for ambiguous questions)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rng = np.random.RandomState(seed)
    if len(data) > n_samples:
        indices = rng.choice(len(data), size=n_samples, replace=False)
        data = [data[i] for i in sorted(indices)]

    # Normalise ground-truth answers to a flat list of strings
    for item in data:
        flat_answers = []
        # Ambiguous: gt_answers is list of list of strings
        gt_nested = item.get("gt_answers")
        if gt_nested and isinstance(gt_nested, list):
            for ans_group in gt_nested:
                if isinstance(ans_group, list):
                    flat_answers.extend(ans_group)
                elif isinstance(ans_group, str):
                    flat_answers.append(ans_group)
        # Unambiguous: gold_standard_answer is list of strings or a string
        gsa = item.get("gold_standard_answer")
        if gsa is not None:
            if isinstance(gsa, list):
                flat_answers.extend(gsa)
            elif isinstance(gsa, str):
                flat_answers.append(gsa)
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for a in flat_answers:
            al = a.strip().lower()
            if al and al not in seen:
                seen.add(al)
                deduped.append(a.strip())
        item["gt_answers"] = deduped if deduped else [""]
        # Ensure is_ambig is present as a bool
        item["is_ambig"] = bool(item.get("is_ambig", False))

    logger.info("Loaded %d AmbigQA samples from %s", len(data), path)
    return data


def load_opennq(path: str, n_samples: int = 200, seed: int = 42) -> List[Dict[str, Any]]:
    """Load Open Natural Questions evaluation data.

    OpenNQ uses the same schema as TriviaQA (factual, unambiguous).

    JSON schema per entry:
        question_id: int
        datapoint_id: int
        question: str
        gt_answer: str
        annotation_idx: int
        annotation_type: str
        short_answer_idx: int
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rng = np.random.RandomState(seed)
    if len(data) > n_samples:
        indices = rng.choice(len(data), size=n_samples, replace=False)
        data = [data[i] for i in sorted(indices)]

    # Normalise ground-truth answers to a list.
    # OpenNQ entries may include gt_aliases with multiple valid answer forms.
    for item in data:
        aliases = item.get("gt_aliases")
        if aliases and isinstance(aliases, list) and len(aliases) > 0:
            seen = set()
            deduped = []
            for a in aliases:
                al = str(a).strip().lower()
                if al and al not in seen:
                    seen.add(al)
                    deduped.append(str(a).strip())
            item["gt_answers"] = deduped if deduped else [""]
        else:
            gt = item.get("gt_answer", item.get("gold_standard_answer", ""))
            if isinstance(gt, str):
                item["gt_answers"] = [gt] if gt else [""]
            elif isinstance(gt, list):
                item["gt_answers"] = [str(a) for a in gt if a]
            else:
                item["gt_answers"] = [str(gt)]
        item["is_ambig"] = False

    logger.info("Loaded %d OpenNQ samples from %s", len(data), path)
    return data


def load_ambiginst(path: str, n_samples: int = 200, seed: int = 42) -> List[Dict[str, Any]]:
    """Load AmbigInst evaluation data.

    AmbigInst uses the same schema as AmbigQA but includes an ``input`` field
    that provides context the question depends on. The input is prepended to
    the question so downstream stages receive a self-contained prompt.

    JSON schema per entry:
        question_id: int
        question: str
        input: str              (context that the question refers to)
        is_ambig: bool
        gold_standard_answer: list[str] | null
        gt_clarifications: list[str] | null
        gt_answers: list[list[str]] | null
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rng = np.random.RandomState(seed)
    if len(data) > n_samples:
        indices = rng.choice(len(data), size=n_samples, replace=False)
        data = [data[i] for i in sorted(indices)]

    for item in data:
        # Prepend the input context to the question
        input_ctx = item.get("input", "")
        if input_ctx:
            item["question"] = f"{input_ctx}\n\n{item['question']}"

        # Normalise ground-truth answers to a flat list of strings
        flat_answers = []
        # Ambiguous: gt_answers is list of list of strings
        gt_nested = item.get("gt_answers")
        if gt_nested and isinstance(gt_nested, list):
            for ans_group in gt_nested:
                if isinstance(ans_group, list):
                    flat_answers.extend(ans_group)
                elif isinstance(ans_group, str):
                    flat_answers.append(ans_group)
        # Unambiguous: gold_standard_answer is list of strings or a string
        gsa = item.get("gold_standard_answer")
        if gsa is not None:
            if isinstance(gsa, list):
                flat_answers.extend(gsa)
            elif isinstance(gsa, str):
                flat_answers.append(gsa)
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for a in flat_answers:
            al = a.strip().lower()
            if al and al not in seen:
                seen.add(al)
                deduped.append(a.strip())
        item["gt_answers"] = deduped if deduped else [""]
        # Ensure is_ambig is present as a bool
        item["is_ambig"] = bool(item.get("is_ambig", False))

    logger.info("Loaded %d AmbigInst samples from %s", len(data), path)
    return data


def load_simpleqa(path: str, n_samples: int | None = None, seed: int = 42) -> List[Dict[str, Any]]:
    """Load SimpleQA evaluation data.

    SimpleQA questions are factual and unambiguous, similar to TriviaQA.

    JSON schema per entry:
        question_id: int
        question: str
        gt_answer: str
        topic: str
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rng = np.random.RandomState(seed)
    if n_samples is not None and len(data) > n_samples:
        indices = rng.choice(len(data), size=n_samples, replace=False)
        data = [data[i] for i in sorted(indices)]

    # Normalise ground-truth answers to a list
    for item in data:
        gt = item.get("gt_answer", item.get("gold_standard_answer", ""))
        if isinstance(gt, str):
            item["gt_answers"] = [gt] if gt else [""]
        elif isinstance(gt, list):
            item["gt_answers"] = [str(a) for a in gt if a]
        else:
            item["gt_answers"] = [str(gt)]
        item["is_ambig"] = False  # SimpleQA questions are factual

    logger.info("Loaded %d SimpleQA samples from %s", len(data), path)
    return data


def load_popqa(path: str, n_samples: int = 2000, seed: int = 42) -> List[Dict[str, Any]]:
    """Load PopQA evaluation data.

    PopQA entries include a ``possible_answers`` field with all acceptable
    answer variants, which is used as the ground-truth answer list instead
    of the single ``gt_answer`` string.

    JSON schema per entry:
        question_id: int
        question: str
        gt_answer: str
        possible_answers: list[str]
        s_pop: int                   (subject popularity score)
        prop: str                    (relation/property type)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rng = np.random.RandomState(seed)
    if len(data) > n_samples:
        indices = rng.choice(len(data), size=n_samples, replace=False)
        data = [data[i] for i in sorted(indices)]

    for item in data:
        # Use possible_answers as the full gt_answers list
        possible = item.get("possible_answers", [])
        if possible and isinstance(possible, list):
            # Deduplicate while preserving order
            seen = set()
            deduped = []
            for a in possible:
                al = str(a).strip().lower()
                if al and al not in seen:
                    seen.add(al)
                    deduped.append(str(a).strip())
            item["gt_answers"] = deduped if deduped else [""]
        else:
            # Fallback to single gt_answer
            gt = item.get("gt_answer", "")
            item["gt_answers"] = [gt] if gt else [""]
        item["is_ambig"] = False  # PopQA questions are factual

    logger.info("Loaded %d PopQA samples from %s", len(data), path)
    return data


def load_condambigqa(path: str, n_samples: int | None = None, seed: int = 42) -> List[Dict[str, Any]]:
    """Load CondAmbigQA evaluation data.

    CondAmbigQA uses the same schema as AmbigQA (after conversion by
    ``prepare_datasets.py``).

    JSON schema per entry:
        question_id: int
        question: str
        is_ambig: bool
        gold_standard_answer: list[str] | null
        gt_clarifications: list[str] | null
        gt_answers: list[list[str]] | null
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rng = np.random.RandomState(seed)
    if n_samples is not None and len(data) > n_samples:
        indices = rng.choice(len(data), size=n_samples, replace=False)
        data = [data[i] for i in sorted(indices)]

    # Normalise ground-truth answers to a flat list of strings
    for item in data:
        flat_answers = []
        # Ambiguous: gt_answers is list of list of strings
        gt_nested = item.get("gt_answers")
        if gt_nested and isinstance(gt_nested, list):
            for ans_group in gt_nested:
                if isinstance(ans_group, list):
                    flat_answers.extend(ans_group)
                elif isinstance(ans_group, str):
                    flat_answers.append(ans_group)
        # Unambiguous: gold_standard_answer is list of strings or a string
        gsa = item.get("gold_standard_answer")
        if gsa is not None:
            if isinstance(gsa, list):
                flat_answers.extend(gsa)
            elif isinstance(gsa, str):
                flat_answers.append(gsa)
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for a in flat_answers:
            al = a.strip().lower()
            if al and al not in seen:
                seen.add(al)
                deduped.append(a.strip())
        item["gt_answers"] = deduped if deduped else [""]
        # Ensure is_ambig is present as a bool
        item["is_ambig"] = bool(item.get("is_ambig", False))

    logger.info("Loaded %d CondAmbigQA samples from %s", len(data), path)
    return data


# ---------------------------------------------------------------------------
# Sentence embeddings
# ---------------------------------------------------------------------------

class SentenceEncoder:
    """Thin wrapper around sentence-transformers for encoding texts."""

    def __init__(self, model_name: str = "all-mpnet-base-v2", device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # Suppress noisy "LOAD REPORT" printed by some model backends
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.model = SentenceTransformer(model_name, device=device)
        self.device = device

    def encode(self, texts: List[str], batch_size: int = 64,
               normalize: bool = True) -> np.ndarray:
        """Encode a list of texts into embeddings.

        Returns:
            np.ndarray of shape (len(texts), dim).
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=normalize,
        )
        return np.asarray(embeddings)


# ---------------------------------------------------------------------------
# Correctness checking via embedding similarity
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """Normalize text for robust matching: strip punctuation, collapse whitespace.

    Transforms like:
      "J.M.W. Turner" -> "j m w turner"
      "CHARLES 2nd"   -> "charles 2nd"
      "(The Beatles)"  -> "the beatles"
      "Škoda"          -> "skoda"
      "John le Carré"  -> "john le carre"

    Args:
        text: Raw text to normalize.

    Returns:
        Lowercased text with diacritics stripped, punctuation removed,
        and whitespace collapsed.
    """
    import re
    import unicodedata
    t = text.lower().strip()
    # Strip diacritics: "Škoda" -> "Skoda", "Carré" -> "Carre"
    t = unicodedata.normalize("NFKD", t)
    t = "".join([c for c in t if unicodedata.category(c) != "Mn"])
    # Remove common enclosing punctuation/quotes
    t = t.strip("\"'`()[]{}*_")
    # Replace punctuation with spaces (preserves word boundaries)
    t = re.sub(r"[^\w\s]", " ", t)
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _token_containment(text: str, answer: str) -> bool:
    """Check if ALL tokens of the shorter string appear in the longer string.

    This catches cases like GT="J M W Turner" appearing inside a sentence
    "The painting by J M W Turner is famous" even when the substring check
    fails because of intervening words.

    Works on normalized text. The shorter string's tokens must all appear
    as a subset of the longer string's tokens.

    Args:
        text: One of the two strings to compare (typically the generated answer).
        answer: The other string (typically the GT answer).

    Returns:
        True if all tokens of the shorter string appear in the longer string.
    """
    t_norm = _normalize_text(text)
    a_norm = _normalize_text(answer)

    t_tokens = set(t_norm.split())
    a_tokens = set(a_norm.split())

    if not t_tokens or not a_tokens:
        return False

    # Determine which is shorter (the "query")
    if len(a_tokens) <= len(t_tokens):
        query, corpus = a_tokens, t_tokens
    else:
        query, corpus = t_tokens, a_tokens

    # Require the query to have at least 1 meaningful token and ALL must appear
    # in corpus. For single-token queries, also require the corpus to be
    # reasonably short to avoid spurious matches (e.g., "the" in any sentence).
    if len(query) == 1:
        # Single-token match: only valid if corpus is also short (<=6 tokens)
        # to avoid matching "Turkey" in "Turkey is a country in..." when GT
        # is actually about a different Turkey.
        return query.issubset(corpus) and len(corpus) <= 6

    return query.issubset(corpus)


def _strip_trailing_s(word: str) -> str:
    """Strip trailing 's' for basic singular/plural normalization.

    Only applies to words >= 4 chars to avoid breaking short words
    like "bus", "gas", "his".  Handles common English plurals:
      "Revelations" -> "Revelation"
      "medals"      -> "medal"
      "Beatles"     -> "Beatle"
    """
    if len(word) >= 4 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _normalize_text_deplural(text: str) -> str:
    """Normalize text AND strip trailing 's' from each word for plural matching."""
    t = _normalize_text(text)
    return " ".join(_strip_trailing_s(w) for w in t.split())


def _word_boundary_contains(haystack: str, needle: str) -> bool:
    """Check if needle appears in haystack as whole word(s), not as a substring.

    Prevents false positives like "doll" matching "dollar" or "fall" matching
    "fallen". Both inputs should already be normalized (lowercase, no punct).

    Args:
        haystack: Text to search in.
        needle: Text to search for.

    Returns:
        True if needle appears as a whole-word match in haystack,
        or if haystack appears as a whole-word match in needle.
    """
    import re
    if not needle or not haystack:
        return False
    # Check both directions (shorter in longer)
    pattern_n = r'\b' + re.escape(needle) + r'\b'
    pattern_h = r'\b' + re.escape(haystack) + r'\b'
    if re.search(pattern_n, haystack):
        return True
    if re.search(pattern_h, needle):
        return True
    return False


def _contains_answer(text: str, answer: str) -> bool:
    """Case-insensitive check whether *answer* appears as substring in *text*."""
    return answer.strip().lower() in text.strip().lower()


def _contains_answer_normalized(text: str, answer: str) -> bool:
    """Normalized substring check: strips punctuation before checking containment.

    Catches cases like "J.M.W. Turner" matching "J M W Turner" where raw
    substring fails due to punctuation differences.

    Args:
        text: Text to search in.
        answer: Answer string to look for.

    Returns:
        True if the normalized answer appears as substring in normalized text.
    """
    return _normalize_text(answer) in _normalize_text(text)


def _rouge_l(candidate: str, reference: str) -> float:
    """Compute Rouge-L F-measure between two strings (word-level LCS)."""
    cand_tokens = candidate.lower().split()
    ref_tokens = reference.lower().split()
    if not cand_tokens or not ref_tokens:
        return 0.0
    # LCS via dynamic programming
    m, n = len(ref_tokens), len(cand_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == cand_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]
    precision = lcs_len / n if n > 0 else 0.0
    recall = lcs_len / m if m > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _jaccard_token_overlap(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity between word-level token sets of two strings."""
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def check_correctness(
    answer_text: str,
    gt_answers: List[str],
    encoder: SentenceEncoder,
    embed_threshold: float = 0.5,
    rouge_threshold: float = 0.2,
    jaccard_threshold: float = 0.4,
) -> bool:
    """Check if *answer_text* matches any ground-truth answer.

    Uses a 7-tier strategy:
      1. Exact substring containment (case-insensitive)
      2. Normalized substring (punctuation-stripped, diacritics-stripped)
      3. Depluralized whole-word match (strips trailing 's', word-boundary safe)
      4. Token containment (all tokens of shorter appear in longer)
      5. Rouge-L F-measure > rouge_threshold
      6. Jaccard token overlap > jaccard_threshold
      7. Embedding cosine similarity > embed_threshold
    """
    if not gt_answers:
        return False

    for gt in gt_answers:
        # Tier 1: exact substring
        if _contains_answer(answer_text, gt) or _contains_answer(gt, answer_text):
            return True
        # Tier 2: normalized substring
        if (_contains_answer_normalized(answer_text, gt)
                or _contains_answer_normalized(gt, answer_text)):
            return True
        # Tier 3: depluralized whole-word match (word-boundary safe)
        a_dp = _normalize_text_deplural(answer_text)
        g_dp = _normalize_text_deplural(gt)
        if g_dp and a_dp and _word_boundary_contains(a_dp, g_dp):
            return True
        # Tier 4: token containment
        if _token_containment(answer_text, gt):
            return True
        # Tier 5: Rouge-L
        if _rouge_l(answer_text, gt) > rouge_threshold:
            return True
        # Tier 6: Jaccard token overlap
        if _jaccard_token_overlap(answer_text, gt) > jaccard_threshold:
            return True

    # Tier 7: embedding similarity
    all_texts = [answer_text] + gt_answers
    embs = encoder.encode(all_texts, normalize=True)
    answer_emb = embs[0:1]
    gt_embs = embs[1:]
    sims = (answer_emb @ gt_embs.T).flatten()
    return bool(np.max(sims) >= embed_threshold)


# Keep old name as alias for backward compat
check_correctness_embedding = check_correctness


def find_correct_cluster(
    cluster_representatives: List[str],
    gt_answers: List[str],
    encoder: SentenceEncoder,
    embed_threshold: float = 0.5,
    rouge_threshold: float = 0.2,
    jaccard_threshold: float = 0.4,
    cluster_members: Optional[List[List[str]]] = None,
) -> int:
    """Return the index of the semantic cluster whose representative matches the GT.

    Uses a 7-tier matching strategy. Lexical and overlap tiers are applied to
    cluster representatives and, when available, all raw cluster members; the
    embedding tier is applied to representatives only:

      1. Exact substring containment (case-insensitive)
      2. Normalized substring (punctuation-stripped, whitespace-collapsed)
      3. Depluralized whole-word match for short/entity answers
      4. Token containment (all GT tokens appear in answer or vice versa)
      5. Rouge-L F-measure > rouge_threshold
      6. Jaccard token overlap > jaccard_threshold
      7. Embedding cosine similarity > embed_threshold (representatives only)

    The implementation applies the text tiers in the order shown below in
    ``_text_match``; the tier labels above describe the paper-level matching
    family rather than a public API contract.

    Args:
        cluster_representatives: Cluster representative answer strings, sorted
            by descending cluster proportion.
        gt_answers: List of acceptable ground-truth answers.
        encoder: SentenceEncoder for tier-6 embedding matching.
        embed_threshold: Cosine similarity threshold for the embedding tier.
        rouge_threshold: Rouge-L threshold for tier 4.
        jaccard_threshold: Jaccard threshold for tier 5.
        cluster_members: Optional list of lists of raw answers per cluster.
            When provided, all raw answers in each cluster are checked in
            lexical/overlap tiers (not just the representative). This significantly
            improves matching when the representative is a full sentence but
            the GT is a short entity (e.g., "Hydrogen" vs "The element is
            Hydrogen.").

    Returns:
        Index of the correct cluster (0-indexed into sorted array), or -1 if
        no cluster matches.
    """
    if not cluster_representatives or not gt_answers:
        return -1

    def _text_match(candidate: str, gt: str) -> bool:
        """Apply tiers 1--6 text-based matching between candidate and GT."""
        # Tier 1: exact substring
        if _contains_answer(candidate, gt) or _contains_answer(gt, candidate):
            return True
        # Tier 2: normalized substring (strips punctuation + diacritics)
        if _contains_answer_normalized(candidate, gt) or _contains_answer_normalized(gt, candidate):
            return True
        # Tier 3: depluralized whole-word match (word-boundary safe)
        c_dp = _normalize_text_deplural(candidate)
        g_dp = _normalize_text_deplural(gt)
        if g_dp and c_dp and _word_boundary_contains(c_dp, g_dp):
            return True
        # Tier 4: token containment
        if _token_containment(candidate, gt):
            return True
        # Tier 5: Rouge-L
        if _rouge_l(candidate, gt) > rouge_threshold:
            return True
        # Tier 6: Jaccard
        if _jaccard_token_overlap(candidate, gt) > jaccard_threshold:
            return True
        return False

    # --- Tiers 1--6: text-based matching on representatives + members ---
    for idx, rep in enumerate(cluster_representatives):
        for gt in gt_answers:
            if _text_match(rep, gt):
                return idx
        # Check all raw answers in the cluster (if available)
        if cluster_members and idx < len(cluster_members):
            for member in cluster_members[idx]:
                if member == rep:
                    continue  # already checked
                for gt in gt_answers:
                    if _text_match(member, gt):
                        return idx

    # --- Tier 7: embedding similarity (representatives first) ---
    rep_embs = encoder.encode(cluster_representatives, normalize=True)
    gt_embs = encoder.encode(gt_answers, normalize=True)
    sim_matrix = rep_embs @ gt_embs.T  # (n_clusters, n_gt)
    max_sim = sim_matrix.max(axis=1)    # best gt match per cluster
    best_cluster = int(np.argmax(max_sim))
    if max_sim[best_cluster] >= embed_threshold:
        return best_cluster

    return -1


# ---------------------------------------------------------------------------
# LLM-as-judge correctness matching
# ---------------------------------------------------------------------------

# Module-level cache for LLM judge results (keyed by (gt, candidate) hash).
# Persists within a single process run to avoid redundant API calls.
_llm_judge_cache: Dict[str, bool] = {}


def _cache_key(gt: str, candidate: str) -> str:
    """Compute a deterministic cache key for a (gt, candidate) pair."""
    raw = f"{gt.strip().lower()}||{candidate.strip().lower()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def llm_judge_correctness(
    gt_answer: str,
    candidate_answer: str,
    client: Any,
    model: str = "gemini-2.5-flash-lite",
    max_retries: int = 3,
    rate_delay: float = 2.0,
) -> bool:
    """Use an LLM to judge whether a candidate answer is semantically equivalent
    to a ground-truth reference answer.

    The judge is queried via an OpenAI-compatible chat endpoint (e.g. Gemini
    Flash Lite). Results are cached in-process to avoid redundant API calls.

    Args:
        gt_answer: The ground-truth reference answer.
        candidate_answer: The candidate answer to evaluate.
        client: An ``openai.OpenAI`` client instance configured for the
            target provider.
        model: Model identifier to use for the judge call.
        max_retries: Maximum number of retry attempts on transient errors.
        rate_delay: Seconds to wait between retries for rate-limit errors.

    Returns:
        True if the judge considers the candidate semantically equivalent to
        the ground truth, False otherwise.
    """
    key = _cache_key(gt_answer, candidate_answer)
    if key in _llm_judge_cache:
        return _llm_judge_cache[key]

    prompt = (
        "Does the candidate answer convey the same factual answer as the reference?\n\n"
        "Say YES if the candidate contains the same entity, person, date, number, "
        "or fact as the reference — even if it uses different wording, "
        "abbreviations, longer phrasing, or includes extra context.\n"
        "Say NO if the candidate gives a different entity, person, date, number, "
        "or fact (e.g., 'gold' vs 'bronze', '1912' vs '1914', different people).\n\n"
        f"Reference: \"{gt_answer}\"\n"
        f"Candidate: \"{candidate_answer}\"\n\n"
        "Answer only YES or NO."
    )

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=8,
            )
            raw = resp.choices[0].message.content
            if raw is None:
                # Gemini sometimes returns None content; treat as NO
                _llm_judge_cache[key] = False
                return False
            text = raw.strip().upper()
            result = text.startswith("YES")
            _llm_judge_cache[key] = result
            return result
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = min(2 ** attempt * rate_delay, 60)
                logger.debug("LLM judge rate-limited, waiting %.1fs...", wait)
                time.sleep(wait)
            elif "500" in err or "503" in err:
                time.sleep(2)
            else:
                if attempt >= max_retries - 1:
                    logger.warning("LLM judge error: %s", err[:120])
                    _llm_judge_cache[key] = False
                    return False
                time.sleep(1)

    _llm_judge_cache[key] = False
    return False


def llm_judge_batch(
    gt_answers: List[str],
    candidates: List[str],
    client: Any,
    model: str = "gemini-2.5-flash-lite",
    max_retries: int = 3,
    rate_delay: float = 2.0,
) -> List[bool]:
    """Judge multiple candidates against multiple ground-truth answers in a
    single API call for efficiency.

    Constructs a prompt that asks the LLM to evaluate each candidate against
    ALL ground-truth answers and return YES/NO per candidate. Falls back to
    individual calls if batch parsing fails.

    Args:
        gt_answers: List of acceptable ground-truth answers.
        candidates: List of candidate answers to evaluate.
        client: An ``openai.OpenAI`` client instance.
        model: Model identifier for the judge.
        max_retries: Maximum retry attempts.
        rate_delay: Seconds between retries on rate-limit errors.

    Returns:
        List of booleans, one per candidate, indicating equivalence.
    """
    if not candidates:
        return []
    if not gt_answers:
        return [False] * len(candidates)

    # Check cache first -- any candidate already judged can be skipped
    results: List[Optional[bool]] = []
    uncached_indices: List[int] = []
    for i, cand in enumerate(candidates):
        cached = None
        for gt in gt_answers:
            key = _cache_key(gt, cand)
            if key in _llm_judge_cache and _llm_judge_cache[key]:
                cached = True
                break
        if cached is True:
            results.append(True)
        else:
            # Check if ALL gt answers have been cached as False
            all_cached_false = all(
                _cache_key(gt, cand) in _llm_judge_cache
                for gt in gt_answers
            )
            if all_cached_false:
                results.append(False)
            else:
                results.append(None)
                uncached_indices.append(i)

    if not uncached_indices:
        return [r if r is not None else False for r in results]

    # Build batch prompt
    gt_str = "; ".join(f'"{gt}"' for gt in gt_answers)
    lines = []
    for idx in uncached_indices:
        lines.append(f"#{idx+1}: \"{candidates[idx]}\"")
    candidates_str = "\n".join(lines)

    prompt = (
        "For each candidate, does it convey the same factual answer as any reference?\n\n"
        "Say YES if the candidate contains the same entity, person, date, number, "
        "or fact — even with different wording, abbreviations, or extra context.\n"
        "Say NO if the candidate gives a different entity, person, date, number, "
        "or fact (e.g., 'gold' vs 'bronze', '1912' vs '1914', different people).\n\n"
        f"Reference answers: {gt_str}\n\n"
        f"Candidates:\n{candidates_str}\n\n"
        "For each candidate, respond with its number followed by YES or NO. "
        "Example:\n#1: YES\n#2: NO\n"
    )

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4 * len(uncached_indices) + 32,
            )
            raw = resp.choices[0].message.content
            if raw is None:
                raise ValueError("Gemini returned None content")
            text = raw.strip()
            # Parse responses
            import re
            parsed = {}
            for match in re.finditer(r"#(\d+)\s*:?\s*(YES|NO)", text, re.IGNORECASE):
                num = int(match.group(1))
                val = match.group(2).upper() == "YES"
                parsed[num] = val

            # Map parsed results back
            for idx in uncached_indices:
                num = idx + 1
                if num in parsed:
                    results[idx] = parsed[num]
                    # Cache the result for each gt answer
                    for gt in gt_answers:
                        _llm_judge_cache[_cache_key(gt, candidates[idx])] = parsed[num]
                else:
                    # Parsing missed this candidate; fall back to individual call
                    result = False
                    for gt in gt_answers:
                        if llm_judge_correctness(
                            gt, candidates[idx], client, model,
                            max_retries=max_retries, rate_delay=rate_delay,
                        ):
                            result = True
                            break
                    results[idx] = result

            return [r if r is not None else False for r in results]

        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = min(2 ** attempt * rate_delay, 60)
                logger.debug("LLM judge batch rate-limited, waiting %.1fs...", wait)
                time.sleep(wait)
            elif "500" in err or "503" in err:
                time.sleep(2)
            else:
                if attempt >= max_retries - 1:
                    logger.warning("LLM judge batch error, falling back to "
                                   "individual calls: %s", err[:120])
                    break
                time.sleep(1)

    # Fallback: individual calls for all uncached
    for idx in uncached_indices:
        if results[idx] is not None:
            continue
        result = False
        for gt in gt_answers:
            if llm_judge_correctness(
                gt, candidates[idx], client, model,
                max_retries=max_retries, rate_delay=rate_delay,
            ):
                result = True
                break
        results[idx] = result

    return [r if r is not None else False for r in results]


def find_correct_cluster_llm(
    cluster_representatives: List[str],
    gt_answers: List[str],
    encoder: SentenceEncoder,
    client: Any,
    model: str = "gemini-2.5-flash-lite",
    rate_delay: float = 2.0,
    embed_threshold: float = 0.5,
    rouge_threshold: float = 0.2,
    jaccard_threshold: float = 0.4,
    cluster_members: Optional[List[List[str]]] = None,
) -> int:
    """Find the correct cluster using heuristic matching with LLM-judge fallback.

    First attempts the fast heuristic 4-tier matching via ``find_correct_cluster``.
    If that returns -1 (no match found), falls back to LLM-as-judge evaluation
    of each cluster representative against all ground-truth answers.

    This two-stage approach minimises API calls: the LLM is only invoked for
    samples where cheaper heuristics fail.

    Args:
        cluster_representatives: Cluster representative answer strings, sorted
            by descending cluster proportion.
        gt_answers: List of acceptable ground-truth answers.
        encoder: SentenceEncoder for heuristic tier-4 matching.
        client: An ``openai.OpenAI`` client instance for the LLM judge.
        model: Model identifier for the LLM judge.
        rate_delay: Seconds to wait between API calls for rate limiting.
        embed_threshold: Cosine similarity threshold for heuristic tier-4.
        rouge_threshold: Rouge-L threshold for heuristic tier-2.
        jaccard_threshold: Jaccard threshold for heuristic tier-3.
        cluster_members: Optional list of lists of raw answers per cluster.

    Returns:
        Index of the correct cluster (0-indexed into sorted array), or -1 if
        no cluster matches even after LLM judging.
    """
    # Stage 1: fast heuristic
    idx = find_correct_cluster(
        cluster_representatives, gt_answers, encoder,
        embed_threshold=embed_threshold,
        rouge_threshold=rouge_threshold,
        jaccard_threshold=jaccard_threshold,
        cluster_members=cluster_members,
    )
    if idx >= 0:
        return idx

    # Stage 2: LLM judge on each cluster representative
    if not cluster_representatives or not gt_answers:
        return -1

    judgments = llm_judge_batch(
        gt_answers, cluster_representatives, client, model,
        rate_delay=rate_delay,
    )

    for i, is_match in enumerate(judgments):
        if is_match:
            return i

    return -1


def save_llm_judge_cache(path: str) -> None:
    """Persist the in-memory LLM judge cache to a JSON file.

    Args:
        path: File path for the cache JSON.
    """
    ensure_dir(os.path.dirname(path) if os.path.dirname(path) else ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_llm_judge_cache, f, indent=2)
    logger.info("Saved LLM judge cache (%d entries) to %s",
                len(_llm_judge_cache), path)


def load_llm_judge_cache(path: str) -> int:
    """Load a previously saved LLM judge cache from a JSON file.

    Args:
        path: File path for the cache JSON.

    Returns:
        Number of entries loaded.
    """
    global _llm_judge_cache
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    _llm_judge_cache.update(loaded)
    logger.info("Loaded LLM judge cache (%d entries) from %s",
                len(loaded), path)
    return len(loaded)


# ---------------------------------------------------------------------------
# Prompt template loading
# ---------------------------------------------------------------------------

def load_prompt_template(path: str) -> str:
    """Read a prompt template file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, path: str) -> None:
    """Save data to a JSON file."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Any:
    """Load data from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_data(
    data: List[Dict],
    val1_frac: float = 0.4,
    val2_frac: float = 0.3,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split data into Val1 (lambda selection), Val2 (conformal threshold), Test.

    Default split: 40 / 30 / 30  (for 200 samples: 80 / 60 / 60).
    """
    rng = np.random.RandomState(seed)
    n = len(data)
    indices = rng.permutation(n)
    n_val1 = int(n * val1_frac)
    n_val2 = int(n * val2_frac)

    val1 = [data[i] for i in indices[:n_val1]]
    val2 = [data[i] for i in indices[n_val1:n_val1 + n_val2]]
    test = [data[i] for i in indices[n_val1 + n_val2:]]

    logger.info("Split %d items -> Val1=%d, Val2=%d, Test=%d",
                n, len(val1), len(val2), len(test))
    return val1, val2, test
