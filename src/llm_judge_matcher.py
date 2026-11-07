"""LLM-as-judge tier 8 for CAELUM correctness matching.

Adds a question-aware LLM judge on top of the existing 7-tier heuristic
matching pipeline (``src/caelum/utils.py:find_correct_cluster``). The judge
is invoked only on ``borderline'' items: those where the heuristic returned
``-1`` AND at least one cluster representative has cosine similarity to the
ground truth in the near-miss zone ``[0.3, 0.4)`` -- i.e., the embedding
felt close enough to be plausibly correct but did not clear the strict
threshold (default 0.5).

Key design choices
------------------
* **One API call per item.** A single prompt enumerates all cluster reps
  (1-indexed) and asks Gemini Flash Lite which one (if any) answers the
  question with the same meaning as one of the ground-truth answers. This
  is materially cheaper than judging each rep individually and lets the
  model use the question + GT set as joint context.
* **Cost control.** The matcher uses ``gemini-2.5-flash-lite`` at
  ``temperature=0`` with a hard cap on output tokens. Audit data
  (latency, token counts, raw response) is returned for every call.
* **Skip-when-trivial.** The function refuses to spend an API call when
  the cluster reps trivially fail to be candidates (no reps, no GT
  answers, or every rep already exact-substring-matches a GT -- the
  latter would have been caught by tier 1 of the heuristic).
* **Deterministic parsing.** A short regex extracts the chosen
  1-indexed cluster number from the model's response. ``0`` (or any
  unparseable text) maps back to ``-1`` (no match).

This module *does not* mutate the existing 7-tier pipeline -- it is
intended to be applied as a post-hoc rematch over cached
``unc_{dataset}_{model}_vne_{n}.json`` files via
``src/run_llm_judge_rematch.py``.
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

# Gemini Flash Lite via OpenAI-compatible chat endpoint. Pricing as of
# Verified 2026-11-04 (Gemini 2.5 Flash-Lite standard paid list rates):
#   input: $0.10 per 1M tokens   output: $0.40 per 1M tokens
# Source: https://ai.google.dev/gemini-api/docs/pricing#gemini-2.5-flash-lite
# Free-tier calls may cost nothing; this deliberately estimates paid usage.
GEMINI_INPUT_USD_PER_MTOK = 0.10
GEMINI_OUTPUT_USD_PER_MTOK = 0.40
GEMINI_BASE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/openai/"
)
DEFAULT_JUDGE_MODEL = "gemini-2.5-flash-lite"

# Borderline-zone bounds for embedding similarity tier-8 trigger.
BORDERLINE_LOW = 0.3
BORDERLINE_HIGH = 0.4


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------

def load_gemini_api_key(env_path: str | Path | None = None) -> str | None:
    """Load ``GEMINI_API_KEY`` from a ``.env`` file or process environment.

    Args:
        env_path: Explicit path to a ``.env`` file. When *None*, the
            workspace root is searched (``<repo>/.env``).

    Returns:
        The API key string, or *None* if no key is found.
    """
    # Try python-dotenv first (matches existing project convention).
    try:
        from dotenv import dotenv_values

        if env_path is None:
            env_path = Path(__file__).resolve().parents[1] / ".env"
        values = dotenv_values(str(env_path))
        val = values.get("GEMINI_API_KEY")
        if val:
            return val
    except ImportError:
        pass

    # Manual parse fallback.
    if env_path is None:
        env_path = Path(__file__).resolve().parents[1] / ".env"
    env_path = Path(env_path)
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("GEMINI_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            pass

    value = os.environ.get("GEMINI_API_KEY")
    return value.strip() if value else None


def make_gemini_client(api_key: str) -> Any:
    """Build an OpenAI-compatible client targeted at the Gemini endpoint.

    Args:
        api_key: A valid ``GEMINI_API_KEY`` string.

    Returns:
        A configured ``openai.OpenAI`` client instance.
    """
    from openai import OpenAI
    # Sourcing a CRLF .env in bash can retain a trailing carriage return.
    # Normalize credentials, not the scientific prompt or judging protocol.
    return OpenAI(api_key=api_key.strip(), base_url=GEMINI_BASE_URL)


# ---------------------------------------------------------------------------
# Borderline detection
# ---------------------------------------------------------------------------

def borderline_max_similarity(
    cluster_reps: list[str],
    gt_answers: list[str],
    encoder: Any,
) -> float:
    """Maximum cosine similarity between any (rep, GT) pair.

    Used to decide whether a sample falls in the ``borderline'' zone for
    LLM-judge tier 8 activation. Returns ``0.0`` for empty inputs.

    Args:
        cluster_reps: Cluster representative strings.
        gt_answers: Ground-truth answer strings.
        encoder: A ``SentenceEncoder`` (or any object with an
            ``encode(texts, normalize=True)`` method returning a numpy
            array of unit-normalised embeddings).

    Returns:
        The maximum cosine similarity over all (rep, GT) pairs.
    """
    import numpy as np

    if not cluster_reps or not gt_answers:
        return 0.0
    rep_embs = encoder.encode(cluster_reps, normalize=True)
    gt_embs = encoder.encode(gt_answers, normalize=True)
    sim = rep_embs @ gt_embs.T  # (n_clusters, n_gt)
    return float(sim.max())


def is_in_borderline_zone(
    max_sim: float,
    low: float = BORDERLINE_LOW,
    high: float = BORDERLINE_HIGH,
) -> bool:
    """Return True iff ``max_sim`` lies in the half-open interval ``[low, high)``."""
    return (max_sim >= low) and (max_sim < high)


# ---------------------------------------------------------------------------
# Trivial-skip helpers
# ---------------------------------------------------------------------------

def _trivial_exact_match_present(
    cluster_reps: list[str], gt_answers: list[str],
) -> bool:
    """True if some rep already trivially matches a GT (case-insensitive substring).

    This duplicates tier 1 of the heuristic. Borderline candidates by
    definition failed tier 1, so this is a defence-in-depth check that
    avoids burning API calls on items the heuristic actually matched
    (e.g., if a future caller forgets the borderline gate).
    """
    if not cluster_reps or not gt_answers:
        return False
    for rep in cluster_reps:
        rep_lc = rep.strip().lower()
        if not rep_lc:
            continue
        for gt in gt_answers:
            gt_lc = gt.strip().lower()
            if not gt_lc:
                continue
            if gt_lc in rep_lc or rep_lc in gt_lc:
                return True
    return False


# ---------------------------------------------------------------------------
# Prompt building & response parsing
# ---------------------------------------------------------------------------

# Match either "ANSWER: <num>" or a leading bare integer (with optional sign).
_ANSWER_RE = re.compile(
    r"(?:ANSWER\s*[:=]?\s*)?(-?\d+)", re.IGNORECASE,
)


def build_judge_prompt(
    question: str,
    gt_answers: list[str],
    cluster_reps: list[str],
    cluster_members: list[list[str]] | None = None,
) -> str:
    """Construct the judge prompt.

    The prompt presents the original question, the set of acceptable
    ground-truth answers, and the candidate cluster representatives
    (1-indexed). The model is asked to return the 1-indexed cluster
    number of the rep that conveys the same factual answer as one of
    the ground truths, or ``0`` if no rep does.

    Args:
        question: The original question string.
        gt_answers: Ground-truth answer strings (any one suffices).
        cluster_reps: Cluster representative answer strings.
        cluster_members: Optional list of lists of raw member answers
            per cluster. When provided, the first few members are
            appended to give the judge more signal about each cluster.

    Returns:
        The fully formatted prompt string.
    """
    gt_block = "\n".join(f"  - {gt!s}" for gt in gt_answers)
    cluster_lines: list[str] = []
    for i, rep in enumerate(cluster_reps, start=1):
        cluster_lines.append(f"  [{i}] {rep!s}")
        if cluster_members and i - 1 < len(cluster_members):
            members = cluster_members[i - 1]
            extras = [m for m in members if m != rep][:2]
            for extra in extras:
                cluster_lines.append(f"      (also: {extra!s})")
    cluster_block = "\n".join(cluster_lines)

    return (
        "You are evaluating short open-domain QA outputs.\n"
        "\n"
        "QUESTION:\n"
        f"  {question.strip()}\n"
        "\n"
        "ACCEPTABLE GROUND-TRUTH ANSWERS (any one suffices):\n"
        f"{gt_block}\n"
        "\n"
        "CANDIDATE CLUSTER ANSWERS:\n"
        f"{cluster_block}\n"
        "\n"
        "TASK: Identify which (if any) of the numbered candidate clusters\n"
        "answers the question with the same meaning as one of the\n"
        "ground-truth answers. The cluster wording may differ -- accept\n"
        "different phrasings, abbreviations, longer descriptions, or\n"
        "extra context, as long as the same entity, person, date, number,\n"
        "or fact is asserted. Reject candidates that name a different\n"
        "entity (e.g., 'gold' vs 'bronze', '1912' vs '1914').\n"
        "\n"
        "Respond with EXACTLY one line of the form:\n"
        "  ANSWER: <N>\n"
        "where N is the cluster number (1-indexed) that matches, or 0 if\n"
        "no candidate matches any of the ground-truth answers.\n"
    )


def parse_judge_response(text: str, n_clusters: int) -> int:
    """Extract the chosen 1-indexed cluster number from a judge response.

    Args:
        text: Raw model response text.
        n_clusters: Number of cluster reps presented to the model.

    Returns:
        ``-1`` if no match (or unparseable / out of range), otherwise the
        zero-indexed cluster index in ``[0, n_clusters)``.
    """
    if not text:
        return -1
    match = _ANSWER_RE.search(text)
    if match is None:
        return -1
    try:
        n = int(match.group(1))
    except (TypeError, ValueError):
        return -1
    if n <= 0 or n > n_clusters:
        return -1
    return n - 1


# ---------------------------------------------------------------------------
# Main judge entry point
# ---------------------------------------------------------------------------

def llm_judge_match(
    question: str,
    gt_answers: list[str],
    cluster_reps: list[str],
    cluster_members: list[list[str]] | None = None,
    *,
    client: Any | None = None,
    model: str = DEFAULT_JUDGE_MODEL,
    max_retries: int = 3,
    rate_delay: float = 1.0,
    skip_trivial: bool = True,
) -> tuple[int, dict[str, Any]]:
    """LLM-as-judge tier 8: question-aware cluster identification via Gemini.

    Sends a single chat-completion request to Gemini Flash Lite asking
    which cluster representative (if any) answers the question with the
    same meaning as a ground-truth answer.

    Args:
        question: The original question string.
        gt_answers: List of acceptable ground-truth answers.
        cluster_reps: Cluster representative answer strings, sorted by
            descending cluster proportion.
        cluster_members: Optional list of lists of raw member answers
            per cluster (used to enrich the prompt with up to 2 extra
            phrasings per cluster).
        client: An ``openai.OpenAI`` client targeted at the Gemini
            endpoint. When *None*, a client is built from
            ``GEMINI_API_KEY``.
        model: Model identifier (default ``gemini-2.5-flash-lite``).
        max_retries: Maximum retry attempts on transient API errors.
        rate_delay: Seconds to wait between retries (exponential
            backoff up to 60 s).
        skip_trivial: When True, return ``(-1, {...})`` immediately if a
            cluster rep already exact-substring-matches a GT (waste
            of API call) or if either reps or gt_answers is empty.

    Returns:
        A tuple ``(cluster_idx, audit)`` where:
          * ``cluster_idx`` is the zero-indexed cluster index in
            ``[0, len(cluster_reps))`` if the judge identified a
            matching cluster, else ``-1``.
          * ``audit`` is a dict with keys
            ``judge_response``, ``prompt_tokens``, ``output_tokens``,
            ``latency_ms``, ``model``, ``called`` (False if the call
            was skipped), and ``error`` (None on success).
    """
    audit: dict[str, Any] = {
        "judge_response": None,
        "prompt_tokens": 0,
        "output_tokens": 0,
        "latency_ms": 0.0,
        "model": model,
        "called": False,
        "error": None,
    }

    if not cluster_reps or not gt_answers:
        audit["error"] = "empty_inputs"
        return -1, audit

    if skip_trivial and _trivial_exact_match_present(cluster_reps, gt_answers):
        # Defence-in-depth: caller should already have gated on this via
        # the borderline zone, but if not, do not waste an API call.
        audit["error"] = "trivial_match_present"
        return -1, audit

    if client is None:
        api_key = load_gemini_api_key()
        if not api_key:
            audit["error"] = "no_api_key"
            return -1, audit
        client = make_gemini_client(api_key)

    prompt = build_judge_prompt(
        question, gt_answers, cluster_reps, cluster_members,
    )

    last_err: str | None = None
    for attempt in range(max_retries):
        t0 = time.perf_counter()
        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 24,
        }
        # Thinking models (e.g. gemini-2.5-flash) spend the token budget on
        # invisible reasoning and return empty text at max_tokens=24; setting
        # CAELUM_JUDGE_REASONING_EFFORT=none disables thinking via the
        # OpenAI-compat layer and leaves room for the visible verdict.
        effort = os.environ.get("CAELUM_JUDGE_REASONING_EFFORT", "").strip()
        if effort:
            create_kwargs["reasoning_effort"] = effort
            create_kwargs["max_tokens"] = 96
        try:
            resp = client.chat.completions.create(**create_kwargs)
        except Exception as exc:  # noqa: BLE001 - transient API errors
            err = str(exc)
            last_err = err
            wait = min(2 ** attempt * rate_delay, 60.0)
            # Try to honour explicit retry hints embedded in API errors.
            m = re.search(r"retry in ([\d.]+)s", err, re.IGNORECASE)
            if m:
                wait = max(float(m.group(1)) + 1.0, wait)
            elif "429" in err or "rate" in err.lower():
                wait = max(wait, 5.0)
            elif "500" in err or "503" in err or "504" in err:
                wait = max(wait, 2.0)
            logger.debug(
                "llm_judge_match transient error (attempt %d/%d): %s ; "
                "sleeping %.1fs", attempt + 1, max_retries, err[:140], wait,
            )
            time.sleep(wait)
            continue

        latency_ms = (time.perf_counter() - t0) * 1000.0
        audit["called"] = True
        audit["latency_ms"] = latency_ms

        text = resp.choices[0].message.content if resp.choices else None
        audit["judge_response"] = text
        usage = getattr(resp, "usage", None)
        if usage is not None:
            audit["prompt_tokens"] = int(getattr(usage, "prompt_tokens", 0) or 0)
            audit["output_tokens"] = int(
                getattr(usage, "completion_tokens", 0) or 0
            )

        cidx = parse_judge_response(text or "", n_clusters=len(cluster_reps))
        return cidx, audit

    # All retries exhausted.
    audit["error"] = f"max_retries_exceeded: {last_err}"
    return -1, audit


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def estimate_cost_usd(prompt_tokens: int, output_tokens: int) -> float:
    """Convert token counts to USD using current Gemini Flash Lite list rates.

    Args:
        prompt_tokens: Cumulative input/prompt tokens.
        output_tokens: Cumulative output/completion tokens.

    Returns:
        Estimated cost in USD.
    """
    in_cost = (prompt_tokens / 1_000_000.0) * GEMINI_INPUT_USD_PER_MTOK
    out_cost = (output_tokens / 1_000_000.0) * GEMINI_OUTPUT_USD_PER_MTOK
    return float(in_cost + out_cost)
