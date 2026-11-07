#!/usr/bin/env python3
"""Download and convert HuggingFace datasets to CAELUM's eval JSON format.

Converts each source dataset into the standardised JSON schemas expected by
the CAELUM pipeline (see ``src/src/utils.py`` loaders). Outputs are written
to ``data/eval/`` by default.

Supported datasets:
    - TriviaQA (factual, unambiguous)        — mandarjoshi/trivia_qa
    - NQ-Open  (factual, unambiguous)        — google-research-datasets/nq_open
    - AmbigQA  (ambiguous + unambiguous)     — sewon/ambig_qa
    - SimpleQA (factual, unambiguous)        — basicv8vc/SimpleQA
    - PopQA    (factual, popularity-graded)  — akariasai/PopQA
    - CondAmbigQA (ambiguous, conditional)   — Apocalypse-AGI-DAO/CondAmbigQA-2K

Note:
    AmbigInst (304 samples) is **already available** in the repository at:
    ``data/eval/AmbigInst.json``
    and does not need downloading.

Usage:
    python src/prepare_datasets.py
    python src/prepare_datasets.py --datasets triviaqa,ambigqa
    python src/prepare_datasets.py --n_triviaqa 500 --dry_run
"""

import argparse
import ast
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("prepare_datasets")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "eval"

# Default sample counts per dataset.
DEFAULT_N_TRIVIAQA = 2000
DEFAULT_N_NQOPEN = 1000
DEFAULT_N_AMBIGQA = 1000
DEFAULT_N_SIMPLEQA = None  # Take all (~4326)
DEFAULT_N_POPQA = 2000
DEFAULT_N_CONDAMBIGQA = None  # Take all (~2000)

ALL_DATASETS = [
    "triviaqa", "nqopen", "ambigqa", "simpleqa", "popqa", "condambigqa",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _subsample(
    dataset_list: list,
    n_samples: int | None,
    seed: int = 42,
) -> list:
    """Randomly subsample a list to *n_samples* entries.

    Args:
        dataset_list: The full list of entries.
        n_samples: Desired number of samples. If None or >= len(dataset_list),
            returns the full list unchanged.
        seed: Random seed for reproducible sampling.

    Returns:
        A subsampled list (order preserved by sorted indices).
    """
    if n_samples is None or n_samples >= len(dataset_list):
        return dataset_list
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset_list), size=n_samples, replace=False)
    return [dataset_list[i] for i in sorted(indices)]


def _save_json(data: list, path: Path) -> None:
    """Write a list of dicts as a pretty-printed JSON file.

    Args:
        data: The list of dictionaries to serialize.
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d entries to %s", len(data), path)


# ---------------------------------------------------------------------------
# TriviaQA
# ---------------------------------------------------------------------------

def prepare_triviaqa(
    output_path: Path,
    n_samples: int | None = DEFAULT_N_TRIVIAQA,
    seed: int = 42,
) -> int:
    """Download TriviaQA (rc.nocontext) and convert to CAELUM format.

    HuggingFace source: ``mandarjoshi/trivia_qa``, config ``rc.nocontext``,
    split ``validation``.

    Output schema per entry:
        question_id (int), datapoint_id (int), question (str), gt_answer (str)

    Args:
        output_path: Path to the output JSON file.
        n_samples: Number of samples to keep (None = all).
        seed: Random seed for subsampling.

    Returns:
        Number of entries written.
    """
    from datasets import load_dataset

    logger.info("Downloading TriviaQA (rc.nocontext, validation)...")
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    logger.info("  Raw size: %d entries", len(ds))

    all_entries = []
    for i, item in enumerate(ds):
        answer = item["answer"]
        # Use normalized_value as primary; fall back to value.
        gt_answer = answer.get("normalized_value") or answer.get("value", "")

        # Collect all answer aliases for robust correctness matching.
        # HuggingFace TriviaQA provides:
        #   - aliases: list of answer surface forms
        #   - normalized_aliases: lowercased/cleaned aliases
        aliases = answer.get("aliases", [])
        normalized_aliases = answer.get("normalized_aliases", [])

        # Build deduplicated alias list (case-insensitive).
        all_aliases = []
        seen = set()
        # Primary answer first
        if gt_answer:
            seen.add(gt_answer.strip().lower())
            all_aliases.append(gt_answer)
        # Then raw aliases
        for a in aliases:
            if a and a.strip().lower() not in seen:
                seen.add(a.strip().lower())
                all_aliases.append(a.strip())
        # Then normalized aliases (catch any not already covered)
        for a in normalized_aliases:
            if a and a.strip().lower() not in seen:
                seen.add(a.strip().lower())
                all_aliases.append(a.strip())

        entry = {
            "question_id": i,
            "datapoint_id": i,
            "question": item["question"],
            "gt_answer": gt_answer,
        }
        # Only include gt_aliases if there are additional forms beyond primary
        if len(all_aliases) > 1:
            entry["gt_aliases"] = all_aliases

        all_entries.append(entry)

    entries = _subsample(all_entries, n_samples, seed)

    # Re-index question_id after subsampling.
    for idx, entry in enumerate(entries):
        entry["question_id"] = idx

    _save_json(entries, output_path)
    return len(entries)


# ---------------------------------------------------------------------------
# NQ-Open
# ---------------------------------------------------------------------------

def prepare_nqopen(
    output_path: Path,
    n_samples: int | None = DEFAULT_N_NQOPEN,
    seed: int = 42,
) -> int:
    """Download NQ-Open and convert to CAELUM OpenNQ format.

    HuggingFace source: ``google-research-datasets/nq_open``,
    split ``validation``.

    Output schema per entry:
        question_id (int), datapoint_id (int), question (str),
        gt_answer (str), annotation_idx (int), annotation_type (str),
        short_answer_idx (int)

    Since NQ-Open may have multiple acceptable answers, we take the first
    answer as ``gt_answer`` and emit one row per question. The full answer
    list is stored in ``all_answers`` for downstream use.

    Args:
        output_path: Path to the output JSON file.
        n_samples: Number of samples to keep (None = all).
        seed: Random seed for subsampling.

    Returns:
        Number of entries written.
    """
    from datasets import load_dataset

    logger.info("Downloading NQ-Open (validation)...")
    ds = load_dataset("google-research-datasets/nq_open", split="validation")
    logger.info("  Raw size: %d entries", len(ds))

    all_entries = []
    for i, item in enumerate(ds):
        answers = item["answer"]  # list of str
        gt_answer = answers[0] if answers else ""

        entry = {
            "question_id": i,
            "datapoint_id": i,
            "question": item["question"],
            "gt_answer": gt_answer,
            "annotation_idx": 0,
            "annotation_type": "short_answer",
            "short_answer_idx": 0,
        }
        # NQ-Open may have multiple acceptable answers; store them all.
        if len(answers) > 1:
            entry["gt_aliases"] = list(dict.fromkeys(answers))  # deduplicate

        all_entries.append(entry)

    entries = _subsample(all_entries, n_samples, seed)

    for idx, entry in enumerate(entries):
        entry["question_id"] = idx

    _save_json(entries, output_path)
    return len(entries)


# ---------------------------------------------------------------------------
# AmbigQA
# ---------------------------------------------------------------------------

def _extract_ambigqa_entry(
    item: dict,
    question_id: int,
) -> dict:
    """Convert a single HuggingFace AmbigQA record to CAELUM format.

    The annotations field may contain multiple annotator records. We merge
    them, preferring ``multipleQAs`` annotations (ambiguous) over
    ``singleAnswer`` ones. If any annotator marked the question as ambiguous,
    we treat it as ambiguous.

    Args:
        item: A single record from the HuggingFace AmbigQA dataset.
        question_id: Sequential ID to assign.

    Returns:
        A dictionary in CAELUM AmbigQA format.
    """
    annotations = item["annotations"]
    types = annotations["type"]        # list of str
    answers = annotations["answer"]    # list of list (for singleAnswer)
    qa_pairs = annotations["qaPairs"]  # list of dict

    is_ambig = False
    gt_clarifications: list[str] = []
    gt_answers: list[list[str]] = []
    gold_standard_answer: list[str] | None = None

    for ann_idx, ann_type in enumerate(types):
        if ann_type == "multipleQAs":
            is_ambig = True
            # Extract from qaPairs at this annotation index.
            qa_pair = qa_pairs[ann_idx]
            questions = qa_pair["question"]  # list of str
            pair_answers = qa_pair["answer"]  # list of list of str
            for q, a in zip(questions, pair_answers):
                if q and q not in gt_clarifications:
                    gt_clarifications.append(q)
                    gt_answers.append(a if isinstance(a, list) else [a])
        elif ann_type == "singleAnswer":
            # Collect single answers (used if no multipleQAs found).
            if not is_ambig and answers[ann_idx]:
                if gold_standard_answer is None:
                    gold_standard_answer = []
                for a in answers[ann_idx]:
                    if a and a not in gold_standard_answer:
                        gold_standard_answer.append(a)

    if is_ambig:
        return {
            "question_id": question_id,
            "question": item["question"],
            "is_ambig": True,
            "gold_standard_answer": None,
            "gt_clarifications": gt_clarifications,
            "gt_answers": gt_answers,
        }
    else:
        return {
            "question_id": question_id,
            "question": item["question"],
            "is_ambig": False,
            "gold_standard_answer": gold_standard_answer,
            "gt_clarifications": None,
            "gt_answers": None,
        }


def prepare_ambigqa(
    output_path: Path,
    n_samples: int | None = DEFAULT_N_AMBIGQA,
    seed: int = 42,
) -> int:
    """Download AmbigQA and convert to CAELUM format.

    HuggingFace source: ``sewon/ambig_qa``, config ``full``,
    split ``validation``.

    Output schema per entry matches the existing AmbigQA.json:
        question_id (int), question (str), is_ambig (bool),
        gold_standard_answer (list[str] | null),
        gt_clarifications (list[str] | null),
        gt_answers (list[list[str]] | null)

    Args:
        output_path: Path to the output JSON file.
        n_samples: Number of samples to keep (None = all).
        seed: Random seed for subsampling.

    Returns:
        Number of entries written.
    """
    from datasets import load_dataset

    logger.info("Downloading AmbigQA (full, validation)...")
    ds = load_dataset("sewon/ambig_qa", "full", split="validation")
    logger.info("  Raw size: %d entries", len(ds))

    all_entries = []
    for i, item in enumerate(ds):
        entry = _extract_ambigqa_entry(item, question_id=i)
        all_entries.append(entry)

    entries = _subsample(all_entries, n_samples, seed)

    for idx, entry in enumerate(entries):
        entry["question_id"] = idx

    _save_json(entries, output_path)

    n_ambig = sum(1 for e in entries if e["is_ambig"])
    logger.info("  Ambiguous: %d / %d (%.1f%%)",
                n_ambig, len(entries), 100.0 * n_ambig / max(len(entries), 1))
    return len(entries)


# ---------------------------------------------------------------------------
# SimpleQA
# ---------------------------------------------------------------------------

def prepare_simpleqa(
    output_path: Path,
    n_samples: int | None = DEFAULT_N_SIMPLEQA,
    seed: int = 42,
) -> int:
    """Download SimpleQA and convert to CAELUM TriviaQA-like format.

    HuggingFace source: ``basicv8vc/SimpleQA``, split ``test``.

    Output schema per entry:
        question_id (int), datapoint_id (int), question (str),
        gt_answer (str), topic (str)

    Args:
        output_path: Path to the output JSON file.
        n_samples: Number of samples to keep (None = all).
        seed: Random seed for subsampling.

    Returns:
        Number of entries written.
    """
    from datasets import load_dataset

    logger.info("Downloading SimpleQA (test)...")
    ds = load_dataset("basicv8vc/SimpleQA", split="test")
    logger.info("  Raw size: %d entries", len(ds))

    all_entries = []
    for i, item in enumerate(ds):
        # metadata is a Python repr string (single quotes), not valid JSON.
        topic = ""
        try:
            meta = ast.literal_eval(item["metadata"]) if item["metadata"] else {}
            topic = meta.get("topic", "")
        except (ValueError, SyntaxError, TypeError):
            pass

        all_entries.append({
            "question_id": i,
            "datapoint_id": i,
            "question": item["problem"],
            "gt_answer": item["answer"],
            "topic": topic,
        })

    entries = _subsample(all_entries, n_samples, seed)

    for idx, entry in enumerate(entries):
        entry["question_id"] = idx

    _save_json(entries, output_path)
    return len(entries)


# ---------------------------------------------------------------------------
# PopQA
# ---------------------------------------------------------------------------

def prepare_popqa(
    output_path: Path,
    n_samples: int | None = DEFAULT_N_POPQA,
    seed: int = 42,
) -> int:
    """Download PopQA and convert to CAELUM TriviaQA-like format with metadata.

    HuggingFace source: ``akariasai/PopQA``, split ``test``.

    Output schema per entry:
        question_id (int), datapoint_id (int), question (str),
        gt_answer (str), s_pop (int), prop (str)

    Args:
        output_path: Path to the output JSON file.
        n_samples: Number of samples to keep (None = all).
        seed: Random seed for subsampling.

    Returns:
        Number of entries written.
    """
    from datasets import load_dataset

    logger.info("Downloading PopQA (test)...")
    ds = load_dataset("akariasai/PopQA", split="test")
    logger.info("  Raw size: %d entries", len(ds))

    all_entries = []
    for i, item in enumerate(ds):
        # possible_answers is a JSON-encoded list string in HF
        possible = item.get("possible_answers")
        if isinstance(possible, str):
            try:
                possible = json.loads(possible)
            except (json.JSONDecodeError, TypeError):
                possible = [item["obj"]]
        if not possible:
            possible = [item["obj"]]

        all_entries.append({
            "question_id": i,
            "datapoint_id": i,
            "question": item["question"],
            "gt_answer": item["obj"],
            "possible_answers": possible,
            "s_pop": item["s_pop"],
            "prop": item["prop"],
        })

    entries = _subsample(all_entries, n_samples, seed)

    for idx, entry in enumerate(entries):
        entry["question_id"] = idx

    _save_json(entries, output_path)
    return len(entries)


# ---------------------------------------------------------------------------
# CondAmbigQA
# ---------------------------------------------------------------------------

def prepare_condambigqa(
    output_path: Path,
    n_samples: int | None = DEFAULT_N_CONDAMBIGQA,
    seed: int = 42,
) -> int:
    """Download CondAmbigQA and convert to CAELUM AmbigQA-like format.

    HuggingFace source: ``Apocalypse-AGI-DAO/CondAmbigQA-2K``,
    split ``train``.

    All entries are treated as ambiguous. The ``properties`` list provides
    condition-grounded interpretations: each property's ``condition`` becomes
    a clarification and ``groundtruth`` becomes the corresponding answer.

    Output schema per entry:
        question_id (int), question (str), is_ambig (bool),
        gold_standard_answer (null),
        gt_clarifications (list[str]),
        gt_answers (list[list[str]])

    Args:
        output_path: Path to the output JSON file.
        n_samples: Number of samples to keep (None = all).
        seed: Random seed for subsampling.

    Returns:
        Number of entries written.
    """
    from datasets import load_dataset

    logger.info("Downloading CondAmbigQA-2K (train)...")
    ds = load_dataset("Apocalypse-AGI-DAO/CondAmbigQA-2K", split="train")
    logger.info("  Raw size: %d entries", len(ds))

    all_entries = []
    for i, item in enumerate(ds):
        properties = item["properties"]
        gt_clarifications = []
        gt_answers = []
        for prop in properties:
            condition = prop.get("condition", "")
            groundtruth = prop.get("groundtruth", "")
            gt_clarifications.append(condition)
            # groundtruth is a single string; wrap in list for consistency.
            if isinstance(groundtruth, list):
                gt_answers.append(groundtruth)
            else:
                gt_answers.append([groundtruth] if groundtruth else [""])

        all_entries.append({
            "question_id": i,
            "question": item["question"],
            "is_ambig": True,
            "gold_standard_answer": None,
            "gt_clarifications": gt_clarifications,
            "gt_answers": gt_answers,
        })

    entries = _subsample(all_entries, n_samples, seed)

    for idx, entry in enumerate(entries):
        entry["question_id"] = idx

    _save_json(entries, output_path)
    return len(entries)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and run dataset preparation."""
    parser = argparse.ArgumentParser(
        description="Download and convert HuggingFace datasets to CAELUM eval JSON format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/prepare_datasets.py\n"
            "  python src/prepare_datasets.py --datasets triviaqa,ambigqa\n"
            "  python src/prepare_datasets.py --n_triviaqa 500 --dry_run\n"
        ),
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for JSON files (default: data/eval).",
    )
    parser.add_argument(
        "--datasets", type=str, default="all",
        help=(
            "Comma-separated list of datasets to prepare. "
            f"Options: {', '.join(ALL_DATASETS)}, all. Default: all."
        ),
    )
    parser.add_argument("--n_triviaqa", type=int, default=DEFAULT_N_TRIVIAQA,
                        help="Number of TriviaQA samples (default: 2000).")
    parser.add_argument("--n_nqopen", type=int, default=DEFAULT_N_NQOPEN,
                        help="Number of NQ-Open samples (default: 1000).")
    parser.add_argument("--n_ambigqa", type=int, default=DEFAULT_N_AMBIGQA,
                        help="Number of AmbigQA samples (default: 1000).")
    parser.add_argument("--n_simpleqa", type=int, default=0,
                        help="Number of SimpleQA samples (0 = all, default: 0).")
    parser.add_argument("--n_popqa", type=int, default=DEFAULT_N_POPQA,
                        help="Number of PopQA samples (default: 2000).")
    parser.add_argument("--n_condambigqa", type=int, default=0,
                        help="Number of CondAmbigQA samples (0 = all, default: 0).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for subsampling (default: 42).")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would be downloaded without actually doing it.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace output files that already exist (refused by default).")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Resolve which datasets to prepare.
    if args.datasets.lower() == "all":
        requested = ALL_DATASETS
    else:
        requested = [d.strip().lower() for d in args.datasets.split(",")]
        for d in requested:
            if d not in ALL_DATASETS:
                parser.error(f"Unknown dataset '{d}'. Choose from: {ALL_DATASETS}")

    # Map dataset name -> (prepare function, output filename, sample count).
    n_counts: dict[str, int | None] = {
        "triviaqa": args.n_triviaqa,
        "nqopen": args.n_nqopen,
        "ambigqa": args.n_ambigqa,
        "simpleqa": args.n_simpleqa if args.n_simpleqa > 0 else None,
        "popqa": args.n_popqa,
        "condambigqa": args.n_condambigqa if args.n_condambigqa > 0 else None,
    }

    prepare_fns = {
        "triviaqa": (prepare_triviaqa, "TriviaQA.json"),
        "nqopen": (prepare_nqopen, "OpenNQ.json"),
        "ambigqa": (prepare_ambigqa, "AmbigQA.json"),
        "simpleqa": (prepare_simpleqa, "SimpleQA.json"),
        "popqa": (prepare_popqa, "PopQA.json"),
        "condambigqa": (prepare_condambigqa, "CondAmbigQA.json"),
    }

    # Print plan.
    logger.info("=" * 60)
    logger.info("CAELUM Dataset Preparation")
    logger.info("=" * 60)
    logger.info("Output directory: %s", output_dir)
    logger.info("Seed: %d", args.seed)
    logger.info("")
    logger.info("%-15s  %-10s  %s", "Dataset", "N samples", "Output file")
    logger.info("-" * 60)
    for name in requested:
        fn, filename = prepare_fns[name]
        n = n_counts[name]
        n_str = str(n) if n is not None else "all"
        logger.info("%-15s  %-10s  %s", name, n_str, output_dir / filename)
    logger.info("")

    if args.dry_run:
        logger.info("[DRY RUN] No downloads performed.")
        return

    # data/eval/ holds the curated TriviaQA/AmbigQA/OpenNQ subsets (300/200/300)
    # that every cached generation was built from; this script's default sizes
    # produce different, larger question sets. Never replace them silently.
    existing = [output_dir / prepare_fns[name][1] for name in requested
                if (output_dir / prepare_fns[name][1]).exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            "Refusing to overwrite existing evaluation files (cached generations depend on them):\n  "
            + "\n  ".join(str(p) for p in existing)
            + "\nUse --output_dir for a new location, or --overwrite to replace them.")

    # Create output directory.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare each dataset.
    results: list[tuple[str, int, str]] = []
    total_start = time.time()

    for name in requested:
        fn, filename = prepare_fns[name]
        out_path = output_dir / filename
        n = n_counts[name]

        logger.info("-" * 40)
        logger.info("Preparing: %s", name)
        start = time.time()
        count = fn(out_path, n_samples=n, seed=args.seed)
        elapsed = time.time() - start
        results.append((name, count, str(out_path)))
        logger.info("  Done in %.1fs", elapsed)

    total_elapsed = time.time() - total_start

    # Print summary table.
    logger.info("")
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info("%-15s  %8s  %s", "Dataset", "Samples", "Path")
    logger.info("-" * 60)
    for name, count, path in results:
        logger.info("%-15s  %8d  %s", name, count, path)
    logger.info("-" * 60)
    logger.info("Total time: %.1fs", total_elapsed)
    logger.info("")
    logger.info(
        "Note: AmbigInst (304 samples) is already available at:\n"
        "  data/eval/AmbigInst.json"
    )


if __name__ == "__main__":
    main()
