"""Align transfer labels by dataset and record ID, preserving repeated text."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable


REFERENCE_FIELDS = ("gt_answer", "gold_standard_answer", "gt_answers")


class TransferAlignmentError(ValueError):
    """The two model corpora do not describe the same evaluation records."""


@dataclass(frozen=True)
class TransferRecord:
    dataset: str
    record_id: str
    question: str
    reference_signature: str
    label: int

    @property
    def identity(self) -> tuple[str, str]:
        return self.dataset, self.record_id


def records_from_items(
    dataset: str, items: Iterable[dict[str, Any]],
) -> list[TransferRecord]:
    """Read dataset-scoped row IDs and the reference metadata to verify later.

    List position is deliberately not an ID: model files may be shuffled.
    Every record must retain its original question_id or datapoint_id.
    """
    if not dataset:
        raise TransferAlignmentError("A dataset namespace is required")
    records = []
    for index, item in enumerate(items):
        record_id = item.get("question_id")
        if record_id is None:
            record_id = item.get("datapoint_id")
        if record_id is None:
            raise TransferAlignmentError(
                f"{dataset} row {index}: missing question_id/datapoint_id"
            )
        question = item.get("question")
        if not isinstance(question, str) or not question:
            raise TransferAlignmentError(
                f"{dataset} record {record_id}: missing question text"
            )
        references = {field: item.get(field) for field in REFERENCE_FIELDS}
        records.append(TransferRecord(
            dataset=dataset,
            record_id=str(record_id),
            question=question,
            reference_signature=json.dumps(
                references, sort_keys=True, ensure_ascii=False,
            ),
            label=int(int(item.get("correct_cluster_idx", -1)) >= 0),
        ))
    return records


def load_transfer_records(path: str | Path, dataset: str) -> list[TransferRecord]:
    with Path(path).open(encoding="utf-8") as stream:
        return records_from_items(dataset, json.load(stream))


def _index_records(
    records: list[TransferRecord], side: str,
) -> dict[tuple[str, str], TransferRecord]:
    indexed = {}
    for record in records:
        if record.identity in indexed:
            raise TransferAlignmentError(
                f"Duplicate {side} record identity: {record.identity!r}"
            )
        indexed[record.identity] = record
    return indexed


def align_transfer_records(
    source: list[TransferRecord], target: list[TransferRecord],
) -> tuple[list[str], list[int], list[int]]:
    """Return questions and per-record labels in target order.

    Require the same record set, question text, and reference/alias metadata.
    Reusing features for identical text is safe; replacing its labels is not.
    """
    source_by_id = _index_records(source, "source")
    target_by_id = _index_records(target, "target")
    if source_by_id.keys() != target_by_id.keys():
        source_only = sorted(source_by_id.keys() - target_by_id.keys())
        target_only = sorted(target_by_id.keys() - source_by_id.keys())
        raise TransferAlignmentError(
            "Different transfer record identities: "
            f"source-only={source_only[:5]!r}, target-only={target_only[:5]!r}"
        )
    questions, source_labels, target_labels = [], [], []
    for target_record in target:
        source_record = source_by_id[target_record.identity]
        if source_record.question != target_record.question:
            raise TransferAlignmentError(
                f"Question mismatch for {target_record.identity!r}"
            )
        if source_record.reference_signature != target_record.reference_signature:
            raise TransferAlignmentError(
                f"Reference/alias mismatch for {target_record.identity!r}"
            )
        questions.append(target_record.question)
        source_labels.append(source_record.label)
        target_labels.append(target_record.label)
    return questions, source_labels, target_labels
