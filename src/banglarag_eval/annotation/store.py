"""JSONL-backed annotation store.

Loads records from a JSONL dataset, provides sanitized views for
annotators (hiding provenance and condition metadata), appends
annotations with schema validation, and supports adjudication.

The store is the single point of truth for read/write of the dataset
file during annotation. It validates every save against the canonical
schema + cross-field rules, so invalid annotations can never corrupt
the dataset.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from banglarag_eval.constants import (
    ADJUDICATION_METHODS,
    FAITHFULNESS_CATEGORIES,
    MIN_INDEPENDENT_ANNOTATORS,
)
from banglarag_eval.dataset import save_dataset
from banglarag_eval.schema import validate_record

# Fields that annotators must NOT see (per docs/annotation_guidelines.md).
_HIDDEN_FIELDS = frozenset({
    "source_text",
    "source_span",
    "intended_answer",
    "evidence_condition",
    "evidence_condition_notes",
    "corruption_metadata",
    "annotations",
    "adjudication",
    "faithfulness_category",
    "hallucination_type",
    "evaluator_outputs",
    "question_generation",
    "data_origin",
    "document_id",
    "source_id",
    "retrieval_configuration",
    "retriever",
    "embedding_model",
    "reranker",
    "generator_model",
    "generator_version",
    "generation_settings",
    "prompt_version",
    "cost",
    "latency",
    "random_seed",
    "dataset_version",
    "schema_version",
    "timestamp",
    "notes",
})


class AnnotationStore:
    """JSONL-backed store for annotation and adjudication."""

    def __init__(self, dataset_path: str | Path) -> None:
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"dataset not found: {self.dataset_path}")
        self._records: list[dict[str, Any]] = self._load_raw()

    def _load_raw(self) -> list[dict[str, Any]]:
        """Load records without validation (we validate on save)."""
        records: list[dict[str, Any]] = []
        with open(self.dataset_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def _save(self) -> None:
        """Validate every record and write back to JSONL."""
        save_dataset(self._records, self.dataset_path, overwrite=True, validate=True)

    def reload(self) -> None:
        """Re-read from disk (pick up changes made by other annotators)."""
        self._records = self._load_raw()

    def get_record(self, example_id: str) -> dict[str, Any] | None:
        """Return the full record for example_id, or None."""
        for record in self._records:
            if record.get("example_id") == example_id:
                return record
        return None

    def get_annotation_view(self, example_id: str) -> dict[str, Any] | None:
        """Return a sanitized record view for the annotator.

        Shows only: example_id, question, question_language,
        retrieved_context, generated_answer, answer_claims.
        Everything else is stripped per the annotation guidelines.
        """
        record = self.get_record(example_id)
        if record is None:
            return None
        return {
            "example_id": record["example_id"],
            "question": record["question"],
            "question_language": record.get("question_language"),
            "retrieved_context": record.get("retrieved_context"),
            "generated_answer": record.get("generated_answer"),
            "answer_claims": record.get("answer_claims"),
        }

    def get_pending_ids(self, annotator_id: str) -> list[str]:
        """Example IDs this annotator hasn't labeled yet and that have
        a generated answer to judge."""
        pending: list[str] = []
        for record in self._records:
            if not record.get("generated_answer"):
                continue
            existing = {
                a["annotator_id"]
                for a in record.get("annotations", [])
            }
            if annotator_id not in existing:
                pending.append(record["example_id"])
        return pending

    def get_completed_ids(self, annotator_id: str) -> list[str]:
        """Example IDs this annotator has already labeled."""
        completed: list[str] = []
        for record in self._records:
            existing = {
                a["annotator_id"]
                for a in record.get("annotations", [])
            }
            if annotator_id in existing:
                completed.append(record["example_id"])
        return completed

    def add_annotation(
        self,
        example_id: str,
        annotator_id: str,
        label: str,
        explanation: str,
    ) -> dict[str, Any]:
        """Append an annotation to the record, validate, and save.

        Returns the annotation dict that was added.
        Raises ValueError if the label is invalid, the explanation is
        empty, the annotator already labeled this record, or validation
        fails.
        """
        if label not in FAITHFULNESS_CATEGORIES:
            raise ValueError(
                f"invalid label '{label}'; must be one of {FAITHFULNESS_CATEGORIES}"
            )
        if not explanation or not explanation.strip():
            raise ValueError("explanation is required and must not be empty")

        record = self.get_record(example_id)
        if record is None:
            raise ValueError(f"record '{example_id}' not found")

        existing_ids = {
            a["annotator_id"] for a in record.get("annotations", [])
        }
        if annotator_id in existing_ids:
            raise ValueError(
                f"annotator '{annotator_id}' has already labeled '{example_id}'"
            )

        annotation = {
            "annotator_id": annotator_id,
            "label": label,
            "explanation": explanation.strip(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        record.setdefault("annotations", []).append(annotation)
        self._save()
        return annotation

    def get_disagreements(self) -> list[dict[str, Any]]:
        """Records with >= 2 annotations whose labels are not unanimous."""
        disagreements: list[dict[str, Any]] = []
        for record in self._records:
            annotations = record.get("annotations", [])
            if len(annotations) < MIN_INDEPENDENT_ANNOTATORS:
                continue
            labels = {a["label"] for a in annotations}
            if len(labels) > 1 and record.get("adjudication") is None:
                disagreements.append({
                    "example_id": record["example_id"],
                    "question": record["question"],
                    "labels": sorted(labels),
                    "annotations": annotations,
                })
        return disagreements

    def get_adjudication_view(self, example_id: str) -> dict[str, Any] | None:
        """Return a view for adjudication: annotator-visible fields
        plus the disagreeing annotations and their explanations."""
        record = self.get_record(example_id)
        if record is None:
            return None
        view = self.get_annotation_view(example_id)
        if view is None:
            return None
        view["annotations"] = record.get("annotations", [])
        view["adjudication"] = record.get("adjudication")
        return view

    def add_adjudication(
        self,
        example_id: str,
        adjudicator_id: str,
        label: str,
        explanation: str,
        method: str = "third_rater",
    ) -> dict[str, Any]:
        """Add adjudication, set gold label, validate, and save.

        Raises ValueError if: label/method invalid, explanation empty,
        record not found, record has < 2 annotations, adjudication
        already exists, or a third_rater adjudicator is also an
        annotator (schema rule R8).
        """
        if label not in FAITHFULNESS_CATEGORIES:
            raise ValueError(
                f"invalid label '{label}'; must be one of {FAITHFULNESS_CATEGORIES}"
            )
        if method not in ADJUDICATION_METHODS:
            raise ValueError(
                f"invalid method '{method}'; must be one of {ADJUDICATION_METHODS}"
            )
        if not explanation or not explanation.strip():
            raise ValueError("explanation is required and must not be empty")

        record = self.get_record(example_id)
        if record is None:
            raise ValueError(f"record '{example_id}' not found")

        annotations = record.get("annotations", [])
        if len(annotations) < MIN_INDEPENDENT_ANNOTATORS:
            raise ValueError(
                f"adjudication requires >= {MIN_INDEPENDENT_ANNOTATORS} annotations, "
                f"found {len(annotations)}"
            )

        if record.get("adjudication") is not None:
            raise ValueError(
                f"record '{example_id}' is already adjudicated"
            )

        annotator_ids = {a["annotator_id"] for a in annotations}
        if method == "third_rater" and adjudicator_id in annotator_ids:
            raise ValueError(
                f"third_rater adjudicator '{adjudicator_id}' must not be one of "
                f"the annotators {sorted(annotator_ids)} (schema rule R8)"
            )

        adjudication = {
            "label": label,
            "explanation": explanation.strip(),
            "adjudicator_id": adjudicator_id,
            "method": method,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        record["adjudication"] = adjudication
        record["faithfulness_category"] = label
        self._save()
        return adjudication

    def get_stats(self) -> dict[str, Any]:
        """Progress statistics for the dashboard."""
        total = len(self._records)
        annotatable = sum(
            1 for r in self._records if r.get("generated_answer")
        )
        annotated_once = sum(
            1 for r in self._records
            if len(r.get("annotations", [])) >= 1
        )
        annotated_twice = sum(
            1 for r in self._records
            if len(r.get("annotations", [])) >= MIN_INDEPENDENT_ANNOTATORS
        )
        adjudicated = sum(
            1 for r in self._records if r.get("adjudication") is not None
        )
        disagreements = len(self.get_disagreements())
        return {
            "total": total,
            "annotatable": annotatable,
            "annotated_once": annotated_once,
            "annotated_twice": annotated_twice,
            "adjudicated": adjudicated,
            "pending_adjudication": disagreements,
        }

    def get_all_annotator_ids(self) -> list[str]:
        """All annotator_ids that have labeled at least one record."""
        ids: set[str] = set()
        for record in self._records:
            for annotation in record.get("annotations", []):
                ids.add(annotation["annotator_id"])
        return sorted(ids)
