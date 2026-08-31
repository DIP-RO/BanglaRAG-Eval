"""Evaluator framework — common abstraction for RAG faithfulness evaluators.

Every evaluator implements a common interface:
    Evaluator.evaluate(record) -> EvaluatorOutput

Outputs append to the record's `evaluator_outputs[]` array — never to
annotation fields. Evaluators must NOT see human labels at inference time.

This module defines the base interface and data structures. Concrete
evaluator adapters are in separate modules:
- lexical_baseline.py: simple lexical overlap baseline (no API needed)
- llm_judge.py: LLM-as-judge via Ollama (cross-model, no paid API)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class EvaluatorOutput:
    """Standardized output from any evaluator.

    Fields match the `evaluator_outputs` schema in
    configs/schema/record.schema.json.
    """

    evaluator_name: str
    evaluator_version: str
    score: float | None = None
    label: str | None = None
    explanation: str | None = None
    latency_seconds: float | None = None
    cost_usd: float | None = None
    configuration: dict[str, Any] | None = None
    timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "evaluator_name": self.evaluator_name,
            "evaluator_version": self.evaluator_version,
            "score": self.score,
            "label": self.label,
            "explanation": self.explanation,
            "latency_seconds": self.latency_seconds,
            "cost_usd": self.cost_usd,
            "configuration": self.configuration,
            "timestamp": self.timestamp,
        }


def _iso_timestamp() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


class Evaluator(ABC):
    """Base class for all RAG faithfulness evaluators.

    Evaluators are the SYSTEMS UNDER EVALUATION — they are never gold
    labels. Their outputs go to `evaluator_outputs[]`, never to
    `annotations[]` or `faithfulness_category`.

    Evaluators must NOT access human labels at inference time. The
    `evaluate()` method receives only the question, retrieved context,
    and generated answer — never annotations or gold labels.
    """

    name: str
    version: str

    @abstractmethod
    def evaluate(self, record: dict[str, Any]) -> EvaluatorOutput:
        """Evaluate a single record.

        Args:
            record: A benchmark record dict. The evaluator should only
                read: question, retrieved_context, generated_answer.
                It must NOT read: annotations, faithfulness_category,
                adjudication, or any human label field.

        Returns:
            EvaluatorOutput with score, label, explanation, and metadata.
        """
        ...

    def evaluate_batch(
        self, records: list[dict[str, Any]]
    ) -> list[EvaluatorOutput]:
        """Evaluate a batch of records.

        Default implementation calls evaluate() for each record.
        Subclasses can override for batch optimization.

        Args:
            records: List of benchmark record dicts.

        Returns:
            List of EvaluatorOutput, one per record.
        """
        return [self.evaluate(r) for r in records]

    def _safe_evaluate(self, record: dict[str, Any]) -> EvaluatorOutput:
        """Wrapper that handles timing and errors.

        Subclasses should call this from their evaluate() implementation
        or implement their own timing.
        """
        start = time.time()
        try:
            result = self.evaluate(record)
            if result.latency_seconds is None:
                result.latency_seconds = round(time.time() - start, 3)
            if result.timestamp is None:
                result.timestamp = _iso_timestamp()
            return result
        except Exception as exc:
            return EvaluatorOutput(
                evaluator_name=self.name,
                evaluator_version=self.version,
                label=None,
                score=None,
                explanation=f"ERROR: {type(exc).__name__}: {exc}",
                latency_seconds=round(time.time() - start, 3),
                timestamp=_iso_timestamp(),
            )


def attach_evaluator_output(
    record: dict[str, Any],
    output: EvaluatorOutput,
) -> dict[str, Any]:
    """Append an EvaluatorOutput to a record's evaluator_outputs[].

    This is the canonical way to add evaluator results to a record.
    It never touches annotations or gold labels.

    Args:
        record: The benchmark record dict.
        output: The EvaluatorOutput to append.

    Returns:
        The record with the output appended (mutated in place).
    """
    if "evaluator_outputs" not in record:
        record["evaluator_outputs"] = []
    record["evaluator_outputs"].append(output.to_dict())
    return record
