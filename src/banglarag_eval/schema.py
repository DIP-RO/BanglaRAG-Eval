"""Record validation: JSON Schema structure plus cross-field rules.

Structural validation delegates to the canonical machine-readable schema
in configs/schema/record.schema.json. Cross-field rules encode
methodological invariants that JSON Schema cannot express cleanly:

  R1  native_bangla records must not have translation provenance.
  R2  translated_bangla records must not claim native authorship.
  R3  faithfulness_category (the gold label) may only exist when it
      comes from human annotation: it must equal the adjudication label
      when an adjudication exists, or the unanimous label of at least
      MIN_INDEPENDENT_ANNOTATORS independent annotators when they agree
      and no adjudication exists. It can never come from an automatic
      evaluator, and disagreement without adjudication yields no gold.
  R4  adjudication requires at least MIN_INDEPENDENT_ANNOTATORS
      independent prior annotations.
  R5  annotator_ids within one record must be distinct (independence).
  R6  corruption_metadata implies the evidence condition is a degraded
      one (never 'correct'), and corruption is therefore never silent.
  R7  the record timestamp and every nested timestamp (annotations,
      adjudication, question_generation, evaluator_outputs) must parse
      as ISO-8601.
  R8  a third-rater adjudicator must be independent: for
      adjudication.method == 'third_rater', adjudicator_id must not
      appear among the record's annotator_ids. (joint_session
      adjudications are exempt: the annotators resolve together, and
      the method field records that mechanism for later analysis.)
  R9  hallucination_type is human-assigned: it may only be non-null
      when the record carries human annotation provenance (an
      adjudication, or unanimous independent annotations per R3).
  R10 source_span must be internally consistent with source_text:
      start < end, end within bounds, and span text matching the slice.

Alignment between language_condition and question_language is reported
as a warning, not an error (see constants.EXPECTED_QUESTION_LANGUAGE).
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Any

import jsonschema

from banglarag_eval.constants import (
    EXPECTED_QUESTION_LANGUAGE,
    MIN_INDEPENDENT_ANNOTATORS,
    RECORD_SCHEMA_PATH,
    TRANSLATED_ORIGINS,
)


@dataclass
class ValidationResult:
    """Outcome of validating a single record."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


@lru_cache(maxsize=1)
def _load_schema_cached() -> dict[str, Any]:
    if not RECORD_SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"canonical record schema not found at {RECORD_SCHEMA_PATH}; "
            "banglarag_eval must run from a repository checkout (editable "
            "install or tests/conftest.py path setup) so that configs/ is "
            "available"
        )
    with open(RECORD_SCHEMA_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def load_record_schema() -> dict[str, Any]:
    """Return a copy of the canonical JSON Schema for benchmark records.

    A deep copy is returned so callers cannot mutate the cached schema
    that the validator is built from.
    """
    return copy.deepcopy(_load_schema_cached())


@lru_cache(maxsize=1)
def _schema_validator() -> jsonschema.Validator:
    schema = _load_schema_cached()
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def _parse_iso8601(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except (ValueError, AttributeError, TypeError):
        return False


def _unanimous_label(annotations: list[dict[str, Any]]) -> str | None:
    """The unanimous annotation label, or None if too few or split."""
    if len(annotations) < MIN_INDEPENDENT_ANNOTATORS:
        return None
    labels = {a["label"] for a in annotations}
    if len(labels) == 1:
        return next(iter(labels))
    return None


def validate_record(record: dict[str, Any]) -> ValidationResult:
    """Validate one record structurally and against cross-field rules."""
    result = ValidationResult()

    for error in sorted(_schema_validator().iter_errors(record), key=str):
        location = "/".join(str(p) for p in error.absolute_path) or "<record>"
        result.errors.append(f"schema: {location}: {error.message}")
    if result.errors:
        # Cross-field rules assume a structurally valid record.
        return result

    language_condition = record["language_condition"]
    data_origin = record["data_origin"]

    # R1 / R2 — never conflate native and translated Bangla.
    if language_condition == "native_bangla" and data_origin in TRANSLATED_ORIGINS:
        result.errors.append(
            "R1: language_condition 'native_bangla' is incompatible with "
            f"translation provenance data_origin '{data_origin}'"
        )
    if language_condition == "translated_bangla" and data_origin == "native_authored":
        result.errors.append(
            "R2: language_condition 'translated_bangla' is incompatible "
            "with data_origin 'native_authored'"
        )

    annotations = record.get("annotations", [])
    adjudication = record.get("adjudication")
    unanimous = _unanimous_label(annotations)

    # R3 — gold label only via human annotation (adjudicated label, or
    # the unanimous label of independent annotators).
    gold = record.get("faithfulness_category")
    if gold is not None:
        if adjudication is not None:
            if adjudication["label"] != gold:
                result.errors.append(
                    f"R3: faithfulness_category '{gold}' does not match "
                    f"adjudication.label '{adjudication['label']}'"
                )
        elif unanimous is None:
            result.errors.append(
                "R3: faithfulness_category is set without human provenance: "
                "no adjudication exists and the independent annotations are "
                "absent, too few, or not unanimous; gold labels come "
                "exclusively from human annotation"
            )
        elif unanimous != gold:
            result.errors.append(
                f"R3: faithfulness_category '{gold}' does not match the "
                f"unanimous annotation label '{unanimous}'"
            )

    # R4 — adjudication requires independent annotations first.
    if adjudication is not None and len(annotations) < MIN_INDEPENDENT_ANNOTATORS:
        result.errors.append(
            f"R4: adjudication requires at least {MIN_INDEPENDENT_ANNOTATORS} "
            f"independent annotations, found {len(annotations)}"
        )

    # R5 — annotator independence.
    annotator_ids = [a["annotator_id"] for a in annotations]
    if len(annotator_ids) != len(set(annotator_ids)):
        result.errors.append(
            "R5: duplicate annotator_id within one record; annotations must "
            "come from independent annotators"
        )

    # R6 — corruption is explicit and only produces degraded evidence.
    if record.get("corruption_metadata") is not None:
        if record["evidence_condition"] == "correct":
            result.errors.append(
                "R6: corruption_metadata present but evidence_condition is "
                "'correct'; corrupted evidence cannot be labeled correct"
            )

    # R7 — the record timestamp and all nested timestamps are ISO-8601.
    if not _parse_iso8601(record["timestamp"]):
        result.errors.append(
            f"R7: timestamp '{record['timestamp']}' is not valid ISO-8601"
        )
    nested_timestamps: list[tuple[str, Any]] = []
    for index, annotation in enumerate(annotations):
        nested_timestamps.append((f"annotations[{index}].timestamp", annotation["timestamp"]))
    if adjudication is not None:
        nested_timestamps.append(("adjudication.timestamp", adjudication["timestamp"]))
    question_generation = record.get("question_generation")
    if question_generation is not None:
        nested_timestamps.append(
            ("question_generation.timestamp", question_generation.get("timestamp"))
        )
    for index, output in enumerate(record.get("evaluator_outputs", [])):
        nested_timestamps.append(
            (f"evaluator_outputs[{index}].timestamp", output.get("timestamp"))
        )
    for location, value in nested_timestamps:
        if value is not None and not _parse_iso8601(value):
            result.errors.append(
                f"R7: {location} '{value}' is not valid ISO-8601"
            )

    # R8 — third-rater adjudicators must be independent of the annotators.
    if adjudication is not None:
        if (
            adjudication["method"] == "third_rater"
            and adjudication["adjudicator_id"] in annotator_ids
        ):
            result.errors.append(
                "R8: third_rater adjudication requires an adjudicator who is "
                f"not one of the annotators; '{adjudication['adjudicator_id']}' "
                "also appears in annotations"
            )

    # R9 — hallucination_type is human-assigned.
    if record.get("hallucination_type") is not None:
        if adjudication is None and unanimous is None:
            result.errors.append(
                "R9: hallucination_type is set without human annotation "
                "provenance (no adjudication and no unanimous independent "
                "annotations)"
            )

    # R10 — source_span consistency with source_text.
    span = record.get("source_span")
    if span is not None:
        source_text = record["source_text"]
        start, end = span["start_char"], span["end_char"]
        if start >= end:
            result.errors.append(
                f"R10: source_span start_char {start} must be < end_char {end}"
            )
        elif end > len(source_text):
            result.errors.append(
                f"R10: source_span end_char {end} exceeds source_text length "
                f"{len(source_text)}"
            )
        elif span.get("text") is not None and source_text[start:end] != span["text"]:
            result.errors.append(
                "R10: source_span.text does not match "
                "source_text[start_char:end_char]"
            )

    # Warning-level: expected question_language for the condition.
    expected = EXPECTED_QUESTION_LANGUAGE.get(language_condition)
    if expected is not None and record["question_language"] != expected:
        result.warnings.append(
            f"question_language '{record['question_language']}' is unusual "
            f"for language_condition '{language_condition}' "
            f"(expected '{expected}'); confirm this is intentional"
        )

    return result


def validate_dataset(records: list[dict[str, Any]]) -> dict[str, ValidationResult]:
    """Validate every record; also enforce example_id uniqueness.

    Returns a mapping from a record key (example_id, or a positional
    placeholder when example_id is missing) to its ValidationResult.
    Duplicate ids are re-keyed with a collision-proof suffix so no
    record's result is ever silently dropped.
    """
    results: dict[str, ValidationResult] = {}
    seen_ids: dict[str, int] = {}
    for index, record in enumerate(records):
        example_id = record.get("example_id") or f"<record #{index}>"
        result = validate_record(record)
        if example_id in seen_ids:
            result.errors.append(
                f"duplicate example_id '{example_id}' "
                f"(first seen at record #{seen_ids[example_id]})"
            )
            key = f"{example_id}#dup{index}"
        else:
            seen_ids[example_id] = index
            key = example_id
        while key in results:
            key += "+"
        results[key] = result
    return results
