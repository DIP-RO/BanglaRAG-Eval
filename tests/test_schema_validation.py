"""Schema and cross-field validation tests."""

from __future__ import annotations

import pytest

from banglarag_eval.schema import validate_dataset, validate_record


def _annotation(annotator_id: str, label: str) -> dict:
    return {
        "annotator_id": annotator_id,
        "label": label,
        "explanation": "test explanation",
        "timestamp": "2026-08-26T10:00:00+06:00",
    }


def _adjudication(label: str, adjudicator_id: str = "adjudicator-1", method: str = "third_rater") -> dict:
    return {
        "label": label,
        "explanation": "adjudicated after disagreement",
        "adjudicator_id": adjudicator_id,
        "method": method,
        "timestamp": "2026-08-27T10:00:00+06:00",
    }


class TestStructuralValidation:
    def test_minimal_valid_record_passes(self, valid_record):
        result = validate_record(valid_record)
        assert result.is_valid, result.errors
        assert result.warnings == []

    @pytest.mark.parametrize(
        "missing_field",
        [
            "schema_version",
            "example_id",
            "document_id",
            "source_id",
            "question",
            "question_language",
            "language_condition",
            "data_origin",
            "source_text",
            "evidence_condition",
            "dataset_version",
            "timestamp",
        ],
    )
    def test_missing_required_field_fails(self, valid_record, missing_field):
        del valid_record[missing_field]
        result = validate_record(valid_record)
        assert not result.is_valid
        assert any(missing_field in e for e in result.errors)

    def test_unknown_language_condition_fails(self, valid_record):
        valid_record["language_condition"] = "hinglish"
        assert not validate_record(valid_record).is_valid

    def test_unknown_evidence_condition_fails(self, valid_record):
        valid_record["evidence_condition"] = "somewhat_relevant"
        assert not validate_record(valid_record).is_valid

    def test_unknown_field_fails(self, valid_record):
        valid_record["evidnce_condition"] = "correct"  # typo must be caught
        assert not validate_record(valid_record).is_valid

    def test_empty_question_fails(self, valid_record):
        valid_record["question"] = ""
        assert not validate_record(valid_record).is_valid

    def test_banglish_is_a_distinct_condition(self, valid_record):
        valid_record["language_condition"] = "banglish"
        valid_record["question_language"] = "banglish"
        valid_record["question"] = "Bangladesh er rajdhani kothay?"
        valid_record["source_text"] = "Bangladesh er rajdhani Dhaka."
        result = validate_record(valid_record)
        assert result.is_valid, result.errors


class TestCrossFieldRules:
    def test_r1_native_bangla_cannot_be_machine_translated(self, valid_record):
        valid_record["data_origin"] = "machine_translated"
        result = validate_record(valid_record)
        assert any(e.startswith("R1") for e in result.errors)

    def test_r2_translated_bangla_cannot_be_native_authored(self, valid_record):
        valid_record["language_condition"] = "translated_bangla"
        valid_record["data_origin"] = "native_authored"
        result = validate_record(valid_record)
        assert any(e.startswith("R2") for e in result.errors)

    def test_r3_gold_label_requires_human_provenance(self, valid_record):
        valid_record["faithfulness_category"] = "faithful"
        result = validate_record(valid_record)
        assert any(e.startswith("R3") for e in result.errors)

    def test_r3_unanimous_agreement_is_a_valid_gold_path(self, valid_record):
        """When independent annotators agree, the unanimous label becomes
        gold without an adjudication record."""
        valid_record["annotations"] = [
            _annotation("ann-1", "faithful"),
            _annotation("ann-2", "faithful"),
        ]
        valid_record["faithfulness_category"] = "faithful"
        result = validate_record(valid_record)
        assert result.is_valid, result.errors

    def test_r3_gold_must_match_unanimous_label(self, valid_record):
        valid_record["annotations"] = [
            _annotation("ann-1", "faithful"),
            _annotation("ann-2", "faithful"),
        ]
        valid_record["faithfulness_category"] = "unsupported"
        result = validate_record(valid_record)
        assert any(e.startswith("R3") for e in result.errors)

    def test_r3_disagreement_without_adjudication_yields_no_gold(self, valid_record):
        valid_record["annotations"] = [
            _annotation("ann-1", "faithful"),
            _annotation("ann-2", "unsupported"),
        ]
        valid_record["faithfulness_category"] = "faithful"
        result = validate_record(valid_record)
        assert any(e.startswith("R3") for e in result.errors)

    def test_r3_gold_label_must_match_adjudication(self, valid_record):
        valid_record["annotations"] = [
            _annotation("ann-1", "faithful"),
            _annotation("ann-2", "unsupported"),
        ]
        valid_record["adjudication"] = _adjudication("unsupported")
        valid_record["faithfulness_category"] = "faithful"
        result = validate_record(valid_record)
        assert any(e.startswith("R3") for e in result.errors)

    def test_gold_label_with_matching_adjudication_passes(self, valid_record):
        valid_record["annotations"] = [
            _annotation("ann-1", "faithful"),
            _annotation("ann-2", "unsupported"),
        ]
        valid_record["adjudication"] = _adjudication("unsupported")
        valid_record["faithfulness_category"] = "unsupported"
        result = validate_record(valid_record)
        assert result.is_valid, result.errors

    def test_r4_adjudication_requires_two_annotations(self, valid_record):
        valid_record["annotations"] = [_annotation("ann-1", "faithful")]
        valid_record["adjudication"] = _adjudication("faithful")
        result = validate_record(valid_record)
        assert any(e.startswith("R4") for e in result.errors)

    def test_r5_annotators_must_be_independent(self, valid_record):
        valid_record["annotations"] = [
            _annotation("ann-1", "faithful"),
            _annotation("ann-1", "unsupported"),
        ]
        result = validate_record(valid_record)
        assert any(e.startswith("R5") for e in result.errors)

    def test_r6_corrupted_evidence_cannot_be_correct(self, valid_record):
        valid_record["corruption_metadata"] = {
            "corruption_type": "context_replacement",
            "original_context_ids": ["doc-001"],
            "modified_context_ids": ["doc-099"],
            "operation_description": "swap in unrelated passage",
            "random_seed": 7,
        }
        result = validate_record(valid_record)
        assert any(e.startswith("R6") for e in result.errors)

    def test_r6_corruption_with_degraded_condition_passes(self, valid_record):
        valid_record["evidence_condition"] = "irrelevant"
        valid_record["corruption_metadata"] = {
            "corruption_type": "context_replacement",
            "original_context_ids": ["doc-001"],
            "modified_context_ids": ["doc-099"],
            "operation_description": "swap in unrelated passage",
            "random_seed": 7,
        }
        result = validate_record(valid_record)
        assert result.is_valid, result.errors

    def test_r7_invalid_timestamp_fails(self, valid_record):
        valid_record["timestamp"] = "yesterday"
        result = validate_record(valid_record)
        assert any(e.startswith("R7") for e in result.errors)

    def test_r7_nested_timestamps_also_checked(self, valid_record):
        bad = _annotation("ann-1", "faithful")
        bad["timestamp"] = "not-a-date"
        valid_record["annotations"] = [bad, _annotation("ann-2", "faithful")]
        result = validate_record(valid_record)
        assert any("annotations[0].timestamp" in e for e in result.errors)

    def test_r7_evaluator_output_timestamp_checked(self, valid_record):
        valid_record["evaluator_outputs"] = [
            {
                "evaluator_name": "test-evaluator",
                "evaluator_version": "1.0",
                "timestamp": "garbage",
            }
        ]
        result = validate_record(valid_record)
        assert any("evaluator_outputs[0].timestamp" in e for e in result.errors)

    def test_r8_third_rater_must_be_independent(self, valid_record):
        valid_record["annotations"] = [
            _annotation("ann-1", "faithful"),
            _annotation("ann-2", "unsupported"),
        ]
        valid_record["adjudication"] = _adjudication(
            "faithful", adjudicator_id="ann-1", method="third_rater"
        )
        result = validate_record(valid_record)
        assert any(e.startswith("R8") for e in result.errors)

    def test_r8_joint_session_may_involve_annotators(self, valid_record):
        valid_record["annotations"] = [
            _annotation("ann-1", "faithful"),
            _annotation("ann-2", "unsupported"),
        ]
        valid_record["adjudication"] = _adjudication(
            "faithful", adjudicator_id="ann-1", method="joint_session"
        )
        result = validate_record(valid_record)
        assert result.is_valid, result.errors

    def test_adjudication_requires_method_field(self, valid_record):
        valid_record["annotations"] = [
            _annotation("ann-1", "faithful"),
            _annotation("ann-2", "unsupported"),
        ]
        adjudication = _adjudication("faithful")
        del adjudication["method"]
        valid_record["adjudication"] = adjudication
        result = validate_record(valid_record)
        assert not result.is_valid

    def test_r9_hallucination_type_requires_human_provenance(self, valid_record):
        valid_record["hallucination_type"] = "evident_conflict"
        result = validate_record(valid_record)
        assert any(e.startswith("R9") for e in result.errors)

    def test_r9_hallucination_type_with_unanimous_annotations_passes(self, valid_record):
        valid_record["annotations"] = [
            _annotation("ann-1", "contradictory"),
            _annotation("ann-2", "contradictory"),
        ]
        valid_record["hallucination_type"] = "evident_conflict"
        result = validate_record(valid_record)
        assert result.is_valid, result.errors

    def test_r10_span_must_be_ordered(self, valid_record):
        valid_record["source_span"] = {"start_char": 5, "end_char": 5, "text": None}
        result = validate_record(valid_record)
        assert any(e.startswith("R10") for e in result.errors)

    def test_r10_span_must_be_in_bounds(self, valid_record):
        length = len(valid_record["source_text"])
        valid_record["source_span"] = {
            "start_char": 0,
            "end_char": length + 10,
            "text": None,
        }
        result = validate_record(valid_record)
        assert any(e.startswith("R10") for e in result.errors)

    def test_r10_span_text_must_match_slice(self, valid_record):
        valid_record["source_span"] = {
            "start_char": 0,
            "end_char": 4,
            "text": "ভুল লেখা",
        }
        result = validate_record(valid_record)
        assert any(e.startswith("R10") for e in result.errors)

    def test_r10_consistent_span_passes(self, valid_record):
        text = valid_record["source_text"]
        valid_record["source_span"] = {
            "start_char": 0,
            "end_char": 4,
            "text": text[0:4],
        }
        result = validate_record(valid_record)
        assert result.is_valid, result.errors

    def test_question_language_mismatch_is_warning_not_error(self, valid_record):
        valid_record["question_language"] = "en"
        result = validate_record(valid_record)
        assert result.is_valid
        assert result.warnings


class TestDatasetValidation:
    def test_duplicate_example_ids_rejected(self, valid_record):
        duplicate = dict(valid_record)
        results = validate_dataset([valid_record, duplicate])
        assert any(
            "duplicate example_id" in e
            for r in results.values()
            for e in r.errors
        )

    def test_distinct_records_pass(self, valid_record):
        second = dict(valid_record)
        second["example_id"] = "test-002"
        results = validate_dataset([valid_record, second])
        assert all(r.is_valid for r in results.values())

    def test_no_result_is_dropped_on_key_collision(self, valid_record):
        """A record whose id collides with a generated duplicate key must
        still get its own ValidationResult."""
        first = dict(valid_record)  # example_id: test-001
        second = dict(valid_record)  # duplicate -> keyed test-001#dup1
        third = dict(valid_record)
        third["example_id"] = "test-001#dup1"  # collides with generated key
        results = validate_dataset([first, second, third])
        assert len(results) == 3
