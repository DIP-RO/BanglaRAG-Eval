"""Additional edge-case tests for annotation store, schema rules, and
config validation.

These tests cover scenarios not exercised in the main test files:
- Annotation store: multiple annotators, pending order, view immutability,
  stats after adjudication, adjudication view content.
- Schema rules: R3 gold without adjudication or annotations, R3 gold
  mismatch, R4 with exactly 1 annotation, R5 with 3 annotators, R6
  corruption with correct evidence, R7 nested timestamp, R9
  hallucination without annotation, R10 span beyond text length.
- Config: extra unknown key, zero pilot_size, negative proportion,
  missing stage field.
- Dataset: save to new path, load empty file, load file with blank lines.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from banglarag_eval.annotation.store import AnnotationStore
from banglarag_eval.config import load_pilot_config, ConfigError
from banglarag_eval.constants import FAITHFULNESS_CATEGORIES
from banglarag_eval.dataset import load_dataset, save_dataset
from banglarag_eval.schema import validate_record


# ── Helpers ─────────────────────────────────────────────────────

def _make_record(
    example_id: str,
    *,
    language_condition: str = "native_bangla",
    data_origin: str = "native_authored",
    evidence_condition: str = "correct",
    annotations: list | None = None,
    adjudication: dict | None = None,
    faithfulness_category: str | None = None,
    hallucination_type: str | None = None,
    corruption_metadata: dict | None = None,
    source_text: str = "বাংলাদেশের রাজধানী ঢাকা।",
    timestamp: str = "2026-08-26T00:00:00+06:00",
) -> dict:
    """A minimal valid record with configurable fields for edge-case tests."""
    return {
        "schema_version": "1.0.0",
        "example_id": example_id,
        "document_id": f"doc-{example_id}",
        "source_id": "test_source",
        "question": "বাংলাদেশের রাজধানীর নাম কী?",
        "question_language": "bn",
        "language_condition": language_condition,
        "data_origin": data_origin,
        "source_text": source_text,
        "evidence_condition": evidence_condition,
        "retrieved_context": source_text,
        "retrieval_documents": [
            {"document_id": f"doc-{example_id}", "text": source_text, "rank": 1}
        ],
        "retrieval_configuration": {"method": "test", "top_k": 1},
        "generated_answer": "বাংলাদেশের রাজধানী ঢাকা।",
        "answer_claims": ["বাংলাদেশের রাজধানী ঢাকা।"],
        "generator_model": "test-generator",
        "generator_version": "1.0",
        "generation_settings": {"temperature": 0},
        "retriever": "test-retriever",
        "prompt_version": "test-v1",
        "question_generation": {
            "method": "human_written",
            "model": None,
            "model_version": None,
            "prompt_version": None,
            "timestamp": "2026-08-26T00:00:00+06:00",
        },
        "annotations": annotations or [],
        "adjudication": adjudication,
        "faithfulness_category": faithfulness_category,
        "hallucination_type": hallucination_type,
        "corruption_metadata": corruption_metadata,
        "evaluator_outputs": [],
        "cost": None,
        "latency": None,
        "random_seed": None,
        "dataset_version": "test-0.1.0",
        "timestamp": timestamp,
        "notes": None,
    }


def _valid_annotation(annotator_id: str, label: str = "faithful") -> dict:
    return {
        "annotator_id": annotator_id,
        "label": label,
        "explanation": "Test explanation.",
        "timestamp": "2026-08-27T10:00:00+00:00",
    }


def _valid_adjudication(label: str = "faithful") -> dict:
    return {
        "label": label,
        "explanation": "Adjudication explanation.",
        "adjudicator_id": "adjudicator-1",
        "method": "third_rater",
        "timestamp": "2026-08-27T11:00:00+00:00",
    }


@pytest.fixture
def dataset_path(tmp_path: Path) -> Path:
    """A 3-record JSONL dataset for testing."""
    records = [_make_record(f"test-{i:03d}") for i in range(3)]
    path = tmp_path / "test_dataset.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


# ── Annotation store edge cases ─────────────────────────────────

class TestAnnotationStoreEdgeCases:
    def test_pending_ids_preserve_record_order(self, dataset_path: Path):
        """Pending IDs should be in the same order as records in the file."""
        store = AnnotationStore(dataset_path)
        pending = store.get_pending_ids("ann-1")
        assert pending == ["test-000", "test-001", "test-002"]

    def test_pending_ids_excludes_records_without_answer(self, tmp_path: Path):
        """Records without a generated_answer should not be annotatable."""
        record = _make_record("test-000")
        record["generated_answer"] = None
        record["answer_claims"] = []
        path = tmp_path / "test.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        store = AnnotationStore(path)
        assert store.get_pending_ids("ann-1") == []

    def test_annotation_view_does_not_mutate_record(self, dataset_path: Path):
        """Getting a view must not modify the underlying record."""
        store = AnnotationStore(dataset_path)
        original = store.get_record("test-000")
        original_annotations = list(original.get("annotations", []))
        store.get_annotation_view("test-000")
        assert original["annotations"] == original_annotations

    def test_three_annotators_on_same_record(self, dataset_path: Path):
        """A record can have more than 2 annotations (R5 only checks
        distinct IDs within one record, not a maximum)."""
        store = AnnotationStore(dataset_path)
        store.add_annotation("test-000", "ann-1", "faithful", "Yes.")
        store.add_annotation("test-000", "ann-2", "faithful", "Yes.")
        store.add_annotation("test-000", "ann-3", "unsupported", "No.")
        record = AnnotationStore(dataset_path).get_record("test-000")
        assert len(record["annotations"]) == 3

    def test_disagreement_excludes_already_adjudicated(self, dataset_path: Path):
        """Once adjudicated, a record should not appear in disagreements."""
        store = AnnotationStore(dataset_path)
        store.add_annotation("test-000", "ann-1", "faithful", "Yes.")
        store.add_annotation("test-000", "ann-2", "unsupported", "No.")
        assert len(store.get_disagreements()) == 1
        store.add_adjudication("test-000", "adj-1", "faithful", "Resolve.", "third_rater")
        assert len(store.get_disagreements()) == 0

    def test_adjudication_view_contains_annotations(self, dataset_path: Path):
        """The adjudication view must include the disagreeing annotations."""
        store = AnnotationStore(dataset_path)
        store.add_annotation("test-000", "ann-1", "faithful", "Yes.")
        store.add_annotation("test-000", "ann-2", "unsupported", "No.")
        view = store.get_adjudication_view("test-000")
        assert view is not None
        assert len(view["annotations"]) == 2
        assert view["annotations"][0]["annotator_id"] == "ann-1"

    def test_adjudication_view_returns_none_for_missing_record(self, dataset_path: Path):
        store = AnnotationStore(dataset_path)
        assert store.get_adjudication_view("nonexistent") is None

    def test_stats_after_adjudication(self, dataset_path: Path):
        """Stats should reflect adjudicated count after adjudication."""
        store = AnnotationStore(dataset_path)
        store.add_annotation("test-000", "ann-1", "faithful", "Yes.")
        store.add_annotation("test-000", "ann-2", "unsupported", "No.")
        store.add_adjudication("test-000", "adj-1", "faithful", "Resolve.", "third_rater")
        stats = store.get_stats()
        assert stats["adjudicated"] == 1
        assert stats["pending_adjudication"] == 0

    def test_stats_annotatable_excludes_no_answer(self, tmp_path: Path):
        """Records without generated_answer should not count as annotatable."""
        record = _make_record("test-000")
        record["generated_answer"] = None
        record["answer_claims"] = []
        path = tmp_path / "test.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        store = AnnotationStore(path)
        stats = store.get_stats()
        assert stats["total"] == 1
        assert stats["annotatable"] == 0

    def test_invalid_adjudication_method_rejected(self, dataset_path: Path):
        store = AnnotationStore(dataset_path)
        store.add_annotation("test-000", "ann-1", "faithful", "Yes.")
        store.add_annotation("test-000", "ann-2", "unsupported", "No.")
        with pytest.raises(ValueError, match="invalid method"):
            store.add_adjudication("test-000", "adj-1", "faithful", "Ok.", "coin_flip")

    def test_adjudication_empty_explanation_rejected(self, dataset_path: Path):
        store = AnnotationStore(dataset_path)
        store.add_annotation("test-000", "ann-1", "faithful", "Yes.")
        store.add_annotation("test-000", "ann-2", "unsupported", "No.")
        with pytest.raises(ValueError, match="explanation is required"):
            store.add_adjudication("test-000", "adj-1", "faithful", "   ", "third_rater")

    def test_adjudication_record_not_found(self, dataset_path: Path):
        store = AnnotationStore(dataset_path)
        with pytest.raises(ValueError, match="not found"):
            store.add_adjudication("nope", "adj-1", "faithful", "Ok.", "third_rater")

    def test_get_all_annotator_ids_sorted(self, dataset_path: Path):
        """Annotator IDs should be returned in sorted order."""
        store = AnnotationStore(dataset_path)
        store.add_annotation("test-000", "charlie", "faithful", "Yes.")
        store.add_annotation("test-001", "alpha", "faithful", "Yes.")
        store.add_annotation("test-002", "bravo", "faithful", "Yes.")
        assert store.get_all_annotator_ids() == ["alpha", "bravo", "charlie"]

    def test_get_all_annotator_ids_empty(self, dataset_path: Path):
        """No annotations means empty list."""
        store = AnnotationStore(dataset_path)
        assert store.get_all_annotator_ids() == []

    def test_reload_picks_up_external_changes(self, dataset_path: Path):
        """reload() should pick up changes made by another process."""
        store = AnnotationStore(dataset_path)
        # Another process writes an annotation directly to the file.
        records = load_dataset(dataset_path, validate=False)
        records[0]["annotations"].append(_valid_annotation("external-ann"))
        save_dataset(records, dataset_path, overwrite=True, validate=True)
        # The store should see the change after reload.
        store.reload()
        record = store.get_record("test-000")
        assert len(record["annotations"]) == 1
        assert record["annotations"][0]["annotator_id"] == "external-ann"


# ── Schema rule edge cases ──────────────────────────────────────

class TestSchemaRuleEdgeCases:
    def test_r3_gold_without_annotations_or_adjudication(self):
        """Gold label with no annotations and no adjudication must fail R3."""
        record = _make_record("test-001", faithfulness_category="faithful")
        result = validate_record(record)
        assert any("R3" in e for e in result.errors)

    def test_r3_gold_mismatches_unanimous_label(self):
        """Gold label that doesn't match unanimous annotations fails R3."""
        record = _make_record(
            "test-001",
            annotations=[
                _valid_annotation("ann-1", "faithful"),
                _valid_annotation("ann-2", "faithful"),
            ],
            faithfulness_category="unsupported",
        )
        result = validate_record(record)
        assert any("R3" in e for e in result.errors)

    def test_r3_gold_matches_unanimous_label_passes(self):
        """Gold label matching unanimous annotations passes R3."""
        record = _make_record(
            "test-001",
            annotations=[
                _valid_annotation("ann-1", "faithful"),
                _valid_annotation("ann-2", "faithful"),
            ],
            faithfulness_category="faithful",
        )
        result = validate_record(record)
        assert not result.errors

    def test_r3_gold_mismatches_adjudication_label(self):
        """Gold label that doesn't match adjudication label fails R3."""
        record = _make_record(
            "test-001",
            annotations=[
                _valid_annotation("ann-1", "faithful"),
                _valid_annotation("ann-2", "unsupported"),
            ],
            adjudication=_valid_adjudication("faithful"),
            faithfulness_category="unsupported",
        )
        result = validate_record(record)
        assert any("R3" in e for e in result.errors)

    def test_r4_adjudication_with_only_one_annotation(self):
        """Adjudication with only 1 annotation fails R4."""
        record = _make_record(
            "test-001",
            annotations=[_valid_annotation("ann-1", "faithful")],
            adjudication=_valid_adjudication("faithful"),
            faithfulness_category="faithful",
        )
        result = validate_record(record)
        assert any("R4" in e for e in result.errors)

    def test_r5_duplicate_annotator_ids(self):
        """Two annotations from the same annotator_id fail R5."""
        record = _make_record(
            "test-001",
            annotations=[
                _valid_annotation("ann-1", "faithful"),
                _valid_annotation("ann-1", "unsupported"),
            ],
        )
        result = validate_record(record)
        assert any("R5" in e for e in result.errors)

    def test_r6_corruption_with_correct_evidence(self):
        """corruption_metadata with evidence_condition='correct' fails R6."""
        record = _make_record(
            "test-001",
            evidence_condition="correct",
            corruption_metadata={
                "corruption_type": "sentence_removal",
                "original_context_ids": ["doc-001"],
                "modified_context_ids": ["doc-001-modified"],
                "operation_description": "Removed supporting sentence.",
                "random_seed": 42,
            },
        )
        result = validate_record(record)
        assert any("R6" in e for e in result.errors)

    def test_r6_corruption_with_degraded_evidence_passes(self):
        """corruption_metadata with degraded evidence condition passes R6."""
        record = _make_record(
            "test-001",
            evidence_condition="irrelevant",
            corruption_metadata={
                "corruption_type": "context_replacement",
                "original_context_ids": ["doc-001"],
                "modified_context_ids": ["doc-001-modified"],
                "operation_description": "Replaced with irrelevant passage.",
                "random_seed": 42,
            },
        )
        result = validate_record(record)
        assert not result.errors

    def test_r7_invalid_nested_timestamp(self):
        """An invalid timestamp in question_generation fails R7."""
        record = _make_record("test-001")
        record["question_generation"]["timestamp"] = "not-a-date"
        result = validate_record(record)
        assert any("R7" in e for e in result.errors)

    def test_r9_hallucination_without_annotation(self):
        """hallucination_type set without annotations fails R9."""
        record = _make_record(
            "test-001",
            hallucination_type="evident_conflict",
        )
        result = validate_record(record)
        assert any("R9" in e for e in result.errors)

    def test_r9_hallucination_with_annotation_passes(self):
        """hallucination_type with annotations present passes R9."""
        record = _make_record(
            "test-001",
            annotations=[
                _valid_annotation("ann-1", "contradictory"),
                _valid_annotation("ann-2", "contradictory"),
            ],
            hallucination_type="evident_conflict",
            faithfulness_category="contradictory",
        )
        result = validate_record(record)
        assert not result.errors

    def test_r10_span_end_exceeds_source_text(self):
        """source_span end_char beyond source_text length fails R10."""
        record = _make_record("test-001")
        record["source_span"] = {
            "start_char": 0,
            "end_char": 999,
            "text": "বাংলাদেশের রাজধানী ঢাকা।",
        }
        result = validate_record(record)
        assert any("R10" in e for e in result.errors)

    def test_r10_span_text_mismatch(self):
        """source_span text not matching source_text substring fails R10."""
        record = _make_record("test-001")
        record["source_span"] = {
            "start_char": 0,
            "end_char": 5,
            "text": "WRONG TEXT",
        }
        result = validate_record(record)
        assert any("R10" in e for e in result.errors)

    def test_r1_native_bangla_with_translated_origin(self):
        """native_bangla with a translated data_origin fails R1."""
        record = _make_record(
            "test-001",
            language_condition="native_bangla",
            data_origin="machine_translated",
        )
        result = validate_record(record)
        assert any("R1" in e for e in result.errors)

    def test_r2_translated_bangla_with_native_origin(self):
        """translated_bangla with native_authored fails R2."""
        record = _make_record(
            "test-001",
            language_condition="translated_bangla",
            data_origin="native_authored",
        )
        result = validate_record(record)
        assert any("R2" in e for e in result.errors)

    def test_all_five_faithfulness_labels_valid(self):
        """Each of the 5 faithfulness categories should be usable."""
        for label in FAITHFULNESS_CATEGORIES:
            record = _make_record(
                "test-001",
                annotations=[
                    _valid_annotation("ann-1", label),
                    _valid_annotation("ann-2", label),
                ],
                faithfulness_category=label,
            )
            result = validate_record(record)
            assert not result.errors, f"label '{label}' should be valid but got: {result.errors}"


# ── Config edge cases ───────────────────────────────────────────

class TestConfigEdgeCases:
    def test_unknown_top_level_key_rejected(self, tmp_path: Path):
        """An unknown top-level key in the config should be rejected."""
        config_text = (
            "stage: methodology_validation\n"
            "pilot_size: 40\n"
            "random_seed: 20260826\n"
            "dataset_version: test-0.1.0\n"
            "unknown_key: should_fail\n"
            "language_proportions:\n"
            "  native_bangla: 0.3\n"
            "  translated_bangla: 0.2\n"
            "  code_mixed: 0.2\n"
            "  banglish: 0.15\n"
            "  english: 0.15\n"
            "evidence_proportions:\n"
            "  correct: 0.3\n"
            "  partially_relevant: 0.2\n"
            "  irrelevant: 0.2\n"
            "  contradictory: 0.15\n"
            "  missing: 0.15\n"
        )
        path = tmp_path / "bad_config.yaml"
        path.write_text(config_text, encoding="utf-8")
        with pytest.raises(ConfigError):
            load_pilot_config(path)

    def test_zero_pilot_size_rejected(self, tmp_path: Path):
        """pilot_size of 0 should be rejected."""
        config_text = (
            "stage: methodology_validation\n"
            "pilot_size: 0\n"
            "random_seed: 20260826\n"
            "dataset_version: test-0.1.0\n"
            "language_proportions:\n"
            "  native_bangla: 0.3\n"
            "  translated_bangla: 0.2\n"
            "  code_mixed: 0.2\n"
            "  banglish: 0.15\n"
            "  english: 0.15\n"
            "evidence_proportions:\n"
            "  correct: 0.3\n"
            "  partially_relevant: 0.2\n"
            "  irrelevant: 0.2\n"
            "  contradictory: 0.15\n"
            "  missing: 0.15\n"
        )
        path = tmp_path / "zero_config.yaml"
        path.write_text(config_text, encoding="utf-8")
        with pytest.raises(ConfigError):
            load_pilot_config(path)

    def test_negative_proportion_rejected(self, tmp_path: Path):
        """A negative proportion should be rejected."""
        config_text = (
            "stage: methodology_validation\n"
            "pilot_size: 40\n"
            "random_seed: 20260826\n"
            "dataset_version: test-0.1.0\n"
            "language_proportions:\n"
            "  native_bangla: -0.1\n"
            "  translated_bangla: 0.5\n"
            "  code_mixed: 0.2\n"
            "  banglish: 0.2\n"
            "  english: 0.2\n"
            "evidence_proportions:\n"
            "  correct: 0.3\n"
            "  partially_relevant: 0.2\n"
            "  irrelevant: 0.2\n"
            "  contradictory: 0.15\n"
            "  missing: 0.15\n"
        )
        path = tmp_path / "neg_config.yaml"
        path.write_text(config_text, encoding="utf-8")
        with pytest.raises(ConfigError):
            load_pilot_config(path)

    def test_missing_stage_field_rejected(self, tmp_path: Path):
        """A config without a stage field should be rejected."""
        config_text = (
            "pilot_size: 40\n"
            "random_seed: 20260826\n"
            "dataset_version: test-0.1.0\n"
            "language_proportions:\n"
            "  native_bangla: 0.3\n"
            "  translated_bangla: 0.2\n"
            "  code_mixed: 0.2\n"
            "  banglish: 0.15\n"
            "  english: 0.15\n"
            "evidence_proportions:\n"
            "  correct: 0.3\n"
            "  partially_relevant: 0.2\n"
            "  irrelevant: 0.2\n"
            "  contradictory: 0.15\n"
            "  missing: 0.15\n"
        )
        path = tmp_path / "no_stage.yaml"
        path.write_text(config_text, encoding="utf-8")
        with pytest.raises(ConfigError):
            load_pilot_config(path)


# ── Dataset I/O edge cases ──────────────────────────────────────

class TestDatasetIOEdgeCases:
    def test_save_to_new_path(self, tmp_path: Path):
        """save_dataset to a new path should create the file."""
        records = [_make_record("test-001")]
        new_path = tmp_path / "new_dataset.jsonl"
        save_dataset(records, new_path, validate=True)
        assert new_path.exists()
        loaded = load_dataset(new_path, validate=True)
        assert len(loaded) == 1
        assert loaded[0]["example_id"] == "test-001"

    def test_load_file_with_blank_lines(self, tmp_path: Path):
        """Blank lines in a JSONL file should be silently skipped."""
        record = _make_record("test-001")
        path = tmp_path / "blanks.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.write("\n")
            fh.write("\n")
            fh.write(json.dumps(record, ensure_ascii=False).replace("test-001", "test-002") + "\n")
            fh.write("\n")
        loaded = load_dataset(path, validate=False)
        assert len(loaded) == 2

    def test_load_empty_file_returns_empty_list(self, tmp_path: Path):
        """An empty file should return an empty list, not raise."""
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        loaded = load_dataset(path, validate=False)
        assert loaded == []

    def test_bangla_unicode_round_trip(self, tmp_path: Path):
        """Bangla text with complex Unicode should survive a round-trip."""
        record = _make_record("test-001")
        record["question"] = "ঢাকা কোন নদীর তীরে অবস্থিত?"
        record["source_text"] = "ঢাকা বুড়িগঙ্গা নদীর তীরে অবস্থিত।"
        record["generated_answer"] = "ঢাকা বুড়িগঙ্গা নদীর তীরে অবস্থিত।"
        path = tmp_path / "bangla.jsonl"
        save_dataset([record], path, validate=True)
        loaded = load_dataset(path, validate=True)
        assert loaded[0]["question"] == "ঢাকা কোন নদীর তীরে অবস্থিত?"
        assert loaded[0]["source_text"] == "ঢাকা বুড়িগঙ্গা নদীর তীরে অবস্থিত।"
        assert loaded[0]["generated_answer"] == "ঢাকা বুড়িগঙ্গা নদীর তীরে অবস্থিত।"

    def test_code_mixed_text_round_trip(self, tmp_path: Path):
        """Code-mixed Bangla-English text should survive a round-trip."""
        record = _make_record("test-001")
        record["language_condition"] = "code_mixed"
        record["question"] = "Bangladesh এর capital city কোনটি?"
        record["generated_answer"] = "Bangladesh এর capital city হলো Dhaka।"
        path = tmp_path / "codemixed.jsonl"
        save_dataset([record], path, validate=True)
        loaded = load_dataset(path, validate=True)
        assert loaded[0]["question"] == "Bangladesh এর capital city কোনটি?"
        assert loaded[0]["generated_answer"] == "Bangladesh এর capital city হলো Dhaka।"

    def test_banglish_text_round_trip(self, tmp_path: Path):
        """Banglish (romanized Bangla) text should survive a round-trip."""
        record = _make_record("test-001")
        record["language_condition"] = "banglish"
        record["question_language"] = "bn"
        record["question"] = "Bangladesh er rajdhani konta?"
        record["generated_answer"] = "Bangladesh er rajdhani Dhaka."
        path = tmp_path / "banglish.jsonl"
        save_dataset([record], path, validate=True)
        loaded = load_dataset(path, validate=True)
        assert loaded[0]["question"] == "Bangladesh er rajdhani konta?"
        assert loaded[0]["generated_answer"] == "Bangladesh er rajdhani Dhaka."
