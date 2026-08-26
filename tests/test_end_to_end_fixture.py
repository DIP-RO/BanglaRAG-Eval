"""End-to-end test on the committed synthetic fixture.

Verifies that the 10-example fixture passes the full schema, that it
covers every language and evidence condition, that no gold labels were
auto-assigned, and that the committed JSONL matches the deterministic
builder output exactly.
"""

from __future__ import annotations

from collections import Counter

from banglarag_eval.constants import (
    EVIDENCE_CONDITIONS,
    LANGUAGE_CONDITIONS,
    REPO_ROOT,
)
from banglarag_eval.dataset import load_dataset
from banglarag_eval.fixtures import build_synthetic_fixture_records
from banglarag_eval.schema import validate_dataset

FIXTURE_PATH = REPO_ROOT / "data" / "fixtures" / "synthetic_fixture_v0.jsonl"


def test_fixture_exists_and_has_ten_records():
    records = load_dataset(FIXTURE_PATH)
    assert len(records) == 10


def test_fixture_passes_full_validation_without_warnings():
    records = load_dataset(FIXTURE_PATH)
    results = validate_dataset(records)
    for example_id, result in results.items():
        assert result.is_valid, f"{example_id}: {result.errors}"
        assert not result.warnings, f"{example_id}: {result.warnings}"


def test_fixture_covers_all_language_conditions():
    records = load_dataset(FIXTURE_PATH)
    counts = Counter(r["language_condition"] for r in records)
    assert set(counts) == set(LANGUAGE_CONDITIONS)
    assert all(count == 2 for count in counts.values())


def test_fixture_covers_all_evidence_conditions():
    records = load_dataset(FIXTURE_PATH)
    counts = Counter(r["evidence_condition"] for r in records)
    assert set(counts) == set(EVIDENCE_CONDITIONS)
    assert all(count == 2 for count in counts.values())


def test_fixture_example_ids_unique():
    records = load_dataset(FIXTURE_PATH)
    ids = [r["example_id"] for r in records]
    assert len(ids) == len(set(ids))


def test_fixture_has_no_auto_assigned_gold_labels():
    """Human annotation fields must be empty: no annotator was involved,
    so no gold labels may exist. Guards the gold-standard rule."""
    records = load_dataset(FIXTURE_PATH)
    for record in records:
        assert record["annotations"] == []
        assert record["adjudication"] is None
        assert record["faithfulness_category"] is None
        assert record["evaluator_outputs"] == []


def test_fixture_provenance_is_complete():
    records = load_dataset(FIXTURE_PATH)
    for record in records:
        assert record["data_origin"] == "synthetic_constructed"
        assert record["source_text"]
        assert record["retrieved_context"]
        assert record["retrieval_documents"]
        assert record["generated_answer"]
        assert record["intended_answer"]
        span = record["source_span"]
        assert span is not None
        assert record["source_text"][span["start_char"] : span["end_char"]] == span["text"]


def test_fixture_corruption_is_explicit_and_logged():
    records = load_dataset(FIXTURE_PATH)
    corrupted = [r for r in records if r["corruption_metadata"] is not None]
    assert len(corrupted) == 2
    for record in corrupted:
        meta = record["corruption_metadata"]
        assert record["evidence_condition"] != "correct"
        assert meta["original_context_ids"]
        assert meta["modified_context_ids"]
        assert meta["operation_description"]


def test_committed_fixture_matches_deterministic_builder():
    """The committed JSONL and the builder must never drift apart."""
    committed = load_dataset(FIXTURE_PATH)
    rebuilt = build_synthetic_fixture_records()
    assert committed == rebuilt
