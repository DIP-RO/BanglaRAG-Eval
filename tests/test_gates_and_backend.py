"""Tests for Stage 1 gates, RAGBackend, and experiment runner utilities."""

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from banglarag_eval.gates import (
    GateResult,
    Stage1GateReport,
    measure_stage1_gates,
    KAPPA_THRESHOLD,
    NO_FIT_THRESHOLD,
    CONDITION_INTEGRITY_THRESHOLD,
    PIPELINE_INTEGRITY_THRESHOLD,
    EVALUATOR_HARNESS_THRESHOLD,
)
from banglarag_eval.rag_backend import (
    RAGBackend,
    RetrievedDocument,
    StubRAGBackend,
    SemFuseRAGBackend,
)
from banglarag_eval.evaluators import LexicalBaselineEvaluator, attach_evaluator_output


# ── RAGBackend tests ────────────────────────────────────────────────


class TestRetrievedDocument:
    """Test RetrievedDocument dataclass."""

    def test_create(self):
        doc = RetrievedDocument(text="hello", document_id="doc-1", score=0.9)
        assert doc.text == "hello"
        assert doc.document_id == "doc-1"
        assert doc.score == 0.9

    def test_defaults(self):
        doc = RetrievedDocument(text="hello")
        assert doc.document_id == ""
        assert doc.score == 0.0
        assert doc.metadata == {}


class TestStubRAGBackend:
    """Test the stub RAG backend."""

    def test_retrieve_empty(self):
        backend = StubRAGBackend()
        results = backend.retrieve("query")
        assert results == []

    def test_retrieve_with_docs(self):
        docs = [{"text": "doc1", "document_id": "d1"},
                {"text": "doc2", "document_id": "d2"}]
        backend = StubRAGBackend(documents=docs)
        results = backend.retrieve("query", top_k=5)
        assert len(results) == 2
        assert results[0].text == "doc1"
        assert results[0].document_id == "d1"
        assert results[0].score > results[1].score  # Sorted by relevance

    def test_retrieve_top_k(self):
        docs = [{"text": f"doc{i}", "document_id": f"d{i}"} for i in range(10)]
        backend = StubRAGBackend(documents=docs)
        results = backend.retrieve("query", top_k=3)
        assert len(results) == 3

    def test_generate(self):
        backend = StubRAGBackend()
        answer = backend.generate("What is X?", "Context about X")
        assert "stub answer" in answer.lower()
        assert "What is X?" in answer

    def test_is_available(self):
        backend = StubRAGBackend()
        assert backend.is_available() is True

    def test_name_and_version(self):
        backend = StubRAGBackend()
        assert backend.name == "stub"
        assert backend.version == "0.1.0"


class TestSemFuseRAGBackend:
    """Test the SemFuse stub backend."""

    def test_not_implemented_retrieve(self):
        backend = SemFuseRAGBackend()
        with pytest.raises(NotImplementedError):
            backend.retrieve("query")

    def test_not_implemented_generate(self):
        backend = SemFuseRAGBackend()
        with pytest.raises(NotImplementedError):
            backend.generate("question", "context")

    def test_not_available_without_config(self):
        backend = SemFuseRAGBackend()
        assert backend.is_available() is False

    def test_available_with_config(self, monkeypatch):
        monkeypatch.setenv("SEMFUSE_URL", "http://localhost:8080")
        monkeypatch.setenv("SEMFUSE_API_KEY", "test-key")
        backend = SemFuseRAGBackend()
        assert backend.is_available() is True

    def test_name_and_version(self):
        backend = SemFuseRAGBackend()
        assert backend.name == "semfuse"
        assert "stub" in backend.version


class TestRAGBackendContract:
    """Test that the interface contract is enforced."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            RAGBackend()

    def test_stub_implements_interface(self):
        backend = StubRAGBackend()
        assert isinstance(backend, RAGBackend)
        assert hasattr(backend, "retrieve")
        assert hasattr(backend, "generate")
        assert hasattr(backend, "is_available")


# ── Gate result tests ───────────────────────────────────────────────


class TestGateResult:
    """Test GateResult dataclass."""

    def test_pass(self):
        g = GateResult(name="test", passed=True, value=0.8, threshold=0.6,
                       details="ok", n=100)
        assert "[PASS]" in str(g)
        assert "test" in str(g)

    def test_fail(self):
        g = GateResult(name="test", passed=False, value=0.3, threshold=0.6,
                       details="not enough", n=10)
        assert "[FAIL]" in str(g)

    def test_to_dict(self):
        g = GateResult(name="test", passed=True, value=0.8, threshold=0.6,
                       details="ok", n=100)
        d = g.to_dict()
        assert d["name"] == "test"
        assert d["passed"] is True
        assert d["n"] == 100


class TestStage1GateReport:
    """Test Stage1GateReport."""

    def test_summary_all_pass(self):
        report = Stage1GateReport(
            gates=[GateResult("g1", True, 0.8, 0.6, "ok", 100)],
            all_passed=True,
            n_records=100,
            decision="ALL GATES PASSED",
        )
        s = report.summary()
        assert "PASS" in s
        assert "ALL GATES PASSED" in s

    def test_summary_with_failures(self):
        report = Stage1GateReport(
            gates=[GateResult("g1", False, 0.3, 0.6, "fail", 10)],
            all_passed=False,
            n_records=10,
            decision="GATES FAILED: g1",
        )
        s = report.summary()
        assert "FAIL" in s
        assert "GATES FAILED" in s

    def test_to_dict(self):
        report = Stage1GateReport(
            gates=[GateResult("g1", True, 0.8, 0.6, "ok", 100)],
            all_passed=True,
            n_records=100,
        )
        d = report.to_dict()
        assert d["all_passed"] is True
        assert len(d["gates"]) == 1


# ── Gate measurement tests ──────────────────────────────────────────


def _make_valid_record(
    example_id="test-001",
    evidence_condition="correct",
    has_annotation=False,
    has_evaluator=False,
    has_gold=False,
):
    """Create a minimal schema-valid record for gate testing."""
    record = {
        "schema_version": "1.0.0",
        "example_id": example_id,
        "document_id": "doc-001",
        "source_id": "doc-001",
        "question": "What is the capital?",
        "question_language": "bn",
        "language_condition": "native_bangla",
        "data_origin": "native_authored",
        "source_text": "বাংলাদেশের রাজধানী ঢাকা।",
        "source_span": {"start_char": 0, "end_char": 20},
        "intended_answer": "ঢাকা",
        "evidence_condition": evidence_condition,
        "evidence_condition_notes": None,
        "retrieved_context": "বাংলাদেশের রাজধানী ঢাকা।" if evidence_condition != "missing" else "",
        "retrieval_documents": [{"document_id": "doc-001", "text": "বাংলাদেশের রাজধানী ঢাকা।", "rank": 1}] if evidence_condition != "missing" else [],
        "retrieval_configuration": {"method": "bm25", "top_k": 5},
        "retriever": "lexical_bm25",
        "generated_answer": "ঢাকা",
        "answer_claims": [],
        "generator_model": "qwen3:8b",
        "generation_settings": {"temperature": 0.0, "seed": 42},
        "prompt_version": "rag_standard_v1",
        "corruption_metadata": None if evidence_condition == "correct" else {"type": "test"},
        "annotations": [],
        "faithfulness_category": "faithful" if has_gold else None,
        "hallucination_type": None,
        "adjudication": None,
        "evaluator_outputs": [],
        "dataset_version": "pilot_stage1_v0",
        "timestamp": "2026-09-01T00:00:00+00:00",
        "random_seed": 42,
        "latency": 1.0,
        "cost": None,
        "notes": None,
    }

    if has_annotation:
        record["annotations"] = [{
            "annotator_id": "annotator1",
            "faithfulness_category": "faithful",
            "explanation": "All claims supported",
            "timestamp": "2026-09-01T00:00:00+00:00",
        }]

    if has_evaluator:
        record["evaluator_outputs"] = [{
            "evaluator_name": "lexical_baseline",
            "evaluator_version": "1.0.0",
            "score": 0.9,
            "label": "faithful",
            "explanation": "High overlap",
            "latency_seconds": 0.01,
            "cost_usd": 0.0,
            "configuration": {},
            "timestamp": "2026-09-01T00:00:00+00:00",
        }]

    return record


def _write_records(records, tmp_path, filename="test.jsonl"):
    """Write records to a temp JSONL file."""
    path = tmp_path / filename
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


class TestMeasureStage1Gates:
    """Test Stage 1 gate measurement."""

    def test_no_annotations(self, tmp_path):
        records = [_make_valid_record(has_annotation=False, has_evaluator=True)
                    for i in range(5)]
        path = _write_records(records, tmp_path)
        report = measure_stage1_gates(path)
        # Annotation agreement gate should fail
        gate_names = [g.name for g in report.gates]
        assert "annotation_agreement" in gate_names
        ann_gate = next(g for g in report.gates if g.name == "annotation_agreement")
        assert ann_gate.passed is False

    def test_pipeline_integrity_pass(self, tmp_path):
        records = [_make_valid_record() for i in range(5)]
        path = _write_records(records, tmp_path)
        report = measure_stage1_gates(path)
        pipeline_gate = next(g for g in report.gates if g.name == "pipeline_integrity")
        assert pipeline_gate.passed is True
        assert pipeline_gate.value == 1.0

    def test_condition_integrity_correct(self, tmp_path):
        records = [_make_valid_record(evidence_condition="correct") for i in range(5)]
        path = _write_records(records, tmp_path)
        report = measure_stage1_gates(path)
        cond_gate = next(g for g in report.gates if g.name == "condition_integrity")
        # correct condition should have no corruption_metadata
        assert cond_gate.value == 1.0

    def test_condition_integrity_missing(self, tmp_path):
        records = [_make_valid_record(evidence_condition="missing") for i in range(5)]
        path = _write_records(records, tmp_path)
        report = measure_stage1_gates(path)
        cond_gate = next(g for g in report.gates if g.name == "condition_integrity")
        # missing should have empty context
        assert cond_gate.value == 1.0

    def test_evaluator_harness_pass(self, tmp_path):
        records = [_make_valid_record(has_evaluator=True) for i in range(10)]
        path = _write_records(records, tmp_path)
        report = measure_stage1_gates(path, evaluator_name="lexical_baseline")
        eval_gate = next(g for g in report.gates if g.name == "evaluator_harness")
        assert eval_gate.passed is True
        assert eval_gate.value == 1.0

    def test_evaluator_harness_fail(self, tmp_path):
        records = [_make_valid_record(has_evaluator=False) for i in range(10)]
        path = _write_records(records, tmp_path)
        report = measure_stage1_gates(path)
        eval_gate = next(g for g in report.gates if g.name == "evaluator_harness")
        assert eval_gate.passed is False
        assert eval_gate.value == 0.0

    def test_taxonomy_no_fit(self, tmp_path):
        records = []
        for i in range(10):
            r = _make_valid_record(has_annotation=True)
            if i < 2:  # 20% have NO-FIT
                r["annotations"][0]["explanation"] = "NO-FIT: cannot determine"
            records.append(r)
        path = _write_records(records, tmp_path)
        report = measure_stage1_gates(path)
        tax_gate = next(g for g in report.gates if g.name == "taxonomy_adequacy")
        assert tax_gate.value == 0.2  # 2/10
        assert tax_gate.passed is False  # 0.2 > 0.10 threshold

    def test_taxonomy_pass(self, tmp_path):
        records = []
        for i in range(20):
            r = _make_valid_record(has_annotation=True)
            if i < 1:  # 5% have NO-FIT
                r["annotations"][0]["explanation"] = "NO-FIT: unclear"
            records.append(r)
        path = _write_records(records, tmp_path)
        report = measure_stage1_gates(path)
        tax_gate = next(g for g in report.gates if g.name == "taxonomy_adequacy")
        assert tax_gate.value == 0.05  # 1/20
        assert tax_gate.passed is True  # 0.05 < 0.10

    def test_all_gates_report(self, tmp_path):
        records = [_make_valid_record(has_evaluator=True) for i in range(5)]
        path = _write_records(records, tmp_path)
        report = measure_stage1_gates(path)
        assert len(report.gates) == 5
        assert report.n_records == 5
        assert isinstance(report.decision, str)

    def test_summary_string(self, tmp_path):
        records = [_make_valid_record() for i in range(3)]
        path = _write_records(records, tmp_path)
        report = measure_stage1_gates(path)
        s = report.summary()
        assert "Stage 1 Gate Report" in s
        assert "annotation_agreement" in s

    def test_with_two_annotators(self, tmp_path):
        """Test annotation agreement gate with two annotators."""
        records = []
        for i in range(10):
            r = _make_valid_record()
            r["annotations"] = [
                {"annotator_id": "ann1", "faithfulness_category": "faithful",
                 "explanation": "ok", "timestamp": "2026-09-01T00:00:00+00:00"},
                {"annotator_id": "ann2", "faithfulness_category": "faithful",
                 "explanation": "ok", "timestamp": "2026-09-01T00:00:00+00:00"},
            ]
            records.append(r)
        path = _write_records(records, tmp_path)
        report = measure_stage1_gates(path)
        ann_gate = next(g for g in report.gates if g.name == "annotation_agreement")
        # Perfect agreement -> kappa = 1.0
        assert ann_gate.value == 1.0
