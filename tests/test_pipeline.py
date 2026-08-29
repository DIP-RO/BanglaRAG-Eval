"""Tests for the Milestone 2 pipeline modules.

Tests cover:
- Source document loading and passage splitting
- Question generation across language conditions
- BM25 lexical retrieval
- Evidence construction and corruption
- Ollama generation adapter (mocked, no real API calls)
- Pipeline orchestrator (offline mode)
- Record schema compliance
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from banglarag_eval.pipeline.sources import (
    SourceDocument,
    load_local_documents,
    load_ragtruth_english,
    split_into_passages,
)
from banglarag_eval.pipeline.questions import (
    generate_questions,
    to_code_mixed,
    to_banglish,
    to_translated_bangla,
    CURATED_BANGLA_DOCUMENTS,
    ENGLISH_BASELINE_DOCUMENTS,
)
from banglarag_eval.pipeline.retriever import (
    LexicalRetriever,
    RetrievedDocument,
    RetrievalResult,
)
from banglarag_eval.pipeline.evidence import (
    build_evidence,
    build_correct_evidence,
    build_partially_relevant_evidence,
    build_irrelevant_evidence,
    build_contradictory_evidence,
    build_missing_evidence,
)
from banglarag_eval.pipeline.generator import (
    generate,
    _strip_thinking,
    check_ollama_available,
    GenerationResult,
)
from banglarag_eval.pipeline.orchestrator import (
    build_pilot_records,
    save_pilot_dataset,
    run_pipeline,
)
from banglarag_eval.schema import validate_record


# ── Source document tests ───────────────────────────────────────


class TestSourceDocument:
    def test_source_document_creation(self):
        doc = SourceDocument(
            document_id="test-001",
            text="এটি একটি পরীক্ষামূলক টেক্সট।",
            language="bn",
            source_collection="local_curated",
        )
        assert doc.document_id == "test-001"
        assert doc.language == "bn"
        assert doc.source_collection == "local_curated"

    def test_source_document_nfc_normalization(self):
        # NFC normalization should be applied
        doc = SourceDocument(
            document_id="test-nfc",
            text="বাংলা",
            language="bn",
            source_collection="test",
        )
        import unicodedata
        assert doc.text == unicodedata.normalize("NFC", "বাংলা")

    def test_source_document_to_dict(self):
        doc = SourceDocument(
            document_id="test-dict",
            text="hello",
            language="en",
            source_collection="test",
            source_url="http://example.com",
            license="MIT",
        )
        d = doc.to_dict()
        assert d["document_id"] == "test-dict"
        assert d["text"] == "hello"
        assert d["language"] == "en"
        assert d["source_url"] == "http://example.com"
        assert d["license"] == "MIT"

    def test_load_local_txt_files(self, tmp_path):
        (tmp_path / "doc1.txt").write_text("এই প্রথম নথি।", encoding="utf-8")
        (tmp_path / "doc2.txt").write_text("এই দ্বিতীয় নথি।", encoding="utf-8")

        docs = load_local_documents(tmp_path)
        assert len(docs) == 2
        assert docs[0].document_id == "doc1"
        assert docs[1].document_id == "doc2"
        assert all(d.language == "bn" for d in docs)

    def test_load_local_jsonl_file(self, tmp_path):
        jsonl_content = json.dumps({
            "document_id": "jdoc-001",
            "text": "JSONL test document",
            "language": "en",
            "source_collection": "test_jsonl",
        }) + "\n"
        (tmp_path / "docs.jsonl").write_text(jsonl_content, encoding="utf-8")

        docs = load_local_documents(tmp_path)
        assert len(docs) == 1
        assert docs[0].document_id == "jdoc-001"
        assert docs[0].language == "en"

    def test_load_local_empty_txt_skipped(self, tmp_path):
        (tmp_path / "empty.txt").write_text("", encoding="utf-8")
        (tmp_path / "real.txt").write_text("real content", encoding="utf-8")

        docs = load_local_documents(tmp_path)
        assert len(docs) == 1
        assert docs[0].document_id == "real"

    def test_load_local_directory_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_local_documents("/nonexistent/path/12345")

    def test_load_local_invalid_json_raises(self, tmp_path):
        (tmp_path / "bad.jsonl").write_text("{invalid json}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="invalid JSON"):
            load_local_documents(tmp_path)

    def test_split_into_passages_short_text(self):
        text = "ছোট টেক্সট।"
        passages = split_into_passages(text, max_chars=500)
        assert len(passages) == 1
        assert passages[0] == text

    def test_split_into_passages_long_text(self):
        text = "প্রথম বাক্য। " * 100
        passages = split_into_passages(text, max_chars=100)
        assert len(passages) > 1
        for p in passages:
            # Each passage should be roughly within limit (may exceed slightly)
            assert len(p) <= 200

    def test_split_into_passages_danda_delimiter(self):
        text = "প্রথম বাক্য। দ্বিতীয় বাক্য। তৃতীয় বাক্য।"
        passages = split_into_passages(text, max_chars=20)
        assert len(passages) > 1


# ── Question generation tests ───────────────────────────────────


class TestQuestionGeneration:
    def test_generate_questions_all_conditions(self):
        docs = CURATED_BANGLA_DOCUMENTS[:1]
        specs = generate_questions(docs)
        # 1 doc, 3 questions, 4 Bangla conditions (english skipped for bn docs)
        assert len(specs) == 12  # 3 questions * 4 conditions

    def test_generate_questions_native_bangla(self):
        docs = CURATED_BANGLA_DOCUMENTS[:1]
        specs = generate_questions(docs, language_conditions=["native_bangla"])
        assert len(specs) == 3
        assert all(s.language_condition == "native_bangla" for s in specs)
        assert all(s.question_language == "bn" for s in specs)
        assert all(s.data_origin == "native_authored" for s in specs)

    def test_generate_questions_code_mixed(self):
        docs = CURATED_BANGLA_DOCUMENTS[:1]
        specs = generate_questions(docs, language_conditions=["code_mixed"])
        assert len(specs) == 3
        assert all(s.language_condition == "code_mixed" for s in specs)
        # Code-mixed should contain English words
        assert any("capital" in s.question or "Bangladesh" in s.question for s in specs)

    def test_generate_questions_banglish(self):
        docs = CURATED_BANGLA_DOCUMENTS[:1]
        specs = generate_questions(docs, language_conditions=["banglish"])
        assert len(specs) == 3
        assert all(s.language_condition == "banglish" for s in specs)
        # Banglish should be romanized
        assert any("rajdhani" in s.question.lower() or "capital" in s.question.lower() for s in specs)

    def test_generate_questions_english_only_from_english_docs(self):
        docs = ENGLISH_BASELINE_DOCUMENTS[:1]
        specs = generate_questions(docs, language_conditions=["english"])
        assert len(specs) == 2
        assert all(s.language_condition == "english" for s in specs)
        assert all(s.question_language == "en" for s in specs)

    def test_generate_questions_english_skipped_for_bangla_docs(self):
        docs = CURATED_BANGLA_DOCUMENTS[:1]
        specs = generate_questions(docs, language_conditions=["english"])
        assert len(specs) == 0

    def test_generate_questions_translated_bangla_origin(self):
        docs = CURATED_BANGLA_DOCUMENTS[:1]
        specs = generate_questions(docs, language_conditions=["translated_bangla"])
        assert len(specs) == 3
        assert all(s.data_origin == "human_translated" for s in specs)

    def test_source_span_found(self):
        docs = CURATED_BANGLA_DOCUMENTS[:1]
        specs = generate_questions(docs, language_conditions=["native_bangla"])
        for spec in specs:
            assert "start_char" in spec.source_span
            assert "end_char" in spec.source_span
            assert "text" in spec.source_span
            assert spec.source_span["start_char"] >= 0
            assert spec.source_span["end_char"] > spec.source_span["start_char"]

    def test_to_code_mixed_known(self):
        result = to_code_mixed("বাংলাদেশের রাজধানীর নাম কী?", "code_mixed")
        assert "Bangladesh" in result
        assert "capital" in result

    def test_to_banglish_known(self):
        result = to_banglish("বাংলাদেশের রাজধানীর নাম কী?", "banglish")
        assert "rajdhani" in result.lower() or "capital" in result.lower()

    def test_to_translated_bangla_known(self):
        result = to_translated_bangla("What is the capital of Bangladesh?")
        assert "রাজধানী" in result

    def test_to_code_mixed_unknown_returns_original(self):
        result = to_code_mixed("unknown question?", "code_mixed")
        assert result == "unknown question?"

    def test_to_banglish_unknown_returns_original(self):
        result = to_banglish("unknown question?", "banglish")
        assert result == "unknown question?"


# ── Retriever tests ─────────────────────────────────────────────


class TestLexicalRetriever:
    def _make_docs(self):
        return [
            SourceDocument(
                document_id="doc-a",
                text="বাংলাদেশের রাজধানী ঢাকা। ঢাকা বুড়িগঙ্গা নদীর তীরে অবস্থিত।",
                language="bn",
                source_collection="test",
            ),
            SourceDocument(
                document_id="doc-b",
                text="পদ্মা সেতু বাংলাদেশের দীর্ঘতম সেতু। এটি পদ্মা নদীর উপরে নির্মিত।",
                language="bn",
                source_collection="test",
            ),
        ]

    def test_retriever_index_and_retrieve(self):
        retriever = LexicalRetriever()
        retriever.index(self._make_docs())
        result = retriever.retrieve("বাংলাদেশের রাজধানী কী?", top_k=1)
        assert len(result.documents) == 1
        assert result.documents[0].rank == 1
        assert result.documents[0].score > 0

    def test_retriever_top_k(self):
        retriever = LexicalRetriever()
        retriever.index(self._make_docs())
        result = retriever.retrieve("বাংলাদেশ", top_k=2)
        assert len(result.documents) == 2
        assert result.documents[0].rank == 1
        assert result.documents[1].rank == 2

    def test_retriever_empty_index(self):
        retriever = LexicalRetriever()
        result = retriever.retrieve("test", top_k=3)
        assert len(result.documents) == 0
        assert result.configuration["method"] == "bm25"

    def test_retriever_configuration(self):
        retriever = LexicalRetriever(k1=1.2, b=0.5)
        retriever.index(self._make_docs())
        result = retriever.retrieve("ঢাকা", top_k=1)
        assert result.configuration["method"] == "bm25"
        assert result.configuration["k1"] == 1.2
        assert result.configuration["b"] == 0.5
        assert result.configuration["top_k"] == 1
        assert result.configuration["num_indexed"] > 0

    def test_retriever_top_context(self):
        retriever = LexicalRetriever()
        retriever.index(self._make_docs())
        result = retriever.retrieve("ঢাকা", top_k=1)
        assert len(result.top_context) > 0
        assert "ঢাকা" in result.top_context

    def test_retriever_english_query(self):
        docs = [
            SourceDocument(
                document_id="en-doc",
                text="The capital of Bangladesh is Dhaka. It is on the Buriganga River.",
                language="en",
                source_collection="test",
            ),
        ]
        retriever = LexicalRetriever()
        retriever.index(docs)
        result = retriever.retrieve("What is the capital of Bangladesh?", top_k=1)
        assert len(result.documents) == 1
        assert "capital" in result.documents[0].text.lower() or "dhaka" in result.documents[0].text.lower()

    def test_retriever_scores_ranked_descending(self):
        retriever = LexicalRetriever()
        retriever.index(self._make_docs())
        result = retriever.retrieve("বাংলাদেশ ঢাকা", top_k=2)
        if len(result.documents) == 2:
            assert result.documents[0].score >= result.documents[1].score


# ── Evidence construction tests ─────────────────────────────────


class TestEvidenceConstruction:
    def _make_retrieval(self, n_docs=3):
        docs = [
            RetrievedDocument(
                document_id=f"doc-{i}",
                text=f"Document {i} content about topic.",
                rank=i + 1,
                score=1.0 - i * 0.1,
                source="test",
            )
            for i in range(n_docs)
        ]
        return RetrievalResult(query="test query", documents=docs)

    def test_correct_evidence(self):
        retrieval = self._make_retrieval()
        result = build_correct_evidence(retrieval)
        assert result.evidence_condition == "correct"
        assert result.corruption_metadata is None
        assert "Document 0" in result.retrieved_context

    def test_partially_relevant_evidence(self):
        retrieval = self._make_retrieval(n_docs=3)
        result = build_partially_relevant_evidence(retrieval, seed=42)
        assert result.evidence_condition == "partially_relevant"
        assert result.corruption_metadata is not None
        assert result.corruption_metadata["corruption_type"] == "document_removal"
        assert len(result.retrieval_documents) == 1

    def test_partially_relevant_single_doc_no_corruption(self):
        retrieval = self._make_retrieval(n_docs=1)
        result = build_partially_relevant_evidence(retrieval, seed=42)
        assert result.evidence_condition == "partially_relevant"
        # With only 1 doc, no corruption metadata needed
        assert result.corruption_metadata is None

    def test_irrelevant_evidence(self):
        retrieval = self._make_retrieval()
        result = build_irrelevant_evidence(retrieval, seed=42)
        assert result.evidence_condition == "irrelevant"
        assert result.corruption_metadata is not None
        assert result.corruption_metadata["corruption_type"] == "context_replacement"
        assert result.retrieval_documents[0]["document_id"] == "irrelevant-passage"

    def test_irrelevant_evidence_bangla_text(self):
        retrieval = self._make_retrieval()
        result = build_irrelevant_evidence(retrieval, seed=42)
        # Should contain Bangla text
        assert any(ord(c) > 0x0980 for c in result.retrieved_context)

    def test_contradictory_evidence(self):
        retrieval = self._make_retrieval()
        result = build_contradictory_evidence(retrieval, seed=42)
        assert result.evidence_condition == "contradictory"
        assert result.corruption_metadata is not None
        assert result.corruption_metadata["corruption_type"] == "contradiction_injection"
        # Should have a contradiction prefix
        assert "ভুল" in result.retrieved_context or "অস্বীকার" in result.retrieved_context or "সঠিক নয়" in result.retrieved_context

    def test_missing_evidence(self):
        result = build_missing_evidence(seed=42)
        assert result.evidence_condition == "missing"
        assert result.retrieved_context == ""
        assert len(result.retrieval_documents) == 0
        assert result.corruption_metadata is not None
        assert result.corruption_metadata["corruption_type"] == "evidence_removal"

    def test_build_evidence_dispatch(self):
        retrieval = self._make_retrieval()
        for condition in ["correct", "partially_relevant", "irrelevant", "contradictory", "missing"]:
            result = build_evidence(condition, retrieval, seed=42)
            assert result.evidence_condition == condition

    def test_build_evidence_unknown_condition(self):
        retrieval = self._make_retrieval()
        with pytest.raises(ValueError, match="unknown evidence condition"):
            build_evidence("unknown_condition", retrieval)

    def test_irrelevant_evidence_reproducible(self):
        retrieval = self._make_retrieval()
        r1 = build_irrelevant_evidence(retrieval, seed=42)
        r2 = build_irrelevant_evidence(retrieval, seed=42)
        assert r1.retrieved_context == r2.retrieved_context

    def test_contradictory_evidence_reproducible(self):
        retrieval = self._make_retrieval()
        r1 = build_contradictory_evidence(retrieval, seed=42)
        r2 = build_contradictory_evidence(retrieval, seed=42)
        assert r1.retrieved_context == r2.retrieved_context


# ── Generator tests (mocked, no real Ollama calls) ──────────────


class TestGenerator:
    def test_strip_thinking_with_tags(self):
        text = "<think>ing\nLet me think about this.\n\n</think>\n\nThe answer is Dhaka."
        result = _strip_thinking(text)
        assert "Dhaka" in result
        assert "Let me think" not in result

    def test_strip_thinking_no_thinking(self):
        text = "The capital is Dhaka."
        result = _strip_thinking(text)
        assert result == "The capital is Dhaka."

    def test_strip_thinking_empty(self):
        result = _strip_thinking("")
        assert result == ""

    def test_strip_thinking_only_thinking(self):
        text = "<think>just thinking\n\n</think>\n\n"
        result = _strip_thinking(text)
        # Should return empty or minimal
        assert "thinking" not in result

    @patch("banglarag_eval.pipeline.generator.urllib.request.urlopen")
    def test_generate_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "response": "The capital of Bangladesh is Dhaka.",
            "eval_count": 10,
        }).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = generate(
            question="What is the capital?",
            context="Bangladesh is a country in South Asia.",
        )

        assert isinstance(result, GenerationResult)
        assert "Dhaka" in result.answer
        assert result.eval_count == 10
        assert result.latency_seconds >= 0

    @patch("banglarag_eval.pipeline.generator.urllib.request.urlopen")
    def test_generate_strips_thinking(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "response": "<think>thinking about it\n\n</think>\n\nThe answer is 42.",
            "eval_count": 20,
        }).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = generate(question="What is the answer?", context="Some context.")
        assert "42" in result.answer
        assert "thinking" not in result.answer

    @patch("banglarag_eval.pipeline.generator.urllib.request.urlopen")
    def test_generate_connection_error(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        with pytest.raises(ConnectionError, match="Cannot connect to Ollama"):
            generate(question="test", context="test")

    @patch("banglarag_eval.pipeline.generator.urllib.request.urlopen")
    def test_check_ollama_available_true(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        assert check_ollama_available() is True

    @patch("banglarag_eval.pipeline.generator.urllib.request.urlopen")
    def test_check_ollama_available_false(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        assert check_ollama_available() is False

    def test_generate_uses_env_model(self):
        # Test that model defaults to env var
        with patch.dict(os.environ, {"OLLAMA_MODEL": "test-model"}):
            with patch("banglarag_eval.pipeline.generator.urllib.request.urlopen") as mock_urlopen:
                mock_resp = MagicMock()
                mock_resp.read.return_value = json.dumps({
                    "response": "test", "eval_count": 1,
                }).encode("utf-8")
                mock_resp.__enter__ = MagicMock(return_value=mock_resp)
                mock_resp.__exit__ = MagicMock(return_value=False)
                mock_urlopen.return_value = mock_resp

                result = generate(question="q", context="c")
                assert result.model == "test-model"


# ── Orchestrator tests (offline mode) ───────────────────────────


class TestOrchestrator:
    def test_build_pilot_records_offline(self):
        records = build_pilot_records(
            language_conditions=["native_bangla"],
            evidence_conditions=["correct"],
            use_ollama=False,
        )
        assert len(records) > 0
        assert all(r["language_condition"] == "native_bangla" for r in records)
        assert all(r["evidence_condition"] == "correct" for r in records)
        assert all(r["generation"]["model"] == "placeholder_offline" for r in records)

    def test_build_pilot_records_all_conditions_offline(self):
        records = build_pilot_records(use_ollama=False)
        # 8 Bangla docs * 3 questions * 4 Bangla conditions = 96
        # 2 English docs * 2 questions * 1 English condition = 4
        # 5 evidence conditions each
        # Total: (96 + 4) * 5 = 500
        assert len(records) == 500

    def test_build_pilot_records_schema_valid(self):
        records = build_pilot_records(
            language_conditions=["native_bangla"],
            evidence_conditions=["correct"],
            use_ollama=False,
        )
        for record in records:
            # Should have required top-level fields
            assert "record_id" in record
            assert "language_condition" in record
            assert "evidence_condition" in record
            assert "source" in record
            assert "question" in record
            assert "generated_answer" in record
            assert "annotations" in record
            assert "gold_label" in record

    def test_build_pilot_records_has_source_span(self):
        records = build_pilot_records(
            language_conditions=["native_bangla"],
            evidence_conditions=["correct"],
            use_ollama=False,
        )
        for record in records:
            assert "source_span" in record["source"]
            assert "start_char" in record["source"]["source_span"]
            assert "end_char" in record["source"]["source_span"]

    def test_build_pilot_records_has_retrieval_config(self):
        records = build_pilot_records(
            language_conditions=["native_bangla"],
            evidence_conditions=["correct"],
            use_ollama=False,
        )
        for record in records:
            assert record["retriever"] == "lexical_bm25"
            assert "retrieval_configuration" in record
            assert record["retrieval_configuration"]["method"] == "bm25"

    def test_build_pilot_records_corruption_metadata(self):
        records = build_pilot_records(
            language_conditions=["native_bangla"],
            evidence_conditions=["irrelevant"],
            use_ollama=False,
        )
        for record in records:
            assert record["corruption_metadata"] is not None
            assert record["corruption_metadata"]["corruption_type"] == "context_replacement"

    def test_build_pilot_records_missing_evidence_empty_context(self):
        records = build_pilot_records(
            language_conditions=["native_bangla"],
            evidence_conditions=["missing"],
            use_ollama=False,
        )
        for record in records:
            assert record["retrieved_context"] == ""
            assert len(record["retrieval_documents"]) == 0

    def test_build_pilot_records_no_gold_labels(self):
        """Gold labels must not be set automatically."""
        records = build_pilot_records(use_ollama=False)
        for record in records:
            assert record["gold_label"] is None
            assert record["adjudication"] is None
            assert record["annotations"] == []

    def test_build_pilot_records_intended_answer_hidden(self):
        """intended_answer should be in question metadata, not in gold_label."""
        records = build_pilot_records(
            language_conditions=["native_bangla"],
            evidence_conditions=["correct"],
            use_ollama=False,
        )
        for record in records:
            assert record["question"]["intended_answer"] is not None
            assert record["gold_label"] is None

    def test_save_pilot_dataset(self, tmp_path):
        records = build_pilot_records(
            language_conditions=["native_bangla"],
            evidence_conditions=["correct"],
            use_ollama=False,
        )
        output = tmp_path / "test_pilot.jsonl"
        save_pilot_dataset(records, output)

        assert output.exists()
        lines = output.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == len(records)
        for line in lines:
            data = json.loads(line)
            assert "record_id" in data

    def test_save_pilot_dataset_creates_parent_dir(self, tmp_path):
        records = [{"record_id": "test", "data": "value"}]
        output = tmp_path / "subdir" / "test.jsonl"
        save_pilot_dataset(records, output)
        assert output.exists()

    def test_run_pipeline_offline(self, tmp_path):
        output = tmp_path / "pipeline_test.jsonl"
        summary = run_pipeline(
            output_path=output,
            language_conditions=["native_bangla"],
            evidence_conditions=["correct"],
            use_ollama=False,
        )
        assert summary["total_records"] > 0
        assert summary["ollama_used"] is False
        assert output.exists()
        assert "native_bangla" in summary["language_distribution"]

    def test_run_pipeline_summary_distributions(self, tmp_path):
        output = tmp_path / "pipeline_full.jsonl"
        summary = run_pipeline(
            output_path=output,
            use_ollama=False,
        )
        assert summary["total_records"] == 500
        assert len(summary["language_distribution"]) == 5
        assert len(summary["evidence_distribution"]) == 5
        # Bangla conditions: 8 docs × 3 questions × 5 evidence = 120 each
        # English condition: 2 docs × 2 questions × 5 evidence = 20
        for lang, count in summary["language_distribution"].items():
            if lang == "english":
                assert count == 20
            else:
                assert count == 120

    def test_record_ids_deterministic(self):
        records1 = build_pilot_records(
            language_conditions=["native_bangla"],
            evidence_conditions=["correct"],
            use_ollama=False,
            seed=42,
        )
        records2 = build_pilot_records(
            language_conditions=["native_bangla"],
            evidence_conditions=["correct"],
            use_ollama=False,
            seed=42,
        )
        ids1 = [r["record_id"] for r in records1]
        ids2 = [r["record_id"] for r in records2]
        assert ids1 == ids2

    def test_bangla_unicode_preserved(self):
        records = build_pilot_records(
            language_conditions=["native_bangla"],
            evidence_conditions=["correct"],
            use_ollama=False,
        )
        for record in records:
            # Source text should contain Bangla characters (native_bangla uses Bangla docs)
            assert any(ord(c) > 0x0980 for c in record["source"]["source_text"])
            # Question should contain Bangla characters
            assert any(ord(c) > 0x0980 for c in record["question"]["text"])

    def test_code_mixed_contains_english(self):
        records = build_pilot_records(
            language_conditions=["code_mixed"],
            evidence_conditions=["correct"],
            use_ollama=False,
        )
        for record in records:
            # Code-mixed questions should contain ASCII letters
            assert any(c.isascii() and c.isalpha() for c in record["question"]["text"])

    def test_banglish_is_romanized(self):
        records = build_pilot_records(
            language_conditions=["banglish"],
            evidence_conditions=["correct"],
            use_ollama=False,
        )
        for record in records:
            # Banglish should not contain Bengali script
            assert not any(ord(c) > 0x0980 for c in record["question"]["text"])
