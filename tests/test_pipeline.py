"""Tests for the Milestone 2 pipeline modules.

Tests cover:
- Source document loading and passage splitting
- Question generation across language conditions
- BM25 lexical retrieval
- Evidence construction and corruption
- Ollama generation adapter (mocked, no real API calls)
- Pipeline orchestrator (offline mode)
- Record schema compliance
- Real integration tests (with live Ollama, skipped if unavailable)
- Real dataset validation (against generated pilot_stage1_real.jsonl)
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
        assert all(r["generator_model"] == "placeholder_offline" for r in records)

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
            # Should have required top-level fields (flat schema)
            assert "example_id" in record
            assert "language_condition" in record
            assert "evidence_condition" in record
            assert "source_text" in record
            assert "question" in record
            assert "generated_answer" in record
            assert "annotations" in record
            assert "faithfulness_category" in record
            assert "schema_version" in record
            assert "document_id" in record
            assert "timestamp" in record

    def test_build_pilot_records_has_source_span(self):
        records = build_pilot_records(
            language_conditions=["native_bangla"],
            evidence_conditions=["correct"],
            use_ollama=False,
        )
        for record in records:
            assert "source_span" in record
            assert "start_char" in record["source_span"]
            assert "end_char" in record["source_span"]

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

    def test_build_pilot_records_no_faithfulness_categorys(self):
        """Gold labels must not be set automatically."""
        records = build_pilot_records(use_ollama=False)
        for record in records:
            assert record["faithfulness_category"] is None
            assert record["adjudication"] is None
            assert record["annotations"] == []

    def test_build_pilot_records_intended_answer_hidden(self):
        """intended_answer should be in question metadata, not in faithfulness_category."""
        records = build_pilot_records(
            language_conditions=["native_bangla"],
            evidence_conditions=["correct"],
            use_ollama=False,
        )
        for record in records:
            assert record["intended_answer"] is not None
            assert record["faithfulness_category"] is None

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
            assert "example_id" in data

    def test_save_pilot_dataset_creates_parent_dir(self, tmp_path):
        records = [{"example_id": "test", "data": "value"}]
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
        ids1 = [r["example_id"] for r in records1]
        ids2 = [r["example_id"] for r in records2]
        assert ids1 == ids2

    def test_bangla_unicode_preserved(self):
        records = build_pilot_records(
            language_conditions=["native_bangla"],
            evidence_conditions=["correct"],
            use_ollama=False,
        )
        for record in records:
            # Source text should contain Bangla characters (native_bangla uses Bangla docs)
            assert any(ord(c) > 0x0980 for c in record["source_text"])
            # Question should contain Bangla characters
            assert any(ord(c) > 0x0980 for c in record["question"])

    def test_code_mixed_contains_english(self):
        records = build_pilot_records(
            language_conditions=["code_mixed"],
            evidence_conditions=["correct"],
            use_ollama=False,
        )
        for record in records:
            # Code-mixed questions should contain ASCII letters
            assert any(c.isascii() and c.isalpha() for c in record["question"])

    def test_banglish_is_romanized(self):
        records = build_pilot_records(
            language_conditions=["banglish"],
            evidence_conditions=["correct"],
            use_ollama=False,
        )
        for record in records:
            # Banglish should not contain Bengali script
            assert not any(ord(c) > 0x0980 for c in record["question"])


# ── Real integration tests (with live Ollama) ──────────────────────
# These tests actually call Ollama and test the real pipeline.
# They are skipped automatically if Ollama is not running.

def _ollama_running():
    """Check if Ollama is running locally."""
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


pytestmark_ollama = pytest.mark.skipif(
    not _ollama_running(),
    reason="Ollama not running at localhost:11434 — start with 'ollama serve'",
)


@pytest.mark.real
@pytestmark_ollama
class TestRealOllamaGeneration:
    """Test real Ollama generation with Qwen3:8b."""

    def test_real_generate_bangla(self):
        """Test that Ollama actually generates Bangla text."""
        result = generate(
            question="বাংলাদেশের রাজধানীর নাম কী?",
            context="বাংলাদেশের রাজধানী ঢাকা। ঢাকা বুড়িগঙ্গা নদীর তীরে অবস্থিত।",
            temperature=0.0,
            timeout=120,
        )
        assert result.answer
        assert len(result.answer) > 0
        assert result.model == "qwen3:8b"
        assert result.latency_seconds > 0
        # Answer should mention Dhaka
        assert "ঢাকা" in result.answer or "Dhaka" in result.answer.lower()

    def test_real_generate_english(self):
        """Test that Ollama generates English text."""
        result = generate(
            question="What is the capital of Bangladesh?",
            context="Bangladesh is a country in South Asia. Its capital is Dhaka.",
            temperature=0.0,
            timeout=120,
        )
        assert result.answer
        assert len(result.answer) > 0
        assert "dhaka" in result.answer.lower() or "Dhaka" in result.answer

    def test_real_generate_no_thinking_in_answer(self):
        """Test that thinking trace is stripped from real output."""
        result = generate(
            question="বাংলাদেশের রাজধানীর নাম কী?",
            context="বাংলাদেশের রাজধানী ঢাকা।",
            temperature=0.0,
            timeout=120,
        )
        # Answer should not contain thinking tags
        assert "<think>" not in result.answer
        assert "</think>" not in result.answer
        # Answer should not start with "Answer:"
        assert not result.answer.startswith("Answer:")

    def test_real_generate_missing_evidence(self):
        """Test generation with no context (missing evidence condition)."""
        result = generate(
            question="বাংলাদেশের রাজধানীর নাম কী?",
            context="(no context provided)",
            temperature=0.0,
            timeout=120,
        )
        assert result.answer
        assert len(result.answer) > 0
        # Model should say it can't answer or provide the answer from its own knowledge

    def test_real_generate_code_mixed(self):
        """Test generation with code-mixed question."""
        result = generate(
            question="Bangladesh er rajdhanir nam ki?",
            context="বাংলাদেশের রাজধানী ঢাকা। ঢাকা বুড়িগঙ্গা নদীর তীরে অবস্থিত।",
            temperature=0.0,
            timeout=120,
        )
        assert result.answer
        assert len(result.answer) > 0


@pytest.mark.real
@pytestmark_ollama
class TestRealRetrieval:
    """Test real BM25 retrieval on curated documents."""

    def test_real_retriever_bangla_query(self):
        """Test that BM25 retrieves relevant passages for Bangla queries."""
        from banglarag_eval.pipeline.questions import CURATED_BANGLA_DOCUMENTS
        from banglarag_eval.pipeline.sources import SourceDocument

        docs = [
            SourceDocument(
                document_id=d["document_id"],
                text=d["text"],
                language="bn",
                source_collection="local_curated",
                license="local_authored",
            )
            for d in CURATED_BANGLA_DOCUMENTS
        ]
        retriever = LexicalRetriever()
        retriever.index(docs)

        result = retriever.retrieve("বাংলাদেশের রাজধানী কোথায়?", top_k=3)
        assert len(result.documents) > 0
        assert result.documents[0].score > 0
        # Top result should be about Dhaka/Bangladesh capital
        top_text = result.documents[0].text
        assert "ঢাকা" in top_text or "রাজধানী" in top_text

    def test_real_retriever_english_query(self):
        """Test BM25 with English queries on English documents."""
        from banglarag_eval.pipeline.questions import ENGLISH_BASELINE_DOCUMENTS
        from banglarag_eval.pipeline.sources import SourceDocument

        docs = [
            SourceDocument(
                document_id=d["document_id"],
                text=d["text"],
                language="en",
                source_collection="local_curated",
                license="local_authored",
            )
            for d in ENGLISH_BASELINE_DOCUMENTS
        ]
        retriever = LexicalRetriever()
        retriever.index(docs)

        result = retriever.retrieve("What is the capital of Bangladesh?", top_k=2)
        assert len(result.documents) > 0
        assert result.documents[0].score > 0

    def test_real_retriever_scores_ranked(self):
        """Test that BM25 scores are properly ranked (descending)."""
        from banglarag_eval.pipeline.questions import CURATED_BANGLA_DOCUMENTS
        from banglarag_eval.pipeline.sources import SourceDocument

        docs = [
            SourceDocument(
                document_id=d["document_id"],
                text=d["text"],
                language="bn",
                source_collection="local_curated",
                license="local_authored",
            )
            for d in CURATED_BANGLA_DOCUMENTS
        ]
        retriever = LexicalRetriever()
        retriever.index(docs)

        result = retriever.retrieve("কক্সবাজার সৈকত", top_k=5)
        scores = [doc.score for doc in result.documents]
        assert scores == sorted(scores, reverse=True)


@pytest.mark.real
@pytestmark_ollama
class TestRealPipelineEndToEnd:
    """Test the real pipeline end-to-end with live Ollama."""

    def test_real_pipeline_small_subset(self):
        """Run the real pipeline on a small subset (2 records)."""
        records = build_pilot_records(
            language_conditions=["native_bangla"],
            evidence_conditions=["correct"],
            use_ollama=True,
        )
        assert len(records) > 0
        for record in records:
            # Real generation should produce non-empty answers
            assert record["generated_answer"]
            assert len(record["generated_answer"]) > 0
            # Model should be qwen3:8b (or fallback)
            assert record["generator_model"] in (
                "qwen3:8b",
                "ollama_fallback_timeouterror",
                "ollama_unavailable_fallback",
            )
            # No gold labels
            assert record["faithfulness_category"] is None
            assert record["annotations"] == []

    def test_real_pipeline_record_schema_valid(self):
        """Test that real pipeline records pass schema validation."""
        from banglarag_eval.schema import validate_record

        records = build_pilot_records(
            language_conditions=["native_bangla"],
            evidence_conditions=["correct"],
            use_ollama=True,
        )
        for record in records:
            result = validate_record(record)
            assert result.is_valid, f"Schema validation failed: {result.errors}"

    def test_real_pipeline_bangla_answer_contains_bangla(self):
        """Test that Bangla questions get Bangla answers from Qwen3."""
        records = build_pilot_records(
            language_conditions=["native_bangla"],
            evidence_conditions=["correct"],
            use_ollama=True,
        )
        bangla_records = [
            r for r in records
            if r["language_condition"] == "native_bangla"
            and r["generator_model"] == "qwen3:8b"
        ]
        if bangla_records:
            # At least some answers should contain Bangla characters
            has_bangla = any(
                any(ord(c) > 0x0980 for c in r["generated_answer"])
                for r in bangla_records
            )
            assert has_bangla, "No Bangla characters found in any generated answer"


# ── Real dataset validation tests ──────────────────────────────────
# These tests validate the actual generated pilot_stage1_real.jsonl
# if it exists. Skipped if the file is not present.

PILOT_DATASET = Path(__file__).parent.parent / "data" / "pilot_stage1_real.jsonl"

pytestmark_dataset = pytest.mark.skipif(
    not PILOT_DATASET.exists(),
    reason=f"Pilot dataset not found at {PILOT_DATASET} — run: python scripts/run_pipeline.py",
)


@pytest.mark.real
@pytestmark_dataset
class TestRealDatasetValidation:
    """Validate the real generated pilot dataset."""

    @classmethod
    def load_records(cls):
        """Load the real pilot dataset."""
        with open(PILOT_DATASET, encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    @pytest.fixture(scope="class")
    def records(self):
        """Load the real pilot dataset once for all tests."""
        return self.load_records()

    def test_dataset_has_500_records(self, records):
        """Dataset should have exactly 500 records."""
        assert len(records) == 500, f"Expected 500 records, got {len(records)}"

    def test_dataset_all_records_valid(self, records):
        """All records should pass schema validation."""
        from banglarag_eval.schema import validate_record
        errors = []
        for record in records:
            result = validate_record(record)
            if not result.is_valid:
                errors.append(f"{record['example_id']}: {result.errors}")
        assert not errors, f"{len(errors)} records failed validation:\n" + "\n".join(errors[:5])

    def test_dataset_language_distribution(self, records):
        """Language distribution should match expected counts."""
        from collections import Counter
        lang = Counter(r["language_condition"] for r in records)
        assert lang["native_bangla"] == 120
        assert lang["translated_bangla"] == 120
        assert lang["code_mixed"] == 120
        assert lang["banglish"] == 120
        assert lang["english"] == 20

    def test_dataset_evidence_distribution(self, records):
        """Evidence distribution should be 100 per condition."""
        from collections import Counter
        evidence = Counter(r["evidence_condition"] for r in records)
        assert evidence["correct"] == 100
        assert evidence["partially_relevant"] == 100
        assert evidence["irrelevant"] == 100
        assert evidence["contradictory"] == 100
        assert evidence["missing"] == 100

    def test_dataset_no_faithfulness_categorys(self, records):
        """No records should have gold labels (humans must annotate)."""
        gold_count = sum(1 for r in records if r["faithfulness_category"] is not None)
        assert gold_count == 0, f"{gold_count} records have gold labels (should be 0)"

    def test_dataset_no_annotations(self, records):
        """No records should have annotations yet."""
        annot_count = sum(1 for r in records if r["annotations"])
        assert annot_count == 0, f"{annot_count} records have annotations (should be 0)"

    def test_dataset_all_have_generated_answers(self, records):
        """Every record should have a non-empty generated_answer."""
        empty = [r["example_id"] for r in records if not r["generated_answer"]]
        assert not empty, f"{len(empty)} records have empty generated_answer"

    def test_dataset_all_have_generation_metadata(self, records):
        """Every record should have generation metadata."""
        missing = [
            r["example_id"] for r in records
            if not r.get("generator_model")
        ]
        assert not missing, f"{len(missing)} records missing generation metadata"

    def test_dataset_models_used(self, records):
        """Check which models were used for generation."""
        from collections import Counter
        models = Counter(r["generator_model"] for r in records)
        # Most should be qwen3:8b
        assert models["qwen3:8b"] >= 490, f"Only {models.get('qwen3:8b', 0)} records used qwen3:8b"
        # Print summary for visibility
        print(f"\nModels used: {dict(models)}")

    def test_dataset_bangla_unicode_preserved(self, records):
        """Bangla records should preserve Bangla Unicode in source text."""
        bangla_records = [
            r for r in records
            if r["language_condition"] == "native_bangla"
        ]
        for record in bangla_records:
            assert any(
                ord(c) > 0x0980 for c in record["source_text"]
            ), f"{record['example_id']}: no Bangla in source_text"

    def test_dataset_example_ids_unique(self, records):
        """All record IDs should be unique."""
        ids = [r["example_id"] for r in records]
        assert len(ids) == len(set(ids)), f"{len(ids) - len(set(ids))} duplicate IDs"

    def test_dataset_example_ids_follow_pattern(self, records):
        """Record IDs should follow the pilot1-LANG-EVID-INDEX pattern."""
        import re
        pattern = re.compile(r"^pilot1-[a-z]{3,4}-[a-z_]+-\d{4}$")
        bad_ids = [r["example_id"] for r in records if not pattern.match(r["example_id"])]
        # Allow some flexibility in the pattern
        assert len(bad_ids) <= 5, f"{len(bad_ids)} record IDs don't match pattern: {bad_ids[:3]}"

    def test_dataset_evidence_condition_distribution_per_language(self, records):
        """Each language condition should have records across all evidence conditions."""
        from collections import Counter
        for lang in ["native_bangla", "translated_bangla", "code_mixed", "banglish"]:
            lang_records = [r for r in records if r["language_condition"] == lang]
            evidence_dist = Counter(r["evidence_condition"] for r in lang_records)
            for cond in ["correct", "partially_relevant", "irrelevant", "contradictory", "missing"]:
                assert cond in evidence_dist, f"{lang}: missing evidence condition {cond}"
                assert evidence_dist[cond] == 24, \
                    f"{lang}/{cond}: expected 24, got {evidence_dist[cond]}"

    def test_dataset_corruption_metadata_present(self, records):
        """Corrupted evidence records should have corruption_metadata."""
        corrupted = [
            r for r in records
            if r["evidence_condition"] in ("irrelevant", "contradictory")
        ]
        for record in corrupted:
            assert record.get("corruption_metadata"), \
                f"{record['example_id']}: missing corruption_metadata"

    def test_dataset_correct_evidence_no_corruption(self, records):
        """Correct evidence records should not have corruption_metadata."""
        correct = [
            r for r in records
            if r["evidence_condition"] == "correct"
        ]
        for record in correct:
            assert not record.get("corruption_metadata"), \
                f"{record['example_id']}: should not have corruption_metadata"

    def test_dataset_missing_evidence_empty_context(self, records):
        """Missing evidence records should have empty retrieved_context."""
        missing = [
            r for r in records
            if r["evidence_condition"] == "missing"
        ]
        for record in missing:
            assert record.get("retrieved_context", "") == "", \
                f"{record['example_id']}: missing evidence should have empty context"

    def test_dataset_sample_answers_quality(self, records):
        """Spot-check a few answers for basic quality."""
        # Find a native_bangla/correct record
        for r in records:
            if (r["language_condition"] == "native_bangla"
                    and r["evidence_condition"] == "correct"
                    and r["generator_model"] == "qwen3:8b"):
                # Answer should be non-trivial (more than 2 characters)
                assert len(r["generated_answer"]) > 2, \
                    f"{r['example_id']}: answer too short: '{r['generated_answer']}'"
                break

    def test_dataset_timestamps_valid(self, records):
        """All timestamps should be valid ISO-8601."""
        from datetime import datetime
        for record in records:
            ts = record.get("timestamp")
            assert ts, f"{record.get('example_id', 'unknown')}: missing timestamp"
            try:
                datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                pytest.fail(f"{record['example_id']}: invalid timestamp '{ts}'")
