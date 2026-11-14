"""Tests for dense and hybrid retrieval configurations."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from banglarag_eval.pipeline import orchestrator
from banglarag_eval.pipeline.retriever import (
    DenseRetriever,
    HybridRetriever,
    LexicalRetriever,
    SemFuseRetriever,
    build_retriever,
)
from banglarag_eval.pipeline.sources import SourceDocument


def _documents() -> list[SourceDocument]:
    return [
        SourceDocument(
            document_id="dhaka",
            text="বাংলাদেশের রাজধানী ঢাকা।",
            language="bn",
            source_collection="test",
        ),
        SourceDocument(
            document_id="bridge",
            text="পদ্মা সেতু বাংলাদেশের দীর্ঘতম সেতু।",
            language="bn",
            source_collection="test",
        ),
        SourceDocument(
            document_id="music",
            text="A guitar is a musical instrument.",
            language="en",
            source_collection="test",
        ),
    ]


def _fake_embeddings(texts):
    vectors = []
    for text in texts:
        lowered = text.lower()
        vectors.append(
            [
                float("ঢাকা" in text or "রাজধানী" in text or "capital" in lowered),
                float("সেতু" in text or "bridge" in lowered),
                float("guitar" in lowered or "music" in lowered),
                1.0,
            ]
        )
    return vectors


class TestDenseRetriever:
    def test_ranks_by_cosine_similarity_and_records_configuration(self):
        retriever = DenseRetriever(embedding_function=_fake_embeddings)
        retriever.index(_documents())

        result = retriever.retrieve("বাংলাদেশের রাজধানী কী?", top_k=2)

        assert [document.document_id for document in result.documents][0] == "dhaka"
        assert result.documents[0].score >= result.documents[1].score
        assert result.configuration["method"] == "dense_cosine"
        assert result.configuration["embedding_model"] == "bge-m3"
        assert result.configuration["embedding_provider"] == "custom"
        assert result.configuration["top_k"] == 2

    def test_empty_index_returns_empty_result(self):
        retriever = DenseRetriever(embedding_function=_fake_embeddings)
        retriever.index([])

        result = retriever.retrieve("query")

        assert result.documents == []
        assert result.configuration["method"] == "dense_cosine"

    def test_rejects_embedding_count_mismatch(self):
        retriever = DenseRetriever(embedding_function=lambda texts: [])
        with pytest.raises(RuntimeError, match="wrong number"):
            retriever.index(_documents())


def test_retriever_index_deduplicates_exact_passages():
    documents = _documents()
    documents.append(
        SourceDocument(
            document_id="dhaka-duplicate",
            text="বাংলাদেশের রাজধানী ঢাকা।",
            language="bn",
            source_collection="test",
        )
    )
    retriever = LexicalRetriever()

    retriever.index(documents)
    result = retriever.retrieve("বাংলাদেশের রাজধানী কী?", top_k=4)

    assert result.configuration["num_indexed"] == 3
    assert len({item.text for item in result.documents}) == len(result.documents)


class TestHybridRetriever:
    def test_fuses_lexical_and_dense_rankings(self):
        retriever = HybridRetriever(embedding_function=_fake_embeddings)
        retriever.index(_documents())

        result = retriever.retrieve("ঢাকা রাজধানী", top_k=2)

        assert result.documents[0].document_id == "dhaka"
        assert result.documents[0].rank == 1
        assert result.configuration["method"] == "hybrid_rrf"
        assert result.configuration["fusion"] == "weighted_reciprocal_rank"
        assert result.configuration["embedding_model"] == "bge-m3"
        assert result.configuration["lexical_configuration"]["method"] == "bm25"
        assert result.configuration["dense_configuration"]["method"] == "dense_cosine"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"lexical_weight": -1.0},
            {"lexical_weight": 0.0, "dense_weight": 0.0},
            {"rrf_k": 0},
            {"candidate_multiplier": 0},
        ],
    )
    def test_rejects_invalid_fusion_settings(self, kwargs):
        with pytest.raises(ValueError):
            HybridRetriever(embedding_function=_fake_embeddings, **kwargs)


class _FakeSemFuse:
    version = "test-0.10"

    def __init__(self, **configuration):
        self.configuration = configuration
        self.documents = []

    def add(self, text, **kwargs):
        self.documents.append({"text": text, **kwargs})
        return 1

    def persist(self):
        return None

    def search(self, query, **kwargs):
        document = self.documents[0]
        return [
            SimpleNamespace(
                document_id=document["document_id"],
                chunk_id="chunk-1",
                text=document["text"],
                score=0.91,
                source=document["source"],
                metadata=document["metadata"],
            )
        ]


class TestSemFuseRetriever:
    def test_maps_semfuse_results_and_configuration(self, tmp_path):
        retriever = SemFuseRetriever(
            client_factory=_FakeSemFuse,
            storage_path=tmp_path,
            mode="hybrid",
            fusion_method="rrf",
        )
        retriever.index(_documents())

        result = retriever.retrieve("বাংলাদেশের রাজধানী কী?", top_k=2)

        assert result.documents[0].document_id == "dhaka"
        assert result.documents[0].source == "test"
        assert result.configuration["method"] == "semfuse"
        assert result.configuration["package_version"] == "test-0.10"
        assert result.configuration["mode"] == "hybrid"
        assert result.configuration["fusion"] == "rrf"
        assert result.configuration["num_indexed"] == 3

    @pytest.mark.parametrize("mode", ["invalid", "dense"])
    def test_rejects_unknown_mode(self, mode):
        with pytest.raises(ValueError, match="mode"):
            SemFuseRetriever(mode=mode)


class TestRetrieverFactory:
    def test_builds_all_supported_retrievers(self):
        assert isinstance(build_retriever("bm25"), LexicalRetriever)
        assert isinstance(
            build_retriever("dense", embedding_function=_fake_embeddings),
            DenseRetriever,
        )
        assert isinstance(
            build_retriever("hybrid", embedding_function=_fake_embeddings),
            HybridRetriever,
        )
        assert isinstance(
            build_retriever("semfuse", semfuse_client_factory=_FakeSemFuse),
            SemFuseRetriever,
        )

    def test_rejects_unknown_retriever(self):
        with pytest.raises(ValueError, match="unknown retrieval method"):
            build_retriever("unknown")


def test_orchestrator_records_dense_retrieval_provenance(monkeypatch):
    monkeypatch.setattr(
        orchestrator,
        "build_retriever",
        lambda method, **kwargs: DenseRetriever(
            model=kwargs["embedding_model"],
            embedding_function=_fake_embeddings,
        ),
    )

    records = orchestrator.build_pilot_records(
        language_conditions=["native_bangla"],
        evidence_conditions=["correct"],
        use_ollama=False,
        retrieval_method="dense",
        retrieval_top_k=2,
    )

    assert records
    for record in records:
        assert record["retriever"] == "dense_bge_m3"
        assert record["embedding_model"] == "bge-m3"
        assert record["reranker"] is None
        assert record["retrieval_configuration"]["method"] == "dense_cosine"
        assert record["retrieval_configuration"]["top_k"] == 2
        assert all("source" in document for document in record["retrieval_documents"])
