"""Configurable lexical, dense, and hybrid retrieval for benchmark runs."""

from __future__ import annotations

import json
import math
import os
import re
import unicodedata
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .sources import SourceDocument, split_into_passages


EmbeddingFunction = Callable[[Sequence[str]], list[list[float]]]
SemFuseFactory = Callable[..., Any]


@dataclass
class RetrievedDocument:
    """A single retrieved passage with score and rank."""

    document_id: str
    text: str
    rank: int
    score: float
    source: str = "local"


@dataclass
class RetrievalResult:
    """Result of a retrieval call."""

    query: str
    documents: list[RetrievedDocument]
    configuration: dict[str, Any] = field(default_factory=dict)

    @property
    def top_context(self) -> str:
        """Return the exact concatenated context shown to the generator."""
        return "\n\n".join(doc.text for doc in self.documents)


class Retriever(ABC):
    """Common interface for independently configurable retrieval systems."""

    name: str
    embedding_model: str | None = None
    reranker: str | None = None

    @abstractmethod
    def index(self, documents: list[SourceDocument], passage_size: int = 500) -> None:
        """Index source documents."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> RetrievalResult:
        """Return ranked passages for a query."""


def _tokenize(text: str) -> list[str]:
    """Tokenize Bangla, English, and code-mixed text."""
    tokens = re.findall(r"[\u0980-\u09FFa-zA-Z0-9]+", text.lower())
    return [token for token in tokens if len(token) > 1]


def _passages(
    documents: list[SourceDocument], passage_size: int
) -> list[dict[str, str]]:
    indexed: list[dict[str, str]] = []
    seen_passages: set[str] = set()
    for document in documents:
        passages = split_into_passages(document.text, max_chars=passage_size)
        for index, passage in enumerate(passages):
            normalized_passage = unicodedata.normalize("NFC", passage).strip()
            if normalized_passage in seen_passages:
                continue
            seen_passages.add(normalized_passage)
            document_id = (
                f"{document.document_id}-p{index:03d}"
                if len(passages) > 1
                else document.document_id
            )
            indexed.append(
                {
                    "document_id": document_id,
                    "text": normalized_passage,
                    "source": document.source_collection,
                }
            )
    return indexed


def _validate_top_k(top_k: int) -> None:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")


def _model_identifier(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


class LexicalRetriever(Retriever):
    """BM25 lexical retriever with Bangla-aware tokenization."""

    name = "lexical_bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._documents: list[dict[str, str]] = []
        self._doc_tokens: list[list[str]] = []
        self._doc_freqs: dict[str, int] = {}
        self._avg_len = 0.0

    def index(self, documents: list[SourceDocument], passage_size: int = 500) -> None:
        self._documents = _passages(documents, passage_size)
        self._doc_tokens = []
        self._doc_freqs = {}

        for document in self._documents:
            tokens = _tokenize(document["text"])
            self._doc_tokens.append(tokens)
            for term in set(tokens):
                self._doc_freqs[term] = self._doc_freqs.get(term, 0) + 1

        self._avg_len = (
            sum(len(tokens) for tokens in self._doc_tokens) / len(self._doc_tokens)
            if self._doc_tokens
            else 0.0
        )

    def retrieve(self, query: str, top_k: int = 3) -> RetrievalResult:
        _validate_top_k(top_k)
        if not self._documents:
            return RetrievalResult(
                query=query, documents=[], configuration={"method": "bm25"}
            )

        query_tokens = _tokenize(query)
        n_documents = len(self._documents)
        scores = [
            self._bm25_score(query_tokens, tokens, n_documents)
            for tokens in self._doc_tokens
        ]
        ranked_indices = sorted(
            range(n_documents), key=lambda index: (-scores[index], index)
        )
        retrieved = [
            RetrievedDocument(
                document_id=self._documents[index]["document_id"],
                text=self._documents[index]["text"],
                rank=rank,
                score=round(scores[index], 6),
                source=self._documents[index]["source"],
            )
            for rank, index in enumerate(ranked_indices[:top_k], start=1)
        ]
        return RetrievalResult(
            query=query,
            documents=retrieved,
            configuration={
                "method": "bm25",
                "top_k": top_k,
                "k1": self.k1,
                "b": self.b,
                "analyzer": "unicode_bn_en_v1",
                "num_indexed": n_documents,
            },
        )

    def _bm25_score(
        self, query_tokens: list[str], doc_tokens: list[str], n_documents: int
    ) -> float:
        if not doc_tokens:
            return 0.0

        document_length = len(doc_tokens)
        term_frequencies: dict[str, int] = {}
        for token in doc_tokens:
            term_frequencies[token] = term_frequencies.get(token, 0) + 1

        score = 0.0
        for term in query_tokens:
            if term not in term_frequencies:
                continue
            frequency = term_frequencies[term]
            document_frequency = self._doc_freqs.get(term, 0)
            if document_frequency == 0:
                continue
            inverse_document_frequency = math.log(
                (n_documents - document_frequency + 0.5)
                / (document_frequency + 0.5)
                + 1.0
            )
            normalized_frequency = (frequency * (self.k1 + 1)) / (
                frequency
                + self.k1
                * (
                    1
                    - self.b
                    + self.b * document_length / self._avg_len
                )
            )
            score += inverse_document_frequency * normalized_frequency
        return score


class OllamaEmbeddingProvider:
    """Batch embedding client for Ollama's local ``/api/embed`` endpoint."""

    def __init__(
        self,
        model: str = "bge-m3",
        base_url: str | None = None,
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.base_url = (
            base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        ).rstrip("/")
        self.timeout = timeout

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = json.dumps(
            {"model": self.model, "input": list(texts), "truncate": True}
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (TimeoutError, urllib.error.URLError) as exc:
            raise ConnectionError(
                f"Cannot embed with Ollama model '{self.model}' at "
                f"{self.base_url}: {exc}"
            ) from exc

        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError(
                f"Ollama returned invalid embeddings for model '{self.model}'"
            )
        return [[float(value) for value in vector] for vector in embeddings]


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("embedding vectors must be non-empty and equally sized")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


class DenseRetriever(Retriever):
    """Dense cosine-similarity retriever using multilingual BGE-M3 embeddings."""

    def __init__(
        self,
        model: str = "bge-m3",
        embedding_function: EmbeddingFunction | None = None,
        base_url: str | None = None,
        timeout: int = 120,
    ) -> None:
        self.name = f"dense_{_model_identifier(model)}"
        self.embedding_model = model
        self._embedding_function = embedding_function or OllamaEmbeddingProvider(
            model=model, base_url=base_url, timeout=timeout
        )
        self._provider = "custom" if embedding_function is not None else "ollama"
        self._documents: list[dict[str, str]] = []
        self._embeddings: list[list[float]] = []

    def index(self, documents: list[SourceDocument], passage_size: int = 500) -> None:
        self._documents = _passages(documents, passage_size)
        self._embeddings = self._embedding_function(
            [document["text"] for document in self._documents]
        )
        if len(self._embeddings) != len(self._documents):
            raise RuntimeError("embedding provider returned the wrong number of vectors")

    def retrieve(self, query: str, top_k: int = 3) -> RetrievalResult:
        _validate_top_k(top_k)
        if not self._documents:
            return RetrievalResult(
                query=query,
                documents=[],
                configuration={
                    "method": "dense_cosine",
                    "embedding_model": self.embedding_model,
                    "embedding_provider": self._provider,
                },
            )

        query_embeddings = self._embedding_function([query])
        if len(query_embeddings) != 1:
            raise RuntimeError("embedding provider did not return one query vector")
        scores = [
            _cosine_similarity(query_embeddings[0], embedding)
            for embedding in self._embeddings
        ]
        ranked_indices = sorted(
            range(len(self._documents)), key=lambda index: (-scores[index], index)
        )
        retrieved = [
            RetrievedDocument(
                document_id=self._documents[index]["document_id"],
                text=self._documents[index]["text"],
                rank=rank,
                score=round(scores[index], 6),
                source=self._documents[index]["source"],
            )
            for rank, index in enumerate(ranked_indices[:top_k], start=1)
        ]
        return RetrievalResult(
            query=query,
            documents=retrieved,
            configuration={
                "method": "dense_cosine",
                "top_k": top_k,
                "embedding_model": self.embedding_model,
                "embedding_provider": self._provider,
                "similarity": "cosine",
                "num_indexed": len(self._documents),
            },
        )


class HybridRetriever(Retriever):
    """Weighted reciprocal-rank fusion of BM25 and BGE-M3 retrieval."""

    def __init__(
        self,
        model: str = "bge-m3",
        embedding_function: EmbeddingFunction | None = None,
        base_url: str | None = None,
        timeout: int = 120,
        lexical_weight: float = 0.5,
        dense_weight: float = 0.5,
        rrf_k: int = 60,
        candidate_multiplier: int = 4,
    ) -> None:
        if lexical_weight < 0 or dense_weight < 0:
            raise ValueError("hybrid weights cannot be negative")
        if lexical_weight + dense_weight == 0:
            raise ValueError("at least one hybrid weight must be positive")
        if rrf_k < 1 or candidate_multiplier < 1:
            raise ValueError("rrf_k and candidate_multiplier must be positive")

        total_weight = lexical_weight + dense_weight
        self.name = f"hybrid_bm25_{_model_identifier(model)}"
        self.lexical_weight = lexical_weight / total_weight
        self.dense_weight = dense_weight / total_weight
        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier
        self.embedding_model = model
        self._lexical = LexicalRetriever()
        self._dense = DenseRetriever(
            model=model,
            embedding_function=embedding_function,
            base_url=base_url,
            timeout=timeout,
        )

    def index(self, documents: list[SourceDocument], passage_size: int = 500) -> None:
        self._lexical.index(documents, passage_size=passage_size)
        self._dense.index(documents, passage_size=passage_size)

    def retrieve(self, query: str, top_k: int = 3) -> RetrievalResult:
        _validate_top_k(top_k)
        candidate_k = top_k * self.candidate_multiplier
        lexical = self._lexical.retrieve(query, top_k=candidate_k)
        dense = self._dense.retrieve(query, top_k=candidate_k)

        documents: dict[str, RetrievedDocument] = {}
        fused_scores: dict[str, float] = {}
        for result, weight in (
            (lexical, self.lexical_weight),
            (dense, self.dense_weight),
        ):
            for document in result.documents:
                documents.setdefault(document.document_id, document)
                fused_scores[document.document_id] = fused_scores.get(
                    document.document_id, 0.0
                ) + weight / (self.rrf_k + document.rank)

        ranked_ids = sorted(
            fused_scores,
            key=lambda document_id: (-fused_scores[document_id], document_id),
        )
        retrieved = [
            RetrievedDocument(
                document_id=document_id,
                text=documents[document_id].text,
                rank=rank,
                score=round(fused_scores[document_id], 8),
                source=documents[document_id].source,
            )
            for rank, document_id in enumerate(ranked_ids[:top_k], start=1)
        ]
        return RetrievalResult(
            query=query,
            documents=retrieved,
            configuration={
                "method": "hybrid_rrf",
                "top_k": top_k,
                "candidate_k": candidate_k,
                "fusion": "weighted_reciprocal_rank",
                "rrf_k": self.rrf_k,
                "lexical_weight": self.lexical_weight,
                "dense_weight": self.dense_weight,
                "embedding_model": self.embedding_model,
                "lexical_configuration": lexical.configuration,
                "dense_configuration": dense.configuration,
            },
        )


class SemFuseRetriever(Retriever):
    """Adapter for SemFuse multilingual semantic/keyword fusion retrieval."""

    def __init__(
        self,
        *,
        mode: str = "hybrid",
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        embedding_provider: str = "local",
        fusion_method: str = "rrf",
        reranker: str | None = None,
        storage_path: str | Path | None = None,
        client_factory: SemFuseFactory | None = None,
    ) -> None:
        normalized_mode = mode.strip().lower()
        if normalized_mode not in {"semantic", "keyword", "hybrid", "auto"}:
            raise ValueError(
                "SemFuse mode must be semantic, keyword, hybrid, or auto"
            )
        if fusion_method not in {"weighted", "rrf"}:
            raise ValueError("SemFuse fusion_method must be weighted or rrf")

        self.mode = normalized_mode
        self.name = f"semfuse_{normalized_mode}_{_model_identifier(embedding_model)}"
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.fusion_method = fusion_method
        self.reranker = reranker
        self._storage_path = Path(storage_path) if storage_path else None
        self._client_factory = client_factory
        self._client: Any | None = None
        self._temporary_storage: TemporaryDirectory[str] | None = None
        self._num_indexed = 0
        self._package_version: str | None = None

    def _build_client(self, passage_size: int, max_passage_size: int) -> Any:
        if self._client_factory is None:
            try:
                from semfuse import SemFuse
            except ImportError as exc:
                raise RuntimeError(
                    "SemFuse retrieval requires the optional dependency; install "
                    "with: pip install 'banglarag-eval[semfuse]'"
                ) from exc
            factory: SemFuseFactory = SemFuse
        else:
            factory = self._client_factory

        if self._storage_path is None:
            self._temporary_storage = TemporaryDirectory(prefix="banglarag-sem-fuse-")
            storage_path = self._temporary_storage.name
        else:
            storage_path = str(self._storage_path)

        return factory(
            embedding_model=self.embedding_model,
            embedding_provider=self.embedding_provider,
            storage_path=storage_path,
            collection="banglarag_eval",
            search_mode=self.mode,
            fusion_method=self.fusion_method,
            reranker=self.reranker,
            chunk_size=max(passage_size, max_passage_size, 1),
            chunk_overlap=0,
        )

    def index(self, documents: list[SourceDocument], passage_size: int = 500) -> None:
        passages = _passages(documents, passage_size)
        max_passage_size = max((len(item["text"]) for item in passages), default=1)
        self._client = self._build_client(passage_size, max_passage_size)

        for passage in passages:
            self._client.add(
                passage["text"],
                metadata={"source_collection": passage["source"]},
                source=passage["source"],
                document_id=passage["document_id"],
                persist=False,
            )
        self._client.persist()
        self._num_indexed = len(passages)
        self._package_version = getattr(self._client, "version", None)

    def retrieve(self, query: str, top_k: int = 3) -> RetrievalResult:
        _validate_top_k(top_k)
        if self._client is None:
            return RetrievalResult(
                query=query,
                documents=[],
                configuration=self._configuration(top_k),
            )

        results = self._client.search(
            query,
            top_k=top_k,
            mode=self.mode,
            rerank=self.reranker is not None,
            include_metadata=True,
        )
        documents = [
            RetrievedDocument(
                document_id=(result.document_id or result.chunk_id or f"semfuse-{rank}"),
                text=result.text,
                rank=rank,
                score=round(float(result.score), 8),
                source=(
                    result.source
                    or result.metadata.get("source_collection")
                    or "semfuse"
                ),
            )
            for rank, result in enumerate(results, start=1)
        ]
        return RetrievalResult(
            query=query,
            documents=documents,
            configuration=self._configuration(top_k),
        )

    def _configuration(self, top_k: int) -> dict[str, Any]:
        return {
            "method": "semfuse",
            "package_version": self._package_version,
            "mode": self.mode,
            "fusion": self.fusion_method,
            "top_k": top_k,
            "embedding_model": self.embedding_model,
            "embedding_provider": self.embedding_provider,
            "reranker": self.reranker,
            "num_indexed": self._num_indexed,
        }


def build_retriever(
    method: str,
    *,
    embedding_model: str = "bge-m3",
    embedding_function: EmbeddingFunction | None = None,
    base_url: str | None = None,
    semfuse_mode: str = "hybrid",
    semfuse_embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
    semfuse_embedding_provider: str = "local",
    semfuse_fusion_method: str = "rrf",
    semfuse_reranker: str | None = None,
    semfuse_storage_path: str | Path | None = None,
    semfuse_client_factory: SemFuseFactory | None = None,
) -> Retriever:
    """Construct a named retrieval configuration."""
    normalized = method.strip().lower().replace("-", "_")
    if normalized in {"bm25", "lexical", "lexical_bm25"}:
        return LexicalRetriever()
    if normalized in {"dense", "bge_m3", "dense_bge_m3"}:
        return DenseRetriever(
            model=embedding_model,
            embedding_function=embedding_function,
            base_url=base_url,
        )
    if normalized in {"hybrid", "hybrid_rrf", "hybrid_bm25_bge_m3"}:
        return HybridRetriever(
            model=embedding_model,
            embedding_function=embedding_function,
            base_url=base_url,
        )
    if normalized in {"semfuse", "sem_fuse"}:
        return SemFuseRetriever(
            mode=semfuse_mode,
            embedding_model=semfuse_embedding_model,
            embedding_provider=semfuse_embedding_provider,
            fusion_method=semfuse_fusion_method,
            reranker=semfuse_reranker,
            storage_path=semfuse_storage_path,
            client_factory=semfuse_client_factory,
        )
    raise ValueError(
        f"unknown retrieval method '{method}'; choose bm25, dense, hybrid, or semfuse"
    )
