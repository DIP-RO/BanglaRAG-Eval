"""Retrieval pipeline for the pilot dataset.

Implements a simple lexical retriever that works for Bangla, English,
and code-mixed text without requiring embedding models. Uses token
overlap scoring (a BM25-like approach) with Bangla-aware tokenization.

For Milestone 2 Stage 1, this is sufficient. Dense retrieval (BGE-M3)
can be added later as an alternative retrieval configuration.
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from typing import Any

from .sources import SourceDocument, split_into_passages


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
        """The concatenated text of the top-k retrieved documents."""
        return "\n\n".join(doc.text for doc in self.documents)


def _tokenize(text: str) -> list[str]:
    """Tokenize text for retrieval.

    Handles Bangla, English, and code-mixed text. Splits on whitespace
    and punctuation, keeping Bangla characters intact.
    """
    # Normalize: lowercase for English, keep Bangla as-is
    text = text.lower()
    # Split on non-alphanumeric (works for both scripts)
    tokens = re.findall(r'[\u0980-\u09FFa-zA-Z0-9]+', text)
    return [t for t in tokens if len(t) > 1]  # drop single chars


class LexicalRetriever:
    """Simple BM25-like lexical retriever.

    Works without embedding models. Suitable for pilot-scale datasets
    (30-500 documents). For larger corpora, a dense retriever (BGE-M3)
    should be added.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """Initialize BM25 parameters.

        Args:
            k1: Term frequency saturation parameter.
            b: Length normalization parameter.
        """
        self.k1 = k1
        self.b = b
        self._documents: list[dict[str, Any]] = []
        self._doc_tokens: list[list[str]] = []
        self._doc_freqs: dict[str, int] = {}  # term -> doc count
        self._avg_len: float = 0.0

    def index(self, documents: list[SourceDocument], passage_size: int = 500) -> None:
        """Index source documents by splitting into passages.

        Args:
            documents: Source documents to index.
            passage_size: Max characters per passage.
        """
        self._documents = []
        self._doc_tokens = []
        self._doc_freqs = {}

        for doc in documents:
            passages = split_into_passages(doc.text, max_chars=passage_size)
            for i, passage in enumerate(passages):
                doc_id = f"{doc.document_id}-p{i:03d}" if len(passages) > 1 else doc.document_id
                self._documents.append({
                    "document_id": doc_id,
                    "text": passage,
                    "source": doc.source_collection,
                    "original_doc_id": doc.document_id,
                })
                tokens = _tokenize(passage)
                self._doc_tokens.append(tokens)
                # Update document frequencies
                unique_terms = set(tokens)
                for term in unique_terms:
                    self._doc_freqs[term] = self._doc_freqs.get(term, 0) + 1

        self._avg_len = (
            sum(len(tokens) for tokens in self._doc_tokens) / len(self._doc_tokens)
            if self._doc_tokens else 0.0
        )

    def retrieve(self, query: str, top_k: int = 3) -> RetrievalResult:
        """Retrieve top-k passages for a query.

        Args:
            query: The search query.
            top_k: Number of passages to retrieve.

        Returns:
            RetrievalResult with ranked documents.
        """
        if not self._documents:
            return RetrievalResult(query=query, documents=[], configuration={"method": "bm25"})

        query_tokens = _tokenize(query)
        n_docs = len(self._documents)
        scores: list[float] = []

        for doc_tokens in self._doc_tokens:
            score = self._bm25_score(query_tokens, doc_tokens, n_docs)
            scores.append(score)

        # Rank by score (descending)
        ranked_indices = sorted(
            range(n_docs), key=lambda i: scores[i], reverse=True
        )

        retrieved: list[RetrievedDocument] = []
        for rank, idx in enumerate(ranked_indices[:top_k]):
            doc = self._documents[idx]
            retrieved.append(RetrievedDocument(
                document_id=doc["document_id"],
                text=doc["text"],
                rank=rank + 1,
                score=round(scores[idx], 4),
                source=doc["source"],
            ))

        return RetrievalResult(
            query=query,
            documents=retrieved,
            configuration={
                "method": "bm25",
                "top_k": top_k,
                "k1": self.k1,
                "b": self.b,
                "num_indexed": n_docs,
            },
        )

    def _bm25_score(
        self,
        query_tokens: list[str],
        doc_tokens: list[str],
        n_docs: int,
    ) -> float:
        """Compute BM25 score for a query-document pair."""
        if not doc_tokens:
            return 0.0

        doc_len = len(doc_tokens)
        doc_term_freq: dict[str, int] = {}
        for token in doc_tokens:
            doc_term_freq[token] = doc_term_freq.get(token, 0) + 1

        score = 0.0
        for term in query_tokens:
            if term not in doc_term_freq:
                continue
            tf = doc_term_freq[term]
            df = self._doc_freqs.get(term, 0)
            if df == 0:
                continue
            # IDF with smoothing
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            # TF saturation
            tf_norm = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_len / self._avg_len)
            )
            score += idf * tf_norm

        return score
