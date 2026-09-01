"""RAGBackend interface — abstraction for retrieval/generation infrastructure.

SemFuse (or any other RAG platform) integrates behind this interface as
retrieval/generation infrastructure ONLY. SemFuse's internal
grounding/faithfulness mechanism must NEVER be used as a gold standard
for the benchmark.

The interface is intentionally minimal:
    RAGBackend.retrieve(query) -> list[RetrievedDocument]
    RAGBackend.generate(question, context) -> str

If no backend is available, the pipeline uses its own LexicalRetriever
and OllamaGenerator (see pipeline/retriever.py and pipeline/generator.py).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievedDocument:
    """A document returned by the retrieval backend.

    Fields:
        text: The passage text.
        document_id: Source document identifier.
        score: Retrieval score (higher = more relevant).
        metadata: Backend-specific metadata.
    """

    text: str
    document_id: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class RAGBackend(ABC):
    """Abstract interface for RAG infrastructure (retrieval + generation).

    SemFuse or any other platform integrates here. This interface
    provides ONLY retrieval and generation — never evaluation.

    Key constraint: Any internal faithfulness/grounding score from the
    backend MUST NOT be exposed as a gold label. The backend is
    infrastructure, not an evaluator.
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        """Retrieve documents for a query.

        Args:
            query: The search query.
            top_k: Maximum number of documents to return.

        Returns:
            List of RetrievedDocument, sorted by relevance (highest first).
        """
        ...

    @abstractmethod
    def generate(self, question: str, context: str) -> str:
        """Generate an answer given a question and context.

        Args:
            question: The question to answer.
            context: The retrieved context to ground the answer.

        Returns:
            Generated answer text.
        """
        ...

    def is_available(self) -> bool:
        """Check if the backend is available for use.

        Returns:
            True if the backend can be used, False otherwise.
        """
        return True


class StubRAGBackend(RAGBackend):
    """Stub implementation that returns placeholder results.

    Used for testing the interface contract and as a fallback when
    no real backend (SemFuse, etc.) is available.

    The stub does NOT call any external API. It returns deterministic
    placeholder data so the pipeline can proceed with its own
    retrieval and generation if needed.
    """

    name = "stub"
    version = "0.1.0"

    def __init__(self, documents: list[dict[str, str]] | None = None):
        """Initialize with optional pre-loaded documents.

        Args:
            documents: List of {text, document_id} dicts for retrieval.
        """
        self._documents = documents or []

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        """Return stub documents (no real retrieval)."""
        results = []
        for i, doc in enumerate(self._documents[:top_k]):
            results.append(RetrievedDocument(
                text=doc.get("text", ""),
                document_id=doc.get("document_id", f"stub-doc-{i}"),
                score=1.0 - (i * 0.1),
                metadata={"backend": "stub"},
            ))
        return results

    def generate(self, question: str, context: str) -> str:
        """Return a stub answer."""
        return f"[stub answer] Question: {question[:50]}..."

    def is_available(self) -> bool:
        """Stub is always available (for testing)."""
        return True


class SemFuseRAGBackend(RAGBackend):
    """SemFuse RAG backend — placeholder for future integration.

    This is a stub that documents the intended SemFuse integration
    point. It does NOT implement any real SemFuse API calls because
    the API is not yet known.

    When SemFuse is available:
    1. Implement retrieve() to call SemFuse's retrieval API
    2. Implement generate() to call SemFuse's generation API
    3. DO NOT expose SemFuse's internal grounding/faithfulness score
       as a gold label — it is infrastructure only

    Configuration:
        base_url: SemFuse API URL (from SEMFUSE_URL env var)
        api_key: API key (from SEMFUSE_API_KEY env var)
    """

    name = "semfuse"
    version = "0.0.1-stub"

    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        import os
        self.base_url = base_url or os.environ.get("SEMFUSE_URL", "")
        self.api_key = api_key or os.environ.get("SEMFUSE_API_KEY", "")

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        """Not implemented — SemFuse API not yet available."""
        raise NotImplementedError(
            "SemFuse integration is not yet available. "
            "Use StubRAGBackend for testing or LexicalRetriever for real retrieval."
        )

    def generate(self, question: str, context: str) -> str:
        """Not implemented — SemFuse API not yet available."""
        raise NotImplementedError(
            "SemFuse integration is not yet available. "
            "Use StubRAGBackend for testing or OllamaGenerator for real generation."
        )

    def is_available(self) -> bool:
        """SemFuse is available only if URL and API key are configured."""
        return bool(self.base_url and self.api_key)
