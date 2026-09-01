"""RAGBackend interface for retrieval/generation infrastructure.

SemFuse or any other RAG platform integrates behind this interface as
infrastructure ONLY — never as a gold-standard evaluator.

Available backends:
- StubRAGBackend: placeholder for testing
- SemFuseRAGBackend: stub for future SemFuse integration
"""

from .base import RAGBackend, RetrievedDocument, StubRAGBackend, SemFuseRAGBackend

__all__ = [
    "RAGBackend",
    "RetrievedDocument",
    "StubRAGBackend",
    "SemFuseRAGBackend",
]
