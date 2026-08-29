"""Evidence construction and controlled corruption module.

Builds the 5 evidence conditions for the pilot dataset:
- correct: retrieved evidence fully supports the answer
- partially_relevant: evidence supports only some claims
- irrelevant: evidence is unrelated to the question
- contradictory: evidence conflicts with the answer
- missing: no sufficient supporting evidence

For corruption conditions (irrelevant, contradictory, missing), the
original evidence is preserved and corruption_metadata is logged.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .retriever import RetrievalResult, RetrievedDocument


@dataclass
class EvidenceResult:
    """Result of evidence construction for one record."""

    evidence_condition: str
    retrieved_context: str
    retrieval_documents: list[dict[str, Any]]
    corruption_metadata: dict[str, Any] | None = None
    evidence_condition_notes: str | None = None


# Pool of irrelevant Bangla passages for the "irrelevant" condition.
# These are topically unrelated to any pilot question.
_IRRELEVANT_PASSAGES = [
    "কম্পিউটার প্রোগ্রামিং একটি জটিল প্রক্রিয়া যেখানে কোড লেখা হয়। প্রোগ্রামাররা বিভিন্ন ভাষা ব্যবহার করেন যেমন পাইথন, জাভা, এবং সি++।",
    "মহাকাশযান মহাশূন্যে ভ্রমণ করে। নাসা এবং অন্যান্য মহাকাশ সংস্থা গ্রহ এবং নক্ষত্র অধ্যয়ন করে। মহাকাশ গবেষণা মানবজাতির জন্য গুরুত্বপূর্ণ।",
    "রান্না একটি শিল্প। ভালো রাঁধুনিরা মসলা এবং উপকরণ ব্যবহার করে সুস্বাদু খাবার তৈরি করেন। বিভিন্ন দেশের রান্নার ধরন আলাদা।",
    "ক্রিকেট একটি জনপ্রিয় খেলা। বাংলাদেশ ক্রিকেট দল আন্তর্জাতিক পর্যায়ে খেলে। টি-২০, ওয়ানডে এবং টেস্ট ক্রিকেট তিন ধরনের ম্যাচ হয়।",
    "গিটার একটি বাদ্যযন্ত্র। এতে ছয়টি তার থাকে। সঙ্গীতজ্ঞরা গিটার বাজিয়ে সুর তৈরি করেন। বিভিন্ন ধরনের গিটার রয়েছে।",
]

# Pool of contradictory statements (will be prepended to real evidence)
_CONTRADICTION_PREFIXES = [
    "উল্লেখযোগ্য গবেষণা অনুসারে উপরের তথ্য ভুল। আসল তথ্য হলো: ",
    "সাম্প্রতিক আবিষ্কার পূর্বের তথ্য অস্বীকার করে। প্রকৃত তথ্য হলো: ",
    "বিশেষজ্ঞদের মতে এই তথ্য সঠিক নয়। সঠিক তথ্য হলো: ",
]


def build_correct_evidence(retrieval: RetrievalResult) -> EvidenceResult:
    """Build correct evidence condition — use retrieved context as-is."""
    docs = [
        {"document_id": d.document_id, "text": d.text, "rank": d.rank, "score": d.score}
        for d in retrieval.documents
    ]
    return EvidenceResult(
        evidence_condition="correct",
        retrieved_context=retrieval.top_context,
        retrieval_documents=docs,
        evidence_condition_notes="Retrieved evidence used as-is, no corruption.",
    )


def build_partially_relevant_evidence(
    retrieval: RetrievalResult,
    seed: int = 42,
) -> EvidenceResult:
    """Build partially_relevant — keep only the first retrieved document."""
    rng = random.Random(seed)
    if not retrieval.documents:
        return build_missing_evidence(seed=seed)

    # Keep only the top document, drop the rest
    kept = retrieval.documents[0]
    docs = [{"document_id": kept.document_id, "text": kept.text, "rank": 1, "score": kept.score}]
    original_ids = [d.document_id for d in retrieval.documents[1:]]

    return EvidenceResult(
        evidence_condition="partially_relevant",
        retrieved_context=kept.text,
        retrieval_documents=docs,
        corruption_metadata={
            "corruption_type": "document_removal",
            "original_context_ids": [d.document_id for d in retrieval.documents],
            "modified_context_ids": [kept.document_id],
            "operation_description": f"Removed {len(original_ids)} of {len(retrieval.documents)} retrieved documents, keeping only top-1.",
            "random_seed": seed,
        } if len(retrieval.documents) > 1 else None,
        evidence_condition_notes="Only top-1 retrieved document kept; rest removed.",
    )


def build_irrelevant_evidence(
    retrieval: RetrievalResult,
    seed: int = 42,
) -> EvidenceResult:
    """Build irrelevant evidence — replace with unrelated passage."""
    rng = random.Random(seed)
    irrelevant_text = rng.choice(_IRRELEVANT_PASSAGES)
    original_ids = [d.document_id for d in retrieval.documents]

    return EvidenceResult(
        evidence_condition="irrelevant",
        retrieved_context=irrelevant_text,
        retrieval_documents=[{
            "document_id": "irrelevant-passage",
            "text": irrelevant_text,
            "rank": 1,
            "score": 0.0,
        }],
        corruption_metadata={
            "corruption_type": "context_replacement",
            "original_context_ids": original_ids,
            "modified_context_ids": ["irrelevant-passage"],
            "operation_description": "Replaced retrieved evidence with an unrelated passage.",
            "random_seed": seed,
        },
        evidence_condition_notes="Retrieved evidence replaced with topically unrelated text.",
    )


def build_contradictory_evidence(
    retrieval: RetrievalResult,
    seed: int = 42,
) -> EvidenceResult:
    """Build contradictory evidence — prepend a contradiction to real evidence."""
    rng = random.Random(seed)
    contradiction = rng.choice(_CONTRADICTION_PREFIXES)
    original_context = retrieval.top_context
    # The contradiction prefix + inverted claim creates a conflict
    # with any answer that follows the original evidence
    modified_context = contradiction + original_context
    original_ids = [d.document_id for d in retrieval.documents]

    return EvidenceResult(
        evidence_condition="contradictory",
        retrieved_context=modified_context,
        retrieval_documents=[{
            "document_id": d.document_id,
            "text": d.text if i == 0 else d.text,
            "rank": d.rank,
            "score": d.score,
        } for i, d in enumerate(retrieval.documents)],
        corruption_metadata={
            "corruption_type": "contradiction_injection",
            "original_context_ids": original_ids,
            "modified_context_ids": original_ids,
            "operation_description": f"Prepended contradiction prefix: '{contradiction.strip()}'",
            "random_seed": seed,
        },
        evidence_condition_notes="Contradiction prefix injected before original evidence.",
    )


def build_missing_evidence(seed: int = 42) -> EvidenceResult:
    """Build missing evidence — empty context."""
    return EvidenceResult(
        evidence_condition="missing",
        retrieved_context="",
        retrieval_documents=[],
        corruption_metadata={
            "corruption_type": "evidence_removal",
            "original_context_ids": [],
            "modified_context_ids": [],
            "operation_description": "No evidence provided to the generator.",
            "random_seed": seed,
        },
        evidence_condition_notes="No retrieved evidence provided.",
    )


def build_evidence(
    condition: str,
    retrieval: RetrievalResult,
    seed: int = 42,
) -> EvidenceResult:
    """Build evidence for a specific condition.

    Args:
        condition: One of "correct", "partially_relevant", "irrelevant",
                   "contradictory", "missing".
        retrieval: The retrieval result to build from.
        seed: Random seed for reproducible corruption.

    Returns:
        EvidenceResult with the constructed evidence and corruption metadata.
    """
    builders = {
        "correct": lambda: build_correct_evidence(retrieval),
        "partially_relevant": lambda: build_partially_relevant_evidence(retrieval, seed),
        "irrelevant": lambda: build_irrelevant_evidence(retrieval, seed),
        "contradictory": lambda: build_contradictory_evidence(retrieval, seed),
        "missing": lambda: build_missing_evidence(seed),
    }

    if condition not in builders:
        raise ValueError(
            f"unknown evidence condition '{condition}'; "
            f"must be one of {list(builders.keys())}"
        )

    return builders[condition]()
