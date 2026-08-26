"""Synthetic fixture: 10 hand-authored examples for pipeline testing.

These records exist ONLY to exercise the schema, validation, and (later)
annotation/evaluation tooling end-to-end without paid APIs. They are not
research data, carry no human annotations, and must never be mixed into
pilot or benchmark datasets. data_origin is 'synthetic_constructed' for
every record regardless of the language condition it simulates.

Coverage: each of the 5 language conditions appears exactly twice and
each of the 5 evidence conditions appears exactly twice. Two records
demonstrate explicit, logged evidence corruption.

The builder is deterministic: fixed timestamps, no randomness. The
committed JSONL under data/fixtures/ must always equal the output of
build_synthetic_fixture_records() (enforced by tests).
"""

from __future__ import annotations

from typing import Any

from banglarag_eval.constants import SCHEMA_VERSION

FIXTURE_DATASET_VERSION = "fixture-0.1.0"
FIXTURE_SOURCE_ID = "synthetic_fixture_v0"
FIXTURE_TIMESTAMP = "2026-08-26T00:00:00+06:00"


def _span(source_text: str, span_text: str) -> dict[str, Any]:
    start = source_text.find(span_text)
    if start < 0:
        raise ValueError(f"span text not found in source: {span_text!r}")
    return {"start_char": start, "end_char": start + len(span_text), "text": span_text}


def _record(
    *,
    number: int,
    language_condition: str,
    question_language: str,
    evidence_condition: str,
    question: str,
    source_text: str,
    intended_answer_span: str,
    intended_answer: str,
    retrieved_documents: list[dict[str, Any]],
    generated_answer: str,
    answer_claims: list[str],
    evidence_condition_notes: str,
    corruption_metadata: dict[str, Any] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "example_id": f"fixture-{number:03d}",
        "document_id": f"fixture-doc-{number:03d}",
        "source_id": FIXTURE_SOURCE_ID,
        "question": question,
        "question_language": question_language,
        "language_condition": language_condition,
        "data_origin": "synthetic_constructed",
        "source_text": source_text,
        "source_span": _span(source_text, intended_answer_span),
        "intended_answer": intended_answer,
        "retrieved_context": " ".join(doc["text"] for doc in retrieved_documents),
        "retrieval_documents": retrieved_documents,
        "retrieval_configuration": {
            "method": "hand_constructed",
            "top_k": len(retrieved_documents),
        },
        "evidence_condition": evidence_condition,
        "evidence_condition_notes": evidence_condition_notes,
        "generated_answer": generated_answer,
        "answer_claims": answer_claims,
        "faithfulness_category": None,
        "hallucination_type": None,
        "annotations": [],
        "adjudication": None,
        "generator_model": "human_fixture_author",
        "generator_version": FIXTURE_DATASET_VERSION,
        "generation_settings": None,
        "embedding_model": None,
        "retriever": "hand_constructed",
        "reranker": None,
        "prompt_version": "fixture-v0",
        "question_generation": {
            "method": "human_written",
            "model": None,
            "model_version": None,
            "prompt_version": None,
            "timestamp": FIXTURE_TIMESTAMP,
        },
        "corruption_metadata": corruption_metadata,
        "evaluator_outputs": [],
        "dataset_version": FIXTURE_DATASET_VERSION,
        "timestamp": FIXTURE_TIMESTAMP,
        "random_seed": None,
        "cost": None,
        "latency": None,
        "notes": notes,
    }


def build_synthetic_fixture_records() -> list[dict[str, Any]]:
    """Return the 10 fixture records, in stable order."""
    records: list[dict[str, Any]] = []

    # 1. native_bangla x correct
    source = (
        "ঢাকা বাংলাদেশের রাজধানী ও বৃহত্তম শহর। "
        "শহরটি বুড়িগঙ্গা নদীর তীরে অবস্থিত।"
    )
    records.append(
        _record(
            number=1,
            language_condition="native_bangla",
            question_language="bn",
            evidence_condition="correct",
            question="বাংলাদেশের রাজধানীর নাম কী?",
            source_text=source,
            intended_answer_span="ঢাকা বাংলাদেশের রাজধানী",
            intended_answer="ঢাকা",
            retrieved_documents=[
                {
                    "document_id": "fixture-doc-001",
                    "text": source,
                    "rank": 1,
                    "score": None,
                    "source": FIXTURE_SOURCE_ID,
                }
            ],
            generated_answer="বাংলাদেশের রাজধানী ঢাকা।",
            answer_claims=["বাংলাদেশের রাজধানী ঢাকা।"],
            evidence_condition_notes=(
                "Retrieved context is the grounding passage; it fully "
                "supports the answer."
            ),
        )
    )

    # 2. native_bangla x contradictory
    source = (
        "পদ্মা সেতুর মোট দৈর্ঘ্য ৬.১৫ কিলোমিটার। "
        "সেতুটি ২০২২ সালের জুন মাসে যান চলাচলের জন্য খুলে দেওয়া হয়।"
    )
    records.append(
        _record(
            number=2,
            language_condition="native_bangla",
            question_language="bn",
            evidence_condition="contradictory",
            question="পদ্মা সেতুর মোট দৈর্ঘ্য কত?",
            source_text=source,
            intended_answer_span="৬.১৫ কিলোমিটার",
            intended_answer="৬.১৫ কিলোমিটার",
            retrieved_documents=[
                {
                    "document_id": "fixture-doc-002",
                    "text": source,
                    "rank": 1,
                    "score": None,
                    "source": FIXTURE_SOURCE_ID,
                }
            ],
            generated_answer="পদ্মা সেতুর মোট দৈর্ঘ্য প্রায় ৯ কিলোমিটার।",
            answer_claims=["পদ্মা সেতুর মোট দৈর্ঘ্য প্রায় ৯ কিলোমিটার।"],
            evidence_condition_notes=(
                "The answer's length claim conflicts with the retrieved "
                "evidence (৬.১৫ কিলোমিটার)."
            ),
        )
    )

    # 3. translated_bangla x partially_relevant
    source = (
        "আইফেল টাওয়ার ফ্রান্সের প্যারিস শহরে অবস্থিত একটি লোহার টাওয়ার। "
        "এটি ১৮৮৯ সালে নির্মিত হয়।"
    )
    records.append(
        _record(
            number=3,
            language_condition="translated_bangla",
            question_language="bn",
            evidence_condition="partially_relevant",
            question="আইফেল টাওয়ার কোথায় অবস্থিত এবং এর উচ্চতা কত?",
            source_text=source,
            intended_answer_span="ফ্রান্সের প্যারিস শহরে",
            intended_answer="প্যারিস, ফ্রান্স",
            retrieved_documents=[
                {
                    "document_id": "fixture-doc-003",
                    "text": source,
                    "rank": 1,
                    "score": None,
                    "source": FIXTURE_SOURCE_ID,
                }
            ],
            generated_answer=(
                "আইফেল টাওয়ার প্যারিসে অবস্থিত এবং এর উচ্চতা ৩৩০ মিটার।"
            ),
            answer_claims=[
                "আইফেল টাওয়ার প্যারিসে অবস্থিত।",
                "আইফেল টাওয়ারের উচ্চতা ৩৩০ মিটার।",
            ],
            evidence_condition_notes=(
                "Evidence supports the location claim but contains nothing "
                "about height."
            ),
            notes=(
                "Simulates translated-Bangla provenance for pipeline "
                "testing; real translated_bangla data must carry "
                "data_origin human_translated or machine_translated."
            ),
        )
    )

    # 4. translated_bangla x missing
    source = (
        "অ্যামাজন বনের প্রায় ৬০ শতাংশ ব্রাজিলে অবস্থিত। "
        "বনটি দক্ষিণ আমেরিকার নয়টি দেশ জুড়ে বিস্তৃত।"
    )
    records.append(
        _record(
            number=4,
            language_condition="translated_bangla",
            question_language="bn",
            evidence_condition="missing",
            question="অ্যামাজন বনের কত শতাংশ ব্রাজিলে অবস্থিত?",
            source_text=source,
            intended_answer_span="প্রায় ৬০ শতাংশ",
            intended_answer="প্রায় ৬০ শতাংশ",
            retrieved_documents=[
                {
                    "document_id": "fixture-doc-004-alt",
                    "text": (
                        "অ্যামাজন বন দক্ষিণ আমেরিকার একটি বিশাল ক্রান্তীয় "
                        "বৃষ্টিবন। এখানে অসংখ্য প্রজাতির উদ্ভিদ ও প্রাণী বাস করে।"
                    ),
                    "rank": 1,
                    "score": None,
                    "source": FIXTURE_SOURCE_ID,
                }
            ],
            generated_answer="অ্যামাজন বনের প্রায় ৬০ শতাংশ ব্রাজিলে অবস্থিত।",
            answer_claims=["অ্যামাজন বনের প্রায় ৬০ শতাংশ ব্রাজিলে অবস্থিত।"],
            evidence_condition_notes=(
                "The retrieved passage is topical but omits the percentage "
                "needed to support the answer."
            ),
            notes=(
                "Simulates translated-Bangla provenance for pipeline "
                "testing; real translated_bangla data must carry "
                "data_origin human_translated or machine_translated."
            ),
        )
    )

    # 5. code_mixed x correct
    source = (
        "বাংলাদেশের economy গত এক দশকে দ্রুত grow করেছে। "
        "Ready-made garments খাত রপ্তানি আয়ের প্রধান উৎস।"
    )
    records.append(
        _record(
            number=5,
            language_condition="code_mixed",
            question_language="bn_en_mixed",
            evidence_condition="correct",
            question="বাংলাদেশের export আয়ের প্রধান source কোন খাত?",
            source_text=source,
            intended_answer_span="Ready-made garments খাত",
            intended_answer="Ready-made garments খাত",
            retrieved_documents=[
                {
                    "document_id": "fixture-doc-005",
                    "text": source,
                    "rank": 1,
                    "score": None,
                    "source": FIXTURE_SOURCE_ID,
                }
            ],
            generated_answer=(
                "Ready-made garments খাত বাংলাদেশের রপ্তানি আয়ের প্রধান উৎস।"
            ),
            answer_claims=[
                "Ready-made garments খাত বাংলাদেশের রপ্তানি আয়ের প্রধান উৎস।"
            ],
            evidence_condition_notes=(
                "Retrieved context fully supports the answer; both are "
                "Bangla-English code-mixed in Bengali script."
            ),
        )
    )

    # 6. code_mixed x irrelevant (explicit corruption)
    source = (
        "ঢাকা Metro Rail-এর প্রথম অংশ ২০২২ সালের December মাসে চালু হয়। "
        "এটি ঢাকার প্রথম mass rapid transit ব্যবস্থা।"
    )
    records.append(
        _record(
            number=6,
            language_condition="code_mixed",
            question_language="bn_en_mixed",
            evidence_condition="irrelevant",
            question="ঢাকা Metro Rail কবে চালু হয়?",
            source_text=source,
            intended_answer_span="২০২২ সালের December মাসে",
            intended_answer="২০২২ সালের ডিসেম্বর মাসে",
            retrieved_documents=[
                {
                    "document_id": "fixture-doc-006-corrupted",
                    "text": (
                        "সুন্দরবন পৃথিবীর বৃহত্তম ম্যানগ্রোভ বন। "
                        "এখানে রয়েল বেঙ্গল টাইগার বাস করে।"
                    ),
                    "rank": 1,
                    "score": None,
                    "source": FIXTURE_SOURCE_ID,
                }
            ],
            generated_answer="ঢাকা Metro Rail ২০২২ সালের December মাসে চালু হয়।",
            answer_claims=[
                "ঢাকা Metro Rail ২০২২ সালের December মাসে চালু হয়।"
            ],
            evidence_condition_notes=(
                "Supporting passage was deliberately replaced with an "
                "unrelated passage; the original evidence is preserved in "
                "source_text."
            ),
            corruption_metadata={
                "corruption_type": "context_replacement",
                "original_context_ids": ["fixture-doc-006"],
                "modified_context_ids": ["fixture-doc-006-corrupted"],
                "operation_description": (
                    "Replaced the supporting passage with an unrelated "
                    "passage about the Sundarbans to construct the "
                    "'irrelevant' evidence condition."
                ),
                "random_seed": 13,
            },
        )
    )

    # 7. banglish x contradictory
    source = (
        "Shapla Bangladesh er jatiyo ful. "
        "Eta sadharonoto bil ebong jhil e phote."
    )
    records.append(
        _record(
            number=7,
            language_condition="banglish",
            question_language="banglish",
            evidence_condition="contradictory",
            question="Bangladesh er jatiyo ful ki?",
            source_text=source,
            intended_answer_span="Shapla",
            intended_answer="Shapla",
            retrieved_documents=[
                {
                    "document_id": "fixture-doc-007",
                    "text": source,
                    "rank": 1,
                    "score": None,
                    "source": FIXTURE_SOURCE_ID,
                }
            ],
            generated_answer="Bangladesh er jatiyo ful holo golap.",
            answer_claims=["Bangladesh er jatiyo ful holo golap."],
            evidence_condition_notes=(
                "The answer names golap (rose) while the evidence states "
                "Shapla (water lily); direct conflict. Entire example is "
                "romanized Bangla (Banglish), a distinct script condition."
            ),
        )
    )

    # 8. banglish x missing (explicit corruption: evidence removal)
    source = (
        "Sundarban prithibir sobcheye boro mangrove bon. "
        "Ekhane biponno Royal Bengal Tiger dekha jay."
    )
    records.append(
        _record(
            number=8,
            language_condition="banglish",
            question_language="banglish",
            evidence_condition="missing",
            question="Sundarban e kon biponno prani dekha jay?",
            source_text=source,
            intended_answer_span="Royal Bengal Tiger",
            intended_answer="Royal Bengal Tiger",
            retrieved_documents=[
                {
                    "document_id": "fixture-doc-008-truncated",
                    "text": "Sundarban prithibir sobcheye boro mangrove bon.",
                    "rank": 1,
                    "score": None,
                    "source": FIXTURE_SOURCE_ID,
                }
            ],
            generated_answer="Sundarban e biponno Royal Bengal Tiger dekha jay.",
            answer_claims=["Sundarban e biponno Royal Bengal Tiger dekha jay."],
            evidence_condition_notes=(
                "The sentence carrying the supporting evidence was removed "
                "from the retrieved context; the answer is no longer "
                "verifiable from it."
            ),
            corruption_metadata={
                "corruption_type": "evidence_removal",
                "original_context_ids": ["fixture-doc-008"],
                "modified_context_ids": ["fixture-doc-008-truncated"],
                "operation_description": (
                    "Removed the second sentence (the supporting evidence) "
                    "from the retrieved passage to construct the 'missing' "
                    "evidence condition."
                ),
                "random_seed": 29,
            },
        )
    )

    # 9. english x partially_relevant
    source = (
        "The University of Dhaka was established in 1921. "
        "It began with three faculties, twelve departments, and 877 students."
    )
    records.append(
        _record(
            number=9,
            language_condition="english",
            question_language="en",
            evidence_condition="partially_relevant",
            question=(
                "When was the University of Dhaka established, and how many "
                "students did it start with?"
            ),
            source_text=source,
            intended_answer_span="established in 1921",
            intended_answer="1921; 877 students",
            retrieved_documents=[
                {
                    "document_id": "fixture-doc-009-partial",
                    "text": (
                        "The University of Dhaka was established in 1921. "
                        "It is the oldest university in Bangladesh."
                    ),
                    "rank": 1,
                    "score": None,
                    "source": FIXTURE_SOURCE_ID,
                }
            ],
            generated_answer=(
                "The University of Dhaka was established in 1921 and "
                "started with 877 students."
            ),
            answer_claims=[
                "The University of Dhaka was established in 1921.",
                "The University of Dhaka started with 877 students.",
            ],
            evidence_condition_notes=(
                "Evidence supports the establishment year but not the "
                "student count."
            ),
        )
    )

    # 10. english x irrelevant
    source = "The Royal Bengal Tiger is the national animal of Bangladesh."
    records.append(
        _record(
            number=10,
            language_condition="english",
            question_language="en",
            evidence_condition="irrelevant",
            question="What is the national animal of Bangladesh?",
            source_text=source,
            intended_answer_span="Royal Bengal Tiger",
            intended_answer="The Royal Bengal Tiger",
            retrieved_documents=[
                {
                    "document_id": "fixture-doc-010-offtopic",
                    "text": (
                        "Cox's Bazar has one of the longest natural sea "
                        "beaches in the world."
                    ),
                    "rank": 1,
                    "score": None,
                    "source": FIXTURE_SOURCE_ID,
                }
            ],
            generated_answer=(
                "The national animal of Bangladesh is the Royal Bengal Tiger."
            ),
            answer_claims=[
                "The national animal of Bangladesh is the Royal Bengal Tiger."
            ],
            evidence_condition_notes=(
                "Retrieved passage is about an unrelated topic and provides "
                "no support for the answer."
            ),
        )
    )

    return records
