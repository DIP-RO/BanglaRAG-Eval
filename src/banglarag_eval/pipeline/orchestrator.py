"""Pipeline orchestrator for building the pilot dataset.

Ties together: source documents → question generation → retrieval →
evidence construction → generation (Ollama/Qwen3) → record assembly.

Produces records in the canonical BanglaRAG-Eval schema format.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .sources import SourceDocument
from .questions import generate_questions, QuestionSpec, CURATED_BANGLA_DOCUMENTS, ENGLISH_BASELINE_DOCUMENTS
from .retriever import LexicalRetriever
from .evidence import build_evidence, EvidenceResult
from .generator import generate as generate_answer, GenerationResult, check_ollama_available


def _iso_timestamp() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _make_record_id(language_condition: str, evidence_condition: str, index: int) -> str:
    """Generate a deterministic record ID."""
    return f"pilot1-{language_condition[:3]}-{evidence_condition[:4]}-{index:04d}"


def build_pilot_records(
    evidence_conditions: list[str] | None = None,
    language_conditions: list[str] | None = None,
    use_ollama: bool = True,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Build pilot records through the full pipeline.

    Args:
        evidence_conditions: Which evidence conditions to build.
            Defaults to all 5: correct, partially_relevant, irrelevant,
            contradictory, missing.
        language_conditions: Which language conditions to build.
            Defaults to all 5: native_bangla, translated_bangla, code_mixed,
            banglish, english.
        use_ollama: If True, generate answers with Ollama. If False, use
            intended_answer as a placeholder (for offline testing).
        seed: Random seed for reproducible corruption.

    Returns:
        List of records in canonical schema format.
    """
    if evidence_conditions is None:
        evidence_conditions = ["correct", "partially_relevant", "irrelevant", "contradictory", "missing"]
    if language_conditions is None:
        language_conditions = ["native_bangla", "translated_bangla", "code_mixed", "banglish", "english"]

    # 1. Load source documents
    bangla_docs = [
        SourceDocument(
            document_id=d["document_id"],
            text=d["text"],
            language="bn",
            source_collection="local_curated",
            license="local_authored",
        )
        for d in CURATED_BANGLA_DOCUMENTS
    ]
    english_docs = [
        SourceDocument(
            document_id=d["document_id"],
            text=d["text"],
            language="en",
            source_collection="local_curated",
            license="local_authored",
        )
        for d in ENGLISH_BASELINE_DOCUMENTS
    ]
    all_docs = bangla_docs + english_docs

    # 2. Index documents for retrieval
    retriever = LexicalRetriever()
    retriever.index(all_docs)

    # 3. Generate questions
    all_doc_dicts = CURATED_BANGLA_DOCUMENTS + ENGLISH_BASELINE_DOCUMENTS
    questions = generate_questions(all_doc_dicts, language_conditions)

    # 4. Check Ollama availability
    ollama_available = check_ollama_available() if use_ollama else False

    # 5. Build records
    records: list[dict[str, Any]] = []
    record_index = 0
    total_expected = len(questions) * len(evidence_conditions)

    print(f"[INFO] Generating {total_expected} records "
          f"({len(questions)} questions x {len(evidence_conditions)} evidence conditions)")

    for q_spec in questions:
        # Find the source document for this question
        source_doc = None
        for doc in all_docs:
            if q_spec.source_span["text"] in doc.text:
                source_doc = doc
                break

        if source_doc is None:
            continue

        # Retrieve evidence for the question
        retrieval = retriever.retrieve(q_spec.question, top_k=3)

        for evidence_condition in evidence_conditions:
            # Build evidence with controlled condition
            evidence = build_evidence(evidence_condition, retrieval, seed=seed + record_index)

            # Generate answer
            if ollama_available and evidence.retrieved_context:
                try:
                    gen_result = generate_answer(
                        question=q_spec.question,
                        context=evidence.retrieved_context,
                        temperature=0.0,
                        timeout=300,
                    )
                    generated_answer = gen_result.answer
                    generation_model = gen_result.model
                    generation_latency = gen_result.latency_seconds
                    generation_tokens = gen_result.eval_count
                except (ConnectionError, RuntimeError, TimeoutError) as exc:
                    print(f"  [WARN] Record {record_index}: generation failed ({type(exc).__name__}), using fallback")
                    generated_answer = q_spec.intended_answer
                    generation_model = f"ollama_fallback_{type(exc).__name__.lower()}"
                    generation_latency = 0.0
                    generation_tokens = 0
            elif ollama_available and not evidence.retrieved_context:
                # Missing evidence — still generate (model should say it can't answer)
                try:
                    gen_result = generate_answer(
                        question=q_spec.question,
                        context="(no context provided)",
                        temperature=0.0,
                        timeout=300,
                    )
                    generated_answer = gen_result.answer
                    generation_model = gen_result.model
                    generation_latency = gen_result.latency_seconds
                    generation_tokens = gen_result.eval_count
                except (ConnectionError, RuntimeError, TimeoutError) as exc:
                    print(f"  [WARN] Record {record_index}: generation failed ({type(exc).__name__}), using fallback")
                    generated_answer = q_spec.intended_answer
                    generation_model = f"ollama_fallback_{type(exc).__name__.lower()}"
                    generation_latency = 0.0
                    generation_tokens = 0
            else:
                # Offline mode — use intended answer as placeholder
                generated_answer = q_spec.intended_answer
                generation_model = "placeholder_offline"
                generation_latency = 0.0
                generation_tokens = 0

            record_id = _make_record_id(q_spec.language_condition, evidence_condition, record_index)

            record = {
                "record_id": record_id,
                "language_condition": q_spec.language_condition,
                "evidence_condition": evidence_condition,
                "source": {
                    "document_id": source_doc.document_id,
                    "source_collection": source_doc.source_collection,
                    "license": source_doc.license,
                    "source_text": source_doc.text,
                    "source_span": q_spec.source_span,
                },
                "question": {
                    "text": q_spec.question,
                    "language": q_spec.question_language,
                    "data_origin": q_spec.data_origin,
                    "intended_answer": q_spec.intended_answer,
                },
                "retriever": "lexical_bm25",
                "retrieval_configuration": retrieval.configuration,
                "retrieved_context": evidence.retrieved_context,
                "retrieval_documents": evidence.retrieval_documents,
                "generation": {
                    "model": generation_model,
                    "temperature": 0.0,
                    "seed": 42,
                    "latency_seconds": generation_latency,
                    "token_count": generation_tokens,
                    "prompt_template": "rag_standard",
                    "timestamp": _iso_timestamp(),
                },
                "generated_answer": generated_answer,
                "answer_claims": [],  # To be filled by claim extraction
                "corruption_metadata": evidence.corruption_metadata,
                "evidence_condition_notes": evidence.evidence_condition_notes,
                "annotations": [],
                "gold_label": None,
                "adjudication": None,
                "metadata": {
                    "pipeline_version": "milestone2-v1",
                    "created_at": _iso_timestamp(),
                    "seed": seed + record_index,
                },
            }

            records.append(record)
            record_index += 1

            if record_index % 10 == 0:
                print(f"  [{record_index}/{total_expected}] "
                      f"{q_spec.language_condition}/{evidence_condition} "
                      f"({generation_latency}s)")

    return records


def save_pilot_dataset(
    records: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """Save pilot records to a JSONL file.

    Args:
        records: List of record dicts.
        output_path: Path to output JSONL file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_pipeline(
    output_path: str | Path = "data/pilot_stage1_real.jsonl",
    evidence_conditions: list[str] | None = None,
    language_conditions: list[str] | None = None,
    use_ollama: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the full pilot pipeline and save results.

    Args:
        output_path: Where to save the JSONL dataset.
        evidence_conditions: Which evidence conditions to build.
        language_conditions: Which language conditions to build.
        use_ollama: Whether to use Ollama for generation.
        seed: Random seed.

    Returns:
        Summary dict with counts and status.
    """
    start_time = time.time()

    records = build_pilot_records(
        evidence_conditions=evidence_conditions,
        language_conditions=language_conditions,
        use_ollama=use_ollama,
        seed=seed,
    )

    save_pilot_dataset(records, output_path)

    # Summary
    from collections import Counter
    lang_counts = Counter(r["language_condition"] for r in records)
    evidence_counts = Counter(r["evidence_condition"] for r in records)
    model_counts = Counter(r["generation"]["model"] for r in records)

    return {
        "total_records": len(records),
        "language_distribution": dict(lang_counts),
        "evidence_distribution": dict(evidence_counts),
        "generation_models": dict(model_counts),
        "output_path": str(output_path),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "ollama_used": use_ollama and check_ollama_available(),
    }
