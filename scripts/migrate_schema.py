"""Migrate old nested-format pilot records to flat schema format.

Usage:
    .venv/bin/python scripts/migrate_schema.py data/pilot_stage1_real.jsonl
"""
import json
import sys
from pathlib import Path


def migrate_record(old: dict) -> dict:
    """Convert old nested format to flat schema format."""
    source = old.get("source", {})
    question = old.get("question", {})
    generation = old.get("generation", {})
    metadata = old.get("metadata", {})

    return {
        "schema_version": "1.0.0",
        "example_id": old.get("record_id", old.get("example_id", "")),
        "document_id": source.get("document_id", old.get("document_id", "")),
        "source_id": source.get("document_id", old.get("source_id", "")),
        "question": question.get("text", old.get("question", "")),
        "question_language": question.get("language", old.get("question_language", "bn")),
        "language_condition": old.get("language_condition", ""),
        "data_origin": question.get("data_origin", old.get("data_origin", "native_authored")),
        "source_text": source.get("source_text", old.get("source_text", "")),
        "source_span": source.get("source_span", old.get("source_span")),
        "intended_answer": question.get("intended_answer", old.get("intended_answer")),
        "evidence_condition": old.get("evidence_condition", ""),
        "evidence_condition_notes": old.get("evidence_condition_notes"),
        "retrieved_context": old.get("retrieved_context", ""),
        "retrieval_documents": old.get("retrieval_documents", []),
        "retrieval_configuration": old.get("retrieval_configuration"),
        "retriever": old.get("retriever", "lexical_bm25"),
        "generated_answer": old.get("generated_answer", ""),
        "answer_claims": old.get("answer_claims", []),
        "generator_model": generation.get("model", old.get("generator_model", "")),
        "generation_settings": {
            "temperature": generation.get("temperature", 0.0),
            "seed": generation.get("seed", 42),
            "latency_seconds": generation.get("latency_seconds", 0.0),
            "token_count": generation.get("token_count", 0),
            "prompt_template": generation.get("prompt_template", "rag_standard"),
        },
        "prompt_version": "rag_standard_v1",
        "corruption_metadata": old.get("corruption_metadata"),
        "annotations": old.get("annotations", []),
        "faithfulness_category": old.get("gold_label", old.get("faithfulness_category")),
        "hallucination_type": None,
        "adjudication": old.get("adjudication"),
        "evaluator_outputs": [],
        "dataset_version": "pilot_stage1_v0",
        "timestamp": generation.get("timestamp", old.get("timestamp", "")),
        "random_seed": metadata.get("seed", old.get("random_seed")),
        "latency": generation.get("latency_seconds", old.get("latency")),
        "cost": None,
        "notes": None,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/migrate_schema.py <file.jsonl>")
        sys.exit(1)

    path = Path(sys.argv[1])
    backup = path.with_suffix(".old.jsonl")

    # Read all records
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    print(f"Read {len(records)} records from {path}")

    # Check if already migrated
    if "example_id" in records[0] and "schema_version" in records[0]:
        print("Records already in flat schema format. Nothing to do.")
        return

    # Backup original
    with open(backup, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Backed up original to {backup}")

    # Migrate
    migrated = [migrate_record(r) for r in records]

    # Write migrated
    with open(path, "w", encoding="utf-8") as f:
        for r in migrated:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(migrated)} migrated records to {path}")


if __name__ == "__main__":
    main()
