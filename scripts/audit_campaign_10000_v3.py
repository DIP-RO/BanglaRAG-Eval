#!/usr/bin/env python3
"""Revalidate the completed unreviewed v3 campaign and write an audit receipt."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from banglarag_eval.pipeline.campaign import finalize_campaign


CONFIG = Path("configs/campaign_10000_v3_unreviewed.yaml")
WORKING = Path("checkpoints/campaign_10000_v3_unreviewed_working.jsonl")
FINAL = Path("data/campaign_10000_v3_unreviewed.jsonl")
CATEGORIES = Path("data/campaign_10000_v3_unreviewed_categories.jsonl")
MANIFEST = Path("data/campaign_10000_v3_unreviewed_manifest.json")
REVIEW_PACKET = Path("data/intermediate/campaign_10000_v3_review.jsonl")
CONTRADICTIONS = Path("data/intermediate/campaign_10000_v3_contradictions.jsonl")
AUDIT = Path("data/campaign_10000_v3_unreviewed_audit.json")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    # Finalization is deliberately repeated here: it performs schema validation,
    # campaign-integrity checks, canonical sorting, and category regeneration.
    manifest = finalize_campaign(WORKING, FINAL, CATEGORIES, MANIFEST, CONFIG)
    records = read_jsonl(FINAL)
    categories = read_jsonl(CATEGORIES)
    review_rows = read_jsonl(REVIEW_PACKET)
    contradiction_rows = read_jsonl(CONTRADICTIONS)

    ids = [row["example_id"] for row in records]
    digest = hashlib.sha256(FINAL.read_bytes()).hexdigest()
    assert len(records) == 10_000
    assert len(set(ids)) == 10_000
    assert manifest["record_count"] == 10_000
    assert manifest["unique_example_ids"] == 10_000
    assert manifest["schema_valid_records"] == 10_000
    assert manifest["human_annotations_present"] == 0
    assert manifest["sha256"] == digest
    assert all(row.get("construction_review_status") == "unreviewed" for row in records)
    assert all(row.get("construction_review") is None for row in records)
    assert all(row.get("annotations") == [] for row in records)
    assert all(row.get("adjudication") is None for row in records)

    assert len(review_rows) == 5_000
    assert len({row["scenario_id"] for row in review_rows}) == 5_000
    assert all(row["review"]["status"] == "pending" for row in review_rows)
    assert len(contradiction_rows) == 622
    assert len({row["qa_id"] for row in contradiction_rows}) == 622
    assert all(row["review_status"] == "unreviewed" for row in contradiction_rows)

    dimensions = {
        "language_condition": "language_condition",
        "evidence_condition": "evidence_condition",
        "generator_model": "generator_model",
        "retriever": "retriever",
    }
    expected_categories = {
        (dimension, value): sorted(
            row["example_id"] for row in records if str(row[field]) == value
        )
        for dimension, field in dimensions.items()
        for value in sorted({str(row[field]) for row in records})
    }
    actual_categories = {
        (row["dimension"], row["category"]): sorted(row["example_ids"])
        for row in categories
    }
    assert actual_categories == expected_categories
    assert all(row["count"] == len(row["example_ids"]) for row in categories)

    receipt = {
        "status": "validated_unreviewed",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records),
        "unique_example_ids": len(set(ids)),
        "schema_valid_records": manifest["schema_valid_records"],
        "sha256": digest,
        "counts": {
            field: dict(sorted(Counter(str(row[field]) for row in records).items()))
            for field in dimensions.values()
        },
        "review_packet": {"records": len(review_rows), "status": "pending"},
        "contradiction_proposals": {
            "records": len(contradiction_rows),
            "status": "unreviewed",
        },
        "files": {
            "records": str(FINAL),
            "categories": str(CATEGORIES),
            "manifest": str(MANIFEST),
            "working_checkpoint": str(WORKING),
        },
    }
    AUDIT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
