#!/usr/bin/env zsh
set -euo pipefail

repo_root="${0:A:h:h}"
cd "$repo_root"

mkdir -p logs checkpoints data/intermediate
exec >> logs/campaign_10000_v3_unreviewed.log 2>&1

echo "[$(date -Iseconds)] campaign v3 pipeline starting"

PYTHONUNBUFFERED=1 .venv/bin/python scripts/run_campaign.py propose-contradictions \
  --config configs/campaign_10000_v3_unreviewed.yaml \
  --questions data/intermediate/campaign_10000_v2_questions.jsonl \
  --contradiction-proposals data/intermediate/campaign_10000_v3_contradictions.jsonl

review_packet="data/intermediate/campaign_10000_v3_review.jsonl"
if PYTHONUNBUFFERED=1 .venv/bin/python -c '
import json
from pathlib import Path

path = Path("data/intermediate/campaign_10000_v3_review.jsonl")
rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
assert len(rows) == 5000
assert len({row["scenario_id"] for row in rows}) == 5000
assert all(row["review"]["status"] == "pending" for row in rows)
'; then
  echo "review packet validated; reusing $review_packet"
else
  PYTHONUNBUFFERED=1 .venv/bin/python scripts/run_campaign.py review \
    --config configs/campaign_10000_v3_unreviewed.yaml \
    --questions data/intermediate/campaign_10000_v2_questions.jsonl \
    --review-packet "$review_packet" \
    --overwrite-review-packet
fi

PYTHONUNBUFFERED=1 .venv/bin/python scripts/run_campaign.py generate \
  --config configs/campaign_10000_v3_unreviewed.yaml \
  --questions data/intermediate/campaign_10000_v2_questions.jsonl \
  --working checkpoints/campaign_10000_v3_unreviewed_working.jsonl

PYTHONUNBUFFERED=1 .venv/bin/python scripts/run_campaign.py finalize \
  --config configs/campaign_10000_v3_unreviewed.yaml \
  --working checkpoints/campaign_10000_v3_unreviewed_working.jsonl \
  --output data/campaign_10000_v3_unreviewed.jsonl \
  --categories data/campaign_10000_v3_unreviewed_categories.jsonl \
  --manifest data/campaign_10000_v3_unreviewed_manifest.json

echo "[$(date -Iseconds)] campaign v3 pipeline completed"
