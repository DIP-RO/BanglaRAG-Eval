#!/usr/bin/env zsh
set -euo pipefail

repo_root="${0:A:h:h}"
cd "$repo_root"

final_output="data/campaign_10000_v3_unreviewed.jsonl"
target_records=10000

audit_and_cleanup() {
  PYTHONUNBUFFERED=1 .venv/bin/python scripts/audit_campaign_10000_v3.py
  rm -f \
    checkpoints/campaign_10000_working.jsonl \
    checkpoints/campaign_10000_v2_working.jsonl
  echo "[$(date -Iseconds)] campaign v3 audit passed; obsolete v1/v2 checkpoints removed"
}

if [[ -f "$final_output" ]]; then
  record_count="$(wc -l < "$final_output" | tr -d '[:space:]')"
  if [[ "$record_count" == "$target_records" ]]; then
    audit_and_cleanup
    exit 0
  fi
fi

/usr/bin/caffeinate -dimsu /bin/zsh scripts/run_campaign_10000_v3.sh
audit_and_cleanup
