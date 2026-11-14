# scripts/

- `build_synthetic_fixture.py` — deterministically regenerates
  `data/fixtures/synthetic_fixture_v0.jsonl` (10 synthetic pipeline-test
  examples). Safe to re-run; tests enforce builder/output sync.

Milestone 2+: pilot generation, annotation export/import, evaluator runs,
analysis scripts.

- `run_campaign.py` — question `prepare`/`repair`, contradiction proposal,
  construction `review`, and resumable `generate`/`finalize` phases. Required
  review gates fail closed and model failures never become placeholder answers.
- `run_campaign_10000_v3.sh` — resumable owner-authorized v3 chain. It writes
  only explicitly unreviewed records until genuine review is completed.
