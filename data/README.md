# data/

- `fixtures/` — small, committed, synthetic datasets used only to test the
  pipeline. `synthetic_fixture_v0.jsonl` (10 examples) is regenerated
  deterministically by `scripts/build_synthetic_fixture.py`; tests enforce
  that the committed file matches the builder.
- `raw/`, `private/`, `intermediate/` — git-ignored. Never commit raw
  scraped corpora, restricted datasets, or private data. Check licensing
  before ingesting any external source.

All datasets are JSON Lines conforming to
`configs/record.schema.json`; released dataset versions are frozen —
corrections create a new `dataset_version`, never an in-place edit.

## Human annotation dataset

The validated campaign file is the annotation source:

`data/campaign_10000_v3_unreviewed.jsonl`

This 209 MB JSONL is kept out of normal Git commits because GitHub's file
limit is 100 MB. Keep the local validated file, or download the matching
release asset from the repository's `dataset-v3` release.

It contains 10,000 generated records, no human labels, and all required
annotation fields. Create a separate working copy before annotating so the
released source remains unchanged:

```bash
.venv/bin/python scripts/setup_annotation.py \
  --source data/campaign_10000_v3_unreviewed.jsonl \
  --target data/human_annotation_v1.jsonl
```

Then start the local annotation UI:

```bash
.venv/bin/python scripts/run_annotation_server.py \
  --dataset data/human_annotation_v1.jsonl \
  --mode both
```

## Run the complete campaign

The resumable command runs contradiction proposals, review-packet
construction, generation, and finalization:

```bash
zsh scripts/run_campaign_10000_v3.sh
```

It requires Ollama to be running with the configured models. The campaign
output is written to `data/campaign_10000_v3_unreviewed.jsonl`.
