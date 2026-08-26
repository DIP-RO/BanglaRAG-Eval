# data/

- `fixtures/` — small, committed, synthetic datasets used only to test the
  pipeline. `synthetic_fixture_v0.jsonl` (10 examples) is regenerated
  deterministically by `scripts/build_synthetic_fixture.py`; tests enforce
  that the committed file matches the builder.
- `raw/`, `private/`, `intermediate/` — git-ignored. Never commit raw
  scraped corpora, restricted datasets, or private data. Check licensing
  before ingesting any external source.

All datasets are JSON Lines conforming to
`configs/schema/record.schema.json`; released dataset versions are frozen —
corrections create a new `dataset_version`, never an in-place edit.
