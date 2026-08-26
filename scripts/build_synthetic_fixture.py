#!/usr/bin/env python3
"""Regenerate the committed synthetic fixture (deterministic).

Usage:
    python scripts/build_synthetic_fixture.py

Writes data/fixtures/synthetic_fixture_v0.jsonl. The builder is fully
deterministic, so regeneration always reproduces the committed file;
tests enforce that the committed JSONL and the builder stay in sync.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from banglarag_eval.dataset import save_dataset  # noqa: E402
from banglarag_eval.fixtures import build_synthetic_fixture_records  # noqa: E402

OUTPUT_PATH = REPO_ROOT / "data" / "fixtures" / "synthetic_fixture_v0.jsonl"


def main() -> None:
    records = build_synthetic_fixture_records()
    # overwrite=True is safe here: the builder is deterministic and the
    # output is version-controlled, so this is regeneration, not loss.
    save_dataset(records, OUTPUT_PATH, overwrite=True, validate=True)
    print(f"Wrote {len(records)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
