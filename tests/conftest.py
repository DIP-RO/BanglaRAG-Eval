"""Test configuration: make the src-layout package importable without
requiring an editable install, and share a canonical valid record."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def valid_record() -> dict:
    """A minimal structurally and methodologically valid record."""
    return {
        "schema_version": "1.0.0",
        "example_id": "test-001",
        "document_id": "doc-001",
        "source_id": "test_source",
        "question": "বাংলাদেশের রাজধানীর নাম কী?",
        "question_language": "bn",
        "language_condition": "native_bangla",
        "data_origin": "native_authored",
        "source_text": "বাংলাদেশের রাজধানী ঢাকা।",
        "evidence_condition": "correct",
        "dataset_version": "test-0.1.0",
        "timestamp": "2026-08-26T00:00:00+06:00",
    }
