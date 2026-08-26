"""Canonical condition vocabularies.

These constants mirror configs/conditions.yaml; tests assert the two
stay in sync so neither can drift silently.
"""

from __future__ import annotations

from pathlib import Path

# Repository root (src/banglarag_eval/constants.py -> repo root)
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
RECORD_SCHEMA_PATH = CONFIGS_DIR / "schema" / "record.schema.json"
CONDITIONS_CONFIG_PATH = CONFIGS_DIR / "conditions.yaml"

SCHEMA_VERSION = "1.0.0"

LANGUAGE_CONDITIONS = (
    "native_bangla",
    "translated_bangla",
    "code_mixed",
    "banglish",
    "english",
)

EVIDENCE_CONDITIONS = (
    "correct",
    "partially_relevant",
    "irrelevant",
    "contradictory",
    "missing",
)

FAITHFULNESS_CATEGORIES = (
    "faithful",
    "partially_faithful",
    "unsupported",
    "contradictory",
    "insufficient_evidence",
)

HALLUCINATION_TYPES = (
    "none",
    "evident_conflict",
    "subtle_conflict",
    "evident_baseless_information",
    "subtle_baseless_information",
    "other",
)

QUESTION_LANGUAGES = ("bn", "en", "bn_en_mixed", "banglish")

DATA_ORIGINS = (
    "native_authored",
    "human_translated",
    "machine_translated",
    "model_generated",
    "synthetic_constructed",
    "adapted_existing_benchmark",
)

# Translation-provenance origins, used by cross-field validation rules.
TRANSLATED_ORIGINS = ("human_translated", "machine_translated")

# Expected question_language for each language_condition. Deviations are
# reported as warnings (not errors) so that future cross-script designs
# (e.g. a Bangla question over English evidence) remain expressible.
EXPECTED_QUESTION_LANGUAGE = {
    "native_bangla": "bn",
    "translated_bangla": "bn",
    "code_mixed": "bn_en_mixed",
    "banglish": "banglish",
    "english": "en",
}

MIN_INDEPENDENT_ANNOTATORS = 2

# How an annotator disagreement was resolved (recorded per record so the
# two mechanisms — with different bias profiles — stay analyzable).
ADJUDICATION_METHODS = ("third_rater", "joint_session")
