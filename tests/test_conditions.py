"""Condition vocabularies must agree across constants, YAML config, and
the JSON Schema.

Coverage: every vocabulary in constants.py is checked against
configs/conditions.yaml, and every enum-bearing schema property that
mirrors a vocabulary (language_condition, evidence_condition,
faithfulness_category, annotations[].label, question_language,
data_origin, hallucination_type, adjudication.method) is checked against
constants. Adding a new vocabulary requires adding its checks here.
"""

from __future__ import annotations

from banglarag_eval import constants
from banglarag_eval.config import load_conditions_config
from banglarag_eval.schema import load_record_schema


def test_language_conditions_match_yaml():
    config = load_conditions_config()
    assert tuple(config["language_conditions"]) == constants.LANGUAGE_CONDITIONS


def test_evidence_conditions_match_yaml():
    config = load_conditions_config()
    assert tuple(config["evidence_conditions"]) == constants.EVIDENCE_CONDITIONS


def test_faithfulness_categories_match_yaml():
    config = load_conditions_config()
    assert tuple(config["faithfulness_categories"]) == constants.FAITHFULNESS_CATEGORIES


def test_hallucination_types_match_yaml():
    config = load_conditions_config()
    assert tuple(config["hallucination_types"]) == constants.HALLUCINATION_TYPES


def test_question_languages_match_yaml():
    config = load_conditions_config()
    assert tuple(config["question_languages"]) == constants.QUESTION_LANGUAGES


def test_data_origins_match_yaml():
    config = load_conditions_config()
    assert tuple(config["data_origins"]) == constants.DATA_ORIGINS


def test_schema_language_condition_enum_matches_constants():
    schema = load_record_schema()
    enum = schema["properties"]["language_condition"]["enum"]
    assert tuple(enum) == constants.LANGUAGE_CONDITIONS


def test_schema_evidence_condition_enum_matches_constants():
    schema = load_record_schema()
    enum = schema["properties"]["evidence_condition"]["enum"]
    assert tuple(enum) == constants.EVIDENCE_CONDITIONS


def test_schema_faithfulness_enum_matches_constants():
    schema = load_record_schema()
    enum = [v for v in schema["properties"]["faithfulness_category"]["enum"] if v]
    assert tuple(enum) == constants.FAITHFULNESS_CATEGORIES


def test_schema_annotation_label_enum_matches_constants():
    schema = load_record_schema()
    enum = schema["properties"]["annotations"]["items"]["properties"]["label"]["enum"]
    assert tuple(enum) == constants.FAITHFULNESS_CATEGORIES


def test_schema_question_language_enum_matches_constants():
    schema = load_record_schema()
    enum = schema["properties"]["question_language"]["enum"]
    assert tuple(enum) == constants.QUESTION_LANGUAGES


def test_schema_data_origin_enum_matches_constants():
    schema = load_record_schema()
    enum = schema["properties"]["data_origin"]["enum"]
    assert tuple(enum) == constants.DATA_ORIGINS


def test_schema_hallucination_type_enum_matches_constants():
    schema = load_record_schema()
    enum = [v for v in schema["properties"]["hallucination_type"]["enum"] if v]
    assert tuple(enum) == constants.HALLUCINATION_TYPES


def test_schema_adjudication_method_enum_matches_constants():
    schema = load_record_schema()
    enum = schema["properties"]["adjudication"]["properties"]["method"]["enum"]
    assert tuple(enum) == constants.ADJUDICATION_METHODS


def test_loaded_schema_is_mutation_safe():
    """Mutating the dict returned by load_record_schema must not affect
    subsequent validation (the cached schema is copy-protected)."""
    schema = load_record_schema()
    schema["properties"]["language_condition"]["enum"].append("hinglish")
    fresh = load_record_schema()
    assert "hinglish" not in fresh["properties"]["language_condition"]["enum"]


def test_banglish_is_distinct_from_code_mixed():
    """Banglish must remain a separate condition, never collapsed."""
    assert "banglish" in constants.LANGUAGE_CONDITIONS
    assert "code_mixed" in constants.LANGUAGE_CONDITIONS
    config = load_conditions_config()
    banglish_desc = config["language_conditions"]["banglish"]["description"]
    assert "distinct" in banglish_desc.lower()
