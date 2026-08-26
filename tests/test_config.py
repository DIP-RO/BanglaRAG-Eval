"""Pilot configuration loading and validation tests."""

from __future__ import annotations

import pytest
import yaml

from banglarag_eval.config import ConfigError, load_pilot_config
from banglarag_eval.constants import REPO_ROOT

STAGE1 = REPO_ROOT / "configs" / "pilot_stage1.yaml"
STAGE2 = REPO_ROOT / "configs" / "pilot_stage2.yaml"


def _write_config(tmp_path, **overrides):
    base = {
        "stage": "methodology_validation",
        "pilot_size": 40,
        "random_seed": 1,
        "dataset_version": "test-0.1.0",
        "language_proportions": {
            "native_bangla": 0.30,
            "translated_bangla": 0.20,
            "code_mixed": 0.20,
            "banglish": 0.15,
            "english": 0.15,
        },
        "evidence_proportions": {
            "correct": 0.30,
            "partially_relevant": 0.20,
            "irrelevant": 0.15,
            "contradictory": 0.20,
            "missing": 0.15,
        },
    }
    base.update(overrides)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    return path


def test_stage1_config_loads():
    config = load_pilot_config(STAGE1)
    assert config.stage == "methodology_validation"
    # Milestone constraint: stage 1 validates methodology on 30-50 examples.
    assert 30 <= config.pilot_size <= 50


def test_stage2_config_loads():
    config = load_pilot_config(STAGE2)
    assert config.stage == "pilot_expansion"
    assert 300 <= config.pilot_size <= 500


def test_pilot_size_is_configurable_not_hardcoded(tmp_path):
    for size in (30, 40, 50, 400, 500):
        config = load_pilot_config(_write_config(tmp_path, pilot_size=size))
        assert config.pilot_size == size


def test_proportions_must_sum_to_one(tmp_path):
    path = _write_config(
        tmp_path,
        language_proportions={
            "native_bangla": 0.5,
            "translated_bangla": 0.2,
            "code_mixed": 0.1,
            "banglish": 0.1,
            "english": 0.05,
        },
    )
    with pytest.raises(ConfigError, match="sum to 1.0"):
        load_pilot_config(path)


def test_missing_condition_rejected(tmp_path):
    """Silently dropping a condition must fail; exclusion is an explicit 0.0."""
    path = _write_config(
        tmp_path,
        language_proportions={"native_bangla": 0.6, "english": 0.4},
    )
    with pytest.raises(ConfigError, match="missing conditions"):
        load_pilot_config(path)


def test_explicit_zero_proportion_allowed(tmp_path):
    path = _write_config(
        tmp_path,
        language_proportions={
            "native_bangla": 0.6,
            "translated_bangla": 0.0,
            "code_mixed": 0.0,
            "banglish": 0.0,
            "english": 0.4,
        },
    )
    config = load_pilot_config(path)
    assert config.language_proportions["banglish"] == 0.0


def test_unknown_condition_rejected(tmp_path):
    path = _write_config(
        tmp_path,
        evidence_proportions={"correct": 0.5, "made_up_condition": 0.5},
    )
    with pytest.raises(ConfigError, match="unknown conditions"):
        load_pilot_config(path)


def test_negative_proportion_rejected(tmp_path):
    path = _write_config(
        tmp_path,
        language_proportions={
            "native_bangla": 1.2,
            "translated_bangla": -0.2,
            "code_mixed": 0.0,
            "banglish": 0.0,
            "english": 0.0,
        },
    )
    with pytest.raises(ConfigError, match="non-negative"):
        load_pilot_config(path)


def test_unknown_key_rejected(tmp_path):
    path = _write_config(tmp_path, pilot_sizes=40)
    with pytest.raises(ConfigError, match="unknown keys"):
        load_pilot_config(path)


def test_missing_key_rejected(tmp_path):
    base_path = _write_config(tmp_path)
    raw = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    del raw["random_seed"]
    base_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="missing keys"):
        load_pilot_config(base_path)


def test_invalid_pilot_size_rejected(tmp_path):
    for bad in (0, -5, "forty", True):
        path = _write_config(tmp_path, pilot_size=bad)
        with pytest.raises(ConfigError, match="pilot_size"):
            load_pilot_config(path)


def test_invalid_stage_rejected(tmp_path):
    path = _write_config(tmp_path, stage="full_benchmark")
    with pytest.raises(ConfigError, match="stage"):
        load_pilot_config(path)
