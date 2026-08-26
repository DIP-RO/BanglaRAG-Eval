"""Loading and validation of pilot and condition configuration files."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from banglarag_eval.constants import (
    CONDITIONS_CONFIG_PATH,
    EVIDENCE_CONDITIONS,
    LANGUAGE_CONDITIONS,
)

_PROPORTION_TOLERANCE = 1e-6

_ALLOWED_PILOT_KEYS = {
    "stage",
    "pilot_size",
    "random_seed",
    "dataset_version",
    "language_proportions",
    "evidence_proportions",
}

_ALLOWED_STAGES = {"methodology_validation", "pilot_expansion"}


class ConfigError(ValueError):
    """Raised when a configuration file is invalid."""


@dataclass(frozen=True)
class PilotConfig:
    stage: str
    pilot_size: int
    random_seed: int
    dataset_version: str
    language_proportions: dict[str, float]
    evidence_proportions: dict[str, float]


def load_conditions_config(path: Path | None = None) -> dict[str, Any]:
    """Load configs/conditions.yaml (the condition vocabulary source)."""
    with open(path or CONDITIONS_CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _check_proportions(
    name: str, proportions: Any, allowed_keys: tuple[str, ...]
) -> dict[str, float]:
    if not isinstance(proportions, dict) or not proportions:
        raise ConfigError(f"{name} must be a non-empty mapping")
    unknown = set(proportions) - set(allowed_keys)
    if unknown:
        raise ConfigError(f"{name} contains unknown conditions: {sorted(unknown)}")
    # Every condition must be listed explicitly. Excluding a condition is
    # a deliberate experimental decision expressed as an explicit 0.0,
    # never a silent omission.
    missing = set(allowed_keys) - set(proportions)
    if missing:
        raise ConfigError(
            f"{name} is missing conditions {sorted(missing)}; "
            "list every condition explicitly (use 0.0 to exclude one)"
        )
    cleaned: dict[str, float] = {}
    for key, value in proportions.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ConfigError(f"{name}[{key}] must be a number, got {value!r}")
        if value < 0:
            raise ConfigError(f"{name}[{key}] must be non-negative, got {value}")
        cleaned[key] = float(value)
    total = sum(cleaned.values())
    if not math.isclose(total, 1.0, abs_tol=_PROPORTION_TOLERANCE):
        raise ConfigError(f"{name} must sum to 1.0, got {total}")
    return cleaned


def load_pilot_config(path: str | Path) -> PilotConfig:
    """Load and validate a pilot configuration file."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: pilot config must be a mapping")

    unknown = set(raw) - _ALLOWED_PILOT_KEYS
    if unknown:
        raise ConfigError(f"{path}: unknown keys: {sorted(unknown)}")
    missing = _ALLOWED_PILOT_KEYS - set(raw)
    if missing:
        raise ConfigError(f"{path}: missing keys: {sorted(missing)}")

    stage = raw["stage"]
    if stage not in _ALLOWED_STAGES:
        raise ConfigError(
            f"{path}: stage must be one of {sorted(_ALLOWED_STAGES)}, got {stage!r}"
        )

    pilot_size = raw["pilot_size"]
    if not isinstance(pilot_size, int) or isinstance(pilot_size, bool) or pilot_size < 1:
        raise ConfigError(f"{path}: pilot_size must be a positive integer")

    random_seed = raw["random_seed"]
    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise ConfigError(f"{path}: random_seed must be an integer")

    dataset_version = raw["dataset_version"]
    if not isinstance(dataset_version, str) or not dataset_version:
        raise ConfigError(f"{path}: dataset_version must be a non-empty string")

    language_proportions = _check_proportions(
        "language_proportions", raw["language_proportions"], LANGUAGE_CONDITIONS
    )
    evidence_proportions = _check_proportions(
        "evidence_proportions", raw["evidence_proportions"], EVIDENCE_CONDITIONS
    )

    return PilotConfig(
        stage=stage,
        pilot_size=pilot_size,
        random_seed=random_seed,
        dataset_version=dataset_version,
        language_proportions=language_proportions,
        evidence_proportions=evidence_proportions,
    )
