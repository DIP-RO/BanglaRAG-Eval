"""JSONL dataset loading and saving with validation.

Datasets are stored as JSON Lines (one record per line, UTF-8, no BOM).
Loading validates every record by default and refuses invalid datasets,
so downstream stages can rely on schema guarantees. Saving never
overwrites an existing file unless explicitly allowed — previous
generations must not be silently destroyed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from banglarag_eval.schema import ValidationResult, validate_dataset


class DatasetError(ValueError):
    """Raised when a dataset file cannot be loaded or is invalid."""


def load_dataset(path: str | Path, validate: bool = True) -> list[dict[str, Any]]:
    """Load a JSONL dataset, validating every record by default."""
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"dataset file does not exist: {path}")

    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise DatasetError(
                    f"{path}:{line_number}: each line must be a JSON object"
                )
            records.append(record)

    if validate:
        results = validate_dataset(records)
        errors = [
            f"{example_id}: {error}"
            for example_id, result in results.items()
            for error in result.errors
        ]
        if errors:
            raise DatasetError(
                f"{path}: {len(errors)} validation error(s):\n" + "\n".join(errors)
            )
    return records


def save_dataset(
    records: list[dict[str, Any]],
    path: str | Path,
    overwrite: bool = False,
    validate: bool = True,
) -> None:
    """Write records as JSONL. Refuses to overwrite unless told to."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise DatasetError(
            f"refusing to overwrite existing dataset {path}; "
            "pass overwrite=True only when replacement is intended"
        )
    if validate:
        results = validate_dataset(records)
        errors = [
            f"{example_id}: {error}"
            for example_id, result in results.items()
            for error in result.errors
        ]
        if errors:
            raise DatasetError(
                f"refusing to save invalid dataset: {len(errors)} error(s):\n"
                + "\n".join(errors)
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def dataset_validation_report(
    records: list[dict[str, Any]],
) -> dict[str, ValidationResult]:
    """Convenience wrapper returning per-record validation results."""
    return validate_dataset(records)
