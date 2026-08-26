"""JSONL dataset loading/saving tests."""

from __future__ import annotations

import json

import pytest

from banglarag_eval.dataset import DatasetError, load_dataset, save_dataset


def test_save_and_load_round_trip(tmp_path, valid_record):
    path = tmp_path / "dataset.jsonl"
    save_dataset([valid_record], path)
    loaded = load_dataset(path)
    assert loaded == [valid_record]


def test_refuses_to_overwrite_by_default(tmp_path, valid_record):
    path = tmp_path / "dataset.jsonl"
    save_dataset([valid_record], path)
    with pytest.raises(DatasetError, match="refusing to overwrite"):
        save_dataset([valid_record], path)
    save_dataset([valid_record], path, overwrite=True)  # explicit is allowed


def test_missing_file_raises(tmp_path):
    with pytest.raises(DatasetError, match="does not exist"):
        load_dataset(tmp_path / "nope.jsonl")


def test_invalid_json_line_raises(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"example_id": "x"}\nnot json\n', encoding="utf-8")
    with pytest.raises(DatasetError, match="invalid JSON"):
        load_dataset(path, validate=False)


def test_non_object_line_raises(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('[1, 2, 3]\n', encoding="utf-8")
    with pytest.raises(DatasetError, match="JSON object"):
        load_dataset(path, validate=False)


def test_invalid_record_rejected_on_load(tmp_path, valid_record):
    del valid_record["question"]
    path = tmp_path / "invalid.jsonl"
    path.write_text(
        json.dumps(valid_record, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with pytest.raises(DatasetError, match="validation error"):
        load_dataset(path)
    # Loading without validation is possible for debugging.
    assert len(load_dataset(path, validate=False)) == 1


def test_refuses_to_save_invalid_dataset(tmp_path, valid_record):
    del valid_record["question"]
    with pytest.raises(DatasetError, match="refusing to save"):
        save_dataset([valid_record], tmp_path / "out.jsonl")


def test_bangla_text_survives_round_trip(tmp_path, valid_record):
    path = tmp_path / "bn.jsonl"
    save_dataset([valid_record], path)
    raw = path.read_text(encoding="utf-8")
    # ensure_ascii=False keeps Bangla readable in version control.
    assert "ঢাকা" in raw
