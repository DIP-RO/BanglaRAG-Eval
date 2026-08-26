# tests/

Run with:

```bash
.venv/bin/python -m pytest tests/
```

No paid APIs, no network access required.

- `test_schema_validation.py` — structural schema + cross-field rules R1–R10
- `test_conditions.py` — constants ↔ `configs/conditions.yaml` ↔ JSON Schema
  vocabulary sync (all six vocabularies against the YAML; language,
  evidence, faithfulness, annotation-label, question-language, data-origin,
  hallucination-type, and adjudication-method enums against the JSON
  Schema); Banglish/code-mixed distinctness; schema mutation safety
- `test_config.py` — pilot config loading/validation; configurable pilot_size
- `test_sampling.py` — deterministic allocation; 10 → 50 → 500 scaling
- `test_dataset_loading.py` — JSONL round-trip, overwrite refusal, validation
- `test_end_to_end_fixture.py` — the committed 10-example fixture passes the
  full pipeline validation and matches the deterministic builder
