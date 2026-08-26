# configs/

- `conditions.yaml` — canonical language/evidence condition vocabularies,
  faithfulness taxonomy labels, question languages, data origins. Single
  source of truth; mirrored constants in `src/banglarag_eval/constants.py`
  are tested against this file.
- `pilot_stage1.yaml` — Stage 1 methodology-validation pilot (30–50 examples;
  `pilot_size` is configurable).
- `pilot_stage2.yaml` — Stage 2 expansion (300–500). Do not run until Stage 1
  passes the gates in `docs/pilot_design.md`.
- `schema/record.schema.json` — canonical machine-readable record schema
  (JSON Schema draft 2020-12), documented in `docs/data_schema.md`.
