# src/

`banglarag_eval` package (src layout; installable via `pip install -e .`,
or importable through `tests/conftest.py` path setup).

Milestone 1 modules:

- `constants.py` — condition vocabularies (tested against `configs/conditions.yaml`)
- `schema.py` — record validation: JSON Schema + cross-field rules R1–R7
  (including the gold-standard rule: `faithfulness_category` only from human
  adjudication)
- `config.py` — pilot/condition configuration loading and validation
- `sampling.py` — deterministic largest-remainder allocation over the
  LANGUAGE × EVIDENCE matrix
- `dataset.py` — JSONL load/save with validation; refuses silent overwrites
- `fixtures.py` — deterministic 10-example synthetic fixture builder

Planned for Milestone 2: source-document interface (local document loading,
referenced by `docs/pilot_design.md` and `RESEARCH_PROTOCOL.md` §14),
question-generation module, pilot generator built on `allocate_cells`,
`Evaluator` abstraction, `RAGBackend` interface (SemFuse adapter point),
annotation tooling, corruption module, metrics and statistics.

Note: the package reads `configs/` at runtime via `constants.REPO_ROOT`, so
it currently requires a repository checkout (editable install, or the
`tests/conftest.py` path setup). A non-editable install raises a clear
`FileNotFoundError`; shipping the configs as package data is a Milestone-2
task.
