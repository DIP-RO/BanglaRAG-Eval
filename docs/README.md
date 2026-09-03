# Documentation Index

This directory contains all project documentation for BanglaRAG-Eval.

## Document Map

| Document | Purpose | Audience |
|---|---|---|
| [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md) | Full research protocol — RQs, hypotheses, benchmark design, metrics, reproducibility | Supervisor, reviewers |
| [pilot_design.md](pilot_design.md) | Two-stage pilot study design with Stage 1 gates | Supervisor, implementers |
| [data_schema.md](data_schema.md) | Canonical record schema, field definitions, validation rules R1–R10 | Implementers |
| [annotation_guidelines.md](annotation_guidelines.md) | Annotator instructions — 5-way taxonomy, decision procedure, worked examples | Annotators |
| [evaluation_protocol.md](evaluation_protocol.md) | Evaluator framework, metrics, statistics, circularity controls | Implementers, reviewers |

## Quick Navigation

### For the research supervisor
1. Read [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md) for the full protocol
2. Read [pilot_design.md](pilot_design.md) for the pilot study design and gates
3. Read [evaluation_protocol.md](evaluation_protocol.md) for evaluator comparison methodology

### For annotators
1. Read [annotation_guidelines.md](annotation_guidelines.md) for labeling instructions
2. Open the annotation server (see scripts/run_annotation_server.py)
3. Login with your assigned annotator ID

### For developers
1. Read [data_schema.md](data_schema.md) for the record format
2. Read [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md) Section 14 for pipeline procedures
3. Check configs/ for experiment configurations
4. Run tests: `.venv/bin/python -m pytest -m "not real"`

### For ICLR 2027 paper writers
1. Read [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md) Sections 1-3 for research gap and RQs
2. Read [evaluation_protocol.md](evaluation_protocol.md) for metrics and statistics
3. Check results/ for experiment outputs

## File Organization

```
docs/
├── README.md                    ← This index
├── RESEARCH_PROTOCOL.md         ← Full research protocol (16 sections)
├── pilot_design.md              ← Two-stage pilot design with gates
├── data_schema.md               ← Record schema + validation rules
├── annotation_guidelines.md     ← Annotator instructions
└── evaluation_protocol.md       ← Evaluator + metrics + statistics
```

## Related Files Outside docs/

| File | Location | Purpose |
|---|---|---|
| Record schema | `configs/schema/record.schema.json` | JSON Schema draft 2020-12 |
| Stage 1 config | `configs/pilot_stage1.yaml` | 40 examples, methodology validation |
| Stage 2 config | `configs/pilot_stage2.yaml` | 400 examples, pilot expansion |
| Conditions | `configs/conditions.yaml` | Language + evidence condition vocabularies |
| Pipeline runner | `scripts/run_pipeline.py` | Generate pilot dataset |
| Evaluator runner | `scripts/run_evaluators.py` | Run evaluators + compute metrics |
| Annotation server | `scripts/run_annotation_server.py` | Flask web UI for annotation |
| Annotation setup | `scripts/setup_annotation.py` | Create annotation working copy |
| Schema migration | `scripts/migrate_schema.py` | Migrate old nested format to flat |
