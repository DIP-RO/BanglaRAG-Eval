# Benchmarking RAG Faithfulness Evaluation in Low-Resource and Code-Mixed Languages: A Bangla Study

> Research repository — ICLR 2027 target

## Research Objective

We investigate whether existing RAG faithfulness evaluators that perform reasonably well in English remain reliable when transferred to:

- Native Bangla RAG
- Translated Bangla data
- Bangla-English code-mixed RAG
- Controlled retrieval conditions

The project focuses on evaluator reliability, not only model performance. Human annotations will serve as the independent reference standard.

## Research Questions

**RQ1.** How well do existing RAG faithfulness evaluators agree with human judgments on native Bangla RAG?

**RQ2.** How does evaluator reliability change under Bangla-English code-mixing, translated versus native data, and controlled retrieval corruption?

**RQ3.** Which evaluation approach provides the best trade-off between human agreement, robustness, ranking stability, cost, and latency?

## Planned Benchmark Conditions

### Language

1. Native Bangla
2. Translated Bangla
3. Bangla-English code-mixed (Bengali script with embedded English)
4. Banglish / romanized Bangla — a distinct exploratory script condition,
   deliberately not collapsed into code-mixing
5. English baseline (control only)

### Evidence

1. Correct/relevant
2. Partially relevant
3. Irrelevant
4. Contradictory
5. Missing/no evidence

The benchmark will separate retrieval quality from downstream answer faithfulness.

## Candidate Automatic Evaluators

These are the systems **under evaluation**, compared against the human
reference standard — they are never gold themselves:

- RAGAS
- Independent LLM-as-a-judge evaluators
- ARES, where technically feasible
- Encoder-based hallucination detectors
- Simple non-neural baselines

The final evaluator set will be determined after the pilot. Human expert
annotation is not in this list: it is the independent reference standard
against which all of the above are measured.

## SemFuse

SemFuse may be used as an experimental RAG platform for controlled retrieval and generation experiments.

**Important:** SemFuse's internal grounding/faithfulness mechanism is NOT the benchmark gold standard.

Gold labels must be independently established through human annotation.

## How the Pilot Pipeline Works (Milestone 1)

The pilot follows a strict two-stage process:

1. **Stage 1 — methodology validation (30–50 examples).** A configurable
   generator (`configs/pilot_stage1.yaml`, currently `pilot_size: 40`)
   allocates examples across the LANGUAGE × EVIDENCE condition matrix using
   deterministic largest-remainder sampling. The complete pipeline —
   construction → retrieval → generation → two-annotator labeling →
   adjudication → automatic evaluation — is exercised end to end and judged
   against pre-registered gates ([docs/pilot_design.md](docs/pilot_design.md)).
2. **Stage 2 — expansion to 300–500 examples** (`configs/pilot_stage2.yaml`)
   only after Stage 1 passes. Scaling is a config change; the schema never
   changes.

Every record follows the canonical schema
([configs/schema/record.schema.json](configs/schema/record.schema.json),
documented in [docs/data_schema.md](docs/data_schema.md)) and carries full
provenance: source span, retrieval configuration, generation settings, prompt
versions, seeds, timestamps, and explicit `evidence_condition` /
`corruption_metadata`. Gold faithfulness labels come exclusively from human
adjudication (schema-enforced); automatic evaluators write only to
`evaluator_outputs`.

### Running locally (no paid APIs required)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/            # full validation suite
.venv/bin/python scripts/build_synthetic_fixture.py   # regenerate fixture
```

A 10-example synthetic fixture
([data/fixtures/synthetic_fixture_v0.jsonl](data/fixtures/synthetic_fixture_v0.jsonl))
covers every language and evidence condition and exists purely to test the
pipeline — it is not research data.

## Documentation

- [docs/RESEARCH_PROTOCOL.md](docs/RESEARCH_PROTOCOL.md) — full research protocol
- [docs/pilot_design.md](docs/pilot_design.md) — pilot stages, sampling, gates
- [docs/data_schema.md](docs/data_schema.md) — record schema and validation rules
- [docs/annotation_guidelines.md](docs/annotation_guidelines.md) — human annotation protocol
- [docs/evaluation_protocol.md](docs/evaluation_protocol.md) — evaluators, metrics, statistics

## Repository Structure

```text
BanglaRAG-Eval/
├── README.md
├── CONTRIBUTING.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── configs/            # conditions, pilot configs, machine-readable schema
├── data/               # fixtures (committed); raw/private data ignored
├── docs/               # protocol, pilot design, schema, annotation, evaluation
├── experiments/        # experiment definitions (Milestone 2+)
├── scripts/            # fixture builder and future pipeline scripts
├── src/                # banglarag_eval package
└── tests/              # schema, config, sampling, dataset, e2e fixture tests
```
