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

---

## Architecture Overview

The repository is organized around a strict separation of concerns:
**retrieval**, **generation**, **annotation**, and **evaluation** are
logically isolated stages that enrich the same record without overwriting
each other's data.

```mermaid
graph TB
    subgraph Configs["Configuration Layer"]
        CONDITIONS["configs/conditions.yaml<br/>Language + Evidence vocabularies"]
        SCHEMA["configs/schema/record.schema.json<br/>Canonical record schema"]
        PILOT1["configs/pilot_stage1.yaml<br/>Stage 1: 40 examples"]
        PILOT2["configs/pilot_stage2.yaml<br/>Stage 2: 400 examples"]
    end

    subgraph Core["Core Package (src/banglarag_eval/)"]
        CONSTANTS["constants.py<br/>Condition vocabularies"]
        CONFIGMOD["config.py<br/>Config loading + validation"]
        SCHEMAMOD["schema.py<br/>10 cross-field rules (R1–R10)"]
        SAMPLING["sampling.py<br/>Deterministic allocation"]
        DATASET["dataset.py<br/>JSONL load/save + validation"]
        FIXTURES["fixtures.py<br/>10-example synthetic builder"]
    end

    subgraph Annotation["Annotation UI (src/banglarag_eval/annotation/)"]
        STORE["store.py<br/>JSONL annotation store"]
        FLASKAPP["app.py<br/>Flask web server"]
        TEMPLATES["templates/<br/>6 HTML templates"]
    end

    subgraph Data["Data Layer"]
        FIXTUREJSONL["data/fixtures/<br/>synthetic_fixture_v0.jsonl"]
        PILOTJSONL["data/pilot_stage1.jsonl<br/>(Milestone 2)"]
    end

    subgraph Tests["Test Suite (tests/)"]
        TESTSCHEMA["test_schema_validation.py<br/>97 tests"]
        TESTANNOT["test_annotation.py<br/>33 tests"]
        TESTOTHER["test_config, test_sampling,<br/>test_dataset, test_conditions,<br/>test_end_to_end"]
    end

    CONDITIONS --> CONSTANTS
    SCHEMA --> SCHEMAMOD
    PILOT1 --> CONFIGMOD
    PILOT2 --> CONFIGMOD

    CONSTANTS --> SCHEMAMOD
    CONFIGMOD --> SAMPLING
    SCHEMAMOD --> DATASET
    SAMPLING --> DATASET
    FIXTURES --> DATASET

    DATASET --> STORE
    STORE --> FLASKAPP
    FLASKAPP --> TEMPLATES

    DATASET --> FIXTUREJSONL
    DATASET --> PILOTJSONL
    STORE --> PILOTJSONL

    SCHEMAMOD --> TESTSCHEMA
    STORE --> TESTANNOT
    FLASKAPP --> TESTANNOT
    DATASET --> TESTOTHER
    SAMPLING --> TESTOTHER
```

---

## Pipeline Flow

Every benchmark record is **created once and enriched by later stages**
without ever being redesigned. Scaling from 10 → 50 → 500 examples changes
only data volume, never structure.

```mermaid
flowchart LR
    subgraph Stage1["1. Data Construction"]
        SOURCE["Source Documents<br/>(local, Bangla Wikipedia, curated)"]
        QUESTION["Question Generation<br/>+ source span + intended answer"]
        EVIDENCE["Evidence Condition<br/>Construction / Corruption"]
    end

    subgraph Stage2["2. Retrieval"]
        RETRIEVE["Retriever<br/>(BM25 / Dense / Hybrid)"]
        RERANK["Reranker<br/>(optional)"]
        CONTEXT["retrieved_context<br/>+ retrieval_documents[]"]
    end

    subgraph Stage3["3. Generation"]
        GENERATOR["Generator<br/>(GPT-4o / Claude / Local)"]
        ANSWER["generated_answer<br/>+ answer_claims[]"]
    end

    subgraph Stage4["4. Human Annotation"]
        ANN1["Annotator 1<br/>(label + explanation)"]
        ANN2["Annotator 2<br/>(label + explanation)"]
        ADJ["Adjudication<br/>(if disagreement)"]
        GOLD["gold label<br/>faithfulness_category"]
    end

    subgraph Stage5["5. Automatic Evaluation"]
        EVAL["Evaluators<br/>(RAGAS, LLM judge, ARES,<br/>encoder detector, baseline)"]
        METRICS["Metrics<br/>(P/R/F1, κ, AUROC,<br/>ranking stability, cost)"]
    end

    SOURCE --> QUESTION --> EVIDENCE
    EVIDENCE --> RETRIEVE --> RERANK --> CONTEXT
    CONTEXT --> GENERATOR --> ANSWER
    ANSWER --> ANN1
    ANSWER --> ANN2
    ANN1 --> ADJ
    ANN2 --> ADJ
    ANN1 --> GOLD
    ANN2 --> GOLD
    ADJ --> GOLD
    ANSWER --> EVAL
    GOLD --> METRICS
    EVAL --> METRICS

    style Stage1 fill:#e0f2fe,stroke:#0284c7
    style Stage2 fill:#f0fdf4,stroke:#16a34a
    style Stage3 fill:#fef3c7,stroke:#d97706
    style Stage4 fill:#fce7f3,stroke:#be185d
    style Stage5 fill:#ede9fe,stroke:#7c3aed
```

### Key pipeline principles

- **Provenance is never lost.** Every question traces to a `source_span`;
  every retrieval stores exact `retrieved_context` + `retrieval_documents[]`;
  every generation stores model, settings, prompt version, seed, timestamp.
- **Corruption is never silent.** Modified evidence carries
  `corruption_metadata` (type, original/modified IDs, operation, seed);
  originals are always preserved in `source_text`.
- **Gold labels come only from humans.** Schema rule R3 mechanically blocks
  any `faithfulness_category` lacking human provenance. Automatic evaluators
  write only to `evaluator_outputs[]`.
- **Previous stages are never overwritten.** `save_dataset()` refuses to
  overwrite by default; corrections create a new `dataset_version`.

---

## Two-Stage Pilot Design

```mermaid
flowchart TB
    START([Start]) --> S1

    subgraph S1["Stage 1 — Methodology Validation"]
        S1BUILD["Build 30–50 examples<br/>configs/pilot_stage1.yaml<br/>pilot_size: 40"]
        S1PIPE["Run complete pipeline<br/>construction → retrieval → generation<br/>→ annotation → adjudication → evaluation"]
        S1GATES{"Pass all gates?"}

        S1BUILD --> S1PIPE --> S1GATES

        S1GATES -->|"No"| REVISE["Revise taxonomy / guidelines<br/>Fix construction / tooling<br/>Re-pilot on fresh batch"]
        REVISE --> S1BUILD

        S1GATES -->|"Yes"| S2
    end

    subgraph S2["Stage 2 — Pilot Expansion"]
        S2SCALE["Scale to 300–500 examples<br/>configs/pilot_stage2.yaml<br/>pilot_size: 400<br/>(config change, no schema change)"]
        S2RUN["Run full experiments<br/>All language × evidence conditions"]
        S2ANALYZE["Analyze evaluator reliability<br/>Ranking stability, cost/latency"]
        S2PAPER["Develop ICLR 2027 paper<br/>around strongest findings"]

        S2SCALE --> S2RUN --> S2ANALYZE --> S2PAPER
    end

    style S1 fill:#e0f2fe,stroke:#0284c7
    style S2 fill:#ede9fe,stroke:#7c3aed
    style REVISE fill:#fef2f2,stroke:#dc2626
```

### Stage 1 gates (pre-registered)

| Gate | Threshold | On failure |
|---|---|---|
| Annotation agreement | Cohen's κ ≥ 0.60 (target ≥ 0.70), 95% bootstrap CI | Revise guidelines/taxonomy |
| Taxonomy adequacy | < 10% `NO-FIT:` markers | Revise taxonomy; version bump |
| Condition integrity | ≥ 90% blind re-label match | Fix construction; regenerate cells |
| Pipeline integrity | 100% records pass validation | Fix tooling |
| Evaluator harness | ≥ 95% evaluator output success | Fix adapters; re-run |

---

## Human Annotation Flow

The annotation web UI enforces the protocol from `docs/annotation_guidelines.md`:
annotators see only the question, retrieved evidence, and generated answer —
never the source document, intended answer, evidence condition, other
annotators' labels, or evaluator outputs.

```mermaid
flowchart TB
    LOGIN["Annotator logs in<br/>with unique ID (e.g. ann-1)"]

    LOGIN --> PENDING{"Pending records<br/>for this annotator?"}

    PENDING -->|"No"| DONE["All Done page<br/>shows completion count"]
    PENDING -->|"Yes"| SHOW["Show next record:<br/>Question + Retrieved Evidence<br/>+ Generated Answer"]

    SHOW --> CHOOSE{"Annotator picks<br/>faithfulness label"}

    CHOOSE --> LABEL["5-way taxonomy:<br/>faithful / partially_faithful<br/>/ unsupported / contradictory<br/>/ insufficient_evidence"]

    LABEL --> EXPLAIN["Write explanation<br/>(required — must reference<br/>specific claims + evidence)"]

    EXPLAIN --> VALIDATE{"Schema validation<br/>passes?"}

    VALIDATE -->|"No"| ERROR["Show error<br/>annotator retries"]
    ERROR --> SHOW

    VALIDATE -->|"Yes"| SAVE["Append to record's<br/>annotations[] array<br/>Save to JSONL"]

    SAVE --> PENDING

    style LOGIN fill:#e0f2fe,stroke:#0284c7
    style SHOW fill:#f0fdf4,stroke:#16a34a
    style LABEL fill:#fef3c7,stroke:#d97706
    style SAVE fill:#ede9fe,stroke:#7c3aed
    style ERROR fill:#fef2f2,stroke:#dc2626
    style DONE fill:#fce7f3,stroke:#be185d
```

### Adjudication flow (when annotators disagree)

```mermaid
flowchart TB
    ANN1["Annotator 1 labels<br/>e.g. faithful"]
    ANN2["Annotator 2 labels<br/>e.g. unsupported"]

    ANN1 --> COMPARE{"Labels match?"}
    ANN2 --> COMPARE

    COMPARE -->|"Yes — unanimous"| AUTOGOLD["Unanimous label<br/>becomes gold label<br/>faithfulness_category = label<br/>(no adjudication record needed)"]

    COMPARE -->|"No — disagreement"| ADJLOGIN["Adjudicator logs in<br/>e.g. adjudicator-1<br/>(must differ from annotators — R8)"]

    ADJLOGIN --> ADJVIEW["Sees: question + evidence<br/>+ answer + both annotations<br/>+ both explanations"]

    ADJVIEW --> ADJCHOOSE{"Adjudicator picks<br/>final gold label"}

    ADJCHOOSE --> ADJMETHOD["Choose method:<br/>third_rater (independent)<br/>or joint_session (consensus)"]

    ADJMETHOD --> ADJEXPLAIN["Write adjudication<br/>explanation (required)"]

    ADJEXPLAIN --> ADJVALIDATE{"Schema validation<br/>passes?<br/>(R3: gold = adjudication label<br/>R4: >= 2 prior annotations<br/>R8: third_rater not an annotator)"}

    ADJVALIDATE -->|"No"| ADJERROR["Show error<br/>adjudicator retries"]
    ADJERROR --> ADJVIEW

    ADJVALIDATE -->|"Yes"| ADJSAVE["Set adjudication record<br/>Set faithfulness_category<br/>Save to JSONL"]

    ADJSAVE --> COMPLETE["Record is complete<br/>with gold label"]

    AUTOGOLD --> COMPLETE

    style ANN1 fill:#e0f2fe,stroke:#0284c7
    style ANN2 fill:#e0f2fe,stroke:#0284c7
    style AUTOGOLD fill:#f0fdf4,stroke:#16a34a
    style ADJLOGIN fill:#fef3c7,stroke:#d97706
    style ADJSAVE fill:#ede9fe,stroke:#7c3aed
    style COMPLETE fill:#fce7f3,stroke:#be185d
    style ADJERROR fill:#fef2f2,stroke:#dc2626
```

---

## Faithfulness Taxonomy — Decision Procedure

Annotators apply this ordered procedure (from `docs/annotation_guidelines.md`).
The procedure is authoritative; the table is a summary.

```mermaid
flowchart TB
    START([Annotator reads<br/>question + evidence + answer]) --> CLAIMS

    CLAIMS["Step 1: Identify substantive claims<br/>Mark central vs. side claims<br/>Ignore hedges, politeness, formatting"]

    CLAIMS --> CONFLICT{"Step 2: Does any<br/>central claim<br/>conflict with evidence?"}

    CONFLICT -->|"Yes"| CONTRADICTORY["Label: contradictory<br/>(conflict takes precedence)"]

    CONFLICT -->|"No"| USABLE{"Step 3: Can evidence<br/>check ANY substantive claim?<br/>(usability test)"}

    USABLE -->|"No — evidence unrelated<br/>or unusable"| INSUFFICIENT["Label: insufficient_evidence"]

    USABLE -->|"Yes"| SUPPORTED{"Step 4: Is any<br/>central claim<br/>supported?"}

    SUPPORTED -->|"No — every central claim<br/>unchecked/unverifiable"| UNSUPPORTED["Label: unsupported<br/>(even if side claims supported)"]

    SUPPORTED -->|"Yes"| ALLSUPPORTED{"Step 5: Is EVERY<br/>substantive claim<br/>supported?"}

    ALLSUPPORTED -->|"Yes"| FAITHFUL["Label: faithful"]
    ALLSUPPORTED -->|"No"| PARTIAL["Label: partially_faithful"]

    CONTRADICTORY --> NOFIT{"No category fits?"}
    INSUFFICIENT --> NOFIT
    UNSUPPORTED --> NOFIT
    FAITHFUL --> NOFIT
    PARTIAL --> NOFIT

    NOFIT -->|"Yes"| NOFITMARK["Choose closest label<br/>Begin explanation with:<br/>NO-FIT: (reason)"]
    NOFIT -->|"No"| FINAL([Final label + explanation<br/>saved to JSONL])

    NOFITMARK --> FINAL

    style CONTRADICTORY fill:#fef2f2,stroke:#dc2626
    style INSUFFICIENT fill:#fef3c7,stroke:#d97706
    style UNSUPPORTED fill:#f0fdf4,stroke:#16a34a
    style FAITHFUL fill:#e0f2fe,stroke:#0284c7
    style PARTIAL fill:#ede9fe,stroke:#7c3aed
    style NOFITMARK fill:#fce7f3,stroke:#be185d
```

### Boundary cases (from annotation guidelines)

- **True-but-unsupported** → `unsupported` (faithfulness is to evidence, not world truth)
- **Faithful-to-wrong-evidence** → `faithful` (if evidence itself is wrong and answer repeats it)
- **Numeric materiality** → compatible if equal after rounding to answer's precision
- **Abstention** → `faithful` if justified, `unsupported` if false refusal (not `contradictory`)
- **Script difference** → "Dhaka" / "ঢাকা" / "Dhaka" are equivalent; script alone ≠ unfaithful

---

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

---

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

---

## Running Locally (No Paid APIs Required)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/            # full validation suite (130 tests)
.venv/bin/python scripts/build_synthetic_fixture.py   # regenerate fixture
```

A 10-example synthetic fixture
([data/fixtures/synthetic_fixture_v0.jsonl](data/fixtures/synthetic_fixture_v0.jsonl))
covers every language and evidence condition and exists purely to test the
pipeline — it is not research data.

---

## Human Annotation (Local Web UI)

A built-in Flask web app lets human annotators label records from their
browser. Annotations are validated against the schema before saving, and
annotators never see provenance metadata (source text, intended answer,
evidence condition, evaluator outputs, or other annotators' labels).

### Quick start

```bash
# Install Flask (if not already installed):
.venv/bin/pip install flask>=3.0

# Start the annotation server on a pilot dataset:
.venv/bin/python scripts/run_annotation_server.py \
    --dataset data/pilot_stage1.jsonl \
    --mode annotate

# For adjudication (resolving annotator disagreements):
.venv/bin/python scripts/run_annotation_server.py \
    --dataset data/pilot_stage1.jsonl \
    --mode adjudicate
```

Then open `http://127.0.0.1:5000` in your browser.

### Annotation workflow

1. **Annotator 1** logs in with their ID (e.g. `ann-1`), sees each
   record one at a time (question, retrieved evidence, generated answer),
   picks a faithfulness label, writes an explanation, and submits.
2. **Annotator 2** logs in with a different ID (e.g. `ann-2`) and labels
   the same records independently — they cannot see annotator 1's labels.
3. When annotators **agree**, the unanimous label becomes the gold label
   automatically (schema rule R3).
4. When annotators **disagree**, an **adjudicator** logs in (e.g.
   `adjudicator-1`), switches to `--mode adjudicate`, sees the
   disagreeing labels + explanations, and assigns the final gold label.
   A `third_rater` adjudicator must be a different person from the
   annotators (schema rule R8).

Set `BANGLARAG_ANNOTATION_SECRET` in your environment for stable
sessions across server restarts.

See [docs/annotation_guidelines.md](docs/annotation_guidelines.md) for
the full annotation protocol, taxonomy, and decision procedure.

---

## Documentation

- [docs/RESEARCH_PROTOCOL.md](docs/RESEARCH_PROTOCOL.md) — full research protocol
- [docs/pilot_design.md](docs/pilot_design.md) — pilot stages, sampling, gates
- [docs/data_schema.md](docs/data_schema.md) — record schema and validation rules
- [docs/annotation_guidelines.md](docs/annotation_guidelines.md) — human annotation protocol
- [docs/evaluation_protocol.md](docs/evaluation_protocol.md) — evaluators, metrics, statistics

---

## Repository Structure

```text
BanglaRAG-Eval/
├── README.md
├── CONTRIBUTING.md
├── LICENSE
├── .env.example        # environment variable template (no secrets committed)
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── configs/            # conditions, pilot configs, machine-readable schema
│   ├── conditions.yaml
│   ├── pilot_stage1.yaml
│   ├── pilot_stage2.yaml
│   └── schema/record.schema.json
├── data/               # fixtures (committed); raw/private data ignored
│   └── fixtures/synthetic_fixture_v0.jsonl
├── docs/               # protocol, pilot design, schema, annotation, evaluation
│   ├── RESEARCH_PROTOCOL.md
│   ├── pilot_design.md
│   ├── data_schema.md
│   ├── annotation_guidelines.md
│   └── evaluation_protocol.md
├── experiments/        # experiment definitions (Milestone 2+)
├── scripts/            # fixture builder, annotation server launcher
│   ├── build_synthetic_fixture.py
│   └── run_annotation_server.py
├── src/                # banglarag_eval package
│   └── banglarag_eval/
│       ├── annotation/ # Flask web UI for human annotation + adjudication
│       │   ├── __init__.py
│       │   ├── app.py       # Flask routes (login, annotate, adjudicate)
│       │   ├── store.py     # JSONL annotation store with validation
│       │   ├── templates/   # 6 HTML templates
│       │   └── static/      # CSS with Bengali font support
│       ├── __init__.py
│       ├── config.py   # pilot config loading + validation
│       ├── constants.py# canonical condition vocabularies
│       ├── dataset.py  # JSONL load/save with validation
│       ├── fixtures.py # 10-example synthetic fixture builder
│       ├── sampling.py # deterministic condition-cell allocation
│       └── schema.py   # record schema + 10 cross-field validation rules
└── tests/              # 130 tests: schema, config, sampling, dataset, e2e, annotation
    ├── conftest.py
    ├── test_schema_validation.py
    ├── test_conditions.py
    ├── test_config.py
    ├── test_sampling.py
    ├── test_dataset_loading.py
    ├── test_end_to_end_fixture.py
    └── test_annotation.py
```
