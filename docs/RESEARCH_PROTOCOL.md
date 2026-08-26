# Research Protocol

## Working Title

**Benchmarking RAG Faithfulness Evaluation in Low-Resource and Code-Mixed Languages: A Bangla Study**

## Target Venue

ICLR 2027 — preliminary target.

The exact title and contribution will be finalized after the pilot experiments.

## 1. Core Research Problem

Do RAG faithfulness evaluators that perform reasonably well in English remain reliable when transferred to native Bangla and Bangla-English code-mixed RAG?

The project studies the reliability of the evaluation methods themselves, rather than simply ranking RAG systems.

### Motivation and Methodological Gap (working statement)

Existing RAG-faithfulness meta-evaluation resources are English-centric or
exclude Bengali (e.g., MEMERAG covers EN/DE/ES/FR/HI only), while existing
Bangla resources cover hallucination without retrieval, or retrieval without
faithfulness. Cross-lingual evidence suggests judge and detector reliability
can degrade sharply outside high-resource languages. Whether this holds for
Bangla — and for code-mixed and romanized Bangla — is an open **empirical**
question. This gap statement remains *working* until the literature review
and pilot evidence support it; it must not be presented as established.

## 2. Preliminary Research Questions

### RQ1

How well do existing RAG faithfulness evaluators agree with human judgments on native Bangla RAG?

### RQ2

How does evaluator reliability change under:

- Bangla-English code-mixing;
- translated versus native data;
- controlled retrieval corruption?

### RQ3

Which evaluation approach provides the best trade-off between:

- human agreement;
- robustness;
- ranking stability;
- cost;
- latency?

## 3. Preliminary Hypotheses

The following are hypotheses only and must not be treated as expected results:

- Existing automatic evaluators may show reduced agreement with human judgments in Bangla.
- Code-mixed inputs may expose additional evaluator weaknesses.
- Different evaluator families may rank RAG systems differently.
- Lower-cost encoder-based methods may provide competitive performance under some conditions.

All hypotheses will be tested empirically.

## 4. Benchmark Language Conditions

### A. Native Bangla

Native Bangla questions, evidence, and responses.

### B. Translated Bangla

A controlled Bangla adaptation of an established English RAG hallucination resource where licensing and methodology permit.

### C. Bangla-English Code-Mixed

Bangla-English mixed-language queries, contexts, and/or responses in Bengali
script (e.g., "বাংলাদেশের economy গত কয়েক বছরে significantly grow করেছে।").

### D. Banglish / Romanized Bangla (exploratory)

Bangla expressed primarily in Latin/Roman characters (e.g., "Bangladesh er
economy goto koyek bochor-e significantly grow koreche."). Banglish is a
**distinct script condition** and is never collapsed into ordinary
code-mixing; script effects and mixing effects must remain separable.

### E. English Baseline (control)

Retained where useful for calibrated comparison. English is a control only;
Bangla remains the primary testbed.

Machine-readable condition definitions: `configs/conditions.yaml`
(mirrored and tested in `src/banglarag_eval/constants.py`).

Native and translated Bangla must never be conflated: `data_origin` metadata
records true provenance, and validation rules R1/R2 reject contradictory
combinations.

## 5. Evidence Conditions

The benchmark should investigate controlled evidence conditions:

1. Correct/relevant evidence
2. Partially relevant evidence
3. Irrelevant evidence
4. Contradictory evidence
5. Missing/no evidence

These conditions allow retrieval quality and answer faithfulness to be analyzed separately.

Each example stores its evidence condition explicitly in metadata
(`evidence_condition`) at construction time — conditions are never inferred
after the fact. Controlled corruption is reproducible and logged
(`corruption_metadata`: corruption type, original/modified context ids,
operation description, random seed); original evidence is always preserved
and records are never silently altered.

## 6. Human Annotation

Human annotation is the independent reference standard.

The annotation protocol must define:

- what constitutes a faithful claim;
- what constitutes an unsupported claim;
- what constitutes contradiction;
- span-level hallucination;
- response-level faithfulness;
- treatment of ambiguous evidence;
- adjudication procedure.

Inter-annotator agreement must be measured before scaling the benchmark.

### Faithfulness Taxonomy

Human annotators apply a five-way response-level taxonomy (definitions,
decision procedure, and boundary cases in
[annotation_guidelines.md](annotation_guidelines.md)):

1. **Faithful** — every substantive claim supported by the retrieved evidence.
2. **Partially Faithful** — some claims supported, at least one not.
3. **Unsupported** — central claims not supported (independent of real-world truth).
4. **Contradictory** — at least one central claim conflicts with the evidence.
5. **Insufficient Evidence** — evidence inadequate to verify or refute.

The taxonomy is defined by the human annotation protocol; automatic
evaluators do not define or modify it. At least **two independent
annotators** label every example (label + explanation + annotator id +
timestamp). When they agree, the unanimous label is the gold label; when
they disagree, an adjudication pass (independent third rater, or a recorded
joint session) assigns it. The gold label (`faithfulness_category`) comes
exclusively from these human paths — enforced mechanically by schema rules
R3/R8.

## 7. Automatic Evaluators

Candidate evaluators include:

- RAGAS;
- independent LLM judges;
- ARES, where technically feasible;
- encoder-based hallucination detectors;
- simple lexical/non-neural baselines.

The final evaluator suite will be determined after the pilot.

## 8. SemFuse

SemFuse may be used to construct controlled RAG experiments.

SemFuse's internal grounding or faithfulness score must **not** be treated as the benchmark gold label.

The benchmark evaluates automatic evaluators against independently produced human annotations.

## 9. Pilot Study

Overall target:

**300–500 carefully selected examples**, reached through a two-stage process
(full design in [pilot_design.md](pilot_design.md)):

- **Stage 1 — methodology validation: 30–50 examples** (configured in
  `configs/pilot_stage1.yaml`), diverse enough to exercise the complete
  pipeline end to end. No dataset is generated at scale before this stage
  passes its pre-registered gates (annotation agreement, taxonomy adequacy,
  condition integrity, pipeline integrity).
- **Stage 2 — pilot expansion toward 300–500 examples** (configured in
  `configs/pilot_stage2.yaml`), only after Stage 1 validation. Scaling is a
  configuration change; the schema and pipeline must not change.

The pilot must answer:

1. Can annotators agree reliably?
2. Are the annotation definitions sufficiently clear?
3. Do automatic evaluators disagree with humans?
4. Does Bangla expose evaluator weaknesses?
5. Does code-mixing introduce additional evaluator errors?
6. Are the evidence corruption conditions useful?
7. Is the benchmark scalable?

## 10. Evaluation Metrics

Candidate metrics include:

### Classification

- Precision
- Recall
- F1
- AUROC
- Balanced Accuracy

### Agreement

- Cohen's kappa where applicable
- Krippendorff's alpha where appropriate
- Human–automatic evaluator agreement

### Correlation

- Spearman correlation
- Pearson correlation where assumptions are appropriate

### Ranking

- Rank correlation
- Pairwise ranking agreement
- Ranking stability across evaluators

### Efficiency

- Cost per 1,000 examples
- Latency
- Throughput
- GPU/CPU requirements

Statistical confidence intervals and appropriate paired tests will be used
(paired bootstrap, McNemar's test where its paired-binary setting applies,
Holm correction across comparison grids). Each test is applied only where its
assumptions hold, with the rationale documented; p-values are never
manufactured, and unmeasured values (including cost/latency) are never
reported. Details: [evaluation_protocol.md](evaluation_protocol.md).

## 11. Reproducibility

Every experiment must record:

- dataset version;
- Git commit;
- configuration;
- model and version;
- prompts;
- retrieval configuration;
- random seed where applicable;
- evaluation implementation/version;
- hardware;
- runtime;
- cost.

## 12. Research Integrity

The project must avoid:

- evaluator circularity;
- leakage between training and evaluation;
- post-hoc hypothesis changes;
- undocumented prompt changes;
- cherry-picked examples;
- unsupported novelty claims.

All major methodological decisions should be documented before the main experiments.

## 13. Decision Gate

The full benchmark should only be constructed after the pilot demonstrates:

- acceptable annotation agreement;
- **measurement sensitivity** — that the pipeline can detect an
  evaluator–human difference of the pre-registered smallest effect of
  interest if one exists. This is a power/instrument criterion, *not* a
  requirement that disagreement be observed: high evaluator–human agreement
  in Bangla is a legitimate, publishable finding, and gating on observing an
  effect would select for a positive result;
- a stable and defensible annotation taxonomy;
- feasibility of the proposed evaluation protocol.

Stage-specific stop/revise/scale gates with pre-registered thresholds are in
[pilot_design.md](pilot_design.md).

The final paper contribution will be determined from the empirical findings.

## 14. Data Construction and Pipeline Procedures

The pipeline keeps retrieval, generation, annotation, and evaluation
logically separated; each stage enriches the same record without overwriting
earlier stages.

- **Data construction.** Source documents will enter through a clean local
  source-document interface (to be implemented in Milestone 2); external
  corpora (Bangla Wikipedia, curated or news documents, benchmark-derived
  material) are added only after licensing review. Restricted datasets are
  not downloaded or redistributed.
- **Question generation.** Modular; every question stores its source document
  id, source span, intended answer, language condition, and generation
  method/model/prompt-version metadata (`question_generation`). Provenance is
  never lost. Where question provenance differs from source-passage
  provenance, `data_origin` describes the source passage and
  `question_generation` records the question's own origin (see
  [data_schema.md](data_schema.md)).
- **Retrieval procedure.** The exact retrieved context, per-document ranks
  and scores, and the full retrieval configuration are stored per record.
- **Generation procedure.** The exact retrieved context, generated answer,
  model/version, settings, prompt version, timestamp, and seed are stored;
  previous generations are never overwritten (enforced by the dataset writer).
- **Evidence construction/corruption.** Per Section 5; corruption operations
  are seeded, logged, and reversible in the sense that originals persist.
- **SemFuse.** Not currently present in this repository. When integrated, it
  connects behind a `RAGBackend` interface as retrieval/generation
  infrastructure only (see Section 8 for the gold-standard prohibition).

Machine-readable schema: `configs/schema/record.schema.json`, documented in
[data_schema.md](data_schema.md).

## 15. Known Limitations (current stage)

- The Milestone-1 fixture is synthetic and hand-authored; it validates the
  pipeline, not any research claim.
- Translated-Bangla data risks translationese artifacts; native vs. translated
  slices are therefore kept separable via `data_origin`.
- Banglish romanization has no standard orthography; annotation guidance
  exists but agreement on Banglish items is itself an empirical question.
- Stage-1 sample sizes support methodology validation only; per-condition
  effect estimates begin at Stage 2 scale at the earliest.
- Evidence-condition constructions are currently manual/synthetic; ecological
  validity relative to real retrieval failures must be assessed during the
  pilot.

## 16. Protocol Artifacts

| Element | Location |
|---|---|
| Machine-readable record schema | `configs/schema/record.schema.json` |
| Condition vocabularies | `configs/conditions.yaml` |
| Stage-1 / Stage-2 pilot configs | `configs/pilot_stage1.yaml`, `configs/pilot_stage2.yaml` |
| Validation & sampling code | `src/banglarag_eval/` |
| Synthetic pipeline fixture | `data/fixtures/synthetic_fixture_v0.jsonl` |
| Pilot design & gates | `docs/pilot_design.md` |
| Annotation guidelines | `docs/annotation_guidelines.md` |
| Data schema documentation | `docs/data_schema.md` |
| Evaluation & statistics protocol | `docs/evaluation_protocol.md` |
