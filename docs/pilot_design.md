# Pilot Design

## Objective

Validate the **methodology** — schema, condition constructions, annotation
guidelines, agreement levels, evaluator harness, and analysis pipeline —
before spending annotation budget at scale. The pilot produces no research
findings; it produces evidence that the pipeline can produce trustworthy
findings.

## Two-stage structure

### Stage 1 — Methodology validation (30–50 examples)

Config: [`configs/pilot_stage1.yaml`](../configs/pilot_stage1.yaml)
(currently `pilot_size: 40`; the size is configuration, not code).

Stage 1 must exercise the complete pipeline end to end:
construction → retrieval/context assembly → generation → **two-annotator
labeling → adjudication** → automatic evaluator runs → metric computation.

### Stage 2 — Pilot expansion (300–500 examples)

Config: [`configs/pilot_stage2.yaml`](../configs/pilot_stage2.yaml)
(currently `pilot_size: 400`). **Stage 2 does not begin until every Stage-1
gate below passes.** Scaling is a config change; the schema and code must not
change (enforced by `test_allocation_scales_without_redesign`).

## Sampling strategy (documented, not idealized)

Targets are configured proportions over LANGUAGE × EVIDENCE:

- Language: native_bangla 0.30, translated_bangla 0.20, code_mixed 0.20,
  banglish 0.15, english 0.15. Bangla is the primary testbed; English is a
  control baseline only; Banglish is a distinct exploratory script condition,
  deliberately not collapsed into code-mixing.
- Evidence: correct 0.30, partially_relevant 0.20, contradictory 0.20,
  irrelevant 0.15, missing 0.15. `correct` is weighted highest because every
  other condition is interpreted relative to it.

Integer cell counts come from **largest-remainder allocation over the joint
distribution** with seeded, reproducible tie-breaking
(`src/banglarag_eval/sampling.py`). A perfectly balanced matrix is *not*
forced; when source-data constraints make a cell infeasible (e.g. too few
natural contradictory contexts in Banglish), the deviation is recorded in the
dataset release notes rather than papered over.

At `pilot_size: 40` the joint targets are small (1–4 per cell); Stage 1
therefore supports only pipeline validation, never per-cell conclusions —
that is by design.

## Source domains

Stage 1 uses locally stored source documents through the source-document
interface (to be implemented in Milestone 2; encyclopedic/factual register,
e.g. hand-curated passages — Bangla Wikipedia integration is planned but not
required for Stage 1).
Licensing is checked before any external corpus is ingested; restricted
datasets are not downloaded or redistributed.

## Annotation process

Per [annotation_guidelines.md](annotation_guidelines.md): two independent
annotators per example with required explanations. Gold labels come from
unanimous agreement directly, or from adjudication (recorded as
`third_rater` — independent of the annotators — or `joint_session`) when the
annotators disagree. Cohen's κ + raw agreement are computed per batch.
Annotators never see condition labels or evaluator outputs.

## Evaluator suite (planned; abstraction lands in Milestone 2)

Candidates, contingent on dependency/API availability: RAGAS faithfulness,
independent LLM-as-judge (cross-model to limit self-preference), an
encoder-based detector, and a simple lexical baseline. Every evaluator runs
behind the common `Evaluator` interface and writes to `evaluator_outputs`.
None of them is ever gold. SemFuse, if integrated, participates only as RAG
infrastructure behind the `RAGBackend` interface.

## Expected outputs of Stage 1

1. A frozen, versioned 30–50 example dataset passing full validation.
2. Two full annotation passes + adjudications with κ and raw agreement.
3. Evaluator outputs + human-agreement metrics for whichever evaluators ran.
4. A written revision log: guideline ambiguities found, taxonomy changes,
   condition-construction problems.
5. A go/no-go decision on each gate below.

## Quality control

- All datasets validate against the schema (CI-testable, no paid APIs).
- Condition constructions are checked by a second person who — blind to the
  stored `evidence_condition` — independently assigns an evidence condition
  from the question, retrieved context, and answer, which is then compared
  to the stored label.
- Corruptions are reproducible from `corruption_metadata` (type, ids,
  operation, seed); originals are always preserved. The boundary between
  corruption (metadata required) and construction (notes sufficient) is
  defined in [data_schema.md](data_schema.md).
- No record is edited in place after annotation begins; corrections create a
  new `dataset_version`.

## Stage-1 gates (stop / revise / scale criteria)

| Gate | Threshold | On failure |
|---|---|---|
| Annotation agreement | Cohen's κ ≥ 0.60 on the 5-way labels (target ≥ 0.70), reported with a 95% bootstrap CI. Scaling additionally requires the CI lower bound ≥ 0.40; at Stage-1 n the CI is wide, so a passing point estimate with a lower bound below 0.40 extends the Stage-1 sample instead of scaling. | Revise guidelines/taxonomy, re-pilot on a fresh sub-batch. κ < 0.40 → redesign taxonomy before any scaling. |
| Taxonomy adequacy | < 10% of examples carry a `NO-FIT:` marker in any annotator's explanation (the marker is defined in [annotation_guidelines.md](annotation_guidelines.md)) | Revise taxonomy; version bump. |
| Condition integrity | **All** Stage-1 examples re-labeled: a checker blind to the stored `evidence_condition` independently assigns a condition from the question/context/answer; ≥ 90% must match the stored label. (Checking 100% is feasible at n ≤ 50; the Stage-2 sampling rate will be pre-registered before Stage 2.) | Fix construction procedures; regenerate affected cells. |
| Pipeline integrity | 100% of records pass validation; every stage's metadata complete | Fix tooling before scaling. |
| Evaluator harness | Each configured evaluator produces standardized output on ≥ 95% of examples (failures logged, not silently dropped) | Fix adapters; re-run. |

Thresholds follow common practice for categorical annotation (κ 0.6–0.8
"substantial"); they are pre-registered here to prevent post-hoc
rationalizing. Three taxonomy boundaries are pre-declared as likely
confusion pairs and tracked explicitly in the Stage-1 revision log:
`partially_faithful` ↔ `unsupported`, `unsupported` ↔
`insufficient_evidence`, and abstention handling — these coincide with the
constructed degraded-evidence conditions, so their agreement is examined
separately.

## What Stage 1 explicitly does NOT claim

No evaluator ranking, no language-condition effect estimates, no
degradation-under-code-mixing conclusion. Sample sizes at Stage 1 cannot
support such claims, and any observed pattern is treated as pipeline
diagnostics only.
