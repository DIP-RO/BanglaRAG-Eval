# Data Schema

Canonical machine-readable schema: [`configs/schema/record.schema.json`](../configs/schema/record.schema.json) (JSON Schema draft 2020-12).
Validation code: `src/banglarag_eval/schema.py`. Datasets are JSON Lines (one record per line, UTF-8, `ensure_ascii=False`).

The schema is designed so that a record is **created once and enriched by later
pipeline stages** — retrieval, generation, human annotation, adjudication,
automatic evaluation — without ever being redesigned. Scaling from 10 → 50 →
500 examples changes only data volume, never structure.

## Design decisions relative to the supervisor's field list

The protocol's flat field list is preserved with these documented adaptations:

1. **`annotator_1_*` / `annotator_2_*` → `annotations[]`.** Human annotations
   are stored as an array of `{annotator_id, label, explanation, timestamp}`
   objects. Rationale: the annotation protocol under consideration (see
   *Faithfulness Benchmarking for Bangla RAG*, §6.7) uses **three** annotators;
   a fixed two-annotator schema cannot store that. No information is lost —
   `annotations[0]`/`annotations[1]` correspond to annotator 1/2, and each
   annotation carries the `annotator_id` and `timestamp` the protocol requires.
   Cross-field rule R4 still enforces ≥ 2 independent annotations before
   adjudication.
2. **`adjudicated_label` → `adjudication` object** carrying label,
   explanation, adjudicator_id, method (`third_rater` | `joint_session`), and
   timestamp (reproducibility requires who, how, and when, not just what).
3. **`cost` / `latency`** at top level refer to *generation*; per-evaluator
   cost/latency live inside each `evaluator_outputs[]` entry, because a record
   is scored by multiple evaluators.
4. Added fields: `schema_version`, `source_span`, `intended_answer`,
   `question_generation`, `generation_settings`, `evidence_condition_notes`,
   `corruption_metadata`, `notes` — all serving provenance or controlled
   corruption requirements in the protocol.

No reproducibility field was removed.

## Field reference

### Identity & provenance (required at creation)

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | string (semver) | Schema this record conforms to (currently `1.0.0`). |
| `example_id` | string | Globally unique example identifier. |
| `document_id` | string | Source document the question was built from. |
| `source_id` | string | Source collection/corpus identifier. |
| `question` | string | Question posed to the RAG system. |
| `question_language` | enum `bn, en, bn_en_mixed, banglish` | Surface language/script of the question text. |
| `language_condition` | enum `native_bangla, translated_bangla, code_mixed, banglish, english` | Experimental language condition of the example. |
| `data_origin` | enum `native_authored, human_translated, machine_translated, model_generated, synthetic_constructed, adapted_existing_benchmark` | True provenance of the example's **source_text** (the grounding passage), independent of the simulated condition. Known limitation: a single record-level value cannot express mixed per-component provenance (e.g. a native question over a translated passage); when components differ, `source_text` provenance governs this field, the question's own provenance lives in `question_generation`, and any remaining nuance is recorded in `notes`. A per-component provenance extension is deferred until the pilot shows it is needed. |
| `source_text` | string | Original grounding passage, preserved verbatim (even under corruption). |
| `evidence_condition` | enum `correct, partially_relevant, irrelevant, contradictory, missing` | Controlled evidence condition, stored explicitly at construction time. |
| `dataset_version` | string | Dataset version this record belongs to. |
| `timestamp` | ISO-8601 string | Record creation time. |

### Question provenance (nullable)

`source_span` (`{start_char, end_char, text}`, NFC-normalized character
offsets into `source_text`), `intended_answer`, `question_generation`
(`{method, model, model_version, prompt_version, timestamp}`). Question
generation must never lose provenance: a question is always traceable to its
source span.

### Retrieval stage (empty/null until run)

`retrieved_context` (exact string shown to the generator),
`retrieval_documents[]` (`{document_id, text, rank, score, source}`),
`retrieval_configuration` (free-form config dump), plus flat convenience
fields `retriever`, `embedding_model`, `reranker`.

### Generation stage (null until run)

`generated_answer`, `answer_claims[]`, `generator_model`,
`generator_version`, `generation_settings`, `prompt_version`, `cost`
(USD), `latency` (seconds), `random_seed`. Existing generations are never
overwritten (`save_dataset` refuses by default).

### Controlled corruption (null unless corrupted)

`corruption_metadata`: `{corruption_type, original_context_ids,
modified_context_ids, operation_description, random_seed}`. Corruption is
never silent: the original evidence is preserved and the operation is logged.

**Corruption vs. construction — the boundary.** `corruption_metadata` is
**required** whenever the retrieved context is derived by *modifying,
truncating, or replacing* evidence that was (or would have been) retrieved
for the example — e.g. removing the supporting sentence from a retrieved
passage, or swapping a retrieved passage for an unrelated one. Selecting or
authoring an independent non-supporting passage directly (no retrieved
evidence was altered) is *construction*, not corruption, and requires only
`evidence_condition_notes` explaining how the condition was realized. Both
paths must leave the condition auditable; a modification without
`corruption_metadata` violates the protocol even though schema validation
cannot detect it mechanically (rule R6 enforces only the converse).

### Human annotation (human-entered only)

`annotations[]`: `{annotator_id, label, explanation, timestamp}` with labels
from the faithfulness taxonomy; `adjudication`: `{label, explanation,
adjudicator_id, method, timestamp}` where `method` records the resolution
mechanism (`third_rater` — independent of the annotators, rule R8 — or
`joint_session`); `faithfulness_category` (gold — the adjudicated label, or
the unanimous label of the independent annotators when they agree and no
adjudication exists, rule R3); `hallucination_type` (optional finer-grained
type; human-assigned only, rule R9).

### Automatic evaluation (systems under test, never gold)

`evaluator_outputs[]`: `{evaluator_name, evaluator_version, score, label,
explanation, latency_seconds, cost_usd, configuration, timestamp}`.

## Cross-field rules (enforced by `validate_record`)

| Rule | Invariant |
|---|---|
| R1 | `native_bangla` records cannot have translation provenance. |
| R2 | `translated_bangla` records cannot claim `native_authored` origin. |
| R3 | `faithfulness_category` may only exist with human provenance: it must equal `adjudication.label` when an adjudication exists, or the unanimous label of ≥ 2 independent annotators otherwise. Disagreement without adjudication yields no gold label. It can never come from an automatic evaluator (RAGAS, LLM judge, SemFuse, encoder detector included). |
| R4 | Adjudication requires ≥ 2 prior independent annotations. |
| R5 | `annotator_id`s within a record must be distinct. |
| R6 | `corruption_metadata` implies a degraded `evidence_condition` (never `correct`). |
| R7 | The record `timestamp` and every nested timestamp (annotations, adjudication, question_generation, evaluator_outputs) must parse as ISO-8601. |
| R8 | A `third_rater` adjudicator must not be one of the record's annotators (`joint_session` adjudications are exempt — the method field records the mechanism). |
| R9 | `hallucination_type` requires human annotation provenance on the record (an adjudication, or unanimous independent annotations). |
| R10 | `source_span` must be internally consistent: `start_char < end_char`, in bounds of `source_text`, and `text` (when present) equal to the slice. |

Warning-level (not error): `question_language` deviating from the value
expected for the `language_condition` (kept expressible for future
cross-script designs, flagged so it is always intentional).

## Versioning

- `schema_version` is bumped semver-style on any schema change; records store
  the version they were written under.
- Datasets carry `dataset_version`; a released dataset version is frozen —
  changes produce a new version, never an in-place edit.
