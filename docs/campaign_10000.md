# 10,000-Record Pre-Annotation Campaign

> **Status: suspended after 325 records.** Methodological review found
> confounded retriever/context-size comparisons, semantically invalid
> contradictions, and unverified partial evidence. The checkpoint is retained
> unchanged for audit only and must not be finalized or analyzed as the study
> dataset. The active next step is the BM25-only, review-gated methodology
> pilot documented in [pilot_design.md](pilot_design.md).

## Purpose

This campaign produces model outputs for later blind human annotation. It does
not create gold labels and does not replace independent annotators or
adjudication.

## Design

The exact factorial is:

- 200 human-authored QA examples from 200 distinct BanglaQuAD v1.0 Wikipedia
  articles;
- 5 question-language conditions: native Bangla, machine back-translated
  Bangla, Bangla-English code-mixed, Banglish, and English;
- 5 controlled evidence conditions: correct, partially relevant, irrelevant,
  contradictory, and missing;
- 2 paired generator families: Qwen3 8B and Gemma 3 4B.

`200 x 5 x 5 x 2 = 10,000` records. BM25 and SemFuse hybrid retrieval are
assigned evenly and deterministically within every language-by-evidence cell;
both generators see identical evidence within a scenario. Analyses must cluster or
bootstrap by base QA, because the 10,000 rows are repeated measures over 200
independent source questions.

English and round-trip Bangla use the pinned, task-specific BanglaT5
`bn-en`/`en-bn` checkpoints with beam-4 decoding. All 200 transformation pairs
were reviewed; 83 source-specific semantic failures are corrected in the
versioned override file. Code-mixing uses the frozen `code_mix_lexical_v2`
substitution rule, while Banglish uses `bangla-nlpkit` v0.1.1 NLC romanization
plus documented Bengali-confusable normalization. These transformations are
recorded per row and in the campaign manifest.

One generator is not sufficient for the paper's RQ3 or for a strong conference
claim about evaluator ranking: it confounds evaluator behavior with one model's
response style. Two independent model families are the minimum defensible
design here; a third family is a useful robustness extension if compute and
annotation budget permit.

## Source And License

BanglaQuAD v1.0 is loaded from
<https://github.com/banglaAI/BanglaQuAD-v1.0>. The repository declares the MIT
license. Source URL, original context, exact answer span, QA identifier, corpus
version, and license are preserved through the campaign artifacts. Selection is
seeded and takes at most one QA per article to maximize topic diversity.

## Condition Integrity

- Correct evidence must contain the exact human-annotated answer span.
- Partially relevant evidence removes the answer-bearing sentence while
  preserving same-article topical context.
- Irrelevant evidence comes from a different selected Wikipedia article.
- Contradictory evidence substitutes the exact answer span with a different
  human-authored BanglaQuAD answer and logs that QA identifier.
- Missing evidence removes all context.

If retrieval misses the source answer, an oracle support passage is inserted
and this is disclosed in `retrieval_configuration.oracle_source_injected`.
This protects evidence-condition validity; raw source-hit status remains in
`retrieval_configuration.raw_source_hit`.

## Reproducibility

Configuration is frozen in `configs/campaign_10000.yaml`. Question preparation
and answer generation write append-only checkpoints and resume by stable IDs.
Finalization refuses incomplete, duplicate, or schema-invalid data and emits a
SHA-256 manifest plus a JSONL category index.

The configuration also pins the local Ollama artifact IDs, weight digests,
parameter counts, quantization, and model licenses for Qwen3 8B and Gemma 3 4B.
It also pins both BanglaT5 revisions and local weight digests, the audited
question-set digest, and the SemFuse embedding revision. BanglaT5's
CC-BY-NC-SA-4.0 terms make this campaign a non-commercial research artifact.

No model error falls back to the intended answer. A failed call leaves the
campaign incomplete so rerunning can retry it honestly.

## Protocol Deviation And Issue Status

The pre-registered protocol says Stage 1 human annotation and all gates precede
scaling. The project owner requested construction before those annotations on
2026-10-11; the config and manifest record that explicit deviation.

Therefore this campaign can complete the engineering/data-construction portion,
but it cannot truthfully close the following research criteria:

- Issue #9 still requires two genuinely independent annotators and adjudication.
- Issue #10 still requires human-derived agreement/taxonomy measurements.
- Issue #16 remains a protocol deviation until the Stage-1 gates pass.
- Issue #17 requires the completed human and evaluator analysis, not only model
  output generation.
# Methodology v2 campaign

The replacement campaign is frozen in `configs/campaign_10000_v2.yaml`:

- 1,000 distinct BanglaQuAD QA IDs from 249 source articles
- one balanced language assignment per QA (200 QAs per language)
- five evidence conditions per QA
- two paired generators per scenario
- BM25 only, with one frozen retrieval result per QA
- exactly `1,000 x 5 x 2 = 10,000` outputs

The 5,000 construction scenarios are stored in
`data/intermediate/campaign_10000_v2_review.jsonl`. On 2026-10-13, the project
owner explicitly authorized pre-annotation generation while those scenarios
were still pending review. These outputs cannot support evaluator conclusions
until review and adjudication are complete. The prior
`checkpoints/campaign_10000_working.jsonl` file is an audit-only artifact and
must not be merged into the v2 checkpoint.

## Corrective v3 pilot

Generation of methodology v2 was halted at 175 rows after inspection exposed
two construction defects: duplicate source passages could occupy multiple BM25
ranks, and open-ended answers could receive semantically nonsensical
contradiction values. The checkpoint is preserved as an audit artifact,
including all incorrect model answers, and is not resumed.

The corrective pilot is frozen in `configs/pilot_methodology_v3.yaml`. Its 50
construction scenarios use 10 underlying QAs, five evidence conditions, one
model, and BM25 over an exact-passage-deduplicated corpus. Open-ended and
appearance contradictions require explicit curated candidates; the first
appearance replacement is `উজ্জ্বল নীল বর্ণের` rather than `ভাইরাস`.

Each v3 record separately stores `evidence_topical_relevance` and
`evidence_answerability`. Partial evidence is represented as topically relevant
but insufficient, while contradictory evidence is relevant but contradicted.
These construction attributes do not populate human gold fields.

Round-trip Bangla records retain the original Bangla question, intermediate
English question, final Bangla question, and explicit `bn->en->bn` path.
`split_group_id` groups every QA and variant from one source article, so all
dataset splitting must operate on that field. JSONL exports use strict
serializer round-trip validation.

The 50-row packet at `data/intermediate/pilot_methodology_v3_review.jsonl` is
pending. Generation remains blocked until two independent reviewers approve
every row.

## Owner-authorized v3 full generation

`configs/campaign_10000_v3_unreviewed.yaml` applies the corrective construction
rules to the full 1,000-QA blocked-language design. The project owner authorized
generation before construction review, so the resulting 10,000 records are
explicitly versioned and labeled `unreviewed`. They are model-output research
artifacts, not gold evaluation data.

Open-ended contradiction candidates are stored separately in
`data/intermediate/campaign_10000_v3_contradictions.jsonl`. Model-proposed
candidates retain model, prompt, settings, timestamp, and `review_status`;
manually configured candidates retain their own provenance. The pending
5,000-scenario packet is written to
`data/intermediate/campaign_10000_v3_review.jsonl` without reviewer decisions.

Run or resume the complete chain with:

```bash
screen -dmS campaign10kv3 zsh scripts/run_campaign_10000_v3.sh
```

The launcher appends to `logs/campaign_10000_v3_unreviewed.log`, resumes by
stable example ID, and finalizes only after all 10,000 rows exist and validate.
Human review fields must remain empty until two genuine independent reviewers
complete the recorded checks and disagreements are adjudicated.
