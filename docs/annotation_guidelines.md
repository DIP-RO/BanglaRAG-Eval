# Annotation Guidelines (v0.1 — to be piloted)

Human annotation is the **independent reference standard** of this benchmark.
No automatic evaluator output (RAGAS, LLM judge, encoder detector, SemFuse
grounding score, or any other) may be shown to annotators, used to pre-fill
labels, or substituted for a human label. These guidelines will be revised
during Stage 1 piloting; every revision gets a new version number, and data
annotated under different guideline versions is never silently mixed.

## What the annotator judges

**Faithfulness to the retrieved evidence — not truth in the world.**
An answer that is factually true but not supported by the retrieved context is
*not* faithful. An answer that repeats an error present in the retrieved
context *is* faithful (to the evidence). Annotators judge only the relation
between the generated answer and the retrieved context shown to them.

## What the annotator sees

1. The question
2. The retrieved evidence (exactly what the generator saw — `retrieved_context`)
3. The generated answer
4. The five category choices below
5. A free-text explanation field (required)

Annotators do **not** see: the original source document, the intended answer,
the evidence-condition label, other annotators' labels, or any automatic
evaluator output.

## Faithfulness taxonomy

**The ordered decision procedure below is authoritative.** The table is a
summary; whenever the two seem to differ, follow the procedure.

| Label | Summary |
|---|---|
| **Faithful** (`faithful`) | Every substantive claim in the answer is supported by the retrieved evidence. |
| **Partially Faithful** (`partially_faithful`) | The central claims are at least partly supported, but at least one substantive claim (central or side) is unsupported or unverifiable. |
| **Unsupported** (`unsupported`) | No central claim is supported by the evidence — regardless of real-world truth. Supported side claims do not change this. |
| **Contradictory** (`contradictory`) | At least one central claim directly conflicts with the retrieved evidence. |
| **Insufficient Evidence** (`insufficient_evidence`) | The evidence cannot be used to check *any* substantive claim (unrelated or essentially unusable). |

### Decision procedure (apply in order; the procedure is authoritative)

1. **Identify the substantive claims** in the answer and mark which are
   *central* (they answer the question) and which are *side* claims
   (additional detail). Ignore hedges, politeness, and formatting.
2. **Does any central claim conflict with the evidence?** → `contradictory`.
   Conflict takes precedence over the remaining steps. *Exception:*
   abstention statements ("the context does not contain the answer") are
   meta-statements about the evidence, not content claims — they are exempt
   from this step and handled in the abstention boundary case below.
3. **Usability test:** could the evidence, read charitably, be used to check
   **any** substantive claim of the answer?
   - **No** (evidence unrelated or essentially unusable for this question)
     → `insufficient_evidence`.
   - **Yes** → continue.
4. **Is any central claim supported?**
   - **No** (every central claim unchecked or unverifiable from the
     evidence) → `unsupported`, even when a side claim happens to be
     supported.
   - **Yes** → continue.
5. **Is every substantive claim (central and side) supported?**
   → `faithful`. Otherwise → `partially_faithful`.

### Worked boundary cases

- **True-but-unsupported.** Evidence: "ঢাকা বুড়িগঙ্গা নদীর তীরে অবস্থিত।"
  Answer: "ঢাকা বাংলাদেশের রাজধানী।" The answer is true in the world but the
  evidence says nothing about capitals → `unsupported`.
- **Faithful-to-wrong-evidence.** If the evidence itself contains an error and
  the answer repeats it → `faithful`. Note the situation in the explanation.
- **Numbers, dates, honorifics.** Bangla numerals (৬.১৫) and Arabic numerals
  (6.15) are equivalent; Bangla and Gregorian calendar dates must match in
  meaning, not script. Honorific variation (তিনি/সে) is not an unfaithfulness.
- **Paraphrase and translation equivalence.** In code-mixed and Banglish
  examples, the same fact may surface in either language or script
  ("Dhaka" / "ঢাকা" / "Dhaka"): script difference alone never makes a claim
  unsupported.
- **Abstention (exempt from step 2's conflict precedence).** If the answer
  declines to answer ("প্রদত্ত তথ্যে এর উত্তর নেই"): when the evidence
  indeed lacks the answer → `faithful`, with "justified abstention" in the
  explanation. When the evidence *does* contain the answer (a false
  refusal) → `unsupported`, with "false refusal" in the explanation — not
  `contradictory`, because the refusal is a meta-statement about the
  evidence rather than a content claim.
- **Numeric materiality.** A figure in the answer is *compatible* with the
  evidence when it equals the evidence's figure after rounding the evidence
  to the precision the answer states (evidence ৬.১৫ কিমি: answer "প্রায়
  ৬.২ কিমি" is compatible at one decimal; answer "প্রায় ৯ কিমি" is not).
  Compatible → supported; incompatible → conflict (`contradictory` if the
  claim is central).
- **No category fits.** If, after the procedure, you genuinely cannot place
  the example in any category, choose the closest label and begin your
  explanation with the marker `NO-FIT:` followed by why no category fits.
  This marker is the instrument for the taxonomy-adequacy gate in
  [pilot_design.md](pilot_design.md) — use it honestly; flagged examples
  drive taxonomy revision.

### Optional finer-grained hallucination type

When the label is not `faithful`, annotators may additionally assign one
provisional hallucination type (RAGTruth-inspired): `evident_conflict`,
`subtle_conflict`, `evident_baseless_information`,
`subtle_baseless_information`, `other`. This field is optional in Stage 1 and
its usefulness is itself under evaluation.

### Token-level / span-level annotation (optional in Stage 1)

The primary annotation is **response-level** (the five-way faithfulness
category above). In addition, annotators *may* mark the specific token or
span in the generated answer that triggers the unfaithfulness — the
**token-/span-level** annotation layer. This serves two purposes:

1. It forces annotators to localize *where* the answer departs from the
   evidence, improving label quality and explanation specificity.
2. It enables token-/span-level evaluator comparison in later analysis
   (e.g., does an encoder detector's per-token score align with the
   human-marked unfaithful span?).

In Stage 1, token-level annotation is **optional and exploratory**: its
annotation reliability and usefulness are themselves under evaluation.
If annotators find it burdensome or unreliable at pilot scale, it may be
dropped or deferred. The schema does not yet have a dedicated field for
token-level spans; if the pilot shows it is useful, a schema extension
(e.g. an `unfaithful_spans[]` array on each annotation) will be added in
a versioned schema bump — never as an ad-hoc field.

## Language-condition-specific notes

- **Native Bangla / Translated Bangla:** judged identically; annotators are
  not told which is which.
- **Code-mixed:** English tokens inside Bengali-script sentences are normal;
  judge meaning, not language purity.
- **Banglish (romanized Bangla):** spelling is not standardized
  ("bochor"/"bosor"). Judge by intended meaning; flag genuinely undecipherable
  romanization as `insufficient_evidence` *only if the evidence (not the
  answer) is undecipherable* — an undecipherable answer with clear evidence is
  `unsupported`, with an explanation.

## Process

1. **Two independent annotators** label every example (label + explanation),
   without seeing each other's work. The infrastructure supports more than
   two; two is the Stage 1 minimum (schema rule R4).
2. Each annotation stores `annotator_id`, `label`, `explanation`, `timestamp`.
3. **Gold label — agreement case:** when the independent annotators assign
   the same label, that unanimous label becomes the gold
   `faithfulness_category` directly; no adjudication record is created
   (schema rule R3 accepts exactly this).
4. **Gold label — disagreement case (adjudication):** when the labels
   disagree, an adjudication pass assigns the gold label. Two mechanisms are
   permitted and must be recorded in `adjudication.method`:
   - `third_rater` — an independent senior annotator who is **not** one of
     the original annotators (enforced by schema rule R8) decides;
   - `joint_session` — the original annotators resolve the case together;
     `adjudicator_id` identifies who recorded the consensus.
   The mechanism is stored per record so the two adjudication styles (with
   different bias profiles) remain analyzable. `third_rater` is the default
   for Stage 1; `joint_session` is reserved for cases the third rater sends
   back for discussion.
   The gold `faithfulness_category` comes from the unanimous label or the
   adjudicated label — and by no other mechanism.
5. **Agreement measurement:** Cohen's κ (two annotators, categorical labels)
   and raw agreement are computed on every batch *before* scaling decisions.
   Stage-1 gates are defined in [pilot_design.md](pilot_design.md).

## Ethics and annotator welfare

Source content is factual/encyclopedic; no offensive-content exposure is
expected in the pilot. Annotators are identified in the data only by opaque
`annotator_id` values; the mapping to persons is kept outside the repository.
