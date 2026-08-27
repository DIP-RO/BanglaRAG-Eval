# Evaluation Protocol

## The gold-standard rule

The reference label for every example is the **human label**
(`faithfulness_category`): the unanimous label when the independent
annotators agree, or the adjudicated label when they disagreed — schema rule
R3 enforces both paths mechanically, and no other path exists. The following
are **never** gold, under any configuration:

- RAGAS or any reference-free metric
- Any LLM-as-judge
- SemFuse's internal grounding/faithfulness mechanism
- Encoder-based hallucination detectors
- Any ensemble or vote of the above

Automatic evaluators are the *objects of study*: the benchmark measures how
well they recover human judgments across language and evidence conditions.

## Evaluator abstraction (Milestone 2 implementation target)

Every evaluator implements a common interface:

```text
Evaluator
  name: str
  version: str
  evaluate(record) -> EvaluatorOutput
      # receives the standardized record (question, retrieved_context,
      # generated_answer, and declared configuration)
      # returns: evaluator_name, evaluator_version, score, label,
      #          explanation, latency_seconds, cost_usd, configuration,
      #          timestamp
```

Outputs append to the record's `evaluator_outputs` array — never to the
annotation fields. Evaluators must not see human labels at inference time.
Planned adapters (implemented only when dependencies/API access exist —
abstraction first): RAGAS faithfulness, independent LLM judges (cross-model to
limit self-preference), ARES where technically feasible, an encoder detector,
and a lexical baseline. SemFuse integrates, if at all, behind a separate
`RAGBackend` interface (retrieval/generation infrastructure only).

## Metrics

### Human agreement (annotation quality)

- Cohen's κ between the two independent annotators (5-way categorical).
- Raw agreement rate.
- Reported per batch and per language condition (descriptively at pilot scale).

### Evaluator vs. human performance

Computed against gold labels (unanimous or adjudicated human labels):

- Precision, recall, F1 (macro and per class)
- Balanced accuracy (imbalance in the gold *faithfulness-label* distribution
  is expected — the label distribution is an empirical outcome, not designed,
  and is unlikely to be uniform)
- AUROC where an evaluator emits scores. **Pre-registered primary
  binarization: `faithful` vs. all other labels.** Alternative binarizations
  (e.g. {faithful, partially_faithful} vs. rest) may be reported, but only as
  clearly labeled exploratory analyses — never selected after seeing which
  performs better.
- Cohen's κ between evaluator labels and gold labels

Score-emitting evaluators additionally get score–label agreement statistics
appropriate to type (e.g. rank correlation between scores and ordinal
severity), chosen and justified case by case.

### Token-level / span-level evaluation (exploratory)

The primary evaluation is **response-level** (the five-way faithfulness
category). If the optional token-/span-level human annotation is collected
(see [annotation_guidelines.md](annotation_guidelines.md)), an additional
**token-level** analysis becomes possible:

- For encoder-based detectors that emit per-token scores: token-level
  precision/recall against human-marked unfaithful spans.
- For LLM judges: whether the judge's explanation localizes the same
  unfaithful span the human marked.
- Span-overlap metrics (e.g. IoU on character offsets) between evaluator-
  identified and human-marked unfaithful regions.

Token-level evaluation is **exploratory in Stage 1**: it is reported only
if the pilot demonstrates sufficient annotation reliability at the span
level. No token-level results are reported without the underlying human
span annotations.

### Ranking stability

Framework requirement (Milestone 2+): given multiple generators/systems or
conditions, compare the *rankings* each evaluator induces — rank correlation
(Spearman/Kendall) between evaluator-induced rankings and human-induced
rankings, and cross-evaluator ranking agreement. Pilot scale permits only the
mechanism to be exercised, not conclusions.

### Efficiency

Measured, never estimated: wall-clock latency per example, cost per example
(API pricing at run time, recorded in the run config), examples/second for
local models. **Values that were not actually measured are not reported.**

## Statistical testing

Configurable analysis module (Milestone 2+). Planned tests, each applied only
where its assumptions hold, with the rationale recorded in the run config:

| Comparison | Test | Why appropriate |
|---|---|---|
| Evaluator A vs B on the same examples (accuracy/F1) | Paired bootstrap over examples | No distributional assumption; respects pairing; works for small n. |
| Evaluator A vs B, binary correctness | McNemar's test | Exactly the paired disagreement-count situation McNemar addresses. |
| Any point estimate | Bootstrap confidence intervals | Uniform, assumption-light uncertainty reporting. |
| Multiple comparisons across the condition grid | Holm correction | Family-wise error control without independence assumptions. |

Rules: no test is run whose assumptions demonstrably fail; p-values are never
manufactured or reported without the underlying measured data; pilot-scale
results carry explicit "underpowered — diagnostic only" caveats.

## Circularity controls

- Judge models are disjoint from generator models wherever possible; any
  overlap is disclosed and analyzed for self-preference.
- Prompt templates for judges are versioned (`prompt_version`); undocumented
  prompt changes are prohibited.
- Human annotators never see evaluator outputs; evaluators never see human
  labels.

## Reproducibility of evaluation runs

Every run records: dataset version, code commit, config file, evaluator
name/version/configuration, model versions, prompt versions, seeds,
timestamps. API keys come from environment variables only and are never
committed.
