"""Exact-match precision evaluator — strict token precision baseline.

This evaluator measures the fraction of answer tokens that appear
exactly in the context. Unlike the lexical baseline (which uses sets),
this evaluator uses term frequency (TF) to account for repeated tokens.

It also computes:
- Precision: fraction of answer token occurrences found in context
- Recall: fraction of context token occurrences found in answer
- F1: harmonic mean of precision and recall

This is the strictest non-neural baseline — no normalization, no
stemming, no semantic matching. It serves as a lower bound for
evaluator performance.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from typing import Any

from .base import Evaluator, EvaluatorOutput, _iso_timestamp


def _tokenize_tf(text: str) -> Counter:
    """Tokenize text into a Counter of token frequencies.

    Handles both Bangla and English text.
    """
    tokens = Counter()
    # English tokens
    for tok in re.findall(r"[a-zA-Z]+", text.lower()):
        if len(tok) > 1:
            tokens[tok] += 1
    # Bangla tokens
    for tok in re.findall(r"[\u0980-\u09FF]+", text):
        if len(tok) > 1:
            tokens[tok] += 1
    # Numbers
    for tok in re.findall(r"\d+", text):
        tokens[tok] += 1
    return tokens


class ExactMatchEvaluator(Evaluator):
    """Exact-match precision evaluator — strict token-level precision.

    Computes TF-based precision, recall, and F1 between answer and context.
    No API needed — runs entirely locally.

    This is the strictest baseline: no normalization, no semantics.
    It serves as a lower bound for evaluator performance.

    Configuration:
        threshold_faithful: F1 >= this -> faithful
        threshold_partial: F1 >= this -> partially_faithful
        threshold_unsupported: F1 >= this -> unsupported
    """

    name = "exact_match_precision"
    version = "1.0.0"

    def __init__(
        self,
        threshold_faithful: float = 0.7,
        threshold_partial: float = 0.4,
        threshold_unsupported: float = 0.15,
    ):
        self.threshold_faithful = threshold_faithful
        self.threshold_partial = threshold_partial
        self.threshold_unsupported = threshold_unsupported
        self.configuration = {
            "threshold_faithful": threshold_faithful,
            "threshold_partial": threshold_partial,
            "threshold_unsupported": threshold_unsupported,
            "method": "tf_precision_recall_f1",
            "tokenizer": "bangla_aware_regex_tf",
        }

    def evaluate(self, record: dict[str, Any]) -> EvaluatorOutput:
        """Evaluate using exact-match TF precision/recall/F1.

        Reads: retrieved_context, generated_answer.
        Does NOT read: annotations, faithfulness_category, or any human label.
        """
        start = time.time()

        answer = record.get("generated_answer", "") or ""
        context = record.get("retrieved_context", "") or ""

        if not context.strip():
            return EvaluatorOutput(
                evaluator_name=self.name,
                evaluator_version=self.version,
                score=0.0,
                label="insufficient_evidence",
                explanation="Empty context — cannot compute precision.",
                latency_seconds=round(time.time() - start, 3),
                cost_usd=0.0,
                configuration=self.configuration,
                timestamp=_iso_timestamp(),
            )

        if not answer.strip():
            return EvaluatorOutput(
                evaluator_name=self.name,
                evaluator_version=self.version,
                score=0.0,
                label="insufficient_evidence",
                explanation="Empty answer — nothing to evaluate.",
                latency_seconds=round(time.time() - start, 3),
                cost_usd=0.0,
                configuration=self.configuration,
                timestamp=_iso_timestamp(),
            )

        # Tokenize with term frequencies
        answer_tf = _tokenize_tf(answer)
        context_tf = _tokenize_tf(context)

        if not answer_tf:
            return EvaluatorOutput(
                evaluator_name=self.name,
                evaluator_version=self.version,
                score=0.0,
                label="insufficient_evidence",
                explanation="No valid tokens in answer.",
                latency_seconds=round(time.time() - start, 3),
                cost_usd=0.0,
                configuration=self.configuration,
                timestamp=_iso_timestamp(),
            )

        # Compute TF-based precision and recall
        # Precision: fraction of answer token occurrences that appear in context
        # For each answer token, count how many occurrences are "covered" by context
        n_answer_total = sum(answer_tf.values())
        n_covered = 0
        for token, count in answer_tf.items():
            context_count = context_tf.get(token, 0)
            n_covered += min(count, context_count)

        precision = n_covered / n_answer_total if n_answer_total > 0 else 0.0

        # Recall: fraction of context token occurrences that appear in answer
        n_context_total = sum(context_tf.values())
        n_recall = 0
        for token, count in answer_tf.items():
            context_count = context_tf.get(token, 0)
            n_recall += min(count, context_count)
        recall = n_recall / n_context_total if n_context_total > 0 else 0.0

        # F1
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        # Map F1 to label
        if f1 >= self.threshold_faithful:
            label = "faithful"
            explanation = f"TF-F1={f1:.3f} (P={precision:.3f}, R={recall:.3f}). High exact match."
        elif f1 >= self.threshold_partial:
            label = "partially_faithful"
            explanation = f"TF-F1={f1:.3f} (P={precision:.3f}, R={recall:.3f}). Moderate exact match."
        elif f1 < self.threshold_unsupported:
            label = "unsupported"
            explanation = f"TF-F1={f1:.3f} (P={precision:.3f}, R={recall:.3f}). Low exact match."
        else:
            label = "unsupported"
            explanation = f"TF-F1={f1:.3f} (P={precision:.3f}, R={recall:.3f}). Below threshold."

        return EvaluatorOutput(
            evaluator_name=self.name,
            evaluator_version=self.version,
            score=round(f1, 4),
            label=label,
            explanation=explanation,
            latency_seconds=round(time.time() - start, 3),
            cost_usd=0.0,
            configuration=self.configuration,
            timestamp=_iso_timestamp(),
        )
