"""Lexical baseline evaluator — simple overlap-based faithfulness checker.

This evaluator requires no API and runs immediately. It checks how much
of the generated answer is supported by the retrieved context using
lexical overlap (token-level). It is intentionally simple — it serves
as a non-neural baseline against which more sophisticated evaluators
are compared.

The evaluator:
1. Tokenizes the answer and context (Bangla-aware tokenization)
2. Computes the fraction of answer tokens found in the context
3. Maps the overlap score to a faithfulness label:
   - score >= 0.8 -> "faithful"
   - score >= 0.5 -> "partially_faithful"
   - score >= 0.2 -> "unsupported"
   - score < 0.2 -> "unsupported" (or "contradictory" if negation detected)
   - empty context -> "insufficient_evidence"
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Any

from .base import Evaluator, EvaluatorOutput, _iso_timestamp


def _tokenize(text: str) -> set[str]:
    """Tokenize text into a set of lowercase tokens.

    Handles both Bangla and English text. Bangla Unicode range:
    U+0980–U+09FF.
    """
    # Split on whitespace and punctuation, keep Bangla and English words
    tokens = set()
    # English tokens
    for tok in re.findall(r"[a-zA-Z]+", text.lower()):
        if len(tok) > 1:
            tokens.add(tok)
    # Bangla tokens (Unicode range U+0980-U+09FF)
    for tok in re.findall(r"[\u0980-\u09FF]+", text):
        if len(tok) > 1:
            tokens.add(tok)
    # Numbers
    for tok in re.findall(r"\d+", text):
        tokens.add(tok)
    return tokens


# Bangla negation words that may indicate contradiction
_BANGLA_NEGATION = {"নয়", "না", "নি", "ভুল", "সঠিক নয়", "গলত"}
_ENGLISH_NEGATION = {"not", "no", "never", "wrong", "incorrect", "false"}


def _detect_negation(answer: str) -> bool:
    """Check if the answer contains negation words."""
    answer_lower = answer.lower()
    for neg in _ENGLISH_NEGATION:
        if neg in answer_lower.split():
            return True
    for neg in _BANGLA_NEGATION:
        if neg in answer:
            return True
    return False


class LexicalBaselineEvaluator(Evaluator):
    """Simple lexical overlap baseline evaluator.

    Computes token-level overlap between generated answer and retrieved
    context. No API needed — runs entirely locally.

    Configuration:
        threshold_faithful: float = 0.8  — score >= this -> faithful
        threshold_partial: float = 0.5  — score >= this -> partially_faithful
        threshold_unsupported: float = 0.2 — score >= this -> unsupported
    """

    name = "lexical_baseline"
    version = "1.0.0"

    def __init__(
        self,
        threshold_faithful: float = 0.8,
        threshold_partial: float = 0.5,
        threshold_unsupported: float = 0.2,
    ):
        self.threshold_faithful = threshold_faithful
        self.threshold_partial = threshold_partial
        self.threshold_unsupported = threshold_unsupported
        self.configuration = {
            "threshold_faithful": threshold_faithful,
            "threshold_partial": threshold_partial,
            "threshold_unsupported": threshold_unsupported,
            "tokenizer": "bangla_aware_regex",
        }

    def evaluate(self, record: dict[str, Any]) -> EvaluatorOutput:
        """Evaluate a record using lexical overlap.

        Reads: question, retrieved_context, generated_answer.
        Does NOT read: annotations, faithfulness_category, or any human label.
        """
        start = time.time()

        answer = record.get("generated_answer", "") or ""
        context = record.get("retrieved_context", "") or ""

        # Empty context -> insufficient evidence
        if not context.strip():
            return EvaluatorOutput(
                evaluator_name=self.name,
                evaluator_version=self.version,
                score=0.0,
                label="insufficient_evidence",
                explanation="Empty retrieved context — cannot verify any claims.",
                latency_seconds=round(time.time() - start, 3),
                configuration=self.configuration,
                timestamp=_iso_timestamp(),
            )

        # Empty answer -> insufficient evidence
        if not answer.strip():
            return EvaluatorOutput(
                evaluator_name=self.name,
                evaluator_version=self.version,
                score=0.0,
                label="insufficient_evidence",
                explanation="Empty generated answer — nothing to evaluate.",
                latency_seconds=round(time.time() - start, 3),
                configuration=self.configuration,
                timestamp=_iso_timestamp(),
            )

        # Tokenize
        answer_tokens = _tokenize(answer)
        context_tokens = _tokenize(context)

        if not answer_tokens:
            return EvaluatorOutput(
                evaluator_name=self.name,
                evaluator_version=self.version,
                score=0.0,
                label="insufficient_evidence",
                explanation="No valid tokens found in answer.",
                latency_seconds=round(time.time() - start, 3),
                configuration=self.configuration,
                timestamp=_iso_timestamp(),
            )

        # Compute overlap: fraction of answer tokens in context
        overlap = len(answer_tokens & context_tokens) / len(answer_tokens)

        # Detect negation for contradiction heuristic
        has_negation = _detect_negation(answer)

        # Map score to label
        if overlap >= self.threshold_faithful:
            label = "faithful"
            explanation = f"High lexical overlap ({overlap:.2f}) — most answer tokens found in context."
        elif overlap >= self.threshold_partial:
            label = "partially_faithful"
            explanation = f"Moderate lexical overlap ({overlap:.2f}) — some answer tokens not in context."
        elif overlap < self.threshold_unsupported and has_negation:
            label = "contradictory"
            explanation = f"Low overlap ({overlap:.2f}) with negation detected — possible contradiction."
        else:
            label = "unsupported"
            explanation = f"Low lexical overlap ({overlap:.2f}) — most answer tokens not found in context."

        return EvaluatorOutput(
            evaluator_name=self.name,
            evaluator_version=self.version,
            score=round(overlap, 4),
            label=label,
            explanation=explanation,
            latency_seconds=round(time.time() - start, 3),
            cost_usd=0.0,  # No API cost
            configuration=self.configuration,
            timestamp=_iso_timestamp(),
        )
