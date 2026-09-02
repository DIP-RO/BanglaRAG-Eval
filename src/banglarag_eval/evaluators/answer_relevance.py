"""Answer relevance evaluator — checks if the answer addresses the question.

This evaluator measures whether the generated answer is relevant to the
asked question. It uses lexical overlap between the question and answer,
plus a simple question-type check (who/what/when/where/why/how).

This is a non-neural evaluator that requires no API — it runs entirely
locally using token overlap and heuristics.

Relevance is different from faithfulness:
- Faithfulness: Is the answer supported by the context?
- Relevance: Does the answer address what was asked?

A faithful answer can be irrelevant (correctly restating context that
doesn't answer the question). An irrelevant answer can be faithful
(supported by context but not answering the question).
"""

from __future__ import annotations

import re
import time
from typing import Any

from .base import Evaluator, EvaluatorOutput, _iso_timestamp
from .lexical_baseline import _tokenize


# Question type indicators
QUESTION_TYPES = {
    "who": ["who", "কে", "কাকে", "কাদের"],
    "what": ["what", "কী", "কি", "কেমন", "কোন"],
    "when": ["when", "কখন", "কবে"],
    "where": ["where", "কোথায়", "কোথা"],
    "why": ["why", "কেন", "কী কারণে"],
    "how": ["how", "কীভাবে", "কিভাবে", "কতটা"],
    "how_many": ["how many", "how much", "কত", "কয়টি", "কতগুলো"],
    "yes_no": ["is", "are", "do", "does", "can", "will", "have",
               "কি", "নাকি", "হবে কি"],
}

# Stopwords to exclude from overlap (low-information words)
STOPWORDS = {
    # English
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "can", "of", "in", "on", "at", "to",
    "for", "with", "by", "from", "as", "and", "or", "but", "not", "no",
    "this", "that", "these", "those", "it", "its", "they", "them", "their",
    "he", "she", "his", "her", "we", "us", "our", "you", "your",
    # Bangla common words
    "এবং", "বা", "কিন্তু", "না", "নি", "এই", "সেই", "তার", "তারা",
    "এটি", "সেটি", "যা", "যে", "এক", "একটি",
}


def _detect_question_type(question: str) -> str:
    """Detect the type of question (who/what/when/where/why/how/yes_no).

    Uses word-boundary matching to avoid false substring matches
    (e.g., "কে" inside "কেন", "how" inside "how many").
    Question words are checked before auxiliary verbs.
    """
    q_lower = question.lower()
    # Priority order: specific question words before auxiliaries
    priority_order = [
        "how_many", "who", "what", "when", "where", "why", "how", "yes_no",
    ]
    for qtype in priority_order:
        for ind in QUESTION_TYPES.get(qtype, []):
            ind_lower = ind.lower()
            # Use word boundary for ASCII indicators
            if ind_lower.isascii():
                pattern = r"\b" + re.escape(ind_lower) + r"\b"
                if re.search(pattern, q_lower):
                    return qtype
            else:
                # For Bangla, check that indicator is not part of a longer word
                # by checking surrounding characters
                idx = q_lower.find(ind_lower)
                while idx >= 0:
                    before = q_lower[idx - 1] if idx > 0 else " "
                    after = q_lower[idx + len(ind_lower)] if idx + len(ind_lower) < len(q_lower) else " "
                    # If surrounding chars are not Bangla letters, it's a word boundary
                    is_bangla = lambda c: "\u0980" <= c <= "\u09FF"
                    if not is_bangla(before) and not is_bangla(after):
                        return qtype
                    idx = q_lower.find(ind_lower, idx + 1)
    return "unknown"


def _content_tokens(tokens: set[str]) -> set[str]:
    """Filter out stopwords from token set."""
    return {t for t in tokens if t not in STOPWORDS and len(t) > 1}


class AnswerRelevanceEvaluator(Evaluator):
    """Answer relevance evaluator — checks question-answer alignment.

    Measures whether the answer is relevant to the question using:
    1. Content token overlap between question and answer
    2. Question type detection (who/what/when/where/why/how)
    3. Answer length adequacy

    No API needed — runs entirely locally.

    Configuration:
        min_answer_length: Minimum answer length to be considered adequate
        relevance_threshold: Score threshold for "relevant" label
    """

    name = "answer_relevance"
    version = "1.0.0"

    def __init__(
        self,
        min_answer_length: int = 10,
        relevance_threshold: float = 0.3,
    ):
        self.min_answer_length = min_answer_length
        self.relevance_threshold = relevance_threshold
        self.configuration = {
            "min_answer_length": min_answer_length,
            "relevance_threshold": relevance_threshold,
            "method": "content_token_overlap + question_type_detection",
        }

    def evaluate(self, record: dict[str, Any]) -> EvaluatorOutput:
        """Evaluate answer relevance.

        Reads: question, generated_answer.
        Does NOT read: annotations, faithfulness_category, or any human label.
        """
        start = time.time()

        question = record.get("question", "") or ""
        answer = record.get("generated_answer", "") or ""

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

        if not question.strip():
            return EvaluatorOutput(
                evaluator_name=self.name,
                evaluator_version=self.version,
                score=0.5,
                label="insufficient_evidence",
                explanation="No question provided — cannot assess relevance.",
                latency_seconds=round(time.time() - start, 3),
                cost_usd=0.0,
                configuration=self.configuration,
                timestamp=_iso_timestamp(),
            )

        # Tokenize and filter
        q_tokens = _content_tokens(_tokenize(question))
        a_tokens = _content_tokens(_tokenize(answer))

        if not q_tokens:
            return EvaluatorOutput(
                evaluator_name=self.name,
                evaluator_version=self.version,
                score=0.5,
                label="insufficient_evidence",
                explanation="No content tokens in question.",
                latency_seconds=round(time.time() - start, 3),
                cost_usd=0.0,
                configuration=self.configuration,
                timestamp=_iso_timestamp(),
            )

        if not a_tokens:
            return EvaluatorOutput(
                evaluator_name=self.name,
                evaluator_version=self.version,
                score=0.0,
                label="unsupported",
                explanation="No content tokens in answer.",
                latency_seconds=round(time.time() - start, 3),
                cost_usd=0.0,
                configuration=self.configuration,
                timestamp=_iso_timestamp(),
            )

        # Compute relevance: fraction of question content tokens in answer
        overlap = len(q_tokens & a_tokens) / len(q_tokens)

        # Question type detection
        q_type = _detect_question_type(question)

        # Answer length adequacy
        answer_len = len(answer.split())
        length_adequate = answer_len >= 3  # At least 3 words

        # Adjust score
        if not length_adequate:
            overlap *= 0.5  # Penalize very short answers

        # Map to label
        if overlap >= self.relevance_threshold and length_adequate:
            label = "faithful"  # Answer is relevant to the question
            explanation = (
                f"Relevance score: {overlap:.2f} (question type: {q_type}). "
                f"Answer addresses the question."
            )
        elif overlap >= self.relevance_threshold * 0.5:
            label = "partially_faithful"
            explanation = (
                f"Relevance score: {overlap:.2f} (question type: {q_type}). "
                f"Answer partially addresses the question."
            )
        else:
            label = "unsupported"
            explanation = (
                f"Relevance score: {overlap:.2f} (question type: {q_type}). "
                f"Answer does not address the question."
            )

        return EvaluatorOutput(
            evaluator_name=self.name,
            evaluator_version=self.version,
            score=round(overlap, 4),
            label=label,
            explanation=explanation,
            latency_seconds=round(time.time() - start, 3),
            cost_usd=0.0,
            configuration=self.configuration,
            timestamp=_iso_timestamp(),
        )
