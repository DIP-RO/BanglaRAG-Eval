"""NLI entailment evaluator — checks if answer is entailed by context.

Uses Ollama to perform a single NLI check: given the context and the
answer, determine if the answer is entailed, contradicted, or neutral
relative to the context.

This is simpler than RAGAS (no claim extraction step) but faster.
It provides a complementary signal to the RAGAS faithfulness evaluator.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
import urllib.error
from typing import Any

from .base import Evaluator, EvaluatorOutput, _iso_timestamp

NLI_PROMPT = """/no_think
You are an NLI (Natural Language Inference) judge. Given a context and an answer, determine the entailment relationship.

Context: {context}

Answer: {answer}

Respond with exactly one line in this format:
LABEL: <entailed|contradicted|neutral>
REASON: <brief explanation>

Judgment:"""


def _strip_thinking(text: str) -> str:
    """Remove Qwen3 thinking trace."""
    text = re.sub(r" IMD.*? IMD", "", text, flags=re.DOTALL)
    text = re.sub(r" IMD", "", text)
    return text.strip()


def _parse_nli_response(response: str) -> tuple[str | None, str | None]:
    """Parse NLI response into (label, reason)."""
    response = _strip_thinking(response)

    label = None
    reason = None

    label_match = re.search(r"LABEL:\s*(\w+)", response, re.IGNORECASE)
    if label_match:
        label = label_match.group(1).lower().strip()

    reason_match = re.search(r"REASON:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
    if reason_match:
        reason = reason_match.group(1).strip()

    # Fallback: search for label keywords (including word stems)
    if label is None:
        lower_resp = response.lower()
        if "entail" in lower_resp or "support" in lower_resp or "agree" in lower_resp:
            label = "entailed"
        elif "contradict" in lower_resp or "conflict" in lower_resp or "wrong" in lower_resp:
            label = "contradicted"
        elif "neutral" in lower_resp or "not address" in lower_resp or "not enough" in lower_resp:
            label = "neutral"

    return label, reason


# Score mapping: entailed=1.0, neutral=0.5, contradicted=0.0
NLI_SCORES = {
    "entailed": 1.0,
    "contradicted": 0.0,
    "neutral": 0.5,
}

# Label mapping to 5-way taxonomy
NLI_TO_LABEL = {
    "entailed": "faithful",
    "contradicted": "contradictory",
    "neutral": "unsupported",
}


class NLIEvaluator(Evaluator):
    """NLI entailment evaluator using Ollama.

    Performs a single NLI check: is the answer entailed by the context?
    Simpler and faster than RAGAS (no claim extraction).

    Configuration:
        judge_model: Ollama model (default: JUDGE_MODEL env or "qwen3:8b")
        base_url: Ollama URL
        timeout: Request timeout
    """

    name = "nli_entailment"
    version = "1.0.0"

    def __init__(
        self,
        judge_model: str | None = None,
        base_url: str | None = None,
        timeout: int = 60,
    ):
        self.judge_model = judge_model or os.environ.get("JUDGE_MODEL", "qwen3:8b")
        self.base_url = base_url or os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self.timeout = timeout
        self.configuration = {
            "judge_model": self.judge_model,
            "method": "single_NLI_check",
            "circularity_note": "Judge model should differ from generator model",
        }

    def evaluate(self, record: dict[str, Any]) -> EvaluatorOutput:
        """Evaluate using NLI entailment check.

        Reads: retrieved_context, generated_answer.
        Does NOT read: annotations, faithfulness_category, or any human label.
        """
        start = time.time()

        answer = record.get("generated_answer", "") or ""
        context = record.get("retrieved_context", "") or ""

        if not answer.strip():
            return EvaluatorOutput(
                evaluator_name=self.name,
                evaluator_version=self.version,
                score=0.0,
                label="insufficient_evidence",
                explanation="Empty answer.",
                latency_seconds=round(time.time() - start, 3),
                cost_usd=0.0,
                configuration=self.configuration,
                timestamp=_iso_timestamp(),
            )

        if not context.strip():
            return EvaluatorOutput(
                evaluator_name=self.name,
                evaluator_version=self.version,
                score=0.0,
                label="insufficient_evidence",
                explanation="Empty context — cannot check entailment.",
                latency_seconds=round(time.time() - start, 3),
                cost_usd=0.0,
                configuration=self.configuration,
                timestamp=_iso_timestamp(),
            )

        prompt = NLI_PROMPT.format(context=context, answer=answer)

        payload = json.dumps({
            "model": self.judge_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "seed": 42},
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            return EvaluatorOutput(
                evaluator_name=self.name,
                evaluator_version=self.version,
                score=None,
                label=None,
                explanation=f"NLI check failed: {type(exc).__name__}",
                latency_seconds=round(time.time() - start, 3),
                cost_usd=0.0,
                configuration=self.configuration,
                timestamp=_iso_timestamp(),
            )

        raw_response = data.get("response", "")
        nli_label, reason = _parse_nli_response(raw_response)

        score = NLI_SCORES.get(nli_label, 0.5)
        faithfulness_label = NLI_TO_LABEL.get(nli_label, "unsupported")

        return EvaluatorOutput(
            evaluator_name=self.name,
            evaluator_version=self.version,
            score=score,
            label=faithfulness_label,
            explanation=reason or f"NLI verdict: {nli_label}",
            latency_seconds=round(time.time() - start, 3),
            cost_usd=0.0,
            configuration=self.configuration,
            timestamp=_iso_timestamp(),
        )
