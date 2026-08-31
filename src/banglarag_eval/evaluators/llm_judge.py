"""LLM-as-judge evaluator — uses a different LLM to judge faithfulness.

This evaluator uses Ollama to run an LLM judge. To avoid circularity
(self-preference), the judge model should be DIFFERENT from the
generator model. If the generator used qwen3:8b, the judge should use
a different model (e.g., qwen3:4b, llama3.2, etc.).

The judge receives:
- The question
- The retrieved context
- The generated answer

The judge does NOT see:
- The source document
- The intended answer
- The evidence condition
- Human annotations or gold labels

The judge returns a faithfulness label and explanation.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Any

from .base import Evaluator, EvaluatorOutput, _iso_timestamp

# Judge prompt template — instructs the LLM to evaluate faithfulness
# using the same 5-way taxonomy as human annotators.
JUDGE_PROMPT_TEMPLATE = """/no_think
You are a faithfulness evaluator. Given a question, retrieved context, and a generated answer, determine if the answer is faithful to the context.

Use this 5-way taxonomy:
- faithful: Every substantive claim in the answer is supported by the context.
- partially_faithful: Some claims are supported, but not all.
- unsupported: No central claim is supported by the context (even if some side claims are).
- contradictory: At least one central claim conflicts with the context.
- insufficient_evidence: The context is empty or cannot check any substantive claim.

Respond in this exact format:
LABEL: <one of the 5 labels>
EXPLANATION: <brief explanation>

Question: {question}

Context: {context}

Answer: {answer}

Evaluation:"""


def _strip_thinking(text: str) -> str:
    """Remove Qwen3 thinking trace from output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def _parse_judge_response(response: str) -> tuple[str | None, str | None]:
    """Parse the judge's response into (label, explanation).

    Expected format:
        LABEL: faithful
        EXPLANATION: All claims are supported...
    """
    response = _strip_thinking(response)

    label = None
    explanation = None

    # Parse LABEL
    label_match = re.search(r"LABEL:\s*(\w+)", response, re.IGNORECASE)
    if label_match:
        label = label_match.group(1).lower().strip()

    # Parse EXPLANATION
    expl_match = re.search(r"EXPLANATION:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
    if expl_match:
        explanation = expl_match.group(1).strip()

    # Fallback: try to find label in the response
    if label is None:
        for cat in ["faithful", "partially_faithful", "unsupported",
                     "contradictory", "insufficient_evidence"]:
            if cat in response.lower():
                label = cat
                break

    return label, explanation


# Valid faithfulness labels
VALID_LABELS = {
    "faithful",
    "partially_faithful",
    "unsupported",
    "contradictory",
    "insufficient_evidence",
}


class LLMJudgeEvaluator(Evaluator):
    """LLM-as-judge evaluator using Ollama.

    The judge model should be DIFFERENT from the generator model to
    avoid self-preference (circularity control).

    Configuration:
        judge_model: Ollama model name (default: from JUDGE_MODEL env or "qwen3:4b")
        base_url: Ollama API URL (default: from OLLAMA_BASE_URL env)
        temperature: Judge temperature (default: 0.0 for determinism)
        timeout: Request timeout in seconds
    """

    name = "llm_judge"
    version = "1.0.0"

    def __init__(
        self,
        judge_model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        timeout: int = 120,
    ):
        self.judge_model = judge_model or os.environ.get("JUDGE_MODEL", "qwen3:4b")
        self.base_url = base_url or os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self.temperature = temperature
        self.timeout = timeout
        self.configuration = {
            "judge_model": self.judge_model,
            "temperature": temperature,
            "prompt_template": "judge_v1",
            "circularity_note": "Judge model should differ from generator model",
        }

    def evaluate(self, record: dict[str, Any]) -> EvaluatorOutput:
        """Evaluate a record using an LLM judge.

        Reads: question, retrieved_context, generated_answer.
        Does NOT read: annotations, faithfulness_category, or any human label.
        """
        start = time.time()

        question = record.get("question", "") or ""
        context = record.get("retrieved_context", "") or "(no context provided)"
        answer = record.get("generated_answer", "") or ""

        if not answer.strip():
            return EvaluatorOutput(
                evaluator_name=self.name,
                evaluator_version=self.version,
                score=None,
                label="insufficient_evidence",
                explanation="Empty generated answer — nothing to evaluate.",
                latency_seconds=round(time.time() - start, 3),
                cost_usd=0.0,
                configuration=self.configuration,
                timestamp=_iso_timestamp(),
            )

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            context=context,
            answer=answer,
        )

        payload = json.dumps({
            "model": self.judge_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "seed": 42,
            },
        }).encode("utf-8")

        url = f"{self.base_url}/api/generate"
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except TimeoutError as exc:
            return EvaluatorOutput(
                evaluator_name=self.name,
                evaluator_version=self.version,
                score=None,
                label=None,
                explanation=f"Judge timed out after {self.timeout}s: {exc}",
                latency_seconds=round(time.time() - start, 3),
                cost_usd=0.0,
                configuration=self.configuration,
                timestamp=_iso_timestamp(),
            )
        except urllib.error.URLError as exc:
            return EvaluatorOutput(
                evaluator_name=self.name,
                evaluator_version=self.version,
                score=None,
                label=None,
                explanation=f"Judge connection error: {exc}",
                latency_seconds=round(time.time() - start, 3),
                cost_usd=0.0,
                configuration=self.configuration,
                timestamp=_iso_timestamp(),
            )

        latency = time.time() - start
        raw_response = data.get("response", "")
        label, explanation = _parse_judge_response(raw_response)

        # Validate label
        if label and label not in VALID_LABELS:
            # Try to normalize common variations
            label_map = {
                "partial": "partially_faithful",
                "partially": "partially_faithful",
                "insufficient": "insufficient_evidence",
                "contradict": "contradictory",
                "support": "unsupported",
            }
            label = label_map.get(label, None)

        # Map label to score (ordinal scale for AUROC)
        label_scores = {
            "faithful": 1.0,
            "partially_faithful": 0.75,
            "unsupported": 0.25,
            "contradictory": 0.0,
            "insufficient_evidence": 0.5,
        }
        score = label_scores.get(label, None) if label else None

        return EvaluatorOutput(
            evaluator_name=self.name,
            evaluator_version=self.version,
            score=score,
            label=label,
            explanation=explanation or raw_response[:200],
            latency_seconds=round(latency, 3),
            cost_usd=0.0,  # Local model, no API cost
            configuration=self.configuration,
            timestamp=_iso_timestamp(),
        )
