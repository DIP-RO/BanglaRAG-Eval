"""RAGAS-style faithfulness evaluator — local implementation using Ollama.

Implements the RAGAS faithfulness metric without requiring the RAGAS
library or paid APIs. Uses a two-step approach:

1. Claim extraction: Extract atomic claims from the generated answer
   using an LLM (Ollama).
2. Claim verification: For each claim, check if it is supported by the
   retrieved context using NLI (entailment).

Faithfulness score = (number of supported claims) / (total claims)

This is a simplified local implementation of the RAGAS faithfulness
metric. The original RAGAS uses OpenAI; this version uses Ollama for
fully local evaluation.

The judge model should be DIFFERENT from the generator model to avoid
self-preference (circularity control).
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

# Step 1: Claim extraction prompt
CLAIM_EXTRACTION_PROMPT = """/no_think
Extract all atomic claims from the following answer. Each claim should be a simple, verifiable statement.

List each claim on a new line starting with "- ".

Answer: {answer}

Claims:"""

# Step 2: Claim verification prompt
CLAIM_VERIFICATION_PROMPT = """/no_think
Determine if the following claim is supported by the context.

Context: {context}

Claim: {claim}

Respond with exactly one word:
- "entailed" if the claim is supported by the context
- "contradicted" if the claim conflicts with the context
- "neutral" if the context does not address the claim

Verdict:"""


def _strip_thinking(text: str) -> str:
    """Remove Qwen3 thinking trace from output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def _ollama_generate(
    model: str,
    prompt: str,
    base_url: str,
    timeout: int = 60,
    temperature: float = 0.0,
) -> str | None:
    """Call Ollama generate API and return the response text."""
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "seed": 42},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "")
    except (TimeoutError, urllib.error.URLError, OSError):
        return None


def _extract_claims(response: str) -> list[str]:
    """Parse claim extraction response into a list of claims."""
    response = _strip_thinking(response)
    claims = []
    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("- "):
            claim = line[2:].strip()
            if claim:
                claims.append(claim)
        elif line and not line.startswith("Claims:") and len(line) > 5:
            # Fallback: treat non-empty lines as claims
            claims.append(line)
    return claims


def _parse_verdict(response: str) -> str:
    """Parse NLI verdict from response."""
    response = _strip_thinking(response).lower().strip()
    if "entail" in response:
        return "entailed"
    elif "contradict" in response:
        return "contradicted"
    elif "neutral" in response or "not" in response:
        return "neutral"
    return "neutral"


class RAGASFaithfulnessEvaluator(Evaluator):
    """RAGAS-style faithfulness evaluator using local Ollama.

    Implements a simplified version of the RAGAS faithfulness metric:
    1. Extract atomic claims from the answer
    2. Verify each claim against the context (entailment/contradiction/neutral)
    3. Score = supported claims / total claims

    Uses Ollama (local LLM) — no paid API required.
    Judge model should differ from generator model (circularity control).

    Configuration:
        judge_model: Ollama model (default: JUDGE_MODEL env or "qwen3:8b")
        base_url: Ollama URL (default: OLLAMA_BASE_URL env)
        timeout: Per-request timeout
        max_claims: Maximum claims to verify (for latency control)
    """

    name = "ragas_faithfulness"
    version = "1.0.0"

    def __init__(
        self,
        judge_model: str | None = None,
        base_url: str | None = None,
        timeout: int = 60,
        max_claims: int = 10,
    ):
        self.judge_model = judge_model or os.environ.get("JUDGE_MODEL", "qwen3:8b")
        self.base_url = base_url or os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self.timeout = timeout
        self.max_claims = max_claims
        self.configuration = {
            "judge_model": self.judge_model,
            "max_claims": max_claims,
            "method": "claim_extraction + NLI_verification",
            "circularity_note": "Judge model should differ from generator model",
        }

    def evaluate(self, record: dict[str, Any]) -> EvaluatorOutput:
        """Evaluate faithfulness using claim extraction + verification.

        Reads: question, retrieved_context, generated_answer.
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
                explanation="Empty answer — nothing to evaluate.",
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
                explanation="Empty context — cannot verify any claims.",
                latency_seconds=round(time.time() - start, 3),
                cost_usd=0.0,
                configuration=self.configuration,
                timestamp=_iso_timestamp(),
            )

        # Step 1: Extract claims
        extract_prompt = CLAIM_EXTRACTION_PROMPT.format(answer=answer)
        extract_response = _ollama_generate(
            self.judge_model, extract_prompt, self.base_url, self.timeout
        )

        if extract_response is None:
            return EvaluatorOutput(
                evaluator_name=self.name,
                evaluator_version=self.version,
                score=None,
                label=None,
                explanation="Claim extraction failed (Ollama unavailable or timeout).",
                latency_seconds=round(time.time() - start, 3),
                cost_usd=0.0,
                configuration=self.configuration,
                timestamp=_iso_timestamp(),
            )

        claims = _extract_claims(extract_response)

        if not claims:
            return EvaluatorOutput(
                evaluator_name=self.name,
                evaluator_version=self.version,
                score=None,
                label=None,
                explanation="No claims extracted from answer.",
                latency_seconds=round(time.time() - start, 3),
                cost_usd=0.0,
                configuration=self.configuration,
                timestamp=_iso_timestamp(),
            )

        # Limit claims for latency
        claims = claims[: self.max_claims]

        # Step 2: Verify each claim
        n_entailed = 0
        n_contradicted = 0
        n_neutral = 0
        verdicts = []

        for claim in claims:
            verify_prompt = CLAIM_VERIFICATION_PROMPT.format(
                context=context, claim=claim
            )
            verify_response = _ollama_generate(
                self.judge_model, verify_prompt, self.base_url, self.timeout
            )

            if verify_response is None:
                verdicts.append("neutral")
                n_neutral += 1
                continue

            verdict = _parse_verdict(verify_response)
            verdicts.append(verdict)

            if verdict == "entailed":
                n_entailed += 1
            elif verdict == "contradicted":
                n_contradicted += 1
            else:
                n_neutral += 1

        # Compute faithfulness score
        total = len(claims)
        score = n_entailed / total if total > 0 else 0.0

        # Map to label
        if n_contradicted > 0 and n_contradicted >= n_entailed:
            label = "contradictory"
        elif score >= 0.8:
            label = "faithful"
        elif score >= 0.5:
            label = "partially_faithful"
        elif score >= 0.2:
            label = "unsupported"
        else:
            label = "unsupported"

        explanation = (
            f"Claims: {total} (entailed: {n_entailed}, "
            f"contradicted: {n_contradicted}, neutral: {n_neutral}). "
            f"Faithfulness = {n_entailed}/{total} = {score:.2f}."
        )

        return EvaluatorOutput(
            evaluator_name=self.name,
            evaluator_version=self.version,
            score=round(score, 4),
            label=label,
            explanation=explanation,
            latency_seconds=round(time.time() - start, 3),
            cost_usd=0.0,
            configuration=self.configuration,
            timestamp=_iso_timestamp(),
        )
