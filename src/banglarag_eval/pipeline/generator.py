"""LLM generation adapter for Ollama (Qwen3:8b).

Uses the Ollama HTTP API (http://localhost:11434) to generate answers
from retrieved context. No paid API key required — Qwen3 runs locally.

Qwen3 produces a "thinking" section before the answer. This adapter
strips the thinking trace and returns only the final answer.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any


@dataclass
class GenerationResult:
    """Result of a single generation call."""

    answer: str
    model: str
    temperature: float
    latency_seconds: float
    eval_count: int  # number of tokens generated
    raw_response: str  # full response including thinking trace


# Default RAG prompt template — works for all language conditions.
# The /no_think flag disables Qwen3's thinking mode for faster generation.
RAG_PROMPT_TEMPLATE = """/no_think
Answer the question based only on the provided context. If the context does not contain enough information to answer, say so explicitly. Do not make up information.

Context:
{context}

Question: {question}

Answer:"""


def _strip_thinking(text: str) -> str:
    """Remove Qwen3's thinking trace from the response.

    Qwen3 outputs a <think>...</think> block before the actual answer.
    We strip it so only the final answer is stored.
    """
    # Remove <think>...</think> blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Also handle the bare thinking format (no tags, just "Thinking..." prefix)
    # Qwen3 sometimes outputs thinking without explicit tags
    lines = text.strip().split('\n')
    # Find the first non-empty line that isn't part of thinking
    cleaned_lines: list[str] = []
    in_thinking = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('<think>'):
            in_thinking = True
            continue
        if '</think>' in stripped:
            in_thinking = False
            continue
        if in_thinking:
            continue
        cleaned_lines.append(line)

    result = '\n'.join(cleaned_lines).strip()
    if not result:
        # If everything was thinking, return the original stripped
        result = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return result


def generate(
    question: str,
    context: str,
    model: str | None = None,
    temperature: float = 0.0,
    base_url: str | None = None,
    timeout: int = 120,
    prompt_template: str = RAG_PROMPT_TEMPLATE,
) -> GenerationResult:
    """Generate an answer using Ollama.

    Args:
        question: The question to answer.
        context: The retrieved evidence/context.
        model: Ollama model name (default: from OLLAMA_MODEL env or "qwen3:8b").
        temperature: Generation temperature (default: 0 for determinism).
        base_url: Ollama API URL (default: from OLLAMA_BASE_URL env or localhost).
        timeout: Request timeout in seconds.
        prompt_template: Prompt template with {context} and {question} placeholders.

    Returns:
        GenerationResult with the answer and metadata.

    Raises:
        ConnectionError: If Ollama is not running.
        RuntimeError: If generation fails.
    """
    model = model or os.environ.get("OLLAMA_MODEL", "qwen3:8b")
    base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    prompt = prompt_template.format(context=context, question=question)

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "seed": 42,  # deterministic for reproducibility
        },
    }).encode("utf-8")

    url = f"{base_url}/api/generate"
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start_time = time.time()

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"Cannot connect to Ollama at {base_url}. "
            f"Is it running? Start with: ollama serve. Error: {exc}"
        ) from exc

    latency = time.time() - start_time
    raw_response = data.get("response", "")
    answer = _strip_thinking(raw_response)
    # Strip leading "Answer:" prefix that the model may echo from the prompt
    answer = re.sub(r'^\s*Answer:\s*', '', answer).strip()

    return GenerationResult(
        answer=answer,
        model=model,
        temperature=temperature,
        latency_seconds=round(latency, 3),
        eval_count=data.get("eval_count", 0),
        raw_response=raw_response,
    )


def check_ollama_available(base_url: str | None = None) -> bool:
    """Check if Ollama is running and accessible.

    Args:
        base_url: Ollama API URL.

    Returns:
        True if Ollama is running, False otherwise.
    """
    base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except (urllib.error.URLError, ConnectionError):
        return False
