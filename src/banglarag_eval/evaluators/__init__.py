"""Evaluator framework for RAG faithfulness evaluation.

Automatic evaluators are the SYSTEMS UNDER EVALUATION — they are never
gold labels. Their outputs go to `evaluator_outputs[]`, never to
`annotations[]` or `faithfulness_category`.

Available evaluators:
- LexicalBaselineEvaluator: simple token overlap (no API needed)
- LLMJudgeEvaluator: LLM-as-judge via Ollama (cross-model)
"""

from .base import Evaluator, EvaluatorOutput, attach_evaluator_output
from .lexical_baseline import LexicalBaselineEvaluator
from .llm_judge import LLMJudgeEvaluator

__all__ = [
    "Evaluator",
    "EvaluatorOutput",
    "attach_evaluator_output",
    "LexicalBaselineEvaluator",
    "LLMJudgeEvaluator",
]
