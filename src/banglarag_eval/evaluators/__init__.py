"""Evaluator framework for RAG faithfulness evaluation.

Automatic evaluators are the SYSTEMS UNDER EVALUATION — they are never
gold labels. Their outputs go to `evaluator_outputs[]`, never to
`annotations[]` or `faithfulness_category`.

Available evaluators (6 total):

No API needed (run immediately):
- LexicalBaselineEvaluator: token set overlap (Bangla-aware)
- AnswerRelevanceEvaluator: question-answer content overlap + question type
- ExactMatchEvaluator: TF-based precision/recall/F1 (strictest baseline)

Ollama-based (local LLM, no paid API):
- LLMJudgeEvaluator: LLM-as-judge with 5-way taxonomy
- RAGASFaithfulnessEvaluator: claim extraction + NLI verification
- NLIEvaluator: single NLI entailment check (faster than RAGAS)

Circularity control: Ollama-based evaluators should use a different
model from the generator (set JUDGE_MODEL env var).
"""

from .base import Evaluator, EvaluatorOutput, attach_evaluator_output
from .lexical_baseline import LexicalBaselineEvaluator
from .llm_judge import LLMJudgeEvaluator
from .ragas_faithfulness import RAGASFaithfulnessEvaluator
from .nli_entailment import NLIEvaluator
from .answer_relevance import AnswerRelevanceEvaluator
from .exact_match import ExactMatchEvaluator

__all__ = [
    "Evaluator",
    "EvaluatorOutput",
    "attach_evaluator_output",
    "LexicalBaselineEvaluator",
    "LLMJudgeEvaluator",
    "RAGASFaithfulnessEvaluator",
    "NLIEvaluator",
    "AnswerRelevanceEvaluator",
    "ExactMatchEvaluator",
]
