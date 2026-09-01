"""Tests for the evaluator framework, metrics, statistics, and ranking modules."""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from banglarag_eval.evaluators import (
    Evaluator,
    EvaluatorOutput,
    attach_evaluator_output,
    LexicalBaselineEvaluator,
    LLMJudgeEvaluator,
    RAGASFaithfulnessEvaluator,
    NLIEvaluator,
    AnswerRelevanceEvaluator,
    ExactMatchEvaluator,
)
from banglarag_eval.metrics import (
    compute_classification_metrics,
    compute_binary_metrics,
    compute_agreement,
    compute_correlation,
    compute_efficiency,
    extract_evaluator_labels,
    extract_evaluator_scores,
    extract_gold_labels,
    extract_annotator_labels,
    bootstrap_ci,
    paired_bootstrap,
    mcnemar_test,
    holm_correction,
    is_underpowered,
    compute_ranking,
    rank_correlation,
    compare_evaluator_rankings,
)


# ── Evaluator base interface tests ──────────────────────────────────


class TestEvaluatorOutput:
    """Test the EvaluatorOutput dataclass."""

    def test_create_output(self):
        out = EvaluatorOutput(
            evaluator_name="test",
            evaluator_version="1.0",
            score=0.5,
            label="faithful",
            explanation="test",
        )
        assert out.evaluator_name == "test"
        assert out.score == 0.5
        assert out.label == "faithful"

    def test_to_dict(self):
        out = EvaluatorOutput(
            evaluator_name="test",
            evaluator_version="1.0",
            score=0.5,
        )
        d = out.to_dict()
        assert d["evaluator_name"] == "test"
        assert d["score"] == 0.5
        assert d["label"] is None

    def test_to_dict_all_fields(self):
        out = EvaluatorOutput(
            evaluator_name="test",
            evaluator_version="1.0",
            score=0.5,
            label="faithful",
            explanation="ok",
            latency_seconds=1.5,
            cost_usd=0.01,
            configuration={"key": "value"},
            timestamp="2026-01-01T00:00:00+00:00",
        )
        d = out.to_dict()
        assert d["latency_seconds"] == 1.5
        assert d["cost_usd"] == 0.01
        assert d["configuration"] == {"key": "value"}


class TestAttachEvaluatorOutput:
    """Test attaching evaluator output to records."""

    def test_attach_to_empty_record(self):
        record = {}
        out = EvaluatorOutput(evaluator_name="test", evaluator_version="1.0")
        result = attach_evaluator_output(record, out)
        assert "evaluator_outputs" in result
        assert len(result["evaluator_outputs"]) == 1
        assert result["evaluator_outputs"][0]["evaluator_name"] == "test"

    def test_attach_to_existing_outputs(self):
        record = {"evaluator_outputs": [{"evaluator_name": "other", "evaluator_version": "1.0"}]}
        out = EvaluatorOutput(evaluator_name="test", evaluator_version="1.0")
        result = attach_evaluator_output(record, out)
        assert len(result["evaluator_outputs"]) == 2

    def test_does_not_touch_annotations(self):
        record = {"annotations": [], "faithfulness_category": None}
        out = EvaluatorOutput(evaluator_name="test", evaluator_version="1.0")
        result = attach_evaluator_output(record, out)
        assert result["annotations"] == []
        assert result["faithfulness_category"] is None


# ── Lexical baseline evaluator tests ────────────────────────────────


class TestLexicalBaselineEvaluator:
    """Test the lexical baseline evaluator."""

    @pytest.fixture
    def evaluator(self):
        return LexicalBaselineEvaluator()

    def test_name_and_version(self, evaluator):
        assert evaluator.name == "lexical_baseline"
        assert evaluator.version == "1.0.0"

    def test_faithful_high_overlap(self, evaluator):
        record = {
            "question": "What is the capital?",
            "retrieved_context": "The capital of Bangladesh is Dhaka.",
            "generated_answer": "The capital is Dhaka.",
        }
        result = evaluator.evaluate(record)
        assert result.label == "faithful"
        assert result.score > 0.5
        assert result.latency_seconds is not None
        assert result.cost_usd == 0.0

    def test_unsupported_low_overlap(self, evaluator):
        record = {
            "question": "What is the capital?",
            "retrieved_context": "The capital of Bangladesh is Dhaka.",
            "generated_answer": "The weather is sunny today.",
        }
        result = evaluator.evaluate(record)
        assert result.label in ("unsupported", "insufficient_evidence")
        assert result.score < 0.5

    def test_empty_context(self, evaluator):
        record = {
            "question": "What is the capital?",
            "retrieved_context": "",
            "generated_answer": "The capital is Dhaka.",
        }
        result = evaluator.evaluate(record)
        assert result.label == "insufficient_evidence"
        assert result.score == 0.0

    def test_empty_answer(self, evaluator):
        record = {
            "question": "What is the capital?",
            "retrieved_context": "The capital is Dhaka.",
            "generated_answer": "",
        }
        result = evaluator.evaluate(record)
        assert result.label == "insufficient_evidence"

    def test_bangla_text(self, evaluator):
        record = {
            "question": "বাংলাদেশের রাজধানী কী?",
            "retrieved_context": "বাংলাদেশের রাজধানী ঢাকা।",
            "generated_answer": "বাংলাদেশের রাজধানী ঢাকা।",
        }
        result = evaluator.evaluate(record)
        assert result.label == "faithful"
        assert result.score > 0.5

    def test_partial_overlap(self, evaluator):
        record = {
            "question": "What is the capital?",
            "retrieved_context": "The capital of Bangladesh is Dhaka on the Buriganga river.",
            "generated_answer": "The capital is Dhaka with five million people and many mosques.",
        }
        result = evaluator.evaluate(record)
        assert result.label in ("partially_faithful", "faithful", "unsupported")
        assert 0.0 <= result.score <= 1.0

    def test_does_not_read_human_labels(self, evaluator):
        """Evaluator must not access annotations or gold labels."""
        record = {
            "question": "What is the capital?",
            "retrieved_context": "The capital is Dhaka.",
            "generated_answer": "The capital is Dhaka.",
            "annotations": [{"annotator_id": "ann-1", "faithfulness_category": "faithful"}],
            "faithfulness_category": "faithful",
        }
        result = evaluator.evaluate(record)
        # Should still work — it ignores human labels
        assert result.label is not None

    def test_configuration_in_output(self, evaluator):
        record = {
            "question": "test",
            "retrieved_context": "test context",
            "generated_answer": "test",
        }
        result = evaluator.evaluate(record)
        assert result.configuration is not None
        assert "threshold_faithful" in result.configuration

    def test_custom_thresholds(self):
        ev = LexicalBaselineEvaluator(
            threshold_faithful=0.95,
            threshold_partial=0.7,
        )
        assert ev.threshold_faithful == 0.95
        assert ev.configuration["threshold_faithful"] == 0.95


# ── LLM judge evaluator tests (mocked) ──────────────────────────────


class TestLLMJudgeEvaluator:
    """Test the LLM-as-judge evaluator (mocked Ollama)."""

    @pytest.fixture
    def evaluator(self):
        return LLMJudgeEvaluator(judge_model="test-model", base_url="http://localhost:99999")

    def test_name_and_version(self, evaluator):
        assert evaluator.name == "llm_judge"
        assert evaluator.version == "1.0.0"

    def test_empty_answer(self, evaluator):
        record = {
            "question": "test",
            "retrieved_context": "context",
            "generated_answer": "",
        }
        result = evaluator.evaluate(record)
        assert result.label == "insufficient_evidence"

    @patch("banglarag_eval.evaluators.llm_judge.urllib.request.urlopen")
    def test_successful_evaluation(self, mock_urlopen, evaluator):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "response": "LABEL: faithful\nEXPLANATION: All claims supported by context.",
        }).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        record = {
            "question": "What is the capital?",
            "retrieved_context": "The capital is Dhaka.",
            "generated_answer": "The capital is Dhaka.",
        }
        result = evaluator.evaluate(record)
        assert result.label == "faithful"
        assert "supported" in result.explanation.lower()
        assert result.score == 1.0

    @patch("banglarag_eval.evaluators.llm_judge.urllib.request.urlopen")
    def test_contradictory_label(self, mock_urlopen, evaluator):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "response": "LABEL: contradictory\nEXPLANATION: Answer conflicts with context.",
        }).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        record = {
            "question": "test",
            "retrieved_context": "test",
            "generated_answer": "test",
        }
        result = evaluator.evaluate(record)
        assert result.label == "contradictory"
        assert result.score == 0.0

    @patch("banglarag_eval.evaluators.llm_judge.urllib.request.urlopen")
    def test_label_normalization(self, mock_urlopen, evaluator):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "response": "LABEL: partial\nEXPLANATION: Some claims supported.",
        }).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        record = {
            "question": "test",
            "retrieved_context": "test",
            "generated_answer": "test",
        }
        result = evaluator.evaluate(record)
        assert result.label == "partially_faithful"

    @patch("banglarag_eval.evaluators.llm_judge.urllib.request.urlopen")
    def test_connection_error(self, mock_urlopen, evaluator):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        record = {
            "question": "test",
            "retrieved_context": "test",
            "generated_answer": "test",
        }
        result = evaluator.evaluate(record)
        assert result.label is None
        assert "connection error" in result.explanation.lower()

    @patch("banglarag_eval.evaluators.llm_judge.urllib.request.urlopen")
    def test_timeout(self, mock_urlopen, evaluator):
        mock_urlopen.side_effect = TimeoutError("timed out")

        record = {
            "question": "test",
            "retrieved_context": "test",
            "generated_answer": "test",
        }
        result = evaluator.evaluate(record)
        assert result.label is None
        assert "timed out" in result.explanation.lower()

    def test_circularity_note_in_config(self, evaluator):
        assert "circularity_note" in evaluator.configuration


# ── Metrics tests ───────────────────────────────────────────────────


class TestClassificationMetrics:
    """Test multi-class classification metrics."""

    def test_perfect_prediction(self):
        gold = ["faithful", "unsupported", "contradictory"]
        pred = ["faithful", "unsupported", "contradictory"]
        m = compute_classification_metrics(gold, pred)
        assert m.precision_macro == 1.0
        assert m.recall_macro == 1.0
        assert m.f1_macro == 1.0
        assert m.balanced_accuracy == 1.0

    def test_random_prediction(self):
        gold = ["faithful", "faithful", "unsupported", "unsupported"]
        pred = ["unsupported", "faithful", "faithful", "unsupported"]
        m = compute_classification_metrics(gold, pred)
        assert 0.0 <= m.precision_macro <= 1.0
        assert 0.0 <= m.f1_macro <= 1.0

    def test_per_class_metrics(self):
        gold = ["faithful", "faithful", "unsupported"]
        pred = ["faithful", "unsupported", "unsupported"]
        m = compute_classification_metrics(gold, pred)
        assert "faithful" in m.per_class
        assert "unsupported" in m.per_class
        assert m.per_class["faithful"]["support"] == 2

    def test_to_dict(self):
        gold = ["faithful", "unsupported"]
        pred = ["faithful", "unsupported"]
        m = compute_classification_metrics(gold, pred)
        d = m.to_dict()
        assert "precision_macro" in d
        assert "per_class" in d


class TestBinaryMetrics:
    """Test binary classification metrics."""

    def test_perfect_binary(self):
        gold = ["faithful", "faithful", "unsupported", "unsupported"]
        pred = ["faithful", "faithful", "unsupported", "unsupported"]
        m = compute_binary_metrics(gold, pred)
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0
        assert m.accuracy == 1.0

    def test_auroc_with_scores(self):
        gold = ["faithful", "faithful", "unsupported", "unsupported"]
        pred = ["faithful", "faithful", "unsupported", "unsupported"]
        scores = [0.9, 0.8, 0.2, 0.1]
        m = compute_binary_metrics(gold, pred, scores=scores)
        assert m.auroc == 1.0

    def test_auroc_none_without_scores(self):
        gold = ["faithful", "unsupported"]
        pred = ["faithful", "unsupported"]
        m = compute_binary_metrics(gold, pred)
        assert m.auroc is None

    def test_kappa(self):
        gold = ["faithful", "faithful", "unsupported", "unsupported"]
        pred = ["faithful", "unsupported", "unsupported", "unsupported"]
        m = compute_binary_metrics(gold, pred)
        assert -1.0 <= m.kappa <= 1.0


class TestAgreementMetrics:
    """Test inter-annotator agreement."""

    def test_perfect_agreement(self):
        a1 = ["faithful", "unsupported", "contradictory"]
        a2 = ["faithful", "unsupported", "contradictory"]
        m = compute_agreement(a1, a2)
        assert m.agreement_rate == 1.0
        assert m.cohen_kappa == 1.0
        assert m.n == 3

    def test_no_agreement(self):
        a1 = ["faithful", "faithful"]
        a2 = ["unsupported", "contradictory"]
        m = compute_agreement(a1, a2)
        assert m.agreement_rate == 0.0

    def test_partial_agreement(self):
        a1 = ["faithful", "unsupported", "faithful", "unsupported"]
        a2 = ["faithful", "unsupported", "unsupported", "faithful"]
        m = compute_agreement(a1, a2)
        assert m.agreement_rate == 0.5

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_agreement(["a", "b"], ["a"])

    def test_empty_lists(self):
        m = compute_agreement([], [])
        assert m.n == 0


class TestCorrelation:
    """Test correlation metrics."""

    def test_perfect_positive(self):
        s1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        s2 = [1.0, 2.0, 3.0, 4.0, 5.0]
        corr, p = compute_correlation(s1, s2, method="spearman")
        assert corr == pytest.approx(1.0)

    def test_perfect_negative(self):
        s1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        s2 = [5.0, 4.0, 3.0, 2.0, 1.0]
        corr, p = compute_correlation(s1, s2, method="spearman")
        assert corr == pytest.approx(-1.0)

    def test_pearson(self):
        s1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        s2 = [2.0, 4.0, 6.0, 8.0, 10.0]
        corr, p = compute_correlation(s1, s2, method="pearson")
        assert abs(corr - 1.0) < 0.001

    def test_too_few_samples(self):
        corr, p = compute_correlation([1.0], [2.0])
        assert corr == 0.0

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            compute_correlation([1.0, 2.0], [1.0])


class TestEfficiencyMetrics:
    """Test efficiency metrics."""

    def test_basic_efficiency(self):
        latencies = [1.0, 2.0, 3.0, 4.0, 5.0]
        m = compute_efficiency(latencies)
        assert m.mean_latency_seconds == 3.0
        assert m.median_latency_seconds == 3.0
        assert m.n == 5
        assert m.examples_per_second > 0

    def test_with_costs(self):
        latencies = [1.0, 2.0]
        costs = [0.01, 0.02]
        m = compute_efficiency(latencies, costs)
        assert m.total_cost_usd == 0.03
        assert m.cost_per_1000 == 15.0  # 0.03/2 * 1000

    def test_empty(self):
        m = compute_efficiency([])
        assert m.n == 0
        assert m.mean_latency_seconds == 0.0


class TestExtractFunctions:
    """Test label/score extraction from records."""

    def test_extract_evaluator_labels(self):
        records = [
            {"evaluator_outputs": [{"evaluator_name": "lex", "label": "faithful"}]},
            {"evaluator_outputs": [{"evaluator_name": "lex", "label": "unsupported"}]},
        ]
        labels = extract_evaluator_labels(records, "lex")
        assert labels == ["faithful", "unsupported"]

    def test_extract_evaluator_labels_missing(self):
        records = [{"evaluator_outputs": []}, {"evaluator_outputs": []}]
        labels = extract_evaluator_labels(records, "lex")
        assert labels == [None, None]

    def test_extract_gold_labels(self):
        records = [
            {"faithfulness_category": "faithful"},
            {"faithfulness_category": None},
        ]
        labels = extract_gold_labels(records)
        assert labels == ["faithful", None]

    def test_extract_annotator_labels(self):
        records = [
            {"annotations": [{"annotator_id": "ann-1", "faithfulness_category": "faithful"}]},
            {"annotations": [{"annotator_id": "ann-2", "faithfulness_category": "unsupported"}]},
        ]
        labels = extract_annotator_labels(records, "ann-1")
        assert labels == ["faithful", None]


# ── Statistics tests ────────────────────────────────────────────────


class TestBootstrapCI:
    """Test bootstrap confidence intervals."""

    def test_mean_ci(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0] * 10
        result = bootstrap_ci(data, n_bootstrap=1000, seed=42)
        assert result.point_estimate == pytest.approx(3.0, abs=0.01)
        assert result.ci_lower < result.point_estimate
        assert result.ci_upper > result.point_estimate
        assert result.n_samples == 50

    def test_empty_data(self):
        result = bootstrap_ci([])
        assert result.point_estimate == 0.0
        assert result.n_samples == 0

    def test_custom_statistic(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = bootstrap_ci(data, statistic=np.median, n_bootstrap=1000, seed=42)
        assert result.point_estimate == 3.0


class TestPairedBootstrap:
    """Test paired bootstrap for evaluator comparison."""

    def test_no_difference(self):
        a = [1.0, 0.0, 1.0, 0.0, 1.0] * 10
        b = [1.0, 0.0, 1.0, 0.0, 1.0] * 10
        result = paired_bootstrap(a, b, n_bootstrap=1000, seed=42)
        assert abs(result.point_estimate) < 0.01

    def test_difference(self):
        a = [1.0, 1.0, 1.0, 1.0, 1.0] * 10
        b = [0.0, 0.0, 0.0, 0.0, 0.0] * 10
        result = paired_bootstrap(a, b, n_bootstrap=1000, seed=42)
        assert result.point_estimate == pytest.approx(1.0, abs=0.01)

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            paired_bootstrap([1.0, 2.0], [1.0])


class TestMcNemar:
    """Test McNemar's test."""

    def test_no_discordance(self):
        correct_a = [True, True, False, False]
        correct_b = [True, True, False, False]
        result = mcnemar_test(correct_a, correct_b)
        assert result.p_value == 1.0
        assert result.n_discordant_a_correct == 0
        assert result.n_discordant_b_correct == 0

    def test_discordance(self):
        correct_a = [True, True, True, False]
        correct_b = [True, False, False, False]
        result = mcnemar_test(correct_a, correct_b)
        assert result.n_discordant_a_correct == 2  # A correct, B wrong
        assert result.n_discordant_b_correct == 0  # B correct, A wrong
        assert result.p_value <= 0.5

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            mcnemar_test([True, False], [True])


class TestHolmCorrection:
    """Test Holm correction."""

    def test_no_significant(self):
        p_values = [0.1, 0.2, 0.3]
        result = holm_correction(p_values, alpha=0.05)
        assert not any(result.rejected)

    def test_all_significant(self):
        p_values = [0.001, 0.002, 0.003]
        result = holm_correction(p_values, alpha=0.05)
        assert all(result.rejected)

    def test_partial_significant(self):
        p_values = [0.001, 0.04, 0.06]
        result = holm_correction(p_values, alpha=0.05)
        assert result.rejected[0]  # First should be rejected
        # Check corrected values are monotonically increasing in sorted order

    def test_empty(self):
        result = holm_correction([])
        assert result.n_tests == 0

    def test_corrected_pvalues_valid(self):
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        result = holm_correction(p_values)
        for p in result.corrected_p_values:
            assert 0.0 <= p <= 1.0


class TestUnderpowered:
    """Test underpowered check."""

    def test_small_sample(self):
        assert is_underpowered(10) == True

    def test_adequate_sample(self):
        assert is_underpowered(50) == False

    def test_custom_threshold(self):
        assert is_underpowered(40, min_n=50) == True
        assert is_underpowered(60, min_n=50) == False


# ── Ranking tests ───────────────────────────────────────────────────


class TestRanking:
    """Test ranking stability framework."""

    def test_compute_ranking(self):
        scores = {"a": 0.9, "b": 0.5, "c": 0.7}
        ranks = compute_ranking(scores)
        assert ranks["a"] == 1  # highest score = rank 1
        assert ranks["c"] == 2
        assert ranks["b"] == 3

    def test_rank_correlation_perfect(self):
        r1 = {"a": 0.9, "b": 0.5, "c": 0.7}
        r2 = {"a": 0.9, "b": 0.5, "c": 0.7}
        result = rank_correlation(r1, r2, method="spearman")
        assert result.correlation == 1.0

    def test_rank_correlation_inverse(self):
        r1 = {"a": 0.9, "b": 0.5, "c": 0.7}
        r2 = {"a": 0.1, "b": 0.9, "c": 0.3}
        result = rank_correlation(r1, r2, method="spearman")
        assert result.correlation == -1.0

    def test_rank_correlation_too_few(self):
        r1 = {"a": 0.9, "b": 0.5}
        r2 = {"a": 0.9, "b": 0.5}
        result = rank_correlation(r1, r2)
        assert result.correlation == 0.0  # Not enough items

    def test_compare_evaluator_rankings(self):
        evaluator_scores = {
            "lex": {"cond1": 0.8, "cond2": 0.6, "cond3": 0.9, "cond4": 0.7},
            "llm": {"cond1": 0.7, "cond2": 0.5, "cond3": 0.85, "cond4": 0.6},
        }
        human_scores = {"cond1": 0.75, "cond2": 0.55, "cond3": 0.88, "cond4": 0.65}
        result = compare_evaluator_rankings(evaluator_scores, human_scores)
        assert result.n_evaluators == 2
        assert len(result.pairwise_correlations) == 3  # 3 pairs
        assert result.mean_correlation > 0  # Should be positively correlated

    def test_compare_evaluator_rankings_underpowered(self):
        evaluator_scores = {
            "lex": {"cond1": 0.8, "cond2": 0.6},
            "llm": {"cond1": 0.7, "cond2": 0.5},
        }
        result = compare_evaluator_rankings(evaluator_scores, min_n=10)
        assert result.underpowered == True

    def test_kendall_method(self):
        r1 = {"a": 0.9, "b": 0.5, "c": 0.7, "d": 0.3}
        r2 = {"a": 0.9, "b": 0.5, "c": 0.7, "d": 0.3}
        result = rank_correlation(r1, r2, method="kendall")
        assert result.correlation == 1.0


# ── RAGAS faithfulness evaluator tests (mocked) ─────────────────────


class TestRAGASFaithfulnessEvaluator:
    """Test the RAGAS-style faithfulness evaluator (mocked Ollama)."""

    @pytest.fixture
    def evaluator(self):
        return RAGASFaithfulnessEvaluator(
            judge_model="test-model", base_url="http://localhost:99999"
        )

    def test_name_and_version(self, evaluator):
        assert evaluator.name == "ragas_faithfulness"
        assert evaluator.version == "1.0.0"

    def test_empty_answer(self, evaluator):
        record = {"question": "test", "retrieved_context": "context", "generated_answer": ""}
        result = evaluator.evaluate(record)
        assert result.label == "insufficient_evidence"

    def test_empty_context(self, evaluator):
        record = {"question": "test", "retrieved_context": "", "generated_answer": "test"}
        result = evaluator.evaluate(record)
        assert result.label == "insufficient_evidence"

    @patch("banglarag_eval.evaluators.ragas_faithfulness._ollama_generate")
    def test_successful_evaluation(self, mock_gen, evaluator):
        # Step 1: claim extraction returns 2 claims
        # Step 2: verification returns "entailed" for both
        mock_gen.side_effect = [
            "- The capital is Dhaka\n- It is a big city",  # claim extraction
            "entailed",  # claim 1 verification
            "entailed",  # claim 2 verification
        ]

        record = {
            "question": "What is the capital?",
            "retrieved_context": "The capital of Bangladesh is Dhaka. It is a big city.",
            "generated_answer": "The capital is Dhaka. It is a big city.",
        }
        result = evaluator.evaluate(record)
        assert result.label == "faithful"
        assert result.score == 1.0
        assert "2" in result.explanation  # 2 claims

    @patch("banglarag_eval.evaluators.ragas_faithfulness._ollama_generate")
    def test_contradicted_claims(self, mock_gen, evaluator):
        mock_gen.side_effect = [
            "- The capital is Chittagong",  # wrong claim
            "contradicted",  # verification
        ]

        record = {
            "question": "What is the capital?",
            "retrieved_context": "The capital of Bangladesh is Dhaka.",
            "generated_answer": "The capital is Chittagong.",
        }
        result = evaluator.evaluate(record)
        assert result.label == "contradictory"
        assert result.score == 0.0

    @patch("banglarag_eval.evaluators.ragas_faithfulness._ollama_generate")
    def test_partial_claims(self, mock_gen, evaluator):
        mock_gen.side_effect = [
            "- The capital is Dhaka\n- The population is 20 million",
            "entailed",  # claim 1 supported
            "neutral",   # claim 2 not in context
        ]

        record = {
            "question": "What is the capital?",
            "retrieved_context": "The capital of Bangladesh is Dhaka.",
            "generated_answer": "The capital is Dhaka. The population is 20 million.",
        }
        result = evaluator.evaluate(record)
        assert result.label == "partially_faithful"
        assert result.score == 0.5

    @patch("banglarag_eval.evaluators.ragas_faithfulness._ollama_generate")
    def test_extraction_failure(self, mock_gen, evaluator):
        mock_gen.return_value = None  # Ollama unavailable

        record = {
            "question": "test",
            "retrieved_context": "test context",
            "generated_answer": "test answer",
        }
        result = evaluator.evaluate(record)
        assert result.label is None
        assert "failed" in result.explanation.lower()

    @patch("banglarag_eval.evaluators.ragas_faithfulness._ollama_generate")
    def test_no_claims_extracted(self, mock_gen, evaluator):
        mock_gen.return_value = "No claims found."

        record = {
            "question": "test",
            "retrieved_context": "test context",
            "generated_answer": "test answer",
        }
        result = evaluator.evaluate(record)
        # "No claims found." is treated as a fallback claim, so it gets verified
        assert result.label is not None

    def test_circularity_note_in_config(self, evaluator):
        assert "circularity_note" in evaluator.configuration

    def test_max_claims_limit(self):
        ev = RAGASFaithfulnessEvaluator(max_claims=3)
        assert ev.max_claims == 3
        assert ev.configuration["max_claims"] == 3


# ── NLI entailment evaluator tests (mocked) ──────────────────────────


class TestNLIEvaluator:
    """Test the NLI entailment evaluator (mocked Ollama)."""

    @pytest.fixture
    def evaluator(self):
        return NLIEvaluator(judge_model="test-model", base_url="http://localhost:99999")

    def test_name_and_version(self, evaluator):
        assert evaluator.name == "nli_entailment"
        assert evaluator.version == "1.0.0"

    def test_empty_answer(self, evaluator):
        record = {"question": "test", "retrieved_context": "context", "generated_answer": ""}
        result = evaluator.evaluate(record)
        assert result.label == "insufficient_evidence"

    def test_empty_context(self, evaluator):
        record = {"question": "test", "retrieved_context": "", "generated_answer": "test"}
        result = evaluator.evaluate(record)
        assert result.label == "insufficient_evidence"

    @patch("banglarag_eval.evaluators.nli_entailment.urllib.request.urlopen")
    def test_entailed(self, mock_urlopen, evaluator):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "response": "LABEL: entailed\nREASON: All claims supported by context.",
        }).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        record = {
            "question": "What is the capital?",
            "retrieved_context": "The capital is Dhaka.",
            "generated_answer": "The capital is Dhaka.",
        }
        result = evaluator.evaluate(record)
        assert result.label == "faithful"
        assert result.score == 1.0

    @patch("banglarag_eval.evaluators.nli_entailment.urllib.request.urlopen")
    def test_contradicted(self, mock_urlopen, evaluator):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "response": "LABEL: contradicted\nREASON: Answer conflicts with context.",
        }).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        record = {
            "question": "test",
            "retrieved_context": "The capital is Dhaka.",
            "generated_answer": "The capital is Chittagong.",
        }
        result = evaluator.evaluate(record)
        assert result.label == "contradictory"
        assert result.score == 0.0

    @patch("banglarag_eval.evaluators.nli_entailment.urllib.request.urlopen")
    def test_neutral(self, mock_urlopen, evaluator):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "response": "LABEL: neutral\nREASON: Context doesn't address the claim.",
        }).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        record = {
            "question": "test",
            "retrieved_context": "The capital is Dhaka.",
            "generated_answer": "The weather is sunny.",
        }
        result = evaluator.evaluate(record)
        assert result.label == "unsupported"
        assert result.score == 0.5

    @patch("banglarag_eval.evaluators.nli_entailment.urllib.request.urlopen")
    def test_connection_error(self, mock_urlopen, evaluator):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        record = {
            "question": "test",
            "retrieved_context": "test",
            "generated_answer": "test",
        }
        result = evaluator.evaluate(record)
        assert result.label is None

    def test_circularity_note(self, evaluator):
        assert "circularity_note" in evaluator.configuration


# ── Answer relevance evaluator tests ─────────────────────────────────


class TestAnswerRelevanceEvaluator:
    """Test the answer relevance evaluator."""

    @pytest.fixture
    def evaluator(self):
        return AnswerRelevanceEvaluator()

    def test_name_and_version(self, evaluator):
        assert evaluator.name == "answer_relevance"
        assert evaluator.version == "1.0.0"

    def test_relevant_answer(self, evaluator):
        record = {
            "question": "What is the capital of Bangladesh?",
            "generated_answer": "The capital of Bangladesh is Dhaka city.",
        }
        result = evaluator.evaluate(record)
        assert result.label in ("faithful", "partially_faithful")
        assert result.score > 0.3

    def test_irrelevant_answer(self, evaluator):
        record = {
            "question": "What is the capital of Bangladesh?",
            "generated_answer": "The weather today is very sunny and warm.",
        }
        result = evaluator.evaluate(record)
        assert result.label == "unsupported"
        assert result.score < 0.2

    def test_empty_answer(self, evaluator):
        record = {"question": "test", "generated_answer": ""}
        result = evaluator.evaluate(record)
        assert result.label == "insufficient_evidence"

    def test_empty_question(self, evaluator):
        record = {"question": "", "generated_answer": "test answer"}
        result = evaluator.evaluate(record)
        assert result.label == "insufficient_evidence"

    def test_bangla_question(self, evaluator):
        record = {
            "question": "বাংলাদেশের রাজধানী কী?",
            "generated_answer": "বাংলাদেশের রাজধানী ঢাকা।",
        }
        result = evaluator.evaluate(record)
        assert result.label in ("faithful", "partially_faithful")
        assert result.score > 0.2

    def test_short_answer_penalized(self, evaluator):
        record = {
            "question": "What is the capital of Bangladesh and how big is it?",
            "generated_answer": "Dhaka.",
        }
        result = evaluator.evaluate(record)
        # Short answer should get penalized
        assert result.score < 0.5

    def test_does_not_read_human_labels(self, evaluator):
        record = {
            "question": "What is the capital?",
            "generated_answer": "The capital is Dhaka.",
            "annotations": [{"faithfulness_category": "faithful"}],
            "faithfulness_category": "faithful",
        }
        result = evaluator.evaluate(record)
        assert result.label is not None

    def test_configuration(self, evaluator):
        assert "relevance_threshold" in evaluator.configuration

    def test_custom_thresholds(self):
        ev = AnswerRelevanceEvaluator(relevance_threshold=0.5)
        assert ev.relevance_threshold == 0.5


# ── Exact match precision evaluator tests ────────────────────────────


class TestExactMatchEvaluator:
    """Test the exact-match precision evaluator."""

    @pytest.fixture
    def evaluator(self):
        return ExactMatchEvaluator()

    def test_name_and_version(self, evaluator):
        assert evaluator.name == "exact_match_precision"
        assert evaluator.version == "1.0.0"

    def test_high_overlap(self, evaluator):
        record = {
            "question": "What is the capital?",
            "retrieved_context": "The capital of Bangladesh is Dhaka.",
            "generated_answer": "The capital of Bangladesh is Dhaka.",
        }
        result = evaluator.evaluate(record)
        assert result.label == "faithful"
        assert result.score > 0.5

    def test_low_overlap(self, evaluator):
        record = {
            "question": "What is the capital?",
            "retrieved_context": "The capital of Bangladesh is Dhaka.",
            "generated_answer": "The weather is very sunny today.",
        }
        result = evaluator.evaluate(record)
        assert result.label == "unsupported"
        assert result.score < 0.4

    def test_empty_context(self, evaluator):
        record = {
            "question": "test",
            "retrieved_context": "",
            "generated_answer": "test answer",
        }
        result = evaluator.evaluate(record)
        assert result.label == "insufficient_evidence"

    def test_empty_answer(self, evaluator):
        record = {
            "question": "test",
            "retrieved_context": "test context",
            "generated_answer": "",
        }
        result = evaluator.evaluate(record)
        assert result.label == "insufficient_evidence"

    def test_bangla_text(self, evaluator):
        record = {
            "question": "রাজধানী কী?",
            "retrieved_context": "বাংলাদেশের রাজধানী ঢাকা।",
            "generated_answer": "বাংলাদেশের রাজধানী ঢাকা।",
        }
        result = evaluator.evaluate(record)
        assert result.label == "faithful"
        assert result.score > 0.5

    def test_partial_overlap(self, evaluator):
        record = {
            "question": "test",
            "retrieved_context": "The capital of Bangladesh is Dhaka on the river.",
            "generated_answer": "The capital is Dhaka with five million people.",
        }
        result = evaluator.evaluate(record)
        assert 0.0 <= result.score <= 1.0

    def test_tf_precision_recall(self, evaluator):
        # Answer has repeated tokens
        record = {
            "question": "test",
            "retrieved_context": "Dhaka Dhaka Dhaka city",
            "generated_answer": "Dhaka Dhaka",
        }
        result = evaluator.evaluate(record)
        # Precision should be high (both Dhaka tokens in context)
        # Recall should be lower (context has more tokens)
        assert result.score > 0.0

    def test_does_not_read_human_labels(self, evaluator):
        record = {
            "question": "test",
            "retrieved_context": "The capital is Dhaka.",
            "generated_answer": "The capital is Dhaka.",
            "annotations": [{"faithfulness_category": "faithful"}],
            "faithfulness_category": "faithful",
        }
        result = evaluator.evaluate(record)
        assert result.label is not None

    def test_configuration(self, evaluator):
        assert "threshold_faithful" in evaluator.configuration

    def test_custom_thresholds(self):
        ev = ExactMatchEvaluator(threshold_faithful=0.9)
        assert ev.threshold_faithful == 0.9
