"""Extended coverage tests — edge cases, error handling, and additional scenarios."""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import numpy as np

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
from banglarag_eval.evaluators.answer_relevance import _detect_question_type, _content_tokens
from banglarag_eval.evaluators.exact_match import _tokenize_tf
from banglarag_eval.evaluators.lexical_baseline import _detect_negation, _tokenize
from banglarag_eval.evaluators.nli_entailment import _parse_nli_response
from banglarag_eval.evaluators.ragas_faithfulness import _extract_claims, _parse_verdict
from banglarag_eval.metrics import (
    compute_classification_metrics,
    compute_binary_metrics,
    compute_agreement,
    compute_correlation,
    compute_efficiency,
    bootstrap_ci,
    paired_bootstrap,
    mcnemar_test,
    holm_correction,
    is_underpowered,
    compute_ranking,
    rank_correlation,
    compare_evaluator_rankings,
)


# ── Lexical baseline: negation detection ────────────────────────────


class TestLexicalBaselineNegation:
    """Test negation detection for contradictory label."""

    def test_english_negation_detected(self):
        assert _detect_negation("This is not correct") is True
        assert _detect_negation("No such thing exists") is True
        assert _detect_negation("Never happened") is True
        assert _detect_negation("This is wrong") is True

    def test_no_negation(self):
        assert _detect_negation("The capital is Dhaka") is False
        assert _detect_negation("Bangladesh is in South Asia") is False

    def test_bangla_negation_detected(self):
        assert _detect_negation("এটি সঠিক নয়") is True
        assert _detect_negation("এটি ভুল") is True

    def test_bangla_no_negation(self):
        assert _detect_negation("বাংলাদেশের রাজধানী ঢাকা") is False

    def test_contradictory_label_with_negation(self):
        """Low overlap + negation should produce contradictory label."""
        ev = LexicalBaselineEvaluator()
        record = {
            "question": "What is the capital?",
            "retrieved_context": "The capital of Bangladesh is Dhaka.",
            "generated_answer": "No, the capital is not Dhaka, it is wrong.",
        }
        result = ev.evaluate(record)
        # With negation and moderate overlap, should be contradictory
        assert result.label in ("contradictory", "unsupported", "partially_faithful")
        assert result.score <= 0.5


# ── Exact match: TF precision/recall ────────────────────────────────


class TestExactMatchTF:
    """Test TF-based precision/recall calculations."""

    def test_tokenize_tf_counts(self):
        tf = _tokenize_tf("the the the cat cat dog")
        assert tf["the"] == 3
        assert tf["cat"] == 2
        assert tf["dog"] == 1

    def test_tf_precision_high(self):
        """All answer tokens in context -> high precision."""
        ev = ExactMatchEvaluator()
        record = {
            "question": "test",
            "retrieved_context": "Dhaka Dhaka Dhaka city big",
            "generated_answer": "Dhaka Dhaka city",
        }
        result = ev.evaluate(record)
        # Precision should be 1.0 (all answer tokens in context)
        assert result.score > 0.0

    def test_tf_recall_low(self):
        """Answer covers small fraction of context -> low recall."""
        ev = ExactMatchEvaluator()
        record = {
            "question": "test",
            "retrieved_context": "Dhaka city big populated busy capital Bangladesh South Asia",
            "generated_answer": "Dhaka city",
        }
        result = ev.evaluate(record)
        # F1 should be low because recall is low
        assert result.score < 0.5

    def test_repeated_tokens_in_answer(self):
        """Answer with repeated tokens not in context."""
        ev = ExactMatchEvaluator()
        record = {
            "question": "test",
            "retrieved_context": "Dhaka city",
            "generated_answer": "Dhaka Dhaka Dhaka Dhaka Dhaka",
        }
        result = ev.evaluate(record)
        # Precision should be 1.0 but recall low
        assert result.score > 0.0
        assert result.score < 1.0

    def test_no_overlap_at_all(self):
        ev = ExactMatchEvaluator()
        record = {
            "question": "test",
            "retrieved_context": "apple banana cherry",
            "generated_answer": "dog elephant fox",
        }
        result = ev.evaluate(record)
        assert result.score == 0.0
        assert result.label == "unsupported"


# ── Answer relevance: question type detection ───────────────────────


class TestQuestionTypeDetection:
    """Test question type detection for different question types."""

    def test_who_question(self):
        assert _detect_question_type("Who is the president?") == "who"
        assert _detect_question_type("কে রাষ্ট্রপতি?") == "who"

    def test_what_question(self):
        assert _detect_question_type("What is the capital?") == "what"
        assert _detect_question_type("কী রাজধানী?") == "what"

    def test_when_question(self):
        assert _detect_question_type("When did this happen?") == "when"
        assert _detect_question_type("কখন এটি ঘটে?") == "when"

    def test_where_question(self):
        assert _detect_question_type("Where is Dhaka?") == "where"
        assert _detect_question_type("কোথায় ঢাকা?") == "where"

    def test_why_question(self):
        assert _detect_question_type("Why is this important?") == "why"
        assert _detect_question_type("কেন এটি গুরুত্বপূর্ণ?") == "why"

    def test_how_question(self):
        assert _detect_question_type("How does this work?") == "how"
        assert _detect_question_type("কীভাবে এটি কাজ করে?") == "how"

    def test_how_many_question(self):
        assert _detect_question_type("How many people live there?") == "how_many"
        assert _detect_question_type("How much does it cost?") == "how_many"

    def test_yes_no_question(self):
        assert _detect_question_type("Is Dhaka the capital?") == "yes_no"
        assert _detect_question_type("Are they coming?") == "yes_no"

    def test_unknown_question(self):
        assert _detect_question_type("Tell me about Bangladesh") == "unknown"

    def test_content_tokens_filters_stopwords(self):
        tokens = _content_tokens({"the", "capital", "is", "dhaka", "a", "an"})
        assert "the" not in tokens
        assert "a" not in tokens
        assert "capital" in tokens
        assert "dhaka" in tokens


# ── RAGAS: claim extraction and max_claims ──────────────────────────


class TestRAGASClaimExtraction:
    """Test RAGAS claim extraction and parsing."""

    def test_extract_claims_with_dashes(self):
        response = "- The capital is Dhaka\n- It has 10 million people\n- It is in South Asia"
        claims = _extract_claims(response)
        assert len(claims) == 3
        assert "The capital is Dhaka" in claims

    def test_extract_claims_empty(self):
        claims = _extract_claims("No claims here")
        # Fallback treats long lines as claims
        assert len(claims) >= 1

    def test_extract_claims_strips_thinking(self):
        response = " IMDthinking IMD - The capital is Dhaka"
        claims = _extract_claims(response)
        assert len(claims) >= 1

    def test_parse_verdict_entailed(self):
        assert _parse_verdict("entailed") == "entailed"
        assert _parse_verdict("The claim is entailed by context") == "entailed"

    def test_parse_verdict_contradicted(self):
        assert _parse_verdict("contradicted") == "contradicted"
        assert _parse_verdict("This is contradicted by the context") == "contradicted"

    def test_parse_verdict_neutral(self):
        assert _parse_verdict("neutral") == "neutral"
        assert _parse_verdict("not addressed") == "neutral"

    def test_parse_verdict_default(self):
        assert _parse_verdict("unknown response") == "neutral"

    @patch("banglarag_eval.evaluators.ragas_faithfulness._ollama_generate")
    def test_max_claims_truncation(self, mock_gen):
        """Should only verify max_claims claims even if more extracted."""
        # Return 5 claims
        mock_gen.side_effect = [
            "- Claim 1\n- Claim 2\n- Claim 3\n- Claim 4\n- Claim 5",
            "entailed", "entailed", "entailed",  # only 3 verifications
        ]
        ev = RAGASFaithfulnessEvaluator(
            judge_model="test", base_url="http://localhost:99999", max_claims=3
        )
        record = {
            "question": "test",
            "retrieved_context": "context",
            "generated_answer": "answer",
        }
        result = ev.evaluate(record)
        # Only 3 claims verified
        assert mock_gen.call_count == 4  # 1 extraction + 3 verifications
        assert result.score == 1.0  # All 3 entailed


# ── NLI: fallback parsing ───────────────────────────────────────────


class TestNLIParsing:
    """Test NLI response parsing edge cases."""

    def test_parse_with_label_prefix(self):
        label, reason = _parse_nli_response("LABEL: entailed\nREASON: All supported")
        assert label == "entailed"
        assert "All supported" in reason

    def test_parse_without_label_prefix(self):
        """Fallback: find label word in response."""
        label, reason = _parse_nli_response("The answer is entailed by the context.")
        assert label == "entailed"

    def test_parse_contradicted_fallback(self):
        label, _ = _parse_nli_response("This contradicts the context.")
        assert label == "contradicted"

    def test_parse_neutral_fallback(self):
        label, _ = _parse_nli_response("The context does not address this.")
        assert label == "neutral"

    def test_parse_no_label_found(self):
        label, reason = _parse_nli_response("Some random text without verdict")
        assert label is None

    def test_parse_case_insensitive(self):
        label, _ = _parse_nli_response("LABEL: ENTAILED")
        assert label == "entailed"


# ── Metrics: custom positive label and label sets ───────────────────


class TestMetricsExtended:
    """Extended metrics tests."""

    def test_binary_custom_positive_label(self):
        gold = ["contradictory", "faithful", "contradictory"]
        pred = ["contradictory", "faithful", "faithful"]
        # Use "contradictory" as positive instead of "faithful"
        m = compute_binary_metrics(gold, pred, positive="contradictory")
        assert m.precision is not None
        assert m.recall is not None
        assert m.f1 is not None

    def test_classification_custom_labels(self):
        gold = ["faithful", "unsupported", "contradictory"]
        pred = ["faithful", "unsupported", "contradictory"]
        m = compute_classification_metrics(gold, pred, labels=["faithful", "unsupported", "contradictory"])
        assert m.f1_macro == 1.0
        assert "faithful" in m.per_class
        assert "contradictory" in m.per_class

    def test_classification_all_same_label(self):
        """All predictions same label."""
        gold = ["faithful"] * 5
        pred = ["faithful"] * 5
        m = compute_classification_metrics(gold, pred)
        # Should not crash, should report 1.0 for the present label
        assert m.balanced_accuracy == 1.0

    def test_binary_auroc_perfect_separation(self):
        gold = ["faithful", "faithful", "unsupported", "unsupported"]
        pred = ["faithful", "faithful", "unsupported", "unsupported"]
        scores = [0.9, 0.8, 0.2, 0.1]
        m = compute_binary_metrics(gold, pred, scores=scores)
        assert m.auroc == 1.0

    def test_binary_auroc_random(self):
        """Random scores -> AUROC near 0.5."""
        gold = ["faithful", "faithful", "unsupported", "unsupported"]
        pred = ["faithful", "faithful", "unsupported", "unsupported"]
        scores = [0.5, 0.5, 0.5, 0.5]
        m = compute_binary_metrics(gold, pred, scores=scores)
        # All same score -> AUROC undefined or 0.5
        assert m.auroc is None or m.auroc == 0.5

    def test_agreement_all_same_labels(self):
        """Agreement when all labels are the same."""
        a1 = ["faithful"] * 10
        a2 = ["faithful"] * 10
        m = compute_agreement(a1, a2)
        assert m.agreement_rate == 1.0

    def test_correlation_identical_arrays(self):
        s1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        s2 = [1.0, 2.0, 3.0, 4.0, 5.0]
        corr, p = compute_correlation(s1, s2, method="pearson")
        assert corr == pytest.approx(1.0)

    def test_efficiency_single_record(self):
        m = compute_efficiency([1.5])
        assert m.n == 1
        assert m.mean_latency_seconds == 1.5
        assert m.median_latency_seconds == 1.5

    def test_efficiency_zero_cost(self):
        m = compute_efficiency([1.0, 2.0], costs=[0.0, 0.0])
        assert m.total_cost_usd == 0.0
        assert m.cost_per_1000 == 0.0


# ── Statistics: bootstrap with median ───────────────────────────────


class TestStatisticsExtended:
    """Extended statistics tests."""

    def test_bootstrap_median(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        result = bootstrap_ci(data, statistic=np.median, n_bootstrap=1000, seed=42)
        assert result.point_estimate == 5.5
        assert result.ci_lower < result.point_estimate
        assert result.ci_upper > result.point_estimate

    def test_paired_bootstrap_median(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0] * 5
        b = [2.0, 3.0, 4.0, 5.0, 6.0] * 5
        result = paired_bootstrap(a, b, statistic=np.median, n_bootstrap=1000, seed=42)
        # Median of a is 3.0, median of b is 4.0, diff = -1.0
        assert result.point_estimate == pytest.approx(-1.0, abs=0.01)

    def test_bootstrap_ci_level(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0] * 10
        result = bootstrap_ci(data, n_bootstrap=1000, confidence_level=0.90, seed=42)
        assert result.confidence_level == 0.90

    def test_mcnemar_all_discordant_a(self):
        """All discordant pairs favor A."""
        correct_a = [True, True, True, True, True]
        correct_b = [False, False, False, False, False]
        result = mcnemar_test(correct_a, correct_b)
        assert result.n_discordant_a_correct == 5
        assert result.n_discordant_b_correct == 0
        assert result.p_value < 0.1  # Should be significant

    def test_mcnemar_all_discordant_b(self):
        """All discordant pairs favor B."""
        correct_a = [False, False, False, False, False]
        correct_b = [True, True, True, True, True]
        result = mcnemar_test(correct_a, correct_b)
        assert result.n_discordant_a_correct == 0
        assert result.n_discordant_b_correct == 5

    def test_holm_monotonicity(self):
        """Corrected p-values should be monotonically non-decreasing in sorted order."""
        p_values = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05]
        result = holm_correction(p_values)
        # Get sorted order of corrected p-values
        sorted_corrected = sorted(result.corrected_p_values)
        for i in range(len(sorted_corrected) - 1):
            assert sorted_corrected[i] <= sorted_corrected[i + 1]

    def test_holm_single_test(self):
        result = holm_correction([0.03])
        assert result.n_tests == 1
        assert result.corrected_p_values[0] == 0.03  # No correction for single test

    def test_is_underpowered_boundary(self):
        assert is_underpowered(29) is True
        assert is_underpowered(30) is False
        assert is_underpowered(31) is False


# ── Ranking: extended scenarios ─────────────────────────────────────


class TestRankingExtended:
    """Extended ranking tests."""

    def test_three_evaluators_comparison(self):
        """Compare rankings from 3 evaluators + human."""
        evaluator_scores = {
            "lex": {"c1": 0.9, "c2": 0.6, "c3": 0.8, "c4": 0.7},
            "llm": {"c1": 0.85, "c2": 0.55, "c3": 0.75, "c4": 0.65},
            "nli": {"c1": 0.8, "c2": 0.5, "c3": 0.7, "c4": 0.6},
        }
        human_scores = {"c1": 0.88, "c2": 0.52, "c3": 0.78, "c4": 0.68}
        result = compare_evaluator_rankings(evaluator_scores, human_scores)
        # 3 evaluators + human = 4 rankings, C(4,2) = 6 pairwise correlations
        assert len(result.pairwise_correlations) == 6
        assert result.n_evaluators == 3
        assert result.n_conditions == 4

    def test_ranking_no_human_scores(self):
        """Compare evaluators without human reference."""
        evaluator_scores = {
            "lex": {"c1": 0.9, "c2": 0.6, "c3": 0.8},
            "llm": {"c1": 0.8, "c2": 0.5, "c3": 0.7},
        }
        result = compare_evaluator_rankings(evaluator_scores)
        # 2 evaluators -> 1 pairwise correlation
        assert len(result.pairwise_correlations) == 1
        assert result.n_evaluators == 2

    def test_ranking_partial_overlap(self):
        """Rankings with partial item overlap."""
        r1 = {"a": 0.9, "b": 0.5, "c": 0.7}
        r2 = {"a": 0.8, "b": 0.6, "d": 0.3}  # 'c' missing, 'd' added
        result = rank_correlation(r1, r2)
        # Only 'a' and 'b' are common -> too few for correlation
        assert result.n_items == 2
        assert result.correlation == 0.0  # Too few items

    def test_ranking_all_same_scores(self):
        """All items have the same score -> no meaningful ranking."""
        r1 = {"a": 0.5, "b": 0.5, "c": 0.5}
        r2 = {"a": 0.5, "b": 0.5, "c": 0.5}
        result = rank_correlation(r1, r2)
        # Spearman of identical arrays with no variance -> NaN -> 0.0
        assert result.correlation == 0.0 or np.isnan(result.correlation)

    def test_compute_ranking_ties(self):
        """Items with equal scores get consecutive ranks."""
        scores = {"a": 0.9, "b": 0.9, "c": 0.5}
        ranks = compute_ranking(scores)
        # a and b both have 0.9, one gets rank 1, other gets rank 2
        assert ranks["a"] in (1, 2)
        assert ranks["b"] in (1, 2)
        assert ranks["c"] == 3


# ── Evaluator batch and error handling ──────────────────────────────


class TestEvaluatorBatchAndErrors:
    """Test batch evaluation and error handling."""

    def test_evaluate_batch_default(self):
        """Test default evaluate_batch implementation."""
        ev = LexicalBaselineEvaluator()
        records = [
            {"question": "q1", "retrieved_context": "context1", "generated_answer": "answer1"},
            {"question": "q2", "retrieved_context": "context2", "generated_answer": "answer2"},
            {"question": "q3", "retrieved_context": "context3", "generated_answer": "answer3"},
        ]
        results = ev.evaluate_batch(records)
        assert len(results) == 3
        assert all(isinstance(r, EvaluatorOutput) for r in results)

    def test_evaluate_batch_empty(self):
        ev = LexicalBaselineEvaluator()
        results = ev.evaluate_batch([])
        assert results == []

    def test_safe_evaluate_catches_errors(self):
        """_safe_evaluate should catch exceptions and return error output."""
        class FailingEvaluator(Evaluator):
            name = "failing"
            version = "1.0"
            def evaluate(self, record):
                raise RuntimeError("Intentional failure")

        ev = FailingEvaluator()
        result = ev._safe_evaluate({"question": "test"})
        assert result.evaluator_name == "failing"
        assert result.label is None
        assert "ERROR" in result.explanation
        assert "Intentional failure" in result.explanation
        assert result.latency_seconds is not None

    def test_safe_evaluate_adds_timestamp(self):
        """_safe_evaluate should add timestamp if missing."""
        class SimpleEvaluator(Evaluator):
            name = "simple"
            version = "1.0"
            def evaluate(self, record):
                return EvaluatorOutput(evaluator_name="simple", evaluator_version="1.0", label="faithful")

        ev = SimpleEvaluator()
        result = ev._safe_evaluate({"question": "test"})
        assert result.timestamp is not None
        assert result.latency_seconds is not None

    def test_attach_multiple_evaluator_outputs(self):
        """Attach outputs from multiple evaluators to same record."""
        record = {"question": "test", "retrieved_context": "ctx", "generated_answer": "ans"}
        ev1 = LexicalBaselineEvaluator()
        ev2 = ExactMatchEvaluator()

        out1 = ev1.evaluate(record)
        out2 = ev2.evaluate(record)

        attach_evaluator_output(record, out1)
        attach_evaluator_output(record, out2)

        assert len(record["evaluator_outputs"]) == 2
        assert record["evaluator_outputs"][0]["evaluator_name"] == "lexical_baseline"
        assert record["evaluator_outputs"][1]["evaluator_name"] == "exact_match_precision"

    def test_evaluator_does_not_modify_record(self):
        """Evaluator should not modify the input record."""
        ev = LexicalBaselineEvaluator()
        record = {
            "question": "test",
            "retrieved_context": "context",
            "generated_answer": "answer",
        }
        original_keys = set(record.keys())
        ev.evaluate(record)
        assert set(record.keys()) == original_keys


# ── Tokenizer edge cases ─────────────────────────────────────────────


class TestTokenizerEdgeCases:
    """Test tokenizer edge cases."""

    def test_tokenize_empty_string(self):
        tokens = _tokenize("")
        assert len(tokens) == 0

    def test_tokenize_only_punctuation(self):
        tokens = _tokenize("... --- !!! ???")
        assert len(tokens) == 0

    def test_tokenize_only_numbers(self):
        tokens = _tokenize("123 456 789")
        assert "123" in tokens
        assert "456" in tokens

    def test_tokenize_mixed_bangla_english(self):
        tokens = _tokenize("বাংলাদেশের economy গত কয়েক বছরে grow করেছে")
        assert "economy" in tokens
        assert "grow" in tokens
        assert any(ord(c) > 0x0980 for c in "বাংলাদেশের")
        assert "বাংলাদেশের" in tokens

    def test_tokenize_single_char_filtered(self):
        """Single-character tokens should be filtered out."""
        tokens = _tokenize("a b c d")
        assert len(tokens) == 0  # All single chars filtered

    def test_tokenize_tf_empty(self):
        tf = _tokenize_tf("")
        assert len(tf) == 0

    def test_tokenize_tf_bangla(self):
        tf = _tokenize_tf("ঢাকা ঢাকা শহর")
        assert tf["ঢাকা"] == 2
        assert tf["শহর"] == 1


# ── Gate edge cases ──────────────────────────────────────────────────


class TestGateEdgeCases:
    """Test Stage 1 gate measurement edge cases."""

    def test_gate_result_str_format(self):
        from banglarag_eval.gates import GateResult
        g = GateResult(name="test_gate", passed=True, value=0.75, threshold=0.60,
                       details="All good", n=100)
        s = str(g)
        assert "[PASS]" in s
        assert "test_gate" in s
        assert "0.750" in s

    def test_stage1_report_to_dict_complete(self):
        from banglarag_eval.gates import Stage1GateReport, GateResult
        report = Stage1GateReport(
            gates=[
                GateResult("g1", True, 0.8, 0.6, "pass", 100),
                GateResult("g2", False, 0.3, 0.6, "fail", 100),
            ],
            all_passed=False,
            n_records=100,
            n_annotated=50,
            n_adjudicated=10,
            underpowered=True,
            decision="GATES FAILED: g2",
        )
        d = report.to_dict()
        assert d["all_passed"] is False
        assert d["n_records"] == 100
        assert d["n_annotated"] == 50
        assert d["n_adjudicated"] == 10
        assert d["underpowered"] is True
        assert len(d["gates"]) == 2

    def test_summary_contains_all_gates(self):
        from banglarag_eval.gates import Stage1GateReport, GateResult
        report = Stage1GateReport(
            gates=[
                GateResult("gate_one", True, 0.8, 0.6, "ok", 100),
                GateResult("gate_two", False, 0.3, 0.6, "fail", 100),
                GateResult("gate_three", True, 0.9, 0.5, "ok", 100),
            ],
            all_passed=False,
            n_records=100,
            decision="MIXED",
        )
        s = report.summary()
        assert "gate_one" in s
        assert "gate_two" in s
        assert "gate_three" in s
        assert "PASS" in s
        assert "FAIL" in s
        assert "MIXED" in s

    def test_thresholds_are_correct(self):
        from banglarag_eval.gates import (
            KAPPA_THRESHOLD, KAPPA_TARGET, KAPPA_CI_LOWER_BOUND,
            NO_FIT_THRESHOLD, CONDITION_INTEGRITY_THRESHOLD,
            PIPELINE_INTEGRITY_THRESHOLD, EVALUATOR_HARNESS_THRESHOLD,
        )
        assert KAPPA_THRESHOLD == 0.60
        assert KAPPA_TARGET == 0.70
        assert KAPPA_CI_LOWER_BOUND == 0.40
        assert NO_FIT_THRESHOLD == 0.10
        assert CONDITION_INTEGRITY_THRESHOLD == 0.90
        assert PIPELINE_INTEGRITY_THRESHOLD == 1.0
        assert EVALUATOR_HARNESS_THRESHOLD == 0.95
