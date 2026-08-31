"""Statistical analysis module — bootstrap, McNemar, Holm correction.

Implements:
- Paired bootstrap (evaluator A vs B on same examples)
- McNemar's test (paired binary correctness)
- Bootstrap confidence intervals for any point estimate
- Holm correction for multiple comparisons

Each test is applied only where its assumptions hold. The rationale for
each test choice should be recorded in the run config. p-values are
never manufactured — only reported with underlying measured data.

Pilot-scale results carry "underpowered — diagnostic only" caveats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy import stats


@dataclass
class BootstrapResult:
    """Result of a bootstrap analysis."""

    point_estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_bootstrap: int
    n_samples: int
    bootstrap_estimates: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_estimate": self.point_estimate,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "confidence_level": self.confidence_level,
            "n_bootstrap": self.n_bootstrap,
            "n_samples": self.n_samples,
        }


@dataclass
class McNemarResult:
    """Result of McNemar's test."""

    statistic: float
    p_value: float
    n_discordant_a_correct: int  # A correct, B wrong
    n_discordant_b_correct: int  # B wrong, A correct
    n_both_correct: int
    n_both_wrong: int
    n_total: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "statistic": self.statistic,
            "p_value": self.p_value,
            "n_discordant_a_correct": self.n_discordant_a_correct,
            "n_discordant_b_correct": self.n_discordant_b_correct,
            "n_both_correct": self.n_both_correct,
            "n_both_wrong": self.n_both_wrong,
            "n_total": self.n_total,
        }


@dataclass
class HolmCorrectionResult:
    """Result of Holm correction for multiple comparisons."""

    original_p_values: list[float]
    corrected_p_values: list[float]
    rejected: list[bool]
    alpha: float
    n_tests: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_p_values": self.original_p_values,
            "corrected_p_values": self.corrected_p_values,
            "rejected": self.rejected,
            "alpha": self.alpha,
            "n_tests": self.n_tests,
        }


def bootstrap_ci(
    data: list[float],
    statistic: Callable[[list[float]], float] = np.mean,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Compute bootstrap confidence interval for any point estimate.

    Args:
        data: The sample data.
        statistic: Function computing the point estimate (default: mean).
        n_bootstrap: Number of bootstrap resamples.
        confidence_level: CI confidence level (default: 0.95).
        seed: Random seed for reproducibility.

    Returns:
        BootstrapResult with point estimate and CI.
    """
    rng = np.random.default_rng(seed)
    data_arr = np.array(data)
    n = len(data_arr)

    if n == 0:
        return BootstrapResult(
            point_estimate=0.0, ci_lower=0.0, ci_upper=0.0,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap, n_samples=0,
        )

    point_estimate = float(statistic(data_arr.tolist()))

    estimates = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data_arr, size=n, replace=True)
        estimates.append(float(statistic(sample.tolist())))

    alpha = 1 - confidence_level
    ci_lower = float(np.percentile(estimates, 100 * alpha / 2))
    ci_upper = float(np.percentile(estimates, 100 * (1 - alpha / 2)))

    return BootstrapResult(
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        n_samples=n,
        bootstrap_estimates=estimates,
    )


def paired_bootstrap(
    metric_a: list[float],
    metric_b: list[float],
    statistic: Callable[[list[float]], float] = np.mean,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Paired bootstrap for comparing evaluator A vs B.

    Resamples pairs (a_i, b_i) with replacement and computes the
    difference in statistic. Respects pairing.

    Args:
        metric_a: Per-example metric values for evaluator A.
        metric_b: Per-example metric values for evaluator B.
        statistic: Function to compute on each resample (default: mean).
        n_bootstrap: Number of bootstrap resamples.
        confidence_level: CI confidence level.
        seed: Random seed.

    Returns:
        BootstrapResult for the difference (A - B).
    """
    if len(metric_a) != len(metric_b):
        raise ValueError(
            f"Length mismatch: {len(metric_a)} vs {len(metric_b)}"
        )

    rng = np.random.default_rng(seed)
    a_arr = np.array(metric_a)
    b_arr = np.array(metric_b)
    n = len(a_arr)

    if n == 0:
        return BootstrapResult(
            point_estimate=0.0, ci_lower=0.0, ci_upper=0.0,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap, n_samples=0,
        )

    point_estimate = float(statistic(a_arr.tolist()) - statistic(b_arr.tolist()))

    estimates = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        sample_a = a_arr[indices]
        sample_b = b_arr[indices]
        diff = float(statistic(sample_a.tolist()) - statistic(sample_b.tolist()))
        estimates.append(diff)

    alpha = 1 - confidence_level
    ci_lower = float(np.percentile(estimates, 100 * alpha / 2))
    ci_upper = float(np.percentile(estimates, 100 * (1 - alpha / 2)))

    return BootstrapResult(
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        n_samples=n,
        bootstrap_estimates=estimates,
    )


def mcnemar_test(
    correct_a: list[bool],
    correct_b: list[bool],
) -> McNemarResult:
    """McNemar's test for paired binary correctness.

    Tests whether evaluators A and B differ in correctness on the same
    examples. Exactly the paired disagreement-count situation.

    Args:
        correct_a: Per-example correctness for evaluator A.
        correct_b: Per-example correctness for evaluator B.

    Returns:
        McNemarResult with statistic, p-value, and discordant counts.
    """
    if len(correct_a) != len(correct_b):
        raise ValueError(
            f"Length mismatch: {len(correct_a)} vs {len(correct_b)}"
        )

    n = len(correct_a)
    both_correct = sum(1 for a, b in zip(correct_a, correct_b) if a and b)
    both_wrong = sum(1 for a, b in zip(correct_a, correct_b) if not a and not b)
    a_correct_b_wrong = sum(1 for a, b in zip(correct_a, correct_b) if a and not b)
    b_correct_a_wrong = sum(1 for a, b in zip(correct_a, correct_b) if not a and b)

    # Discordant pairs
    n01 = b_correct_a_wrong  # B correct, A wrong
    n10 = a_correct_b_wrong  # A correct, B wrong

    # Use exact binomial test for small samples, chi-square for large
    if n01 + n10 == 0:
        statistic = 0.0
        p_value = 1.0
    elif n01 + n10 < 25:
        # Exact binomial test
        # Under null, discordant pairs are equally likely either way
        result = stats.binomtest(min(n01, n10), n01 + n10, p=0.5)
        statistic = float(abs(n01 - n10))
        p_value = float(result.pvalue)
    else:
        # Chi-square approximation with continuity correction
        statistic = float((abs(n01 - n10) - 1) ** 2 / (n01 + n10))
        p_value = float(1 - stats.chi2.cdf(statistic, df=1))

    return McNemarResult(
        statistic=statistic,
        p_value=p_value,
        n_discordant_a_correct=n10,
        n_discordant_b_correct=n01,
        n_both_correct=both_correct,
        n_both_wrong=both_wrong,
        n_total=n,
    )


def holm_correction(
    p_values: list[float],
    alpha: float = 0.05,
) -> HolmCorrectionResult:
    """Holm step-down correction for multiple comparisons.

    Controls family-wise error rate without independence assumptions.

    Args:
        p_values: List of original p-values.
        alpha: Family-wise significance level.

    Returns:
        HolmCorrectionResult with corrected p-values and rejection decisions.
    """
    n = len(p_values)
    if n == 0:
        return HolmCorrectionResult(
            original_p_values=[], corrected_p_values=[],
            rejected=[], alpha=alpha, n_tests=0,
        )

    # Sort p-values with original indices
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    corrected = [0.0] * n
    rejected = [False] * n

    prev_corrected = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        # Holm correction: p_adj = p * (n - rank)
        adj_p = min(1.0, p * (n - rank))
        # Enforce monotonicity
        adj_p = max(adj_p, prev_corrected)
        corrected[orig_idx] = adj_p
        prev_corrected = adj_p

        # Rejection: reject if adjusted p <= alpha
        rejected[orig_idx] = adj_p <= alpha

    return HolmCorrectionResult(
        original_p_values=list(p_values),
        corrected_p_values=corrected,
        rejected=rejected,
        alpha=alpha,
        n_tests=n,
    )


def is_underpowered(n: int, min_n: int = 30) -> bool:
    """Check if sample size is too small for reliable conclusions.

    Pilot-scale results with n < min_n should carry
    "underpowered — diagnostic only" caveats.

    Args:
        n: Sample size.
        min_n: Minimum sample size for adequate power.

    Returns:
        True if underpowered.
    """
    return n < min_n
