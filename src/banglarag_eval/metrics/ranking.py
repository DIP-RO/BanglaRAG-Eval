"""Ranking stability framework — evaluator-induced ranking comparison.

Given multiple evaluators or conditions, compares the rankings each
evaluator induces. Uses Spearman and Kendall rank correlation.

Pilot scale permits only the mechanism to be exercised, not conclusions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats


@dataclass
class RankingCorrelation:
    """Result of a ranking correlation analysis."""

    method: str
    correlation: float
    p_value: float
    n_items: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "correlation": self.correlation,
            "p_value": self.p_value,
            "n_items": self.n_items,
        }


@dataclass
class RankingStabilityResult:
    """Result of a full ranking stability analysis."""

    pairwise_correlations: list[dict[str, Any]] = field(default_factory=list)
    mean_correlation: float = 0.0
    min_correlation: float = 0.0
    max_correlation: float = 0.0
    n_evaluators: int = 0
    n_conditions: int = 0
    underpowered: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "pairwise_correlations": self.pairwise_correlations,
            "mean_correlation": self.mean_correlation,
            "min_correlation": self.min_correlation,
            "max_correlation": self.max_correlation,
            "n_evaluators": self.n_evaluators,
            "n_conditions": self.n_conditions,
            "underpowered": self.underpowered,
        }


def compute_ranking(
    scores: dict[str, float],
) -> dict[str, int]:
    """Compute rankings from scores.

    Args:
        scores: Dict mapping item names to scores.

    Returns:
        Dict mapping item names to ranks (1 = best).
    """
    # Sort by score descending (higher = better)
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ranks = {}
    for rank, (name, _) in enumerate(sorted_items, 1):
        ranks[name] = rank
    return ranks


def rank_correlation(
    ranking1: dict[str, float],
    ranking2: dict[str, float],
    method: str = "spearman",
) -> RankingCorrelation:
    """Compute rank correlation between two rankings.

    Args:
        ranking1: First ranking (item -> score).
        ranking2: Second ranking (item -> score).
        method: "spearman" or "kendall".

    Returns:
        RankingCorrelation with correlation and p-value.
    """
    # Get common items
    common = set(ranking1.keys()) & set(ranking2.keys())
    if len(common) < 3:
        return RankingCorrelation(
            method=method, correlation=0.0, p_value=1.0, n_items=len(common)
        )

    items = sorted(common)
    scores1 = [ranking1[item] for item in items]
    scores2 = [ranking2[item] for item in items]

    if method == "spearman":
        result = stats.spearmanr(scores1, scores2)
    elif method == "kendall":
        result = stats.kendalltau(scores1, scores2)
    else:
        raise ValueError(f"Unknown method: {method}")

    return RankingCorrelation(
        method=method,
        correlation=float(result[0]),
        p_value=float(result[1]),
        n_items=len(common),
    )


def compare_evaluator_rankings(
    evaluator_scores: dict[str, dict[str, float]],
    human_scores: dict[str, float] | None = None,
    method: str = "spearman",
    min_n: int = 30,
) -> RankingStabilityResult:
    """Compare rankings induced by multiple evaluators.

    Args:
        evaluator_scores: Dict mapping evaluator name to
            {condition/system -> score}.
        human_scores: Optional human-induced ranking scores
            {condition/system -> score}.
        method: Correlation method ("spearman" or "kendall").
        min_n: Minimum number of items for adequate power.

    Returns:
        RankingStabilityResult with pairwise correlations.
    """
    all_rankings = dict(evaluator_scores)
    if human_scores is not None:
        all_rankings["human_gold"] = human_scores

    evaluator_names = list(evaluator_scores.keys())
    all_names = list(all_rankings.keys())

    correlations = []
    for i, name1 in enumerate(all_names):
        for j, name2 in enumerate(all_names):
            if i >= j:
                continue
            result = rank_correlation(
                all_rankings[name1], all_rankings[name2], method=method
            )
            correlations.append({
                "evaluator_a": name1,
                "evaluator_b": name2,
                **result.to_dict(),
            })

    corr_values = [c["correlation"] for c in correlations] if correlations else [0.0]

    # Count conditions (items being ranked)
    n_conditions = 0
    if evaluator_scores:
        first_eval = next(iter(evaluator_scores.values()))
        n_conditions = len(first_eval)

    return RankingStabilityResult(
        pairwise_correlations=correlations,
        mean_correlation=float(np.mean(corr_values)),
        min_correlation=float(np.min(corr_values)),
        max_correlation=float(np.max(corr_values)),
        n_evaluators=len(evaluator_names),
        n_conditions=n_conditions,
        underpowered=n_conditions < min_n,
    )
