"""Metrics, statistical analysis, and ranking stability for RAG faithfulness evaluation.

Modules:
- metrics: Evaluator-vs-human metrics, human agreement, correlation, efficiency
- statistics: Bootstrap CI, paired bootstrap, McNemar's test, Holm correction
- ranking: Evaluator-induced ranking comparison (Spearman/Kendall)
"""

from .metrics import (
    ClassificationMetrics,
    BinaryMetrics,
    AgreementMetrics,
    EfficiencyMetrics,
    compute_classification_metrics,
    compute_binary_metrics,
    compute_agreement,
    compute_correlation,
    compute_efficiency,
    extract_evaluator_labels,
    extract_evaluator_scores,
    extract_gold_labels,
    extract_annotator_labels,
    ALL_LABELS,
    PRIMARY_POSITIVE,
)
from .statistics import (
    BootstrapResult,
    McNemarResult,
    HolmCorrectionResult,
    bootstrap_ci,
    paired_bootstrap,
    mcnemar_test,
    holm_correction,
    is_underpowered,
)
from .ranking import (
    RankingCorrelation,
    RankingStabilityResult,
    compute_ranking,
    rank_correlation,
    compare_evaluator_rankings,
)

__all__ = [
    # Metrics
    "ClassificationMetrics",
    "BinaryMetrics",
    "AgreementMetrics",
    "EfficiencyMetrics",
    "compute_classification_metrics",
    "compute_binary_metrics",
    "compute_agreement",
    "compute_correlation",
    "compute_efficiency",
    "extract_evaluator_labels",
    "extract_evaluator_scores",
    "extract_gold_labels",
    "extract_annotator_labels",
    "ALL_LABELS",
    "PRIMARY_POSITIVE",
    # Statistics
    "BootstrapResult",
    "McNemarResult",
    "HolmCorrectionResult",
    "bootstrap_ci",
    "paired_bootstrap",
    "mcnemar_test",
    "holm_correction",
    "is_underpowered",
    # Ranking
    "RankingCorrelation",
    "RankingStabilityResult",
    "compute_ranking",
    "rank_correlation",
    "compare_evaluator_rankings",
]
