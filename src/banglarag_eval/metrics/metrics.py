"""Metrics module — evaluator-vs-human and human agreement metrics.

Implements:
- Evaluator vs. human: precision, recall, F1 (macro and per-class),
  balanced accuracy, AUROC, Cohen's kappa
- Human agreement: Cohen's kappa between annotators, raw agreement rate
- Correlation: Spearman, Pearson
- Efficiency: cost per 1k examples, latency per example, examples/second

Pre-registered primary binarization: `faithful` vs. all other labels.
Alternative binarizations are labeled exploratory only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    balanced_accuracy_score,
    roc_auc_score,
    cohen_kappa_score,
)
from scipy import stats


# All valid faithfulness labels
ALL_LABELS = [
    "faithful",
    "partially_faithful",
    "unsupported",
    "contradictory",
    "insufficient_evidence",
]

# Pre-registered primary binarization
PRIMARY_POSITIVE = "faithful"


@dataclass
class ClassificationMetrics:
    """Container for classification metrics."""

    precision_macro: float
    recall_macro: float
    f1_macro: float
    balanced_accuracy: float
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "balanced_accuracy": self.balanced_accuracy,
            "per_class": self.per_class,
        }


@dataclass
class BinaryMetrics:
    """Container for binary classification metrics."""

    precision: float
    recall: float
    f1: float
    accuracy: float
    auroc: float | None = None  # None if evaluator doesn't emit scores
    kappa: float = 0.0
    support: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "auroc": self.auroc,
            "kappa": self.kappa,
            "support": self.support,
        }


@dataclass
class AgreementMetrics:
    """Container for inter-annotator agreement metrics."""

    cohen_kappa: float
    agreement_rate: float
    n: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "cohen_kappa": self.cohen_kappa,
            "agreement_rate": self.agreement_rate,
            "n": self.n,
        }


@dataclass
class EfficiencyMetrics:
    """Container for efficiency metrics."""

    mean_latency_seconds: float
    median_latency_seconds: float
    total_cost_usd: float
    cost_per_1000: float
    examples_per_second: float
    n: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_latency_seconds": self.mean_latency_seconds,
            "median_latency_seconds": self.median_latency_seconds,
            "total_cost_usd": self.total_cost_usd,
            "cost_per_1000": self.cost_per_1000,
            "examples_per_second": self.examples_per_second,
            "n": self.n,
        }


def _binarize(labels: list[str], positive: str = PRIMARY_POSITIVE) -> list[int]:
    """Binarize labels: positive=1, all others=0."""
    return [1 if l == positive else 0 for l in labels]


def compute_classification_metrics(
    gold: list[str],
    pred: list[str],
    labels: list[str] | None = None,
) -> ClassificationMetrics:
    """Compute multi-class classification metrics.

    Args:
        gold: Gold (human) labels.
        pred: Evaluator predicted labels.
        labels: Label set (defaults to ALL_LABELS).

    Returns:
        ClassificationMetrics with macro P/R/F1, balanced accuracy, per-class.
    """
    if labels is None:
        labels = ALL_LABELS

    # Filter to labels present in data
    present = sorted(set(gold) | set(pred))
    labels_filtered = [l for l in labels if l in present] or present

    p, r, f1, support = precision_recall_fscore_support(
        gold, pred, labels=labels_filtered, zero_division=0
    )

    bal_acc = balanced_accuracy_score(gold, pred)

    per_class = {}
    for i, label in enumerate(labels_filtered):
        per_class[label] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    return ClassificationMetrics(
        precision_macro=float(np.mean(p)),
        recall_macro=float(np.mean(r)),
        f1_macro=float(np.mean(f1)),
        balanced_accuracy=float(bal_acc),
        per_class=per_class,
    )


def compute_binary_metrics(
    gold: list[str],
    pred: list[str],
    scores: list[float] | None = None,
    positive: str = PRIMARY_POSITIVE,
) -> BinaryMetrics:
    """Compute binary classification metrics.

    Pre-registered primary binarization: `faithful` vs. all other labels.

    Args:
        gold: Gold (human) labels.
        pred: Evaluator predicted labels.
        scores: Optional evaluator scores for AUROC.
        positive: The positive class label.

    Returns:
        BinaryMetrics with P/R/F1, accuracy, AUROC, kappa.
    """
    gold_bin = _binarize(gold, positive)
    pred_bin = _binarize(pred, positive)

    p, r, f1, _ = precision_recall_fscore_support(
        gold_bin, pred_bin, labels=[1, 0], zero_division=0
    )

    accuracy = float(np.mean(np.array(gold_bin) == np.array(pred_bin)))
    kappa = float(cohen_kappa_score(gold_bin, pred_bin))

    auroc = None
    if scores is not None and len(set(gold_bin)) > 1:
        try:
            auroc = float(roc_auc_score(gold_bin, scores))
        except (ValueError, TypeError):
            auroc = None

    return BinaryMetrics(
        precision=float(p[0]),
        recall=float(r[0]),
        f1=float(f1[0]),
        accuracy=accuracy,
        auroc=auroc,
        kappa=kappa,
        support=len(gold),
    )


def compute_agreement(
    annotator1: list[str],
    annotator2: list[str],
) -> AgreementMetrics:
    """Compute inter-annotator agreement.

    Args:
        annotator1: Labels from annotator 1.
        annotator2: Labels from annotator 2.

    Returns:
        AgreementMetrics with Cohen's kappa and raw agreement rate.
    """
    if len(annotator1) != len(annotator2):
        raise ValueError(
            f"Length mismatch: {len(annotator1)} vs {len(annotator2)}"
        )

    n = len(annotator1)
    if n == 0:
        return AgreementMetrics(cohen_kappa=0.0, agreement_rate=0.0, n=0)

    agreement = sum(1 for a, b in zip(annotator1, annotator2) if a == b) / n
    kappa = float(cohen_kappa_score(annotator1, annotator2))

    return AgreementMetrics(
        cohen_kappa=kappa,
        agreement_rate=float(agreement),
        n=n,
    )


def compute_correlation(
    scores1: list[float],
    scores2: list[float],
    method: str = "spearman",
) -> tuple[float, float]:
    """Compute correlation between two score arrays.

    Args:
        scores1: First score array (e.g., evaluator scores).
        scores2: Second score array (e.g., human severity scores).
        method: "spearman" or "pearson".

    Returns:
        (correlation, p_value)
    """
    if len(scores1) != len(scores2):
        raise ValueError(
            f"Length mismatch: {len(scores1)} vs {len(scores2)}"
        )

    if len(scores1) < 3:
        return 0.0, 1.0  # Not enough data

    if method == "spearman":
        result = stats.spearmanr(scores1, scores2)
    elif method == "pearson":
        result = stats.pearsonr(scores1, scores2)
    else:
        raise ValueError(f"Unknown method: {method}")

    return float(result[0]), float(result[1])


def compute_efficiency(
    latencies: list[float],
    costs: list[float] | None = None,
) -> EfficiencyMetrics:
    """Compute efficiency metrics from measured latencies and costs.

    Args:
        latencies: Per-example latency in seconds.
        costs: Per-example cost in USD (optional, defaults to 0).

    Returns:
        EfficiencyMetrics with mean/median latency, cost per 1k, throughput.
    """
    if not latencies:
        return EfficiencyMetrics(
            mean_latency_seconds=0.0,
            median_latency_seconds=0.0,
            total_cost_usd=0.0,
            cost_per_1000=0.0,
            examples_per_second=0.0,
            n=0,
        )

    latencies_arr = np.array(latencies)
    total_cost = sum(costs) if costs else 0.0
    total_time = float(np.sum(latencies_arr))

    return EfficiencyMetrics(
        mean_latency_seconds=float(np.mean(latencies_arr)),
        median_latency_seconds=float(np.median(latencies_arr)),
        total_cost_usd=float(total_cost),
        cost_per_1000=float(total_cost / len(latencies) * 1000) if costs else 0.0,
        examples_per_second=float(len(latencies) / total_time) if total_time > 0 else 0.0,
        n=len(latencies),
    )


def extract_evaluator_labels(
    records: list[dict[str, Any]],
    evaluator_name: str,
) -> list[str | None]:
    """Extract evaluator labels from records.

    Args:
        records: List of benchmark records.
        evaluator_name: Name of the evaluator to extract.

    Returns:
        List of evaluator labels (or None if not found).
    """
    labels = []
    for r in records:
        found = None
        for eo in r.get("evaluator_outputs", []):
            if eo.get("evaluator_name") == evaluator_name:
                found = eo.get("label")
                break
        labels.append(found)
    return labels


def extract_evaluator_scores(
    records: list[dict[str, Any]],
    evaluator_name: str,
) -> list[float | None]:
    """Extract evaluator scores from records."""
    scores = []
    for r in records:
        found = None
        for eo in r.get("evaluator_outputs", []):
            if eo.get("evaluator_name") == evaluator_name:
                found = eo.get("score")
                break
        scores.append(found)
    return scores


def extract_gold_labels(records: list[dict[str, Any]]) -> list[str | None]:
    """Extract gold (human) labels from records."""
    return [r.get("faithfulness_category") for r in records]


def extract_annotator_labels(
    records: list[dict[str, Any]],
    annotator_id: str,
) -> list[str | None]:
    """Extract labels from a specific annotator."""
    labels = []
    for r in records:
        found = None
        for ann in r.get("annotations", []):
            if ann.get("annotator_id") == annotator_id:
                found = ann.get("faithfulness_category")
                break
        labels.append(found)
    return labels
