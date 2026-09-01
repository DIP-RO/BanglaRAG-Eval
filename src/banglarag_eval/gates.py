"""Stage 1 Gate measurement — computes all five pilot quality gates.

Implements the five gates from docs/pilot_design.md:

1. Annotation agreement: Cohen's kappa >= 0.60, 95% bootstrap CI
2. Taxonomy adequacy: < 10% NO-FIT markers in explanations
3. Condition integrity: >= 90% blind re-label match
4. Pipeline integrity: 100% records pass validation
5. Evaluator harness: >= 95% evaluator output success

Usage:
    from banglarag_eval.gates import measure_stage1_gates
    result = measure_stage1_gates("data/pilot_stage1_annotated_v0.jsonl")
    print(result.summary())
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from banglarag_eval.schema import validate_record
from banglarag_eval.metrics import (
    compute_agreement,
    bootstrap_ci,
    extract_annotator_labels,
    extract_evaluator_labels,
    ALL_LABELS,
)
from banglarag_eval.metrics.statistics import is_underpowered
import numpy as np


# Gate thresholds (pre-registered in pilot_design.md)
KAPPA_THRESHOLD = 0.60
KAPPA_TARGET = 0.70
KAPPA_CI_LOWER_BOUND = 0.40
NO_FIT_THRESHOLD = 0.10  # < 10% of examples
CONDITION_INTEGRITY_THRESHOLD = 0.90  # >= 90% match
PIPELINE_INTEGRITY_THRESHOLD = 1.0  # 100%
EVALUATOR_HARNESS_THRESHOLD = 0.95  # >= 95%


@dataclass
class GateResult:
    """Result of a single gate measurement."""

    name: str
    passed: bool
    value: float
    threshold: float
    details: str
    n: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "value": self.value,
            "threshold": self.threshold,
            "details": self.details,
            "n": self.n,
        }

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.value:.3f} (threshold: {self.threshold:.3f}) — {self.details}"


@dataclass
class Stage1GateReport:
    """Full Stage 1 gate report."""

    gates: list[GateResult] = field(default_factory=list)
    all_passed: bool = False
    n_records: int = 0
    n_annotated: int = 0
    n_adjudicated: int = 0
    underpowered: bool = False
    decision: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "gates": [g.to_dict() for g in self.gates],
            "all_passed": self.all_passed,
            "n_records": self.n_records,
            "n_annotated": self.n_annotated,
            "n_adjudicated": self.n_adjudicated,
            "underpowered": self.underpowered,
            "decision": self.decision,
        }

    def summary(self) -> str:
        """Human-readable summary of all gates."""
        lines = ["=" * 60, "Stage 1 Gate Report", "=" * 60, ""]
        for gate in self.gates:
            lines.append(str(gate))
        lines.append("")
        lines.append(f"Records: {self.n_records}")
        lines.append(f"Annotated: {self.n_annotated}")
        lines.append(f"Adjudicated: {self.n_adjudicated}")
        lines.append(f"Underpowered: {self.underpowered}")
        lines.append(f"All gates passed: {self.all_passed}")
        lines.append(f"Decision: {self.decision}")
        lines.append("=" * 60)
        return "\n".join(lines)


def _load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL records from file."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _count_no_fit(records: list[dict[str, Any]]) -> tuple[int, int]:
    """Count NO-FIT markers in annotator explanations.

    Returns:
        (n_no_fit, n_total_annotations)
    """
    n_no_fit = 0
    n_total = 0
    for record in records:
        for ann in record.get("annotations", []):
            n_total += 1
            explanation = ann.get("explanation", "") or ""
            if "NO-FIT:" in explanation.upper():
                n_no_fit += 1
    return n_no_fit, n_total


def _check_condition_integrity(records: list[dict[str, Any]]) -> tuple[int, int]:
    """Check condition integrity — blind re-label match rate.

    This is a placeholder that checks if the evidence_condition field
    is consistent with the corruption_metadata. A real blind re-label
    requires a human checker who doesn't see the stored condition.

    For now, we check that:
    - correct evidence has no corruption_metadata
    - corrupted evidence has corruption_metadata
    - missing evidence has empty retrieved_context

    Returns:
        (n_match, n_total)
    """
    n_match = 0
    n_total = len(records)

    for record in records:
        condition = record.get("evidence_condition", "")
        corruption = record.get("corruption_metadata")
        context = record.get("retrieved_context", "") or ""

        matches = True

        if condition == "correct":
            if corruption is not None:
                matches = False
        elif condition == "missing":
            if context.strip():
                matches = False
        else:
            # partially_relevant, irrelevant, contradictory should have corruption_metadata
            if corruption is None:
                matches = False

        if matches:
            n_match += 1

    return n_match, n_total


def _check_pipeline_integrity(records: list[dict[str, Any]]) -> tuple[int, int]:
    """Check pipeline integrity — all records pass schema validation.

    Returns:
        (n_valid, n_total)
    """
    n_valid = 0
    n_total = len(records)

    for record in records:
        result = validate_record(record)
        if result.is_valid:
            n_valid += 1

    return n_valid, n_total


def _check_evaluator_harness(
    records: list[dict[str, Any]],
    evaluator_name: str | None = None,
) -> tuple[int, int]:
    """Check evaluator harness — success rate of evaluator outputs.

    Args:
        records: List of records with evaluator_outputs.
        evaluator_name: Specific evaluator to check (None = any).

    Returns:
        (n_success, n_total)
    """
    n_success = 0
    n_total = len(records)

    for record in records:
        outputs = record.get("evaluator_outputs", [])
        if evaluator_name:
            outputs = [o for o in outputs if o.get("evaluator_name") == evaluator_name]

        # Success = has a non-null label
        if outputs and any(o.get("label") is not None for o in outputs):
            n_success += 1

    return n_success, n_total


def measure_stage1_gates(
    dataset_path: str | Path,
    evaluator_name: str | None = None,
    n_bootstrap: int = 10000,
) -> Stage1GateReport:
    """Measure all five Stage 1 gates on an annotated dataset.

    Args:
        dataset_path: Path to the annotated JSONL dataset.
        evaluator_name: Evaluator to check for harness gate (None = any).
        n_bootstrap: Bootstrap iterations for kappa CI.

    Returns:
        Stage1GateReport with all gate results and a go/no-go decision.
    """
    records = _load_records(dataset_path)
    n_records = len(records)

    report = Stage1GateReport(
        n_records=n_records,
        n_annotated=sum(1 for r in records if r.get("annotations")),
        n_adjudicated=sum(1 for r in records if r.get("adjudication")),
        underpowered=is_underpowered(n_records),
    )

    # Gate 1: Annotation agreement (Cohen's kappa)
    if report.n_annotated > 0:
        # Get annotator IDs
        annotator_ids = set()
        for r in records:
            for ann in r.get("annotations", []):
                if ann.get("annotator_id"):
                    annotator_ids.add(ann["annotator_id"])

        if len(annotator_ids) >= 2:
            ids = sorted(annotator_ids)
            labels1 = extract_annotator_labels(records, ids[0])
            labels2 = extract_annotator_labels(records, ids[1])

            # Filter to records where both annotators labeled
            paired = [(l1, l2) for l1, l2 in zip(labels1, labels2) if l1 and l2]
            if len(paired) >= 2:
                l1_list = [p[0] for p in paired]
                l2_list = [p[1] for p in paired]

                agreement = compute_agreement(l1_list, l2_list)
                kappa = agreement.cohen_kappa

                # Handle NaN kappa (happens when all labels are the same)
                if np.isnan(kappa):
                    kappa = 1.0  # Perfect agreement on single label = kappa 1.0

                # Bootstrap CI for kappa
                # We bootstrap over the paired labels
                rng = np.random.default_rng(42)
                kappa_estimates = []
                n = len(paired)
                for _ in range(n_bootstrap):
                    indices = rng.integers(0, n, size=n)
                    sample_l1 = [l1_list[i] for i in indices]
                    sample_l2 = [l2_list[i] for i in indices]
                    try:
                        from sklearn.metrics import cohen_kappa_score
                        k = cohen_kappa_score(sample_l1, sample_l2)
                        if np.isnan(k):
                            k = 1.0  # Single-label agreement = perfect
                        kappa_estimates.append(k)
                    except Exception:
                        pass

                if kappa_estimates:
                    ci_lower = float(np.percentile(kappa_estimates, 2.5))
                    ci_upper = float(np.percentile(kappa_estimates, 97.5))
                else:
                    ci_lower = 0.0
                    ci_upper = 0.0

                # Gate passes if kappa >= 0.60 AND CI lower bound >= 0.40
                passed = kappa >= KAPPA_THRESHOLD and ci_lower >= KAPPA_CI_LOWER_BOUND

                # Decision logic
                if kappa < 0.40:
                    decision_note = "kappa < 0.40 -> redesign taxonomy"
                elif kappa < 0.60:
                    decision_note = "kappa 0.40-0.60 -> revise guidelines, re-pilot"
                elif ci_lower < KAPPA_CI_LOWER_BOUND:
                    decision_note = "kappa >= 0.60 but CI lower < 0.40 -> extend Stage 1 sample"
                else:
                    decision_note = "kappa >= 0.60 with CI lower >= 0.40 -> can scale"

                report.gates.append(GateResult(
                    name="annotation_agreement",
                    passed=passed,
                    value=kappa,
                    threshold=KAPPA_THRESHOLD,
                    details=f"κ={kappa:.3f}, 95% CI=[{ci_lower:.3f}, {ci_upper:.3f}], n={n}. {decision_note}",
                    n=n,
                ))
            else:
                report.gates.append(GateResult(
                    name="annotation_agreement",
                    passed=False,
                    value=0.0,
                    threshold=KAPPA_THRESHOLD,
                    details=f"Insufficient paired annotations ({len(paired)}). Need at least 2.",
                    n=len(paired),
                ))
        else:
            report.gates.append(GateResult(
                name="annotation_agreement",
                passed=False,
                value=0.0,
                threshold=KAPPA_THRESHOLD,
                details=f"Need at least 2 annotators, found {len(annotator_ids)}.",
                n=0,
            ))
    else:
        report.gates.append(GateResult(
            name="annotation_agreement",
            passed=False,
            value=0.0,
            threshold=KAPPA_THRESHOLD,
            details="No annotated records found. Run annotators first (Issue #9).",
            n=0,
        ))

    # Gate 2: Taxonomy adequacy (NO-FIT markers)
    n_no_fit, n_total_annotations = _count_no_fit(records)
    if n_total_annotations > 0:
        no_fit_rate = n_no_fit / n_total_annotations
        passed = no_fit_rate < NO_FIT_THRESHOLD
        report.gates.append(GateResult(
            name="taxonomy_adequacy",
            passed=passed,
            value=no_fit_rate,
            threshold=NO_FIT_THRESHOLD,
            details=f"{n_no_fit}/{n_total_annotations} annotations have NO-FIT markers ({no_fit_rate:.1%})",
            n=n_total_annotations,
        ))
    else:
        report.gates.append(GateResult(
            name="taxonomy_adequacy",
            passed=False,
            value=0.0,
            threshold=NO_FIT_THRESHOLD,
            details="No annotations found to check for NO-FIT markers.",
            n=0,
        ))

    # Gate 3: Condition integrity (blind re-label match)
    n_match, n_total_conditions = _check_condition_integrity(records)
    match_rate = n_match / n_total_conditions if n_total_conditions > 0 else 0.0
    passed = match_rate >= CONDITION_INTEGRITY_THRESHOLD
    report.gates.append(GateResult(
        name="condition_integrity",
        passed=passed,
        value=match_rate,
        threshold=CONDITION_INTEGRITY_THRESHOLD,
        details=f"{n_match}/{n_total_conditions} conditions match ({match_rate:.1%}). Note: automated check; full blind re-label requires human checker.",
        n=n_total_conditions,
    ))

    # Gate 4: Pipeline integrity (schema validation)
    n_valid, n_total_pipeline = _check_pipeline_integrity(records)
    valid_rate = n_valid / n_total_pipeline if n_total_pipeline > 0 else 0.0
    passed = valid_rate >= PIPELINE_INTEGRITY_THRESHOLD
    report.gates.append(GateResult(
        name="pipeline_integrity",
        passed=passed,
        value=valid_rate,
        threshold=PIPELINE_INTEGRITY_THRESHOLD,
        details=f"{n_valid}/{n_total_pipeline} records pass schema validation ({valid_rate:.1%})",
        n=n_total_pipeline,
    ))

    # Gate 5: Evaluator harness (output success rate)
    n_success, n_total_eval = _check_evaluator_harness(records, evaluator_name)
    success_rate = n_success / n_total_eval if n_total_eval > 0 else 0.0
    passed = success_rate >= EVALUATOR_HARNESS_THRESHOLD
    eval_name = evaluator_name or "any"
    report.gates.append(GateResult(
        name="evaluator_harness",
        passed=passed,
        value=success_rate,
        threshold=EVALUATOR_HARNESS_THRESHOLD,
        details=f"{n_success}/{n_total_eval} records have successful {eval_name} evaluator output ({success_rate:.1%})",
        n=n_total_eval,
    ))

    # Overall decision
    report.all_passed = all(g.passed for g in report.gates)
    if report.all_passed:
        report.decision = "ALL GATES PASSED — approved to scale to Stage 2"
    else:
        failed = [g.name for g in report.gates if not g.passed]
        report.decision = f"GATES FAILED: {', '.join(failed)} — revise before scaling"

    if report.underpowered and not report.all_passed:
        report.decision += " (CAVEAT: underpowered — diagnostic only)"

    return report
