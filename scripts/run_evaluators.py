"""Run evaluators on a dataset and compute metrics.

Usage:
    # Run lexical baseline on all records (no API needed)
    .venv/bin/python scripts/run_evaluators.py \\
        --dataset data/pilot_stage1_real.jsonl \\
        --evaluators lexical_baseline \\
        --output data/pilot_stage1_evaluated.jsonl

    # Run lexical baseline + LLM judge (needs Ollama)
    .venv/bin/python scripts/run_evaluators.py \\
        --dataset data/pilot_stage1_real.jsonl \\
        --evaluators lexical_baseline llm_judge \\
        --output data/pilot_stage1_evaluated.jsonl

    # Compute metrics on annotated + evaluated dataset
    .venv/bin/python scripts/run_evaluators.py \\
        --dataset data/pilot_stage1_annotated_v0.jsonl \\
        --compute-metrics \\
        --output results/metrics.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from banglarag_eval.evaluators import (
    LexicalBaselineEvaluator,
    LLMJudgeEvaluator,
    attach_evaluator_output,
)
from banglarag_eval.dataset import load_dataset, save_dataset
from banglarag_eval.metrics import (
    compute_classification_metrics,
    compute_binary_metrics,
    compute_agreement,
    compute_efficiency,
    extract_evaluator_labels,
    extract_evaluator_scores,
    extract_gold_labels,
    extract_annotator_labels,
)
from banglarag_eval.gates import measure_stage1_gates


def build_evaluator(name: str, **kwargs):
    """Build an evaluator by name."""
    if name == "lexical_baseline":
        return LexicalBaselineEvaluator()
    elif name == "llm_judge":
        return LLMJudgeEvaluator()
    else:
        raise ValueError(f"Unknown evaluator: {name}")


def run_evaluators(
    dataset_path: str,
    evaluator_names: list[str],
    output_path: str,
):
    """Run evaluators on all records in a dataset."""
    print(f"Loading dataset from {dataset_path}...")
    records = load_dataset(dataset_path)
    print(f"Loaded {len(records)} records")

    for eval_name in evaluator_names:
        print(f"\nRunning {eval_name} evaluator...")
        evaluator = build_evaluator(eval_name)
        n_success = 0
        n_fail = 0
        start = time.time()

        for i, record in enumerate(records):
            try:
                output = evaluator.evaluate(record)
                attach_evaluator_output(record, output)
                if output.label is not None:
                    n_success += 1
                else:
                    n_fail += 1
            except Exception as e:
                n_fail += 1
                print(f"  ERROR on record {i}: {e}")

            if (i + 1) % 50 == 0:
                elapsed = time.time() - start
                print(f"  [{i+1}/{len(records)}] {eval_name}: "
                      f"{n_success} success, {n_fail} fail ({elapsed:.1f}s)")

        elapsed = time.time() - start
        print(f"  Done: {n_success}/{len(records)} success, "
              f"{n_fail} fail in {elapsed:.1f}s")

    print(f"\nSaving evaluated dataset to {output_path}...")
    save_dataset(records, output_path)
    print(f"Saved {len(records)} records with evaluator outputs")

    return records


def compute_metrics(
    dataset_path: str,
    output_path: str,
):
    """Compute metrics on an annotated + evaluated dataset."""
    print(f"Loading dataset from {dataset_path}...")
    records = load_dataset(dataset_path)
    print(f"Loaded {len(records)} records")

    # Get gold labels (from annotations/adjudication)
    gold = extract_gold_labels(records)
    n_gold = sum(1 for g in gold if g is not None)
    print(f"Records with gold labels: {n_gold}/{len(records)}")

    # Get annotator IDs
    annotator_ids = set()
    for r in records:
        for ann in r.get("annotations", []):
            if ann.get("annotator_id"):
                annotator_ids.add(ann["annotator_id"])

    results = {
        "n_records": len(records),
        "n_gold_labels": n_gold,
        "annotator_ids": sorted(annotator_ids),
        "evaluators": {},
        "agreement": {},
        "gates": {},
    }

    # Inter-annotator agreement
    if len(annotator_ids) >= 2:
        ids = sorted(annotator_ids)
        labels1 = extract_annotator_labels(records, ids[0])
        labels2 = extract_annotator_labels(records, ids[1])
        paired = [(l1, l2) for l1, l2 in zip(labels1, labels2) if l1 and l2]
        if paired:
            l1 = [p[0] for p in paired]
            l2 = [p[1] for p in paired]
            agreement = compute_agreement(l1, l2)
            results["agreement"] = {
                "annotator_1": ids[0],
                "annotator_2": ids[1],
                "cohen_kappa": agreement.cohen_kappa,
                "agreement_rate": agreement.agreement_rate,
                "n_paired": agreement.n,
            }
            print(f"  Agreement: κ={agreement.cohen_kappa:.3f}, "
                  f"rate={agreement.agreement_rate:.3f}, n={agreement.n}")

    # Per-evaluator metrics
    evaluator_names = set()
    for r in records:
        for eo in r.get("evaluator_outputs", []):
            evaluator_names.add(eo.get("evaluator_name"))

    for eval_name in sorted(evaluator_names):
        print(f"\n  Computing metrics for {eval_name}...")
        pred = extract_evaluator_labels(records, eval_name)
        scores = extract_evaluator_scores(records, eval_name)

        # Filter to records with both gold and pred
        paired = [(g, p, s) for g, p, s in zip(gold, pred, scores)
                  if g is not None and p is not None]

        if not paired:
            results["evaluators"][eval_name] = {
                "n_paired": 0,
                "note": "No paired gold+pred labels found",
            }
            continue

        gold_paired = [p[0] for p in paired]
        pred_paired = [p[1] for p in paired]
        scores_paired = [p[2] for p in paired if p[2] is not None]

        # Multi-class metrics
        class_metrics = compute_classification_metrics(gold_paired, pred_paired)

        # Binary metrics (faithful vs rest)
        binary = compute_binary_metrics(
            gold_paired, pred_paired,
            scores=scores_paired if len(scores_paired) == len(paired) else None,
        )

        # Efficiency
        latencies = []
        for r in records:
            for eo in r.get("evaluator_outputs", []):
                if eo.get("evaluator_name") == eval_name:
                    if eo.get("latency_seconds") is not None:
                        latencies.append(eo["latency_seconds"])

        efficiency = compute_efficiency(latencies) if latencies else None

        results["evaluators"][eval_name] = {
            "n_paired": len(paired),
            "classification": class_metrics.to_dict(),
            "binary": binary.to_dict(),
            "efficiency": efficiency.to_dict() if efficiency else None,
        }

        print(f"    n={len(paired)}, F1_macro={class_metrics.f1_macro:.3f}, "
              f"binary_F1={binary.f1:.3f}, kappa={binary.kappa:.3f}")

    # Stage 1 gates
    print("\n  Measuring Stage 1 gates...")
    gate_report = measure_stage1_gates(dataset_path)
    results["gates"] = gate_report.to_dict()
    print(gate_report.summary())

    # Save results
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nMetrics saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluators and compute metrics on a dataset"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to JSONL dataset",
    )
    parser.add_argument(
        "--evaluators",
        nargs="*",
        default=["lexical_baseline"],
        help="Evaluators to run (lexical_baseline, llm_judge)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for evaluated dataset or metrics JSON",
    )
    parser.add_argument(
        "--compute-metrics",
        action="store_true",
        help="Compute metrics instead of running evaluators",
    )
    args = parser.parse_args()

    if args.compute_metrics:
        compute_metrics(args.dataset, args.output)
    else:
        run_evaluators(args.dataset, args.evaluators, args.output)


if __name__ == "__main__":
    main()
