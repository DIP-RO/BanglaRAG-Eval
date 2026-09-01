"""Set up annotation working copy from the generated pilot dataset.

Creates a versioned annotation copy so the original generated dataset
is preserved. Then the annotation server can be started on the copy.

Usage:
    .venv/bin/python scripts/setup_annotation.py \\
        --source data/pilot_stage1_real.jsonl \\
        --target data/pilot_stage1_annotated_v0.jsonl

    # Then start the annotation server:
    .venv/bin/python scripts/run_annotation_server.py \\
        --dataset data/pilot_stage1_annotated_v0.jsonl \\
        --mode annotate
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from banglarag_eval.dataset import load_dataset
from banglarag_eval.schema import validate_record


def setup_annotation_copy(source_path: str, target_path: str):
    """Create an annotation working copy from the source dataset.

    Args:
        source_path: Path to the generated pilot dataset.
        target_path: Path for the annotation working copy.
    """
    source = Path(source_path)
    target = Path(target_path)

    if not source.exists():
        print(f"ERROR: Source dataset not found: {source}")
        sys.exit(1)

    if target.exists():
        print(f"WARNING: Target already exists: {target}")
        response = input("Overwrite? (y/N): ")
        if response.lower() != "y":
            print("Aborted.")
            return

    # Load and validate source
    print(f"Loading source dataset: {source}")
    records = load_dataset(str(source))
    print(f"Loaded {len(records)} records")

    # Validate all records
    n_valid = 0
    n_invalid = 0
    for record in records:
        result = validate_record(record)
        if result.is_valid:
            n_valid += 1
        else:
            n_invalid += 1
            if n_invalid <= 3:
                print(f"  INVALID record {record.get('example_id', '?')}: "
                      f"{result.errors[:2]}")

    print(f"Validation: {n_valid}/{len(records)} valid, {n_invalid} invalid")

    if n_invalid > 0:
        print(f"WARNING: {n_invalid} records failed validation. "
              f"Annotation copy will still be created.")

    # Check no existing annotations
    n_annotated = sum(1 for r in records if r.get("annotations"))
    if n_annotated > 0:
        print(f"WARNING: {n_annotated} records already have annotations. "
              f"Creating a fresh copy will lose them.")

    # Ensure annotations and adjudication fields exist
    for record in records:
        if "annotations" not in record:
            record["annotations"] = []
        if "adjudication" not in record:
            record["adjudication"] = None
        if "faithfulness_category" not in record:
            record["faithfulness_category"] = None

    # Save annotation copy
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nAnnotation copy created: {target}")
    print(f"  Records: {len(records)}")
    print(f"  Schema valid: {n_valid}/{len(records)}")
    print(f"  Annotations: 0 (fresh copy)")
    print(f"\nNext steps:")
    print(f"  1. Start annotation server:")
    print(f"     .venv/bin/python scripts/run_annotation_server.py \\")
    print(f"         --dataset {target} --mode annotate")
    print(f"  2. Annotator 1: http://127.0.0.1:5000 (login: annotator1)")
    print(f"  3. Annotator 2: http://127.0.0.1:5000 (login: annotator2)")
    print(f"  4. After both annotators finish, run adjudication:")
    print(f"     .venv/bin/python scripts/run_annotation_server.py \\")
    print(f"         --dataset {target} --mode adjudicate")
    print(f"  5. Measure Stage 1 gates:")
    print(f"     .venv/bin/python scripts/run_evaluators.py \\")
    print(f"         --dataset {target} --compute-metrics \\")
    print(f"         --output results/stage1_gates.json")


def main():
    parser = argparse.ArgumentParser(
        description="Set up annotation working copy from pilot dataset"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to source (generated) dataset",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Path for annotation working copy",
    )
    args = parser.parse_args()

    setup_annotation_copy(args.source, args.target)


if __name__ == "__main__":
    main()
