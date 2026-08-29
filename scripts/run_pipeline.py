#!/usr/bin/env python3
"""Run the Milestone 2 pilot pipeline.

Builds a real pilot dataset with:
- Curated Bangla + English source documents
- Question generation across 5 language conditions
- BM25 lexical retrieval
- Controlled evidence corruption (5 conditions)
- Answer generation via Ollama/Qwen3 (or offline fallback)

Usage:
    .venv/bin/python scripts/run_pipeline.py
    .venv/bin/python scripts/run_pipeline.py --no-ollama
    .venv/bin/python scripts/run_pipeline.py --languages native_bangla code_mixed
    .venv/bin/python scripts/run_pipeline.py --output data/pilot_stage1_real.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from banglarag_eval.pipeline.orchestrator import run_pipeline
from banglarag_eval.pipeline.generator import check_ollama_available


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Milestone 2 pilot pipeline"
    )
    parser.add_argument(
        "--output",
        default="data/pilot_stage1_real.jsonl",
        help="Output JSONL path (default: data/pilot_stage1_real.jsonl)",
    )
    parser.add_argument(
        "--no-ollama",
        action="store_true",
        help="Skip Ollama generation, use intended answers as placeholders",
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help="Language conditions to build (default: all 5)",
    )
    parser.add_argument(
        "--evidence",
        nargs="*",
        default=None,
        help="Evidence conditions to build (default: all 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    use_ollama = not args.no_ollama

    if use_ollama:
        if check_ollama_available():
            print("[OK] Ollama is running. Generation will use Qwen3:8b.")
        else:
            print("[WARN] Ollama is not running. Falling back to placeholder answers.")
            print("       Start Ollama with: ollama serve")
            use_ollama = False
    else:
        print("[INFO] Ollama disabled. Using intended answers as placeholders.")

    print(f"[INFO] Building pilot dataset → {args.output}")
    print()

    summary = run_pipeline(
        output_path=args.output,
        evidence_conditions=args.evidence,
        language_conditions=args.languages,
        use_ollama=use_ollama,
        seed=args.seed,
    )

    print("=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Total records:     {summary['total_records']}")
    print()
    print("Language distribution:")
    for lang, count in sorted(summary["language_distribution"].items()):
        print(f"  {lang:25s} {count:4d}")
    print()
    print("Evidence distribution:")
    for cond, count in sorted(summary["evidence_distribution"].items()):
        print(f"  {cond:25s} {count:4d}")
    print()
    print("Generation models:")
    for model, count in sorted(summary["generation_models"].items()):
        print(f"  {model:30s} {count:4d}")
    print()
    print(f"Ollama used:       {summary['ollama_used']}")
    print(f"Elapsed:           {summary['elapsed_seconds']}s")
    print(f"Output:            {summary['output_path']}")
    print()
    print("Next steps:")
    print(f"  1. Review: {args.output}")
    print(f"  2. Copy for annotation: cp {args.output} data/pilot_stage1_annotated_v0.jsonl")
    print(f"  3. Start annotation UI: .venv/bin/python scripts/run_annotation_server.py --dataset data/pilot_stage1_annotated_v0.jsonl")


if __name__ == "__main__":
    main()
