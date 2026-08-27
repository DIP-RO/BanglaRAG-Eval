#!/usr/bin/env python3
"""Launch the local annotation web server.

Usage:
    python scripts/run_annotation_server.py --dataset data/pilot.jsonl
    python scripts/run_annotation_server.py --dataset data/pilot.jsonl --mode adjudicate
    python scripts/run_annotation_server.py --dataset data/pilot.jsonl --mode both --port 5050

The server runs locally (127.0.0.1) by default. Set
BANGLARAG_ANNOTATION_SECRET in your environment for stable sessions
across restarts; otherwise a random secret is generated per run.

Annotators browse to http://127.0.0.1:5000, enter their annotator ID,
and label records one at a time. The dataset JSONL is updated in place
with validated annotations.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from banglarag_eval.annotation import create_app  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the BanglaRAG-Eval annotation web server."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the JSONL dataset to annotate.",
    )
    parser.add_argument(
        "--mode",
        choices=["annotate", "adjudicate", "both"],
        default="annotate",
        help="UI mode: annotate (default), adjudicate, or both.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1 — local only).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port (default: 5000).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode (auto-reload). Not for production use.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    app = create_app(dataset_path, mode=args.mode)
    print(f"\n  Dataset: {dataset_path}")
    print(f"  Mode:    {args.mode}")
    print(f"  URL:     http://{args.host}:{args.port}")
    print(f"\n  Open the URL in your browser to start annotating.\n")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
