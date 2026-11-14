#!/usr/bin/env python3
"""Prepare, generate, resume, and finalize the 10,000-record campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from banglarag_eval.pipeline.campaign import (
    finalize_campaign,
    generate_campaign,
    load_campaign_config,
    prepare_contradiction_proposals,
    prepare_review_packet,
    prepare_questions,
    prepare_questions_banglat5,
    verify_source_file,
    repair_prepared_question_variants,
)
from banglarag_eval.pipeline.generator import check_ollama_available


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "prepare",
            "repair",
            "propose-contradictions",
            "review",
            "generate",
            "finalize",
            "all",
        ),
        help="Campaign phase; prepare/generate are append-only and resumable.",
    )
    parser.add_argument("--config", default="configs/campaign_10000_v3_unreviewed.yaml")
    parser.add_argument("--source", default="data/raw/BanglaQuADv1.0-train.json")
    parser.add_argument(
        "--questions", default="data/intermediate/campaign_10000_questions_final_v2.jsonl"
    )
    parser.add_argument(
        "--reuse-translations",
        help="Reuse validated translation pairs from an earlier audited checkpoint.",
    )
    parser.add_argument("--working", default="checkpoints/campaign_10000_working.jsonl")
    parser.add_argument("--review-packet")
    parser.add_argument("--overwrite-review-packet", action="store_true")
    parser.add_argument("--output", default="data/campaign_10000_final.jsonl")
    parser.add_argument(
        "--categories", default="data/campaign_10000_v3_unreviewed_categories.jsonl"
    )
    parser.add_argument("--manifest", default="data/campaign_10000_manifest.json")
    parser.add_argument("--transform-model", default="qwen3:8b")
    parser.add_argument("--fallback-transform-model", default="gemma3:4b")
    parser.add_argument(
        "--translation-backend", choices=("banglat5", "ollama"), default="banglat5"
    )
    parser.add_argument("--bn-en-model", default="models/banglat5_nmt_bn_en")
    parser.add_argument("--en-bn-model", default="models/banglat5_nmt_en_bn")
    parser.add_argument("--translation-overrides", default=None)
    parser.add_argument("--repair-output")
    parser.add_argument(
        "--contradiction-proposals",
        default="data/intermediate/campaign_10000_v3_contradictions.jsonl",
    )
    args = parser.parse_args()

    config = load_campaign_config(args.config)
    needs_ollama = args.phase in {"propose-contradictions", "generate", "all"} or (
        args.phase == "prepare" and args.translation_backend == "ollama"
    )
    if needs_ollama and not check_ollama_available():
        raise SystemExit("Ollama is required and is not available; no placeholder outputs are permitted.")

    if args.phase in {"prepare", "all"}:
        verify_source_file(args.source, config)
        if args.translation_backend == "banglat5":
            questions = prepare_questions_banglat5(
                args.source,
                args.questions,
                count=config["base_examples"],
                seed=config["random_seed"],
                bn_en_model_path=args.bn_en_model,
                en_bn_model_path=args.en_bn_model,
                overrides_path=args.translation_overrides,
            )
        else:
            questions = prepare_questions(
                args.source,
                args.questions,
                count=config["base_examples"],
                seed=config["random_seed"],
                transform_model=args.transform_model,
                fallback_transform_model=args.fallback_transform_model,
                reuse_translations_path=args.reuse_translations,
            )
        print(f"Prepared {len(questions)} source questions: {args.questions}")

    if args.phase == "repair":
        repair_output = args.repair_output or args.questions
        repaired = repair_prepared_question_variants(
            args.questions,
            repair_output,
            args.translation_overrides,
        )
        print(f"Repaired {len(repaired)} prepared questions: {repair_output}")

    if args.phase == "propose-contradictions":
        status = prepare_contradiction_proposals(
            args.questions,
            args.contradiction_proposals,
            args.config,
        )
        print(json.dumps(status, ensure_ascii=False, indent=2))

    if args.phase in {"review", "all"}:
        review_packet = args.review_packet or config.get("review_gate", {}).get("file")
        if not review_packet:
            raise SystemExit("--review-packet or review_gate.file is required")
        status = prepare_review_packet(
            args.questions,
            review_packet,
            args.config,
            overwrite=args.overwrite_review_packet,
        )
        print(json.dumps(status, ensure_ascii=False, indent=2))

    if args.phase in {"generate", "all"}:
        status = generate_campaign(args.questions, args.working, args.config)
        print(json.dumps(status, ensure_ascii=False, indent=2))

    if args.phase in {"finalize", "all"}:
        manifest = finalize_campaign(
            args.working,
            args.output,
            args.categories,
            args.manifest,
            args.config,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
