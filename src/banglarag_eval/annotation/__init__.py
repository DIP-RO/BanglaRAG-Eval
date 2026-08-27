"""Local web annotation UI for human annotators.

Provides a Flask app that presents records one at a time (question,
retrieved context, generated answer) and lets an annotator choose a
faithfulness label and write an explanation. Annotators never see the
source document, intended answer, evidence-condition label, other
annotators' labels, or any evaluator output. Annotations are validated
against the schema before saving.

Run with: python scripts/run_annotation_server.py --dataset path.jsonl
"""

from banglarag_eval.annotation.app import create_app

__all__ = ["create_app"]
