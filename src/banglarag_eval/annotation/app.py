"""Flask app for human annotation and adjudication.

Routes:
  GET  /              → redirect to login or annotate
  GET  /login         → annotator ID entry form
  POST /login         → set session, redirect to annotate
  POST /logout        → clear session, redirect to login
  GET  /annotate      → show next pending record (or complete page)
  POST /annotate      → save annotation, advance to next
  GET  /annotate/<id> → show a specific record
  GET  /progress      → dashboard with stats
  GET  /adjudicate    → list records needing adjudication
  GET  /adjudicate/<id> → adjudication form for one record
  POST /adjudicate/<id> → save adjudication

Annotators see only: question, retrieved_context, generated_answer.
Adjudicators see the above plus the disagreeing annotations.
Neither sees: source_text, intended_answer, evidence_condition,
evaluator_outputs, or any other hidden provenance field.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from flask import (
    Flask,
    abort,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from banglarag_eval.annotation.store import AnnotationStore
from banglarag_eval.constants import (
    ADJUDICATION_METHODS,
    FAITHFULNESS_CATEGORIES,
)

_LABEL_DESCRIPTIONS = {
    "faithful": "Every substantive claim is supported by the retrieved evidence.",
    "partially_faithful": "Some claims supported; at least one substantive claim is not.",
    "unsupported": "No central claim is supported (regardless of real-world truth).",
    "contradictory": "At least one central claim directly conflicts with the evidence.",
    "insufficient_evidence": "Evidence cannot verify or refute any substantive claim.",
}


def create_app(
    dataset_path: str | Path,
    *,
    secret_key: str | None = None,
    mode: str = "annotate",
) -> Flask:
    """Create the annotation Flask app.

    Args:
        dataset_path: Path to the JSONL dataset.
        secret_key: Flask session secret. If None, reads from
            BANGLARAG_ANNOTATION_SECRET env var, or generates a random one
            (with a warning printed to stderr).
        mode: "annotate" for the annotation UI, "adjudicate" for the
            adjudication UI, or "both" for combined access.
    """
    template_folder = Path(__file__).parent / "templates"
    static_folder = Path(__file__).parent / "static"
    app = Flask(
        __name__,
        template_folder=str(template_folder),
        static_folder=str(static_folder),
    )

    if secret_key is None:
        secret_key = os.environ.get("BANGLARAG_ANNOTATION_SECRET")
    if not secret_key:
        import secrets as _secrets
        secret_key = _secrets.token_hex(32)
        import sys
        print(
            "WARNING: no BANGLARAG_ANNOTATION_SECRET set; using a random "
            "session key (sessions will not survive restart).",
            file=sys.stderr,
        )
    app.secret_key = secret_key
    app.config["DATASET_PATH"] = str(dataset_path)
    app.config["MODE"] = mode

    # One store per app; reload from disk on each request to pick up
    # changes from other annotators working concurrently.
    store = AnnotationStore(dataset_path)

    def _get_store() -> AnnotationStore:
        store.reload()
        return store

    def _require_login() -> str | None:
        """Return annotator_id from session, or None."""
        return session.get("annotator_id")

    # ── Root ──────────────────────────────────────────────────────
    @app.route("/")
    def index():
        if not _require_login():
            return redirect(url_for("login"))
        if mode in ("adjudicate", "both"):
            return redirect(url_for("adjudicate_list"))
        return redirect(url_for("annotate_next"))

    # ── Login / Logout ────────────────────────────────────────────
    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            annotator_id = request.form.get("annotator_id", "").strip()
            if not annotator_id:
                flash("Please enter your annotator ID.", "error")
                return render_template("login.html"), 400
            session["annotator_id"] = annotator_id
            if mode in ("adjudicate", "both"):
                return redirect(url_for("adjudicate_list"))
            return redirect(url_for("annotate_next"))
        return render_template("login.html")

    @app.route("/logout", methods=["POST"])
    def logout():
        session.clear()
        return redirect(url_for("login"))

    # ── Annotation ────────────────────────────────────────────────
    @app.route("/annotate")
    def annotate_next():
        annotator_id = _require_login()
        if not annotator_id:
            return redirect(url_for("login"))
        if mode == "adjudicate":
            return redirect(url_for("adjudicate_list"))

        s = _get_store()
        pending = s.get_pending_ids(annotator_id)
        if not pending:
            stats = s.get_stats()
            completed = s.get_completed_ids(annotator_id)
            return render_template(
                "complete.html",
                annotator_id=annotator_id,
                completed_count=len(completed),
                total=stats["annotatable"],
            )
        # Show the first pending record.
        return redirect(url_for("annotate_record", example_id=pending[0]))

    @app.route("/annotate/<example_id>", methods=["GET", "POST"])
    def annotate_record(example_id: str):
        annotator_id = _require_login()
        if not annotator_id:
            return redirect(url_for("login"))
        if mode == "adjudicate":
            return redirect(url_for("adjudicate_list"))

        s = _get_store()

        if request.method == "POST":
            label = request.form.get("label", "").strip()
            explanation = request.form.get("explanation", "").strip()
            try:
                s.add_annotation(example_id, annotator_id, label, explanation)
                flash(f"Saved annotation for {example_id}.", "success")
            except ValueError as exc:
                flash(str(exc), "error")
                view = s.get_annotation_view(example_id)
                if view is None:
                    abort(404)
                pending = s.get_pending_ids(annotator_id)
                return render_template(
                    "annotate.html",
                    record=view,
                    labels=FAITHFULNESS_CATEGORIES,
                    label_descriptions=_LABEL_DESCRIPTIONS,
                    annotator_id=annotator_id,
                    pending=pending,
                    current_index=pending.index(example_id) if example_id in pending else 0,
                    pending_count=len(pending),
                ), 400
            return redirect(url_for("annotate_next"))

        view = s.get_annotation_view(example_id)
        if view is None:
            abort(404)
        pending = s.get_pending_ids(annotator_id)
        if example_id not in pending:
            flash(f"You have already labeled {example_id}.", "info")
            return redirect(url_for("annotate_next"))
        return render_template(
            "annotate.html",
            record=view,
            labels=FAITHFULNESS_CATEGORIES,
            label_descriptions=_LABEL_DESCRIPTIONS,
            annotator_id=annotator_id,
            pending=pending,
            current_index=pending.index(example_id),
            pending_count=len(pending),
        )

    # ── Progress dashboard ────────────────────────────────────────
    @app.route("/progress")
    def progress():
        annotator_id = _require_login()
        if not annotator_id:
            return redirect(url_for("login"))
        s = _get_store()
        stats = s.get_stats()
        completed = s.get_completed_ids(annotator_id)
        return render_template(
            "progress.html",
            stats=stats,
            annotator_id=annotator_id,
            completed_count=len(completed),
            all_annotators=s.get_all_annotator_ids(),
        )

    # ── Adjudication ──────────────────────────────────────────────
    @app.route("/adjudicate")
    def adjudicate_list():
        annotator_id = _require_login()
        if not annotator_id:
            return redirect(url_for("login"))
        if mode == "annotate":
            return redirect(url_for("annotate_next"))

        s = _get_store()
        disagreements = s.get_disagreements()
        return render_template(
            "adjudicate_list.html",
            disagreements=disagreements,
            annotator_id=annotator_id,
            count=len(disagreements),
        )

    @app.route("/adjudicate/<example_id>", methods=["GET", "POST"])
    def adjudicate_record(example_id: str):
        annotator_id = _require_login()
        if not annotator_id:
            return redirect(url_for("login"))
        if mode == "annotate":
            return redirect(url_for("annotate_next"))

        s = _get_store()

        if request.method == "POST":
            label = request.form.get("label", "").strip()
            explanation = request.form.get("explanation", "").strip()
            method = request.form.get("method", "third_rater").strip()
            try:
                s.add_adjudication(
                    example_id, annotator_id, label, explanation, method
                )
                flash(f"Adjudication saved for {example_id}.", "success")
            except ValueError as exc:
                flash(str(exc), "error")
                view = s.get_adjudication_view(example_id)
                if view is None:
                    abort(404)
                return render_template(
                    "adjudicate.html",
                    record=view,
                    labels=FAITHFULNESS_CATEGORIES,
                    label_descriptions=_LABEL_DESCRIPTIONS,
                    methods=ADJUDICATION_METHODS,
                    annotator_id=annotator_id,
                ), 400
            return redirect(url_for("adjudicate_list"))

        view = s.get_adjudication_view(example_id)
        if view is None:
            abort(404)
        if view.get("adjudication") is not None:
            flash(f"{example_id} is already adjudicated.", "info")
            return redirect(url_for("adjudicate_list"))
        return render_template(
            "adjudicate.html",
            record=view,
            labels=FAITHFULNESS_CATEGORIES,
            label_descriptions=_LABEL_DESCRIPTIONS,
            methods=ADJUDICATION_METHODS,
            annotator_id=annotator_id,
        )

    return app
