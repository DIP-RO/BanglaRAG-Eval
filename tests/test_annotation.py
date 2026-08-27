"""Tests for the annotation store and Flask app.

These tests use a tiny synthetic dataset (built from the conftest
valid_record) and do NOT require any paid APIs. Flask tests use the
test client, which runs in-process without starting a real server.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Skip the entire module if Flask is not installed.
flask = pytest.importorskip("flask")

from banglarag_eval.annotation.app import create_app
from banglarag_eval.annotation.store import AnnotationStore


# ── Fixtures ────────────────────────────────────────────────────

def _make_record(
    example_id: str,
    *,
    question: str = "বাংলাদেশের রাজধানীর নাম কী?",
    generated_answer: str = "বাংলাদেশের রাজধানী ঢাকা।",
    retrieved_context: str = "বাংলাদেশের রাজধানী ঢাকা।",
    annotations: list | None = None,
    adjudication: dict | None = None,
    faithfulness_category: str | None = None,
) -> dict:
    """A minimal valid record with a generated answer (annotatable)."""
    return {
        "schema_version": "1.0.0",
        "example_id": example_id,
        "document_id": f"doc-{example_id}",
        "source_id": "test_source",
        "question": question,
        "question_language": "bn",
        "language_condition": "native_bangla",
        "data_origin": "native_authored",
        "source_text": "বাংলাদেশের রাজধানী ঢাকা।",
        "evidence_condition": "correct",
        "retrieved_context": retrieved_context,
        "retrieval_documents": [
            {"document_id": f"doc-{example_id}", "text": retrieved_context, "rank": 1}
        ],
        "retrieval_configuration": {"method": "test", "top_k": 1},
        "generated_answer": generated_answer,
        "answer_claims": [generated_answer],
        "generator_model": "test-generator",
        "generator_version": "1.0",
        "generation_settings": {"temperature": 0},
        "retriever": "test-retriever",
        "prompt_version": "test-v1",
        "question_generation": {
            "method": "human_written",
            "model": None,
            "model_version": None,
            "prompt_version": None,
            "timestamp": "2026-08-26T00:00:00+06:00",
        },
        "annotations": annotations or [],
        "adjudication": adjudication,
        "faithfulness_category": faithfulness_category,
        "hallucination_type": None,
        "corruption_metadata": None,
        "evaluator_outputs": [],
        "cost": None,
        "latency": None,
        "random_seed": None,
        "dataset_version": "test-0.1.0",
        "timestamp": "2026-08-26T00:00:00+06:00",
        "notes": None,
    }


@pytest.fixture
def dataset_path(tmp_path: Path) -> Path:
    """A 3-record JSONL dataset for testing."""
    records = [_make_record(f"test-{i:03d}") for i in range(3)]
    path = tmp_path / "test_dataset.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


# ── Store tests ─────────────────────────────────────────────────

class TestAnnotationStore:
    def test_load_and_get_record(self, dataset_path: Path):
        store = AnnotationStore(dataset_path)
        record = store.get_record("test-000")
        assert record is not None
        assert record["example_id"] == "test-000"

    def test_get_record_not_found(self, dataset_path: Path):
        store = AnnotationStore(dataset_path)
        assert store.get_record("nonexistent") is None

    def test_annotation_view_hides_provenance(self, dataset_path: Path):
        """The annotator view must NOT expose hidden fields."""
        store = AnnotationStore(dataset_path)
        view = store.get_annotation_view("test-000")
        assert view is not None
        assert "source_text" not in view
        assert "intended_answer" not in view
        assert "evidence_condition" not in view
        assert "annotations" not in view
        assert "evaluator_outputs" not in view
        assert "corruption_metadata" not in view
        assert "data_origin" not in view
        # Visible fields:
        assert view["example_id"] == "test-000"
        assert view["question"]
        assert view["retrieved_context"]
        assert view["generated_answer"]

    def test_pending_ids_for_new_annotator(self, dataset_path: Path):
        store = AnnotationStore(dataset_path)
        pending = store.get_pending_ids("ann-1")
        assert pending == ["test-000", "test-001", "test-002"]

    def test_add_annotation_appends_and_saves(self, dataset_path: Path):
        store = AnnotationStore(dataset_path)
        annotation = store.add_annotation(
            "test-000", "ann-1", "faithful", "All claims are supported."
        )
        assert annotation["annotator_id"] == "ann-1"
        assert annotation["label"] == "faithful"
        assert annotation["explanation"] == "All claims are supported."
        assert "timestamp" in annotation

        # Verify it persisted.
        store2 = AnnotationStore(dataset_path)
        record = store2.get_record("test-000")
        assert len(record["annotations"]) == 1
        assert record["annotations"][0]["annotator_id"] == "ann-1"

    def test_pending_excludes_already_annotated(self, dataset_path: Path):
        store = AnnotationStore(dataset_path)
        store.add_annotation("test-000", "ann-1", "faithful", "Supported.")
        pending = store.get_pending_ids("ann-1")
        assert "test-000" not in pending
        assert "test-001" in pending

    def test_duplicate_annotation_rejected(self, dataset_path: Path):
        store = AnnotationStore(dataset_path)
        store.add_annotation("test-000", "ann-1", "faithful", "First.")
        with pytest.raises(ValueError, match="already labeled"):
            store.add_annotation("test-000", "ann-1", "unsupported", "Second.")

    def test_invalid_label_rejected(self, dataset_path: Path):
        store = AnnotationStore(dataset_path)
        with pytest.raises(ValueError, match="invalid label"):
            store.add_annotation("test-000", "ann-1", "perfect", "Test.")

    def test_empty_explanation_rejected(self, dataset_path: Path):
        store = AnnotationStore(dataset_path)
        with pytest.raises(ValueError, match="explanation is required"):
            store.add_annotation("test-000", "ann-1", "faithful", "   ")

    def test_record_not_found(self, dataset_path: Path):
        store = AnnotationStore(dataset_path)
        with pytest.raises(ValueError, match="not found"):
            store.add_annotation("nope", "ann-1", "faithful", "Test.")

    def test_disagreements_detected(self, dataset_path: Path):
        store = AnnotationStore(dataset_path)
        store.add_annotation("test-000", "ann-1", "faithful", "Supported.")
        store.add_annotation("test-000", "ann-2", "unsupported", "Not supported.")
        disagreements = store.get_disagreements()
        assert len(disagreements) == 1
        assert disagreements[0]["example_id"] == "test-000"
        assert "faithful" in disagreements[0]["labels"]
        assert "unsupported" in disagreements[0]["labels"]

    def test_no_disagreement_when_unanimous(self, dataset_path: Path):
        store = AnnotationStore(dataset_path)
        store.add_annotation("test-000", "ann-1", "faithful", "Supported.")
        store.add_annotation("test-000", "ann-2", "faithful", "Also supported.")
        assert store.get_disagreements() == []

    def test_adjudication_sets_gold_label(self, dataset_path: Path):
        store = AnnotationStore(dataset_path)
        store.add_annotation("test-000", "ann-1", "faithful", "Supported.")
        store.add_annotation("test-000", "ann-2", "unsupported", "Not supported.")
        adjudication = store.add_adjudication(
            "test-000", "adjudicator-1", "faithful",
            "The evidence supports the answer.", "third_rater"
        )
        assert adjudication["label"] == "faithful"
        assert adjudication["method"] == "third_rater"

        record = AnnotationStore(dataset_path).get_record("test-000")
        assert record["faithfulness_category"] == "faithful"
        assert record["adjudication"]["label"] == "faithful"

    def test_third_rater_must_be_independent(self, dataset_path: Path):
        store = AnnotationStore(dataset_path)
        store.add_annotation("test-000", "ann-1", "faithful", "Supported.")
        store.add_annotation("test-000", "ann-2", "unsupported", "Not supported.")
        with pytest.raises(ValueError, match="must not be one of"):
            store.add_adjudication(
                "test-000", "ann-1", "faithful", "Test.", "third_rater"
            )

    def test_joint_session_allows_annotator(self, dataset_path: Path):
        store = AnnotationStore(dataset_path)
        store.add_annotation("test-000", "ann-1", "faithful", "Supported.")
        store.add_annotation("test-000", "ann-2", "unsupported", "Not supported.")
        adjudication = store.add_adjudication(
            "test-000", "ann-1", "faithful",
            "Consensus after discussion.", "joint_session"
        )
        assert adjudication["method"] == "joint_session"

    def test_adjudication_requires_two_annotations(self, dataset_path: Path):
        store = AnnotationStore(dataset_path)
        store.add_annotation("test-000", "ann-1", "faithful", "Supported.")
        with pytest.raises(ValueError, match=">= 2"):
            store.add_adjudication(
                "test-000", "adjudicator-1", "faithful", "Test.", "third_rater"
            )

    def test_double_adjudication_rejected(self, dataset_path: Path):
        store = AnnotationStore(dataset_path)
        store.add_annotation("test-000", "ann-1", "faithful", "Supported.")
        store.add_annotation("test-000", "ann-2", "unsupported", "Not supported.")
        store.add_adjudication("test-000", "adj-1", "faithful", "First.", "third_rater")
        with pytest.raises(ValueError, match="already adjudicated"):
            store.add_adjudication("test-000", "adj-2", "unsupported", "Second.", "third_rater")

    def test_stats(self, dataset_path: Path):
        store = AnnotationStore(dataset_path)
        stats = store.get_stats()
        assert stats["total"] == 3
        assert stats["annotatable"] == 3
        assert stats["annotated_once"] == 0
        assert stats["annotated_twice"] == 0
        assert stats["adjudicated"] == 0

        store.add_annotation("test-000", "ann-1", "faithful", "Yes.")
        store.add_annotation("test-000", "ann-2", "unsupported", "No.")
        stats = store.get_stats()
        assert stats["annotated_once"] == 1
        assert stats["annotated_twice"] == 1
        assert stats["pending_adjudication"] == 1

    def test_get_all_annotator_ids(self, dataset_path: Path):
        store = AnnotationStore(dataset_path)
        store.add_annotation("test-000", "ann-1", "faithful", "Yes.")
        store.add_annotation("test-001", "ann-2", "unsupported", "No.")
        assert store.get_all_annotator_ids() == ["ann-1", "ann-2"]


# ── Flask app tests ─────────────────────────────────────────────

class TestFlaskApp:
    @pytest.fixture
    def client(self, dataset_path: Path):
        app = create_app(dataset_path, secret_key="test-secret", mode="both")
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client

    def test_root_redirects_to_login_when_not_logged_in(self, client):
        resp = client.get("/")
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]

    def test_login_sets_session(self, client):
        resp = client.post("/login", data={"annotator_id": "ann-1"})
        assert resp.status_code == 302
        # Should redirect to annotate (mode=both defaults to adjudicate list,
        # but annotate is also accessible).
        assert resp.headers["Location"] in [
            "http://localhost/adjudicate",
            "/adjudicate",
        ]

    def test_login_empty_id_rejected(self, client):
        resp = client.post("/login", data={"annotator_id": ""})
        assert resp.status_code == 400

    def test_annotate_redirects_without_login(self, client):
        resp = client.get("/annotate")
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]

    def test_annotate_shows_record_when_logged_in(self, client):
        client.post("/login", data={"annotator_id": "ann-1"})
        resp = client.get("/annotate")
        assert resp.status_code == 302
        # Should redirect to the first pending record.
        assert "test-000" in resp.headers["Location"]

        resp = client.get("/annotate/test-000")
        assert resp.status_code == 200
        body = resp.data.decode("utf-8")
        assert "test-000" in body
        # Annotator-visible fields:
        assert "বাংলাদেশের রাজধানীর নাম কী?" in body
        assert "বাংলাদেশের রাজধানী ঢাকা।" in body
        # Hidden fields must NOT appear:
        assert "evidence_condition" not in body.lower()
        assert "source_text" not in body.lower()
        assert "intended_answer" not in body.lower()

    def test_submit_annotation(self, client):
        client.post("/login", data={"annotator_id": "ann-1"})
        resp = client.post("/annotate/test-000", data={
            "label": "faithful",
            "explanation": "The answer is fully supported by the evidence.",
        })
        assert resp.status_code == 302
        # Should redirect to next pending.
        assert "/annotate" in resp.headers["Location"]

    def test_submit_invalid_label(self, client):
        client.post("/login", data={"annotator_id": "ann-1"})
        resp = client.post("/annotate/test-000", data={
            "label": "perfect",
            "explanation": "Test.",
        })
        assert resp.status_code == 400

    def test_submit_empty_explanation(self, client):
        client.post("/login", data={"annotator_id": "ann-1"})
        resp = client.post("/annotate/test-000", data={
            "label": "faithful",
            "explanation": "",
        })
        assert resp.status_code == 400

    def test_complete_page_when_all_done(self, client):
        client.post("/login", data={"annotator_id": "ann-1"})
        for eid in ["test-000", "test-001", "test-002"]:
            client.post(f"/annotate/{eid}", data={
                "label": "faithful",
                "explanation": "Supported.",
            })
        resp = client.get("/annotate")
        assert resp.status_code == 200
        assert b"All Done" in resp.data

    def test_adjudicate_list_shows_disagreements(self, client):
        # Create a disagreement.
        client.post("/login", data={"annotator_id": "ann-1"})
        client.post("/annotate/test-000", data={
            "label": "faithful", "explanation": "Supported."
        })
        client.post("/login", data={"annotator_id": "ann-2"})
        client.post("/annotate/test-000", data={
            "label": "unsupported", "explanation": "Not supported."
        })
        # Now adjudicate.
        client.post("/login", data={"annotator_id": "adjudicator-1"})
        resp = client.get("/adjudicate")
        assert resp.status_code == 200
        assert b"test-000" in resp.data

    def test_adjudicate_record(self, client):
        client.post("/login", data={"annotator_id": "ann-1"})
        client.post("/annotate/test-000", data={
            "label": "faithful", "explanation": "Supported."
        })
        client.post("/login", data={"annotator_id": "ann-2"})
        client.post("/annotate/test-000", data={
            "label": "unsupported", "explanation": "Not supported."
        })
        client.post("/login", data={"annotator_id": "adjudicator-1"})
        resp = client.post("/adjudicate/test-000", data={
            "label": "faithful",
            "explanation": "The evidence supports the answer.",
            "method": "third_rater",
        })
        assert resp.status_code == 302
        assert "/adjudicate" in resp.headers["Location"]

    def test_progress_page(self, client):
        client.post("/login", data={"annotator_id": "ann-1"})
        resp = client.get("/progress")
        assert resp.status_code == 200
        assert b"Annotation Progress" in resp.data

    def test_logout_clears_session(self, client):
        client.post("/login", data={"annotator_id": "ann-1"})
        client.post("/logout")
        resp = client.get("/annotate")
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]

    def test_annotation_persists_across_reloads(self, dataset_path: Path):
        """Annotation saved by one session must be visible to a fresh store."""
        app = create_app(dataset_path, secret_key="test-secret", mode="both")
        app.config["TESTING"] = True
        with app.test_client() as c1:
            c1.post("/login", data={"annotator_id": "ann-1"})
            c1.post("/annotate/test-000", data={
                "label": "faithful", "explanation": "Supported."
            })
        # Fresh app instance reads from the same file.
        app2 = create_app(dataset_path, secret_key="test-secret", mode="both")
        app2.config["TESTING"] = True
        with app2.test_client() as c2:
            c2.post("/login", data={"annotator_id": "ann-2"})
            resp = c2.get("/annotate/test-000")
            assert resp.status_code == 200
            # ann-2 should still be able to annotate (it's a different annotator).
            resp = c2.post("/annotate/test-000", data={
                "label": "unsupported", "explanation": "Not supported."
            })
            assert resp.status_code == 302
