"""Tests for the resumable 10,000-record campaign builder."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from banglarag_eval.pipeline import campaign
from banglarag_eval.pipeline.generator import GenerationResult
from banglarag_eval.pipeline.retriever import RetrievedDocument, RetrievalResult


def _source_payload() -> dict:
    articles = []
    for index in range(2):
        answer = f"উত্তর{index}"
        context = f"বিষয় {index} সম্পর্কে প্রথম বাক্য। সঠিক তথ্য হলো {answer}। অতিরিক্ত প্রাসঙ্গিক বাক্য।"
        articles.append(
            {
                "title": f"নিবন্ধ {index}",
                "url": f"https://bn.wikipedia.org/wiki/{index}",
                "paragraphs": [
                    {
                        "context": context,
                        "qas": [
                            {
                                "id": f"qa-{index}",
                                "question": f"বিষয় {index} এর সঠিক তথ্য কী?",
                                "answers": [{"answer_start": context.index(answer), "text": answer}],
                            }
                        ],
                    }
                ],
            }
        )
    return {"version": "1.0", "split": "train", "data": articles}


def _variants(question: str, model: str) -> dict[str, str]:
    return {
        "translated_bangla": question.replace("সঠিক", "যথাযথ"),
        "code_mixed": question.replace("তথ্য", "information"),
        "banglish": "Bishoytir shothik tottho ki?",
        "english": "What is the correct information about the topic?",
    }


def _campaign_config() -> dict:
    return {
        "dataset_version": "test-campaign-1.0.0",
        "target_records": 100,
        "base_examples": 2,
        "random_seed": 7,
        "source": {"name": "fixture", "license": "MIT"},
        "language_conditions": list(campaign.LANGUAGE_METADATA),
        "evidence_conditions": ["correct", "partially_relevant", "irrelevant", "contradictory", "missing"],
        "generator_models": ["model-a", "model-b"],
        "retrieval_methods": ["bm25", "semfuse"],
        "retrieval_top_k": 2,
        "semfuse": {
            "mode": "hybrid",
            "embedding_model": "test-embedding",
            "embedding_provider": "hashing",
            "fusion_method": "rrf",
            "reranker": None,
        },
        "generation": {
            "temperature": 0.0,
            "max_tokens": 32,
            "prompt_version": "test-v1",
            "concurrency": 4,
        },
        "contradiction_replacements": {
            "qa-0": "বিকল্প তথ্য শূন্য",
            "qa-1": "বিকল্প তথ্য এক",
        },
        "protocol_deviation": {"approved_by": "test"},
    }


class _FakeRetriever:
    embedding_model = None
    reranker = None

    def __init__(self, method: str):
        self.name = f"fake_{method}"
        self.documents = []

    def index(self, documents):
        self.documents = documents

    def retrieve(self, query, top_k=2):
        documents = [
            RetrievedDocument(item.document_id, item.text, rank, 1.0 / rank, item.source_collection)
            for rank, item in enumerate(self.documents[:top_k], start=1)
        ]
        return RetrievalResult(query, documents, {"method": self.name, "top_k": top_k})


def _fake_generate(**kwargs) -> GenerationResult:
    return GenerationResult(
        answer=f"{kwargs['model']}: generated answer",
        model=kwargs["model"],
        temperature=kwargs["temperature"],
        latency_seconds=0.01,
        eval_count=4,
        raw_response="generated answer",
        max_tokens=kwargs["max_tokens"],
    )


def _fake_contradiction_generate(**kwargs) -> GenerationResult:
    kwargs["prompt_template"].format(
        context=kwargs["context"], question=kwargs["question"]
    )
    return GenerationResult(
        answer='{"replacement": "বিকল্প উত্তর"}',
        model=kwargs["model"],
        temperature=kwargs["temperature"],
        latency_seconds=0.01,
        eval_count=8,
        raw_response='{"replacement": "বিকল্প উত্তর"}',
        max_tokens=kwargs["max_tokens"],
        finish_reason="stop",
    )


def test_prepare_generate_and_finalize_campaign(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.json"
    source.write_text(json.dumps(_source_payload(), ensure_ascii=False), encoding="utf-8")
    questions = tmp_path / "questions.jsonl"
    prepared = campaign.prepare_questions(
        source, questions, count=2, seed=7, transformer=_variants
    )
    assert len(prepared) == 2
    assert len(campaign.prepare_questions(source, questions, count=2, seed=7, transformer=_variants)) == 2
    assert len(questions.read_text(encoding="utf-8").splitlines()) == 2

    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(yaml.safe_dump(_campaign_config(), allow_unicode=True), encoding="utf-8")
    monkeypatch.setattr(
        campaign,
        "build_retriever",
        lambda method, **kwargs: _FakeRetriever(method),
    )
    working = tmp_path / "working.jsonl"
    status = campaign.generate_campaign(
        questions, working, config_path, generation_function=_fake_generate
    )
    assert status == {"expected": 100, "completed": 100, "missing": 0, "output": str(working)}
    generated = [json.loads(line) for line in working.read_text(encoding="utf-8").splitlines()]
    assert all("raw_retrieval_documents" in row for row in generated)
    assert all("finish_reason" in row["generation_settings"] for row in generated)
    assert all("truncated" in row["generation_settings"] for row in generated)
    assert all(row["split_group_id"].startswith("article-") for row in generated)
    assert all(row["evidence_topical_relevance"] for row in generated)
    assert all(row["evidence_answerability"] for row in generated)
    assert [row["generator_model"] for row in generated[:50]] == ["model-a"] * 50
    assert [row["generator_model"] for row in generated[50:]] == ["model-b"] * 50

    final = tmp_path / "final.jsonl"
    categories = tmp_path / "categories.jsonl"
    manifest_file = tmp_path / "manifest.json"
    manifest = campaign.finalize_campaign(
        working, final, categories, manifest_file, config_path
    )
    assert manifest["record_count"] == 100
    assert manifest["schema_valid_records"] == 100
    assert manifest["human_annotations_present"] == 0
    assert sum(manifest["counts"]["generator_model"].values()) == 100
    assert manifest["counts"]["retriever"] == {"fake_bm25": 50, "fake_semfuse": 50}
    assert len(final.read_text(encoding="utf-8").splitlines()) == 100
    category_rows = [json.loads(line) for line in categories.read_text(encoding="utf-8").splitlines()]
    assert all(row["count"] == len(row["example_ids"]) for row in category_rows)


def test_variant_validation_rejects_script_leakage():
    variants = _variants("সঠিক তথ্য কী?", "model")
    variants["banglish"] = "বাংলা script"
    try:
        campaign.validate_variants(variants)
    except ValueError as exc:
        assert "banglish" in str(exc)
    else:
        raise AssertionError("Bengali characters in banglish must be rejected")


def test_complete_variants_normalizes_assamese_confusables_before_romanization():
    variants = campaign._complete_variants(
        "কোন সময়ে তেওঁ কাম কৰে?",
        {
            "translated_bangla": "তিনি কখন কাজ করেন?",
            "english": "When does he work?",
        },
    )

    assert not any("\u0980" <= char <= "\u09ff" for char in variants["banglish"])
    assert any("\u0980" <= char <= "\u09ff" for char in variants["code_mixed"])
    assert any("A" <= char <= "z" for char in variants["code_mixed"])


def test_diverse_selection_uses_multiple_balanced_rounds():
    examples = []
    for title_index in range(2):
        for qa_index in range(3):
            answer = f"উত্তর {title_index}-{qa_index}"
            context = f"প্রসঙ্গ {answer}।"
            examples.append(
                campaign.SourceQA(
                    qa_id=f"qa-{title_index}-{qa_index}",
                    title=f"নিবন্ধ {title_index}",
                    source_url="https://example.test",
                    context=context,
                    question=f"নিবন্ধ {title_index} এর উত্তর কী?",
                    answer=answer,
                    answer_start=context.index(answer),
                )
            )

    selected = campaign.select_diverse_examples(examples, count=5, seed=7)

    assert len(selected) == 5
    assert len({item.qa_id for item in selected}) == 5
    counts = Counter(item.title for item in selected)
    assert sorted(counts.values()) == [2, 3]


def test_blocked_language_full_campaign_arithmetic(tmp_path: Path):
    config = _campaign_config()
    config.update(
        {
            "experimental_design": "blocked_language",
            "fixed_retrieval_per_question": True,
            "target_records": 10_000,
            "base_examples": 1_000,
            "retrieval_methods": ["bm25"],
        }
    )
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    loaded = campaign.load_campaign_config(config_path)

    assert loaded["target_records"] == 10_000


def test_banglat5_preparation_resumes_an_incomplete_checkpoint(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.json"
    source.write_text(json.dumps(_source_payload(), ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "questions.jsonl"
    existing = {
        "qa_id": "qa-0",
        "questions": {
            "native_bangla": "বিষয় 0 এর সঠিক তথ্য কী?",
            **_variants("বিষয় 0 এর সঠিক তথ্য কী?", "model"),
        },
    }
    output.write_text(json.dumps(existing, ensure_ascii=False) + "\n", encoding="utf-8")

    class _Tokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def __call__(self, batch, **kwargs):
            return {"input_ids": batch}

        def batch_decode(self, outputs, **kwargs):
            return outputs

    class _Model:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def generate(self, input_ids, **kwargs):
            return ["What is the correct info about the topic?"] * len(input_ids)

    monkeypatch.setattr("transformers.AutoTokenizer", _Tokenizer)
    monkeypatch.setattr("transformers.AutoModelForSeq2SeqLM", _Model)
    monkeypatch.setattr(
        campaign,
        "_translate_batch",
        lambda texts, tokenizer, model: [
            "What is the correct info about the topic?"
            if not any("A" <= char <= "z" for char in text)
            else "বিষয়টির যথাযথ তথ্য কী?"
            for text in texts
        ],
    )

    prepared = campaign.prepare_questions_banglat5(
        source,
        output,
        count=2,
        seed=7,
        bn_en_model_path="unused",
        en_bn_model_path="unused",
    )

    assert len(prepared) == 2
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2


def test_verify_question_file_rejects_digest_mismatch(tmp_path: Path):
    questions = tmp_path / "questions.jsonl"
    questions.write_text("{}\n", encoding="utf-8")

    try:
        campaign.verify_question_file(
            questions, {"question_preparation": {"sha256": "0" * 64}}
        )
    except ValueError as exc:
        assert "question-set SHA-256 mismatch" in str(exc)
    else:
        raise AssertionError("a modified question set must be rejected")


def test_partial_evidence_masks_answer_when_every_sentence_contains_it():
    retrieval = RetrievalResult(
        query="প্রশ্ন?",
        documents=[
            RetrievedDocument("doc", "বন্যা উত্তর।", 1, 1.0, "fixture")
        ],
        configuration={"method": "fixture"},
    )
    row = {"answer": "বন্যা", "title": "বন্যা"}

    evidence = campaign.construct_evidence(
        "partially_relevant", retrieval, row, {"answer": "অন্য"}, 7
    )

    assert evidence["context"] == "এই উৎস নিবন্ধের বিষয়: [বিষয়]।"
    assert "বন্যা" not in evidence["context"]


def test_partial_evidence_removes_answer_from_every_retrieved_passage():
    retrieval = RetrievalResult(
        query="প্রশ্ন?",
        documents=[
            RetrievedDocument("doc-1", "ভূমিকা। উত্তর প্রথম স্থানে।", 1, 1.0, "fixture"),
            RetrievedDocument("doc-2", "উত্তর দ্বিতীয় স্থানেও আছে। শেষ।", 2, 0.5, "fixture"),
        ],
        configuration={"method": "fixture"},
    )
    row = {"answer": "উত্তর", "title": "পরীক্ষা"}

    evidence = campaign.construct_evidence(
        "partially_relevant", retrieval, row, {"answer": "অন্য"}, 7
    )

    assert len(evidence["documents"]) == 2
    assert "উত্তর" not in evidence["context"]


def test_partial_evidence_handles_punctuation_inside_answer_span():
    retrieval = RetrievalResult(
        query="প্রশ্ন?",
        documents=[
            RetrievedDocument(
                "doc",
                "ভূমিকা। E. coli, DNA Pol III হলো প্রধান এনজাইম। শেষ।",
                1,
                1.0,
                "fixture",
            )
        ],
        configuration={"method": "fixture"},
    )
    row = {"answer": "E. coli, DNA Pol III", "title": "ডিএনএ"}

    evidence = campaign.construct_evidence(
        "partially_relevant", retrieval, row, {"answer": "অন্য"}, 7
    )

    assert "E. coli, DNA Pol III" not in evidence["context"]
    assert "প্রধান এনজাইম" not in evidence["context"]


def test_answer_type_matching_rejects_nonsensical_year_replacement():
    assert campaign._same_answer_type("১৭৮৫", "২০১৬", "কত সালে?")
    assert not campaign._same_answer_type("১৭৮৫", "বিকিরণের পথের", "কত সালে?")


def test_answer_type_matching_accepts_different_length_lists_of_same_family():
    assert campaign._same_answer_type(
        "এক, দুই, তিন, চার, পাঁচ, ছয়, সাত",
        "আট, নয়",
        "কোনগুলো?",
        "কোনগুলো?",
    )
    assert not campaign._same_answer_type(
        "পাক্সিল ও প্রোজ্যাক",
        "আট ও নয়",
        "কোন ওষুধ ব্যবহার নিষেধ?",
        "কোনগুলো?",
    )


def test_appearance_contradiction_requires_curated_replacement():
    retrieval = RetrievalResult(
        query="উপরিপৃষ্ঠ কেমন?",
        documents=[
            RetrievedDocument(
                "doc", "উপরিপৃষ্ঠ ইষদ মরিচা ধরা বর্নের।", 1, 1.0, "fixture"
            )
        ],
        configuration={"method": "fixture"},
    )
    row = {
        "answer": "ইষদ মরিচা ধরা বর্নের",
        "title": "আলকুসুম",
        "active_question": "উপরিপৃষ্ঠ কেমন?",
        "questions": {"native_bangla": "উপরিপৃষ্ঠ কেমন?"},
    }
    distractors = [
        {
            "answer": "ভাইরাস",
            "source_text": "ভিন্ন নিবন্ধ।",
            "document_id": "d1",
            "qa_id": "q1",
            "questions": {"native_bangla": "রোগটি কী?"},
        }
    ]

    try:
        campaign.construct_evidence("contradictory", retrieval, row, distractors, 7)
    except ValueError as exc:
        assert "requires a curated contradiction replacement" in str(exc)
    else:
        raise AssertionError("appearance contradictions must fail closed")

    row["contradiction_replacement"] = "উজ্জ্বল নীল বর্ণের"
    evidence = campaign.construct_evidence(
        "contradictory", retrieval, row, distractors, 7
    )
    assert "উজ্জ্বল নীল বর্ণের" in evidence["context"]
    assert evidence["answerability"] == "contradicted"


def test_source_documents_are_deduplicated_by_exact_passage():
    rows = [
        {
            "document_id": "doc-a",
            "source_text": "একই উৎস পাঠ।",
            "source_url": "https://example.test/article",
            "title": "নিবন্ধ",
            "qa_id": "qa-a",
        },
        {
            "document_id": "doc-b",
            "source_text": "একই উৎস পাঠ।",
            "source_url": "https://example.test/article",
            "title": "নিবন্ধ",
            "qa_id": "qa-b",
        },
    ]

    documents = campaign._build_source_documents(rows)

    assert len(documents) == 1
    assert documents[0].document_id == "doc-a"


def test_round_trip_question_provenance_retains_intermediate_question():
    row = {
        "questions": {
            "native_bangla": "মূল প্রশ্ন?",
            "english": "Original question?",
            "translated_bangla": "ফিরে আসা প্রশ্ন?",
        },
        "transformation_prompt_version": "test-v1",
        "translation_models": {"translated_bangla": "bn-en+en-bn"},
        "transformation_model": "bn-en+en-bn",
        "timestamp": "2026-10-13T00:00:00+00:00",
    }

    metadata = campaign._question_generation_metadata(row, "translated_bangla")

    assert metadata["original_question"] == "মূল প্রশ্ন?"
    assert metadata["intermediate_question"] == "Original question?"
    assert metadata["final_question"] == "ফিরে আসা প্রশ্ন?"
    assert metadata["translation_path"] == "bn->en->bn"


def test_strict_jsonl_export_preserves_quotes_and_url(tmp_path: Path):
    output = tmp_path / "export.jsonl"
    value = {"quote": 'তিনি বললেন "হ্যাঁ"', "url": "https://example.test/a:b"}

    campaign._append_jsonl(output, value)

    assert campaign.validate_jsonl_export(output) == 1
    assert json.loads(output.read_text(encoding="utf-8")) == value
    assert "https://" in output.read_text(encoding="utf-8")


def test_contradiction_proposals_are_resumable_and_marked_unreviewed(
    tmp_path: Path,
):
    source = tmp_path / "source.json"
    source.write_text(json.dumps(_source_payload(), ensure_ascii=False), encoding="utf-8")
    questions = tmp_path / "questions.jsonl"
    campaign.prepare_questions(source, questions, count=2, seed=7, transformer=_variants)
    config = _campaign_config()
    config.pop("contradiction_replacements")
    config["contradiction_generation"] = {"models": ["proposal-model"], "max_tokens": 32}
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    proposals = tmp_path / "proposals.jsonl"

    first = campaign.prepare_contradiction_proposals(
        questions,
        proposals,
        config_path,
        generation_function=_fake_contradiction_generate,
    )
    second = campaign.prepare_contradiction_proposals(
        questions,
        proposals,
        config_path,
        generation_function=_fake_contradiction_generate,
    )

    assert first["required"] == 2
    assert first["completed"] == 2
    assert second["completed"] == 2
    rows = [json.loads(line) for line in proposals.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert all(row["review_status"] == "unreviewed" for row in rows)


def test_contradiction_proposal_retries_after_rejected_candidate(tmp_path: Path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps(_source_payload(), ensure_ascii=False), encoding="utf-8")
    questions = tmp_path / "questions.jsonl"
    campaign.prepare_questions(source, questions, count=1, seed=7, transformer=_variants)
    config = _campaign_config()
    config.pop("contradiction_replacements")
    config["base_examples"] = 1
    config["target_records"] = 50
    config["contradiction_generation"] = {"models": ["proposal-model"], "max_tokens": 32}
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    calls = 0

    def generate(**kwargs) -> GenerationResult:
        nonlocal calls
        calls += 1
        answer = '{"replacement": "উত্তর0"}' if calls == 1 else '{"replacement": "বিকল্প উত্তর"}'
        return GenerationResult(
            answer=answer,
            model=kwargs["model"],
            temperature=kwargs["temperature"],
            latency_seconds=0.01,
            eval_count=8,
            raw_response=answer,
            max_tokens=kwargs["max_tokens"],
        )

    proposals = tmp_path / "proposals.jsonl"
    result = campaign.prepare_contradiction_proposals(
        questions,
        proposals,
        config_path,
        generation_function=generate,
    )

    assert result["completed"] == 1
    assert calls == 2
    assert json.loads(proposals.read_text(encoding="utf-8"))["replacement"] == "বিকল্প উত্তর"


def test_contradiction_proposals_collect_all_unresolved_rows(tmp_path: Path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps(_source_payload(), ensure_ascii=False), encoding="utf-8")
    questions = tmp_path / "questions.jsonl"
    campaign.prepare_questions(source, questions, count=2, seed=7, transformer=_variants)
    config = _campaign_config()
    config.pop("contradiction_replacements")
    config["contradiction_generation"] = {
        "models": ["proposal-model"],
        "max_tokens": 32,
        "attempts_per_model": 1,
    }
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    def repeat_original(**kwargs) -> GenerationResult:
        replacement = "উত্তর0" if "উত্তর0" in kwargs["context"] else "উত্তর1"
        answer = json.dumps({"replacement": replacement}, ensure_ascii=False)
        return GenerationResult(
            answer=answer,
            model=kwargs["model"],
            temperature=kwargs["temperature"],
            latency_seconds=0.01,
            eval_count=8,
            raw_response=answer,
            max_tokens=kwargs["max_tokens"],
        )

    proposals = tmp_path / "proposals.jsonl"
    with pytest.raises(RuntimeError, match="2 contradiction proposals remain unresolved"):
        campaign.prepare_contradiction_proposals(
            questions,
            proposals,
            config_path,
            generation_function=repeat_original,
        )

    unresolved = tmp_path / "proposals_unresolved.jsonl"
    rows = [json.loads(line) for line in unresolved.read_text(encoding="utf-8").splitlines()]
    assert [row["qa_id"] for row in rows] == ["qa-0", "qa-1"]


def test_nonmissing_corruptions_preserve_passage_count():
    retrieval = RetrievalResult(
        query="কত সালে?",
        documents=[
            RetrievedDocument("support", "ঘটনাটি ১৭৮৫ সালে ঘটে।", 1, 1.0, "fixture"),
            RetrievedDocument("other", "অন্য একটি প্রাসঙ্গিক বাক্য।", 2, 0.5, "fixture"),
        ],
        configuration={"method": "fixture"},
    )
    row = {
        "answer": "১৭৮৫",
        "title": "ঘটনা",
        "question": "ঘটনাটি কত সালে ঘটে?",
        "questions": {"native_bangla": "ঘটনাটি কত সালে ঘটে?"},
    }
    distractors = [
        {"answer": "২০১৬", "source_text": "ভিন্ন নিবন্ধ এক।", "document_id": "d1", "qa_id": "q1", "questions": {"native_bangla": "কত সালে?"}},
        {"answer": "১৯৬৯", "source_text": "ভিন্ন নিবন্ধ দুই।", "document_id": "d2", "qa_id": "q2", "questions": {"native_bangla": "কত সালে?"}},
    ]

    for condition in ("correct", "partially_relevant", "irrelevant", "contradictory"):
        evidence = campaign.construct_evidence(condition, retrieval, row, distractors, 7)
        assert len(evidence["documents"]) == 2


def test_contradictory_evidence_replaces_answer_in_every_passage():
    retrieval = RetrievalResult(
        query="কত সালে?",
        documents=[
            RetrievedDocument("doc-1", "ঘটনাটি ১৭৮৫ সালে ঘটে।", 1, 1.0, "fixture"),
            RetrievedDocument("doc-2", "অন্য উৎসেও ১৭৮৫ লেখা আছে।", 2, 0.5, "fixture"),
        ],
        configuration={"method": "fixture"},
    )
    row = {
        "answer": "১৭৮৫",
        "question": "ঘটনাটি কত সালে ঘটে?",
        "questions": {"native_bangla": "ঘটনাটি কত সালে ঘটে?"},
    }
    distractors = [
        {
            "answer": "২০১৬",
            "source_text": "ভিন্ন নিবন্ধ।",
            "document_id": "d1",
            "qa_id": "q1",
            "questions": {"native_bangla": "কত সালে?"},
        }
    ]

    evidence = campaign.construct_evidence(
        "contradictory", retrieval, row, distractors, 7
    )

    assert len(evidence["documents"]) == 2
    assert "১৭৮৫" not in evidence["context"]
    assert evidence["context"].count("২০১৬") == 2


def test_review_packet_fails_closed_until_human_approved(tmp_path: Path):
    packet = tmp_path / "review.jsonl"
    packet.write_text(json.dumps({
        "scenario_id": "qa|native_bangla|correct",
        "evidence_condition": "correct",
        "review": {
            "status": "pending",
            "reviewer_id": None,
            "timestamp": None,
            "checks": {},
        },
    }) + "\n", encoding="utf-8")

    try:
        campaign._approved_review_scenarios(packet)
    except ValueError as exc:
        assert "not human-approved" in str(exc)
    else:
        raise AssertionError("pending review must block generation")


def test_review_packet_requires_two_independent_decisions(tmp_path: Path):
    packet = tmp_path / "review.jsonl"
    checks = {
        "question_meaning_preserved": True,
        "answer_not_leaked_by_question": True,
        "evidence_condition_verified": True,
    }
    packet.write_text(json.dumps({
        "scenario_id": "qa|native_bangla|correct",
        "evidence_condition": "correct",
        "review": {
            "status": "approved",
            "decisions": [
                {"reviewer_id": "r1", "decision": "approved", "timestamp": "2026-10-12T00:00:00Z", "checks": checks},
                {"reviewer_id": "r2", "decision": "approved", "timestamp": "2026-10-12T00:01:00Z", "checks": checks},
            ],
        },
    }) + "\n", encoding="utf-8")

    approved = campaign._approved_review_scenarios(packet)

    assert set(approved) == {"qa|native_bangla|correct"}
