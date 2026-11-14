"""Reproducible construction of the 10,000-record pre-annotation campaign."""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from bangla_nlpkit import romanize

from banglarag_eval.schema import validate_record

from .generator import GenerationResult, generate
from .retriever import RetrievedDocument, RetrievalResult, build_retriever
from .sources import SourceDocument


LANGUAGE_METADATA = {
    "native_bangla": ("bn", "adapted_existing_benchmark"),
    "translated_bangla": ("bn", "machine_translated"),
    "code_mixed": ("bn_en_mixed", "synthetic_constructed"),
    "banglish": ("banglish", "synthetic_constructed"),
    "english": ("en", "model_generated"),
}

QUESTION_GENERATION_METHOD = {
    "native_bangla": "human_authored_existing_benchmark",
    "translated_bangla": "deterministic_round_trip_machine_translation",
    "code_mixed": "deterministic_lexical_substitution",
    "banglish": "deterministic_nlc_romanization",
    "english": "model_translation",
}

EVIDENCE_PROPERTIES = {
    "correct": ("relevant", "answerable"),
    "partially_relevant": ("relevant", "insufficient"),
    "irrelevant": ("irrelevant", "insufficient"),
    "contradictory": ("relevant", "contradicted"),
    "missing": ("none", "no_context"),
}

TRANSFORM_PROMPT = """/no_think
Transform the Bengali question below without changing its factual meaning.
Return ONLY one valid JSON object with exactly these string keys:
translated_bangla, english.

Rules:
- translated_bangla: translate to English internally, then naturally back-translate to Bengali script.
- english: faithful English translation; do not answer the question.
- Preserve names, dates, numbers, and what the question asks.
- Both values MUST remain questions and end with a question mark; never answer them.

Bengali question: {question}
JSON:"""

LEAKAGE_REPAIR_PROMPT = """/no_think
Translate the Bengali question below without answering it. Return ONLY one valid
JSON object with exactly two string keys: translated_bangla and english.
Both values must be faithful questions ending in a question mark. The gold answer
is shown only as a forbidden string: do not include it, translate it, reveal it,
or turn the question into a statement.

Forbidden gold answer: {answer}
Bengali question: {question}
JSON:"""

QUESTION_TRANSFORM_PROMPT_VERSION = "question_translation_v4_question_contract"
LEAKAGE_REPAIR_PROMPT_VERSION = "question_translation_v5_no_answer_leak"

CAMPAIGN_RAG_PROMPT = """/no_think
Answer the question using only the provided context. Give one or two concise
sentences in the question's language. If the context is missing or insufficient,
say that explicitly. Do not use outside knowledge or invent facts.

Context:
{context}

Question: {question}
Answer:"""

CONTRADICTION_PROPOSAL_PROMPT = """/no_think
Create one plausible but factually false replacement for the original answer.
The replacement will be inserted verbatim into the Bangla source sentence, so
it must fit the same grammatical slot, semantic category, suffix/case, number,
and writing system. Do not repeat the original answer or explain your choice.
Return ONLY one JSON object: {{"replacement": "..."}}

Bangla source:
{context}

Question: {question}
Original answer: {answer}
JSON:"""

CONTRADICTION_MUTATION_PROMPT = """/no_think
Act as a Bengali lexical mutation engine, not as a question-answering system.
Rewrite the original answer by changing its factual content words to a
different plausible value. Preserve its grammatical slot, semantic category,
suffix/case, number, and writing system. The output must differ from the
original and must not add a negation or explanation. Return ONLY one JSON
object: {{"replacement": "..."}}

Question (for answer type only): {question}
Original answer: {answer}
JSON:"""


@dataclass(frozen=True)
class SourceQA:
    qa_id: str
    title: str
    source_url: str
    context: str
    question: str
    answer: str
    answer_start: int

    @property
    def document_id(self) -> str:
        return f"bq-{self.qa_id}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_campaign_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    language_factor = (
        1
        if config.get("experimental_design", "full_factorial") == "blocked_language"
        else len(config["language_conditions"])
    )
    expected = (
        config["base_examples"]
        * language_factor
        * len(config["evidence_conditions"])
        * len(config["generator_models"])
    )
    if expected != config["target_records"]:
        raise ValueError(
            f"campaign factorial produces {expected} records, not "
            f"target_records={config['target_records']}"
        )
    if not config["retrieval_methods"]:
        raise ValueError("at least one retrieval method is required")
    if config.get("fixed_retrieval_per_question") and len(config["retrieval_methods"]) != 1:
        raise ValueError("fixed_retrieval_per_question requires exactly one retriever")
    return config


def verify_source_file(path: str | Path, config: dict[str, Any]) -> str:
    """Verify the raw source against the frozen SHA-256 digest."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    expected = config["source"].get("sha256")
    if expected and digest != expected:
        raise ValueError(f"source SHA-256 mismatch: expected {expected}, found {digest}")
    return digest


def verify_question_file(path: str | Path, config: dict[str, Any]) -> str:
    """Verify prepared questions when the campaign config pins their digest."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    expected = config.get("question_preparation", {}).get("sha256")
    if expected and digest != expected:
        raise ValueError(
            f"question-set SHA-256 mismatch: expected {expected}, found {digest}"
        )
    return digest


def load_banglaquad(path: str | Path) -> list[SourceQA]:
    """Load valid, span-grounded examples from BanglaQuAD's SQuAD JSON."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    examples: list[SourceQA] = []
    seen_ids: set[str] = set()
    seen_triples: set[tuple[str, str, str]] = set()
    for article in payload["data"]:
        title = unicodedata.normalize("NFC", article["title"].strip())
        source_url = article.get("url", "")
        for paragraph in article["paragraphs"]:
            context = unicodedata.normalize("NFC", paragraph["context"].strip())
            for qa in paragraph.get("qas", []):
                if not qa.get("answers"):
                    continue
                qa_id = str(qa["id"])
                question = unicodedata.normalize("NFC", qa["question"].strip())
                answer = unicodedata.normalize("NFC", qa["answers"][0]["text"].strip())
                if not question or not answer or not context or qa_id in seen_ids:
                    continue
                answer_start = context.find(answer)
                triple = (question, answer, context)
                if answer_start < 0 or triple in seen_triples:
                    continue
                if not re.search(r"[\u0980-\u09ff]", question):
                    continue
                seen_ids.add(qa_id)
                seen_triples.add(triple)
                examples.append(
                    SourceQA(
                        qa_id=qa_id,
                        title=title,
                        source_url=source_url,
                        context=context,
                        question=question,
                        answer=answer,
                        answer_start=answer_start,
                    )
                )
    return examples


def select_diverse_examples(
    examples: list[SourceQA], count: int, seed: int
) -> list[SourceQA]:
    """Select distinct QAs in seeded, article-balanced rounds."""
    by_title: dict[str, list[SourceQA]] = defaultdict(list)
    for example in examples:
        by_title[example.title].append(example)
    if count > len(examples):
        raise ValueError(f"only {len(examples)} distinct QAs available; need {count}")
    rng = random.Random(seed)
    titles = sorted(by_title)
    rng.shuffle(titles)
    ranked = {
        title: sorted(
            candidates,
            key=lambda item: (_question_quality_penalty(item), item.qa_id),
        )
        for title, candidates in by_title.items()
    }
    selected: list[SourceQA] = []
    round_index = 0
    while len(selected) < count:
        added = 0
        for title in titles:
            candidates = ranked[title]
            if round_index >= len(candidates):
                continue
            selected.append(candidates[round_index])
            added += 1
            if len(selected) == count:
                return sorted(selected, key=lambda item: item.qa_id)
        if not added:
            break
        round_index += 1
    raise ValueError(f"only {len(selected)} distinct QAs available; need {count}")


def _question_quality_penalty(example: SourceQA) -> int:
    """Prefer concise, punctuated questions with nontrivial answer spans."""
    question = example.question
    length = len(question)
    penalty = abs(length - 70)
    if not question.rstrip().endswith(("?", "？")):
        penalty += 1_000
    if not 25 <= length <= 130:
        penalty += 300
    penalty += question.count("  ") * 20
    if len(example.answer.strip()) < 2:
        penalty += 200
    return penalty


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("transformation model did not return a JSON object")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("transformation output must be an object")
    return value


def validate_variants(variants: dict[str, Any]) -> dict[str, str]:
    required = {"translated_bangla", "code_mixed", "banglish", "english"}
    if set(variants) != required:
        raise ValueError(f"question variants must have exactly {sorted(required)}")
    cleaned = {key: unicodedata.normalize("NFC", str(value).strip()) for key, value in variants.items()}
    if any(not value for value in cleaned.values()):
        raise ValueError("question variants cannot be empty")
    has_bn = lambda value: bool(re.search(r"[\u0980-\u09ff]", value))
    has_latin = lambda value: bool(re.search(r"[A-Za-z]", value))
    if not has_bn(cleaned["translated_bangla"]):
        raise ValueError("translated_bangla must use Bengali script")
    if not (has_bn(cleaned["code_mixed"]) and has_latin(cleaned["code_mixed"])):
        raise ValueError("code_mixed must contain Bengali and Latin scripts")
    if has_bn(cleaned["banglish"]) or not has_latin(cleaned["banglish"]):
        raise ValueError("banglish must use Latin script without Bengali characters")
    if has_bn(cleaned["english"]) or not has_latin(cleaned["english"]):
        raise ValueError("english must use Latin script without Bengali characters")
    for key in ("translated_bangla", "english"):
        if not cleaned[key].endswith(("?", "？")):
            raise ValueError(f"{key} must remain a question ending in a question mark")
    return cleaned


def _repair_code_mixed(value: str, original_question: str) -> str:
    """Construct controlled lexical code-mixing from the original question."""
    del value
    candidate = original_question
    already_mixed = bool(
        re.search(r"[\u0980-\u09ff]", candidate) and re.search(r"[A-Za-z]", candidate)
    )
    replacements = (
        ("কোনগুলি", "which ones"),
        ("কোনটিকেও", "which one-ও"),
        ("কোনটিকে", "which one-কে"),
        ("কোনগুলো", "which ones"),
        ("কতজন", "how many people"),
        ("কতক্ষণ", "how long"),
        ("কতবার", "how many times"),
        ("কতরও বেশি", "more than how many"),
        ("কোথাকার", "from where"),
        ("কোথা থেকে", "from where"),
        ("কয় তালার", "how many floors"),
        ("কি কি", "what things"),
        ("কী কী", "what things"),
        ("কতগুলি", "how many"),
        ("কতগুলো", "how many"),
        ("কতটি", "how many"),
        ("কয়টি", "how many"),
        ("কতদিন", "how long"),
        ("কততম", "which numbered"),
        ("কতটুকু", "how much"),
        ("কততে", "what number-এ"),
        ("কত সালে", "which year-এ"),
        ("কোন সালে", "which year-এ"),
        ("কত শতকের", "which century-র"),
        ("কোনটি জন্য", "what purpose-এ"),
        ("কোনটির", "which one-এর"),
        ("কোনটি", "which one"),
        ("কোনগুলোকে", "which ones-কে"),
        ("কোণটি", "which one"),
        ("কিসে কিসে", "in what things"),
        ("কিসের", "what"),
        ("কিসে", "in what"),
        ("কাকে", "whom"),
        ("কাদের", "whose"),
        ("কারা", "who"),
        ("কার", "whose"),
        ("কীভাবে", "how"),
        ("কিভাবে", "how"),
        ("রাজধানী", "capital city"),
        ("নাম", "name"),
        ("কোথায়", "where"),
        ("কবে", "when"),
        ("কখন", "when"),
        ("কেন", "why"),
        ("কেমন", "how"),
        ("কত", "how much"),
        ("কোন", "which"),
        ("তথ্য", "information"),
        ("অর্থ", "meaning"),
        ("ব্যবহার", "use"),
        ("কাজ", "work"),
        ("নতুন", "new"),
        ("প্রধান", "main"),
        ("প্রকাশিত", "published"),
        ("প্রভাবিত", "influenced"),
        ("বাজারজাত", "marketed"),
        ("কয়েকটি বিষয়ে", "several topics-এ"),
        ("বছর", "year"),
        ("শতাব্দী", "century"),
        ("কে", "who"),
        ("কি", "what"),
        ("কী", "what"),
    )
    for bangla, english in replacements:
        pattern = rf"(?<![\u0980-\u09ff]){re.escape(bangla)}(?![\u0980-\u09ff])"
        if re.search(pattern, candidate):
            return re.sub(pattern, english, candidate, count=1)
    if already_mixed:
        return candidate
    raise ValueError(f"no controlled code-mix substitution matched {original_question!r}")


def _complete_variants(question: str, translations: dict[str, Any]) -> dict[str, str]:
    if set(translations) != {"translated_bangla", "english"}:
        raise ValueError(
            "transformation output must contain exactly translated_bangla and english"
        )
    variants = dict(translations)
    variants["code_mixed"] = _repair_code_mixed("", question)
    normalized_question = question.translate(str.maketrans({"ৰ": "র", "ৱ": "ব"}))
    variants["banglish"] = unicodedata.normalize(
        "NFC", romanize(normalized_question).strip()
    )
    return validate_variants(variants)


def transform_question(question: str, model: str, retries: int = 1) -> dict[str, str]:
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            result = generate(
                question=question,
                context="",
                model=model,
                temperature=0.0,
                max_tokens=256,
                timeout=300,
                prompt_template=TRANSFORM_PROMPT,
            )
            return _complete_variants(question, _extract_json_object(result.answer))
        except (ValueError, json.JSONDecodeError, RuntimeError, TimeoutError) as exc:
            last_error = exc
    raise RuntimeError(f"failed to transform question after {retries} attempts: {last_error}")


def _answer_leaks(
    answer: str, native_question: str, variants: dict[str, str]
) -> bool:
    """Detect a newly introduced exact answer token in a transformed question."""
    answer = answer.strip()
    if not answer or answer.casefold() in native_question.casefold():
        return False
    pattern = rf"(?<![\w\u0980-\u09ff]){re.escape(answer)}(?![\w\u0980-\u09ff])"
    return any(re.search(pattern, value, flags=re.I) for value in variants.values())


def transform_question_avoiding_answer(
    question: str, answer: str, model: str, retries: int = 2
) -> dict[str, str]:
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            result = generate(
                question=question,
                context="",
                model=model,
                temperature=0.0,
                max_tokens=256,
                timeout=300,
                prompt_template=LEAKAGE_REPAIR_PROMPT.replace(
                    "{answer}", answer.replace("{", "{{").replace("}", "}}")
                ),
            )
            variants = _complete_variants(question, _extract_json_object(result.answer))
            if _answer_leaks(answer, question, variants):
                raise ValueError("transformed question reveals the forbidden answer")
            return variants
        except (ValueError, json.JSONDecodeError, RuntimeError, TimeoutError) as exc:
            last_error = exc
    raise RuntimeError(f"answer-safe translation failed after {retries} attempts: {last_error}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return records


def _serialize_json_line(value: dict[str, Any]) -> str:
    """Serialize one strict JSON object and verify its round trip."""
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    decoded = json.loads(encoded)
    if decoded != value:
        raise ValueError("JSON serialization did not round-trip exactly")
    return encoded + "\n"


def validate_jsonl_export(path: str | Path) -> int:
    """Strictly parse every nonblank JSONL row and return the row count."""
    records = _read_jsonl(Path(path))
    if any(not isinstance(record, dict) for record in records):
        raise ValueError(f"{path}: every JSONL row must be an object")
    return len(records)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(_serialize_json_line(value))
        handle.flush()


def prepare_questions(
    source_path: str | Path,
    output_path: str | Path,
    *,
    count: int,
    seed: int,
    transform_model: str = "qwen3:8b",
    fallback_transform_model: str | None = "gemma3:4b",
    reuse_translations_path: str | Path | None = None,
    transformer: Callable[[str, str], dict[str, str]] = transform_question,
) -> list[dict[str, Any]]:
    """Select and transform source QAs, resuming from an append-only JSONL."""
    output = Path(output_path)
    existing = _read_jsonl(output)
    by_id = {item["qa_id"]: item for item in existing}
    selected = select_diverse_examples(load_banglaquad(source_path), count, seed)
    selected_ids = {item.qa_id for item in selected}
    unknown = set(by_id) - selected_ids
    if unknown:
        raise ValueError(f"question checkpoint contains non-selected ids: {sorted(unknown)[:3]}")
    reuse_by_id: dict[str, dict[str, Any]] = {}
    if reuse_translations_path:
        reuse_by_id = {
            item["qa_id"]: item for item in _read_jsonl(Path(reuse_translations_path))
        }

    for index, example in enumerate(selected, start=1):
        if example.qa_id in by_id:
            continue
        actual_transform_model = transform_model
        transformation_prompt_version = QUESTION_TRANSFORM_PROMPT_VERSION
        reused = reuse_by_id.get(example.qa_id)
        variants: dict[str, str] | None = None
        if reused:
            try:
                variants = _complete_variants(
                    example.question,
                    {
                        "translated_bangla": reused["questions"]["translated_bangla"],
                        "english": reused["questions"]["english"],
                    },
                )
                actual_transform_model = reused["transformation_model"]
                transformation_prompt_version = reused["transformation_prompt_version"]
            except (KeyError, ValueError):
                variants = None
        if variants is None:
            try:
                variants = transformer(example.question, transform_model)
            except RuntimeError:
                if not fallback_transform_model or fallback_transform_model == transform_model:
                    raise
                variants = transformer(example.question, fallback_transform_model)
                actual_transform_model = fallback_transform_model
        variants = validate_variants(variants)
        if _answer_leaks(example.answer, example.question, variants):
            try:
                variants = transform_question_avoiding_answer(
                    example.question, example.answer, transform_model
                )
                actual_transform_model = transform_model
            except RuntimeError:
                if not fallback_transform_model or fallback_transform_model == transform_model:
                    raise
                variants = transform_question_avoiding_answer(
                    example.question, example.answer, fallback_transform_model
                )
                actual_transform_model = fallback_transform_model
            transformation_prompt_version = LEAKAGE_REPAIR_PROMPT_VERSION
        record = {
            "qa_id": example.qa_id,
            "document_id": example.document_id,
            "title": example.title,
            "source_url": example.source_url,
            "source_text": example.context,
            "answer": example.answer,
            "answer_start": example.answer_start,
            "questions": {"native_bangla": example.question, **variants},
            "transformation_model": actual_transform_model,
            "transformation_primary_model": transform_model,
            "transformation_prompt_version": transformation_prompt_version,
            "timestamp": _now(),
        }
        _append_jsonl(output, record)
        by_id[example.qa_id] = record
        print(f"[questions {index}/{count}] {example.qa_id}")
    return [by_id[item.qa_id] for item in selected]


def _load_translation_overrides(path: str | Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    overrides = payload.get("overrides", payload)
    if not isinstance(overrides, dict):
        raise ValueError("translation overrides must be a mapping keyed by qa_id")
    return overrides


def _translate_batch(
    texts: list[str], tokenizer: Any, model: Any, batch_size: int = 16
) -> list[str]:
    translated: list[str] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**encoded, max_length=128, num_beams=4)
        translated.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    return [unicodedata.normalize("NFC", value.strip()) for value in translated]


def prepare_questions_banglat5(
    source_path: str | Path,
    output_path: str | Path,
    *,
    count: int,
    seed: int,
    bn_en_model_path: str | Path,
    en_bn_model_path: str | Path,
    overrides_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Prepare audited variants with pinned bidirectional BanglaT5 checkpoints."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    output = Path(output_path)
    selected = select_diverse_examples(load_banglaquad(source_path), count, seed)
    existing = _read_jsonl(output)
    by_id = {item["qa_id"]: item for item in existing}
    if len(by_id) != len(existing):
        raise ValueError("existing MT checkpoint contains duplicate QA IDs")
    selected_ids = {item.qa_id for item in selected}
    unknown = set(by_id) - selected_ids
    if unknown:
        raise ValueError(f"MT checkpoint contains non-selected ids: {sorted(unknown)[:3]}")
    missing = [item for item in selected if item.qa_id not in by_id]
    if not missing:
        return [by_id[item.qa_id] for item in selected]

    bn_en_path = str(bn_en_model_path)
    en_bn_path = str(en_bn_model_path)
    bn_en_tokenizer = AutoTokenizer.from_pretrained(
        bn_en_path, use_fast=False, local_files_only=True
    )
    bn_en_model = AutoModelForSeq2SeqLM.from_pretrained(
        bn_en_path, local_files_only=True
    )
    english = _translate_batch(
        [item.question for item in missing], bn_en_tokenizer, bn_en_model
    )
    del bn_en_model

    en_bn_tokenizer = AutoTokenizer.from_pretrained(
        en_bn_path, use_fast=False, local_files_only=True
    )
    en_bn_model = AutoModelForSeq2SeqLM.from_pretrained(
        en_bn_path, local_files_only=True
    )
    round_trip = _translate_batch(english, en_bn_tokenizer, en_bn_model)
    del en_bn_model

    overrides = _load_translation_overrides(overrides_path)
    model_id = "csebuetnlp/banglat5_nmt_bn_en+csebuetnlp/banglat5_nmt_en_bn"
    selected_positions = {item.qa_id: index for index, item in enumerate(selected, start=1)}
    for index, (example, english_question, translated_bangla) in enumerate(
        zip(missing, english, round_trip, strict=True), start=1
    ):
        override = overrides.get(example.qa_id, {})
        translations = {
            "translated_bangla": override.get("translated_bangla", translated_bangla),
            "english": override.get("english", english_question),
        }
        for key in translations:
            if not translations[key].endswith(("?", "？")):
                translations[key] += "?"
        variants = _complete_variants(example.question, translations)
        for key in ("code_mixed", "banglish"):
            if key in override:
                variants[key] = override[key]
        variants = validate_variants(variants)
        if _answer_leaks(example.answer, example.question, variants):
            raise ValueError(f"translation reveals answer for QA {example.qa_id}")
        timestamp = _now()
        record = {
            "qa_id": example.qa_id,
            "document_id": example.document_id,
            "title": example.title,
            "source_url": example.source_url,
            "source_text": example.context,
            "answer": example.answer,
            "answer_start": example.answer_start,
            "questions": {"native_bangla": example.question, **variants},
            "transformation_model": model_id,
            "transformation_primary_model": "csebuetnlp/banglat5_nmt_bn_en",
            "translation_models": {
                "english": "csebuetnlp/banglat5_nmt_bn_en",
                "translated_bangla": model_id,
            },
            "transformation_prompt_version": (
                "banglat5_beam4_manual_override_v1"
                if override
                else "banglat5_beam4_v1"
            ),
            "timestamp": timestamp,
        }
        _append_jsonl(output, record)
        by_id[example.qa_id] = record
        completed = len(existing) + index
        if completed % 25 == 0:
            position = selected_positions[example.qa_id]
            print(f"[BanglaT5 questions {completed}/{count}; selected position {position}]")
    return [by_id[item.qa_id] for item in selected]


def repair_prepared_question_variants(
    input_path: str | Path,
    output_path: str | Path,
    overrides_path: str | Path,
) -> list[dict[str, Any]]:
    """Rebuild deterministic variants from frozen translations plus reviewed overrides."""
    rows = _read_jsonl(Path(input_path))
    overrides = _load_translation_overrides(overrides_path)
    repaired: list[dict[str, Any]] = []
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    for row in rows:
        override = overrides.get(row["qa_id"], {})
        translations = {
            "translated_bangla": override.get(
                "translated_bangla", row["questions"]["translated_bangla"]
            ),
            "english": override.get("english", row["questions"]["english"]),
        }
        variants = _complete_variants(row["questions"]["native_bangla"], translations)
        for key in ("code_mixed", "banglish"):
            if key in override:
                variants[key] = override[key]
        row["questions"] = {
            "native_bangla": row["questions"]["native_bangla"],
            **validate_variants(variants),
        }
        row["transformation_prompt_version"] = "question_variants_reviewed_v3"
        row["variant_repair_timestamp"] = _now()
        _append_jsonl(output, row)
        repaired.append(row)
    return repaired


def _supporting_passage(source_text: str, answer: str, window: int = 700) -> str:
    start = source_text.find(answer)
    if start < 0:
        raise ValueError("answer is not present in source text")
    left = max(0, start - window)
    right = min(len(source_text), start + len(answer) + window)
    return source_text[left:right].strip()


def _ground_retrieval(
    retrieval: RetrievalResult,
    question_row: dict[str, Any],
    top_k: int,
) -> tuple[RetrievalResult, bool]:
    answer = question_row["answer"]
    source_text = unicodedata.normalize("NFC", question_row["source_text"]).strip()
    if any(
        unicodedata.normalize("NFC", document.text).strip() == source_text
        and answer in document.text
        for document in retrieval.documents
    ):
        return retrieval, False
    injected = RetrievedDocument(
        document_id=f"{question_row['document_id']}-oracle",
        text=_supporting_passage(question_row["source_text"], answer),
        rank=1,
        score=1.0,
        source="BanglaQuAD-v1.0:oracle_support",
    )
    documents = [injected]
    for document in retrieval.documents[: max(0, top_k - 1)]:
        documents.append(
            RetrievedDocument(
                document_id=document.document_id,
                text=document.text,
                rank=len(documents) + 1,
                score=document.score,
                source=document.source,
            )
        )
    configuration = dict(retrieval.configuration)
    configuration["oracle_source_injected"] = True
    return RetrievalResult(retrieval.query, documents, configuration), True


def _document_dict(document: RetrievedDocument, text: str | None = None) -> dict[str, Any]:
    return {
        "document_id": document.document_id,
        "text": document.text if text is None else text,
        "rank": document.rank,
        "score": document.score,
        "source": document.source,
    }


_BN_DIGIT_TRANSLATION = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def _answer_type(answer: str, question: str = "") -> str:
    """Return a conservative answer type used only to propose contradictions."""
    normalized = answer.translate(_BN_DIGIT_TRANSLATION).strip()
    lowered_question = question.translate(_BN_DIGIT_TRANSLATION).lower()
    if re.fullmatch(r"(?:1[0-9]{3}|20[0-9]{2})", normalized):
        return "year"
    if re.search(r"[0-9]{1,2}\s+(?:জানুয়ারি|ফেব্রুয়ারি|মার্চ|এপ্রিল|মে|জুন|জুলাই|আগস্ট|সেপ্টেম্বর|অক্টোবর|নভেম্বর|ডিসেম্বর)", normalized):
        return "date"
    if normalized in {"এক", "একটি", "দুই", "দুটি", "দুটো", "তিন", "তিনটি", "চার", "চারটি"}:
        return "number"
    if re.fullmatch(r"[0-9][0-9.,:/%-]*(?:\s+\S+){0,2}", normalized):
        return "number"
    list_parts = [part.strip() for part in re.split(r"\s+(?:ও|and)\s+|[,;]", answer) if part.strip()]
    if len(list_parts) > 1:
        prefix = "medicine_" if re.search(r"ওষুধ|ঔষধ|medicine|drug", lowered_question) else ""
        return f"{prefix}list:{len(list_parts)}"
    if re.search(r"কত সালে|কোন সালে|কবে|what year|when", lowered_question):
        return "date"
    if re.search(
        r"(?<![\u0980-\u09ff])(?:কে|কার|কাকে)(?![\u0980-\u09ff])|\bwho\b|\bwhom\b",
        lowered_question,
    ):
        return "person"
    if re.search(r"কোথায়|কোথায়|কোন স্থানে|where", lowered_question):
        return "location"
    if re.search(
        r"কেমন|রঙ|রং|বর্ণ|বর্ন|দেখতে|colour|color|appearance",
        lowered_question,
    ) or re.search(
        r"রঙের|রংয়ের|বর্ণের|বর্নের|কালো|সাদা|লাল|নীল|সবুজ|হলুদ|ধূসর|মরিচা",
        normalized,
    ):
        return "appearance"
    return "text"


def _same_answer_type(
    answer: str,
    candidate: str,
    question: str,
    candidate_question: str = "",
) -> bool:
    expected = _answer_type(answer, question)
    observed = _answer_type(candidate, candidate_question)
    expected_family = expected.split(":", 1)[0]
    observed_family = observed.split(":", 1)[0]
    return expected_family == observed_family and answer.strip() != candidate.strip()


def _materialize_documents(
    retrieval: RetrievalResult,
    replacements: dict[str, tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    replacements = replacements or {}
    documents = []
    for document in retrieval.documents:
        text, suffix = replacements.get(document.document_id, (document.text, ""))
        item = _document_dict(document, text)
        if suffix:
            item["document_id"] = f"{document.document_id}-{suffix}"
            item["source"] = "controlled_corruption"
        documents.append(item)
    return documents


def _context_from_documents(documents: list[dict[str, Any]]) -> str:
    return "\n\n".join(document["text"] for document in documents)


def _detect_answer_language(text: str) -> str:
    has_bangla = any("\u0980" <= char <= "\u09ff" for char in text)
    has_latin = bool(re.search(r"[A-Za-z]", text))
    if has_bangla and has_latin:
        return "bn_en_mixed"
    if has_bangla:
        return "bn"
    if has_latin:
        return "en_or_banglish"
    return "und"


def construct_evidence(
    condition: str,
    retrieval: RetrievalResult,
    question_row: dict[str, Any],
    distractor_row: dict[str, Any] | list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    """Create auditable evidence whose stored condition is mechanically true."""
    answer = question_row["answer"]
    original_ids = [document.document_id for document in retrieval.documents]
    question = question_row.get(
        "active_question",
        question_row.get("questions", {}).get("native_bangla", question_row.get("question", "")),
    )
    if condition == "correct":
        documents = _materialize_documents(retrieval)
        return {
            "context": _context_from_documents(documents),
            "documents": documents,
            "topical_relevance": EVIDENCE_PROPERTIES[condition][0],
            "answerability": EVIDENCE_PROPERTIES[condition][1],
            "notes": "Retrieved evidence contains the exact human-authored answer span.",
            "corruption": None,
        }
    if condition == "missing":
        return {
            "context": "",
            "documents": [],
            "topical_relevance": EVIDENCE_PROPERTIES[condition][0],
            "answerability": EVIDENCE_PROPERTIES[condition][1],
            "notes": "All evidence removed before generation.",
            "corruption": {
                "corruption_type": "evidence_removal",
                "original_context_ids": original_ids,
                "modified_context_ids": [],
                "operation_description": "Removed every retrieved passage.",
                "random_seed": seed,
            },
        }
    if condition == "irrelevant":
        distractors = distractor_row if isinstance(distractor_row, list) else [distractor_row]
        if len(distractors) < len(retrieval.documents):
            distractors = (distractors * len(retrieval.documents))[: len(retrieval.documents)]
        documents = []
        for rank, row in enumerate(distractors[: len(retrieval.documents)], start=1):
            text = row["source_text"]
            if answer in text:
                raise ValueError("irrelevant evidence contains the intended answer span")
            documents.append({
                "document_id": f"{row['document_id']}-irrelevant-{rank}",
                "text": text,
                "rank": rank,
                "score": 0.0,
                "source": "BanglaQuAD-v1.0:controlled_irrelevant",
            })
        return {
            "context": _context_from_documents(documents),
            "documents": documents,
            "topical_relevance": EVIDENCE_PROPERTIES[condition][0],
            "answerability": EVIDENCE_PROPERTIES[condition][1],
            "notes": "Replaced each retrieved passage with a passage from a different article while preserving passage count.",
            "corruption": {
                "corruption_type": "cross_article_replacement",
                "original_context_ids": original_ids,
                "modified_context_ids": [item["document_id"] for item in documents],
                "operation_description": "Replaced every passage with a deterministically selected different-article passage while preserving rank count.",
                "random_seed": seed,
            },
        }

    support = next((item for item in retrieval.documents if answer in item.text), None)
    if support is None:
        raise ValueError("grounded retrieval does not contain the answer span")
    if condition == "partially_relevant":
        replacements: dict[str, tuple[str, str]] = {}
        masked_title = question_row["title"].replace(answer, "[বিষয়]")
        for document in retrieval.documents:
            if answer not in document.text:
                continue
            sentinel = "__ANNOTATED_ANSWER_SPAN__"
            masked_text = document.text.replace(answer, sentinel)
            sentences = re.split(r"(?<=[।.!?])\s+", masked_text)
            kept = [sentence for sentence in sentences if sentinel not in sentence]
            modified = " ".join(kept).strip()
            if not modified:
                modified = f"এই উৎস নিবন্ধের বিষয়: {masked_title}।"
            replacements[document.document_id] = (modified, "partial")
        documents = _materialize_documents(retrieval, replacements)
        if answer in _context_from_documents(documents):
            raise ValueError("partial evidence still contains the answer span")
        return {
            "context": _context_from_documents(documents),
            "documents": documents,
            "topical_relevance": EVIDENCE_PROPERTIES[condition][0],
            "answerability": EVIDENCE_PROPERTIES[condition][1],
            "notes": "Same-article context remains topically related, but every answer-bearing sentence was removed; the context is insufficient for this question and requires human verification.",
            "corruption": {
                "corruption_type": "answer_sentence_removal",
                "original_context_ids": original_ids,
                "modified_context_ids": [item["document_id"] for item in documents],
                "operation_description": "Removed every sentence containing the exact annotated answer span.",
                "random_seed": seed,
            },
        }
    if condition == "contradictory":
        candidates = distractor_row if isinstance(distractor_row, list) else [distractor_row]
        manual_replacement = question_row.get("contradiction_replacement")
        answer_type = _answer_type(answer, question)
        answer_family = answer_type.split(":", 1)[0]
        if manual_replacement:
            distractor = manual_replacement
            if answer_family != "text" and not _same_answer_type(
                answer, distractor, question, question
            ):
                raise ValueError("manual contradiction replacement is not answer-type compatible")
            selected_qa_id = "manual_review_candidate"
        else:
            if answer_family in {"text", "appearance"}:
                raise ValueError(
                    f"answer type {answer_type!r} requires a curated contradiction replacement"
                )
            typed = [
                row
                for row in candidates
                if _same_answer_type(
                    answer,
                    row["answer"],
                    question,
                    row.get("questions", {}).get("native_bangla", ""),
                )
            ]
            if not typed:
                raise ValueError(
                    f"no type-compatible contradiction for answer type {_answer_type(answer, question)!r}"
                )
            selected = typed[0]
            distractor = selected["answer"]
            selected_qa_id = selected["qa_id"]
        replacements = {
            document.document_id: (
                document.text.replace(answer, distractor),
                "contradiction",
            )
            for document in retrieval.documents
            if answer in document.text
        }
        documents = _materialize_documents(retrieval, replacements)
        if answer in _context_from_documents(documents):
            raise ValueError("contradictory evidence still contains the intended answer span")
        return {
            "context": _context_from_documents(documents),
            "documents": documents,
            "topical_relevance": EVIDENCE_PROPERTIES[condition][0],
            "answerability": EVIDENCE_PROPERTIES[condition][1],
            "notes": "Every occurrence of the answer span was replaced by a semantically typed candidate; plausibility, grammar, and factual contradiction require human review.",
            "corruption": {
                "corruption_type": "answer_span_substitution",
                "original_context_ids": original_ids,
                "modified_context_ids": [item["document_id"] for item in documents],
                "operation_description": f"Replaced {_answer_type(answer, question)} answer {answer!r} with {distractor!r} from {selected_qa_id}.",
                "random_seed": seed,
                "replacement_provenance": question_row.get(
                    "contradiction_replacement_metadata"
                ),
            },
        }
    raise ValueError(f"unknown evidence condition: {condition}")


def _scenario_assignments(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> list[tuple[dict[str, Any], str, str, str]]:
    if config.get("experimental_design", "full_factorial") == "blocked_language":
        languages = config["language_conditions"]
        scenarios = [
            (row, languages[index % len(languages)], evidence)
            for index, row in enumerate(rows)
            for evidence in config["evidence_conditions"]
        ]
    else:
        scenarios = [
            (row, language, evidence)
            for row in rows
            for language in config["language_conditions"]
            for evidence in config["evidence_conditions"]
        ]
    methods = config["retrieval_methods"]
    by_cell: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, (_, language, evidence) in enumerate(scenarios):
        by_cell[(language, evidence)].append(index)
    method_by_index: dict[int, str] = {}
    for cell in sorted(by_cell):
        cell_seed = int(
            hashlib.sha256(f"{config['random_seed']}|{cell[0]}|{cell[1]}".encode()).hexdigest()[:16],
            16,
        )
        indices = by_cell[cell]
        random.Random(cell_seed).shuffle(indices)
        for position, scenario_index in enumerate(indices):
            method_by_index[scenario_index] = methods[position % len(methods)]
    return [(*scenario, method_by_index[index]) for index, scenario in enumerate(scenarios)]


def _example_id(
    base_index: int, language: str, evidence: str, model: str, prefix: str = "c10k"
) -> str:
    digest = hashlib.sha1(model.encode("utf-8")).hexdigest()[:6]
    return f"{prefix}-{base_index:03d}-{language[:4]}-{evidence[:4]}-{digest}"


def _build_source_documents(rows: list[dict[str, Any]]) -> list[SourceDocument]:
    """Build a deterministic corpus with exact duplicate passages removed."""
    documents: list[SourceDocument] = []
    seen_texts: set[str] = set()
    for row in rows:
        normalized_text = unicodedata.normalize("NFC", row["source_text"]).strip()
        if normalized_text in seen_texts:
            continue
        seen_texts.add(normalized_text)
        documents.append(
            SourceDocument(
                document_id=row["document_id"],
                text=row["source_text"],
                language="bn",
                source_collection="BanglaQuAD-v1.0",
                source_url=row["source_url"],
                license="MIT",
                metadata={"title": row["title"], "qa_id": row["qa_id"]},
            )
        )
    return documents


def _select_distractor(
    rows: list[dict[str, Any]], scenario_index: int, answer: str
) -> dict[str, Any]:
    """Choose a deterministic cross-article distractor without answer leakage."""
    for shift in range(len(rows)):
        candidate = rows[(scenario_index + 73 + shift) % len(rows)]
        if answer in candidate["source_text"] or answer in candidate["answer"]:
            continue
        return candidate
    raise ValueError(f"no leakage-free distractor available for answer {answer!r}")


def _select_distractors(
    rows: list[dict[str, Any]],
    scenario_index: int,
    question_row: dict[str, Any],
    count: int,
    question: str | None = None,
) -> list[dict[str, Any]]:
    """Select distinct cross-article candidates, preferring answer-type matches."""
    answer = question_row["answer"]
    question = question or question_row["questions"]["native_bangla"]
    candidates = []
    for shift in range(len(rows)):
        candidate = rows[(scenario_index + 73 + shift) % len(rows)]
        if candidate["document_id"] == question_row["document_id"]:
            continue
        if answer in candidate["source_text"] or answer in candidate["answer"]:
            continue
        candidates.append(candidate)
    typed = [
        candidate
        for candidate in candidates
        if _same_answer_type(
            answer,
            candidate["answer"],
            question,
            candidate.get("questions", {}).get("native_bangla", ""),
        )
    ]
    typed_qa_ids = {candidate["qa_id"] for candidate in typed}
    ordered = typed + [candidate for candidate in candidates if candidate["qa_id"] not in typed_qa_ids]
    if not ordered:
        raise ValueError(f"no leakage-free distractors for answer {answer!r}")
    if len(ordered) < count:
        ordered = (ordered * ((count + len(ordered) - 1) // len(ordered)))[:count]
    return ordered[:count]


def _question_generation_metadata(row: dict[str, Any], language: str) -> dict[str, Any]:
    model_generated = language in {"translated_bangla", "english"}
    prompt_versions = {
        "translated_bangla": row["transformation_prompt_version"],
        "english": row["transformation_prompt_version"],
        "code_mixed": "code_mix_lexical_v2",
        "banglish": "bangla_nlpkit_nlc_v0.1.1_confusable_normalization_v1",
        "native_bangla": None,
    }
    translation_paths = {
        "translated_bangla": "bn->en->bn",
        "english": "bn->en",
        "code_mixed": None,
        "banglish": None,
        "native_bangla": None,
    }
    return {
        "method": QUESTION_GENERATION_METHOD[language],
        "model": (
            row.get("translation_models", {}).get(language, row["transformation_model"])
            if model_generated
            else None
        ),
        "model_version": (
            row.get("translation_models", {}).get(language, row["transformation_model"])
            if model_generated
            else None
        ),
        "prompt_version": prompt_versions[language],
        "timestamp": row["timestamp"],
        "original_question": row["questions"]["native_bangla"],
        "intermediate_question": (
            row["questions"]["english"] if language == "translated_bangla" else None
        ),
        "final_question": row["questions"][language],
        "translation_path": translation_paths[language],
    }


def _split_group_id(row: dict[str, Any]) -> str:
    """Group every QA and variant from one source article into one split unit."""
    article_key = row.get("source_url") or row["title"]
    digest = hashlib.sha256(article_key.encode("utf-8")).hexdigest()[:16]
    return f"article-{digest}"


def _load_contradiction_replacements(
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    replacements: dict[str, dict[str, Any]] = {
        qa_id: {
            "replacement": replacement,
            "origin": "curated_config",
            "review_status": "pending",
        }
        for qa_id, replacement in config.get("contradiction_replacements", {}).items()
    }
    proposal_path_value = config.get("contradiction_replacements_file")
    if not proposal_path_value:
        return replacements
    proposal_path = Path(proposal_path_value)
    proposals = _read_jsonl(proposal_path)
    expected_digest = config.get("contradiction_replacements_sha256")
    actual_digest = hashlib.sha256(proposal_path.read_bytes()).hexdigest()
    if expected_digest and actual_digest != expected_digest:
        raise ValueError(
            "contradiction-proposal SHA-256 mismatch: "
            f"expected {expected_digest}, found {actual_digest}"
        )
    for proposal in proposals:
        qa_id = proposal["qa_id"]
        if qa_id in replacements:
            raise ValueError(f"duplicate contradiction replacement for {qa_id}")
        if (
            proposal.get("review_status") == "unreviewed"
            and not config.get("allow_unreviewed_contradiction_proposals", False)
        ):
            raise ValueError(
                f"unreviewed contradiction proposal is not allowed for {qa_id}"
            )
        replacements[qa_id] = {
            "replacement": proposal["replacement"],
            "origin": proposal.get("origin", "model_proposed"),
            "review_status": proposal.get("review_status", "unreviewed"),
            "generator_model": proposal.get("generator_model"),
            "prompt_version": proposal.get("prompt_version"),
            "proposal_file_sha256": actual_digest,
        }
    return replacements


def prepare_contradiction_proposals(
    questions_path: str | Path,
    output_path: str | Path,
    config_path: str | Path,
    *,
    generation_function: Callable[..., GenerationResult] = generate,
) -> dict[str, Any]:
    """Propose resumable, explicitly unreviewed contradictions for open answers."""
    config = load_campaign_config(config_path)
    rows = _campaign_rows(questions_path, config)[: config["base_examples"]]
    output = Path(output_path)
    unresolved_output = output.with_name(f"{output.stem}_unresolved.jsonl")
    existing = _read_jsonl(output)
    proposals: dict[str, dict[str, Any]] = {}
    for proposal in existing:
        qa_id = proposal["qa_id"]
        if qa_id in proposals:
            raise ValueError(f"duplicate contradiction proposal for {qa_id}")
        proposals[qa_id] = proposal

    languages = config["language_conditions"]
    configured = config.get("contradiction_replacements", {})
    required_rows: list[tuple[dict[str, Any], str, str]] = []
    for index, row in enumerate(rows):
        language = languages[index % len(languages)]
        question = row["questions"][language]
        answer_type = _answer_type(row["answer"], question)
        if answer_type.split(":", 1)[0] not in {"text", "appearance"}:
            continue
        if row["qa_id"] in configured:
            continue
        required_rows.append((row, question, answer_type))

    models = config.get("contradiction_generation", {}).get(
        "models", [config["generator_models"][0]]
    )
    contradiction_generation = config.get("contradiction_generation", {})
    max_tokens = contradiction_generation.get("max_tokens", 96)
    temperature = float(contradiction_generation.get("temperature", 0.2))
    attempts_per_model = int(contradiction_generation.get("attempts_per_model", 5))
    unresolved: list[dict[str, Any]] = []
    for position, (row, question, answer_type) in enumerate(required_rows, start=1):
        if row["qa_id"] in proposals:
            proposal = proposals[row["qa_id"]]
            if proposal.get("original_answer") != row["answer"]:
                raise ValueError(f"stale contradiction proposal for {row['qa_id']}")
            existing_replacement = str(proposal.get("replacement", "")).strip()
            if not existing_replacement or existing_replacement == row["answer"]:
                raise ValueError(f"invalid contradiction proposal for {row['qa_id']}")
            if answer_type.split(":", 1)[0] == "appearance" and not _same_answer_type(
                row["answer"], existing_replacement, question, question
            ):
                raise ValueError(f"type-incompatible contradiction proposal for {row['qa_id']}")
            continue
        last_error: Exception | None = None
        replacement = ""
        result: GenerationResult | None = None
        used_model = ""
        rejected_candidates: list[str] = []
        for model_index, model in enumerate(models):
            for attempt_index in range(attempts_per_model):
                try:
                    prompt_template = (
                        CONTRADICTION_PROPOSAL_PROMPT
                        if attempt_index == 0
                        else CONTRADICTION_MUTATION_PROMPT
                    )
                    prompt = prompt_template.replace(
                        "{answer}", row["answer"].replace("{", "{{").replace("}", "}}")
                    )
                    rejected_note = ", ".join(rejected_candidates[-5:]) or "none"
                    rejected_note = rejected_note.replace("{", "{{").replace("}", "}}")
                    prompt += (
                        f"\nDeterministic attempt: {model_index + 1}.{attempt_index + 1}."
                        f"\nRejected candidates (never return these): {rejected_note}"
                    )
                    result = generation_function(
                        question=question,
                        context=_supporting_passage(row["source_text"], row["answer"], 350),
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=300,
                        prompt_template=prompt,
                    )
                    payload = _extract_json_object(result.answer)
                    replacement = str(payload.get("replacement", "")).strip()
                    if not replacement or "\n" in replacement or len(replacement) > 240:
                        raise ValueError("replacement is empty, multiline, or too long")
                    if row["answer"] in replacement or replacement == row["answer"]:
                        raise ValueError("replacement repeats the original answer")
                    if answer_type.split(":", 1)[0] == "appearance" and not _same_answer_type(
                        row["answer"], replacement, question, question
                    ):
                        raise ValueError("appearance replacement is not type compatible")
                    used_model = model
                    break
                except (ValueError, RuntimeError, TimeoutError, ConnectionError) as exc:
                    last_error = exc
                    if replacement and replacement not in rejected_candidates:
                        rejected_candidates.append(replacement)
                    replacement = ""
            if replacement:
                break
        if not replacement or result is None:
            unresolved.append({
                "qa_id": row["qa_id"],
                "document_id": row["document_id"],
                "question": question,
                "original_answer": row["answer"],
                "answer_type": answer_type,
                "error": str(last_error),
                "timestamp": _now(),
            })
            print(
                f"[contradictions unresolved] {row['qa_id']}: {last_error}"
            )
            continue
        proposal = {
            "qa_id": row["qa_id"],
            "document_id": row["document_id"],
            "question": question,
            "original_answer": row["answer"],
            "answer_type": answer_type,
            "replacement": replacement,
            "origin": "model_proposed",
            "review_status": "unreviewed",
            "generator_model": used_model,
            "prompt_version": "contradiction_candidate_v1",
            "generation_settings": {
                "temperature": result.temperature,
                "max_tokens": result.max_tokens,
                "finish_reason": result.finish_reason,
                "truncated": result.truncated,
            },
            "timestamp": _now(),
        }
        _append_jsonl(output, proposal)
        proposals[row["qa_id"]] = proposal
        if position % 25 == 0:
            print(f"[contradictions {position}/{len(required_rows)}]")

    validate_jsonl_export(output)
    if unresolved:
        unresolved_output.parent.mkdir(parents=True, exist_ok=True)
        with open(unresolved_output, "w", encoding="utf-8") as handle:
            for item in unresolved:
                handle.write(_serialize_json_line(item))
        validate_jsonl_export(unresolved_output)
        raise RuntimeError(
            f"{len(unresolved)} contradiction proposals remain unresolved; "
            f"see {unresolved_output}"
        )
    if unresolved_output.exists():
        unresolved_output.unlink()
    return {
        "required": len(required_rows),
        "completed": len(proposals),
        "output": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def _campaign_rows(questions_path: str | Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    verify_question_file(questions_path, config)
    rows = _read_jsonl(Path(questions_path))
    if len(rows) < config["base_examples"]:
        raise ValueError(
            f"expected at least {config['base_examples']} prepared questions, found {len(rows)}"
        )
    return rows


def _build_campaign_retrievers(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    retrievers: dict[str, Any] = {}
    documents = _build_source_documents(rows)
    for method in config["retrieval_methods"]:
        kwargs: dict[str, Any] = {}
        if method == "semfuse":
            semfuse = config["semfuse"]
            kwargs = {
                "semfuse_mode": semfuse["mode"],
                "semfuse_embedding_model": semfuse["embedding_model"],
                "semfuse_embedding_provider": semfuse["embedding_provider"],
                "semfuse_fusion_method": semfuse["fusion_method"],
                "semfuse_reranker": semfuse["reranker"],
            }
        retriever = build_retriever(method, **kwargs)
        retriever.index(documents)
        retrievers[method] = retriever
    return retrievers


def _prepare_scenarios(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    retrievers: dict[str, Any],
    *,
    candidate_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    candidate_rows = candidate_rows or rows
    contradiction_replacements = _load_contradiction_replacements(config)
    retrieval_cache: dict[tuple[str, str, str], tuple[RetrievalResult, RetrievalResult, bool]] = {}
    prepared: list[dict[str, Any]] = []
    for scenario_index, (row, language, evidence_condition, method) in enumerate(
        _scenario_assignments(rows, config)
    ):
        question = row["questions"][language]
        cache_key = (method, row["qa_id"], language)
        if cache_key not in retrieval_cache:
            raw = retrievers[method].retrieve(question, top_k=config["retrieval_top_k"])
            grounded, injected = _ground_retrieval(raw, row, config["retrieval_top_k"])
            grounded.configuration["raw_source_hit"] = not injected
            grounded.configuration.setdefault("oracle_source_injected", injected)
            retrieval_cache[cache_key] = (raw, grounded, injected)
        raw_retrieval, grounded, injected = retrieval_cache[cache_key]
        distractors = _select_distractors(
            candidate_rows,
            scenario_index,
            row,
            max(config["retrieval_top_k"], len(grounded.documents)),
            question,
        )
        construction_seed = config["random_seed"] + scenario_index
        replacement_entry = contradiction_replacements.get(row["qa_id"])
        scenario_row = {
            **row,
            "active_question": question,
            "contradiction_replacement": (
                replacement_entry["replacement"] if replacement_entry else None
            ),
            "contradiction_replacement_metadata": replacement_entry,
        }
        try:
            evidence = construct_evidence(
                evidence_condition, grounded, scenario_row, distractors, construction_seed
            )
        except ValueError as exc:
            raise ValueError(
                f"failed to construct {row['qa_id']}|{language}|{evidence_condition}: {exc}"
            ) from exc
        prepared.append({
            "scenario_id": f"{row['qa_id']}|{language}|{evidence_condition}",
            "row": row,
            "language": language,
            "evidence_condition": evidence_condition,
            "method": method,
            "question": question,
            "raw_retrieval": raw_retrieval,
            "retrieval": grounded,
            "oracle_source_injected": injected,
            "evidence": evidence,
            "construction_seed": construction_seed,
        })
    return prepared


def prepare_review_packet(
    questions_path: str | Path,
    output_path: str | Path,
    config_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write construction candidates that humans must approve before generation."""
    config = load_campaign_config(config_path)
    corpus_rows = _campaign_rows(questions_path, config)
    rows = corpus_rows[: config["base_examples"]]
    retrievers = _build_campaign_retrievers(corpus_rows, config)
    scenarios = _prepare_scenarios(
        rows, config, retrievers, candidate_rows=corpus_rows
    )
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"review packet already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        for scenario in scenarios:
            row = scenario["row"]
            raw = scenario["raw_retrieval"]
            packet = {
                "scenario_id": scenario["scenario_id"],
                "qa_id": row["qa_id"],
                "document_id": row["document_id"],
                "split_group_id": _split_group_id(row),
                "language_condition": scenario["language"],
                "evidence_condition": scenario["evidence_condition"],
                "question": scenario["question"],
                "native_question": row["questions"]["native_bangla"],
                "question_generation": _question_generation_metadata(
                    row, scenario["language"]
                ),
                "intended_answer": row["answer"],
                "answer_type": _answer_type(row["answer"], scenario["question"]),
                "raw_retrieval_documents": [_document_dict(item) for item in raw.documents],
                "raw_retrieval_configuration": raw.configuration,
                "oracle_source_injected": scenario["oracle_source_injected"],
                "evidence": scenario["evidence"],
                "review": {
                    "status": "pending",
                    "required_checks": [
                        "question_meaning_preserved",
                        "answer_not_leaked_by_question",
                        "evidence_condition_verified",
                        "answer_not_inferable_for_partial",
                        "contradiction_is_type_compatible",
                        "result_is_grammatical",
                    ],
                    "decisions": [],
                    "notes": None,
                },
            }
            handle.write(_serialize_json_line(packet))
    validate_jsonl_export(output)
    return {
        "records": len(scenarios),
        "pending_review": len(scenarios),
        "output": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def _approved_review_scenarios(path: str | Path) -> dict[str, dict[str, Any]]:
    packets = _read_jsonl(Path(path))
    approved: dict[str, dict[str, Any]] = {}
    for packet in packets:
        review = packet.get("review", {})
        if review.get("status") != "approved":
            raise ValueError(f"scenario {packet.get('scenario_id')} is not human-approved")
        decisions = review.get("decisions", [])
        reviewer_ids = [decision.get("reviewer_id") for decision in decisions]
        if len(decisions) < 2 or len(reviewer_ids) != len(set(reviewer_ids)) or None in reviewer_ids:
            raise ValueError(
                f"scenario {packet.get('scenario_id')} requires two independent reviewers"
            )
        required_checks = {"question_meaning_preserved", "answer_not_leaked_by_question", "evidence_condition_verified"}
        if packet.get("evidence_condition") == "partially_relevant":
            required_checks.add("answer_not_inferable_for_partial")
        if packet.get("evidence_condition") == "contradictory":
            required_checks.update({"contradiction_is_type_compatible", "result_is_grammatical"})
        for decision in decisions:
            if decision.get("decision") != "approved" or not decision.get("timestamp"):
                raise ValueError(
                    f"scenario {packet.get('scenario_id')} has an incomplete reviewer decision"
                )
            checks = decision.get("checks", {})
            missing = sorted(key for key in required_checks if checks.get(key) is not True)
            if missing:
                raise ValueError(
                    f"scenario {packet.get('scenario_id')} failed review checks for "
                    f"{decision.get('reviewer_id')}: {missing}"
                )
        scenario_id = packet["scenario_id"]
        if scenario_id in approved:
            raise ValueError(f"duplicate scenario in review packet: {scenario_id}")
        approved[scenario_id] = packet
    return approved


def generate_campaign(
    questions_path: str | Path,
    output_path: str | Path,
    config_path: str | Path,
    *,
    generation_function: Callable[..., GenerationResult] = generate,
) -> dict[str, Any]:
    """Generate paired model outputs, resuming safely by example_id."""
    config = load_campaign_config(config_path)
    if str(config.get("campaign_status", "")).startswith("paused"):
        raise ValueError(
            "campaign is paused for methodology repair and cannot be resumed"
        )
    corpus_rows = _campaign_rows(questions_path, config)
    rows = corpus_rows[: config["base_examples"]]
    retrievers = _build_campaign_retrievers(corpus_rows, config)

    output = Path(output_path)
    existing = _read_jsonl(output)
    mismatched_versions = {
        record.get("dataset_version")
        for record in existing
        if record.get("dataset_version") != config["dataset_version"]
    }
    if mismatched_versions:
        raise ValueError(
            "working checkpoint belongs to a different dataset version: "
            f"{sorted(str(value) for value in mismatched_versions)}"
        )
    completed = {record["example_id"] for record in existing}
    expected_ids: set[str] = set()
    total = config["target_records"]
    generated = len(existing)
    generation_concurrency = int(config["generation"].get("concurrency", 1))
    if generation_concurrency < 1:
        raise ValueError("generation.concurrency must be at least 1")
    prepared_scenarios = _prepare_scenarios(
        rows, config, retrievers, candidate_rows=corpus_rows
    )
    review_config = config.get("review_gate", {})
    approved_reviews: dict[str, dict[str, Any]] = {}
    review_digest = None
    if review_config.get("required"):
        review_path = Path(review_config["file"])
        approved_reviews = _approved_review_scenarios(review_path)
        review_digest = hashlib.sha256(review_path.read_bytes()).hexdigest()
        expected_review_digest = review_config.get("sha256")
        if expected_review_digest and review_digest != expected_review_digest:
            raise ValueError(
                "review-packet SHA-256 mismatch: "
                f"expected {expected_review_digest}, found {review_digest}"
            )
        expected_scenarios = {item["scenario_id"] for item in prepared_scenarios}
        if set(approved_reviews) != expected_scenarios:
            missing = sorted(expected_scenarios - set(approved_reviews))
            extra = sorted(set(approved_reviews) - expected_scenarios)
            raise ValueError(f"review packet scenario mismatch; missing={missing[:3]}, extra={extra[:3]}")

    # Process one model at a time so a 16 GB local machine does not swap
    # Qwen and Gemma weights for every paired scenario.
    row_indices = {id(row): index for index, row in enumerate(rows)}
    for model in config["generator_models"]:
        pending: list[tuple[dict[str, Any], str]] = []
        for scenario in prepared_scenarios:
            row = scenario["row"]
            base_index = row_indices[id(row)]
            example_id = _example_id(
                base_index,
                scenario["language"],
                scenario["evidence_condition"],
                model,
                config.get("example_id_prefix", "c10k"),
            )
            if example_id in expected_ids:
                raise RuntimeError(f"duplicate expected example_id: {example_id}")
            expected_ids.add(example_id)
            if example_id not in completed:
                pending.append((scenario, example_id))

        def generate_record(
            job: tuple[dict[str, Any], str],
        ) -> tuple[dict[str, Any], str, str, str]:
            scenario, example_id = job
            row = scenario["row"]
            language = scenario["language"]
            evidence_condition = scenario["evidence_condition"]
            method = scenario["method"]
            question = scenario["question"]
            raw_retrieval = scenario["raw_retrieval"]
            retrieval = scenario["retrieval"]
            evidence = scenario["evidence"]
            construction_seed = scenario["construction_seed"]
            construction_review = None
            if approved_reviews:
                packet = approved_reviews[scenario["scenario_id"]]
                if packet["question"] != question:
                    raise ValueError(f"reviewed question changed for {scenario['scenario_id']}")
                evidence = packet["evidence"]
                construction_review = {
                    **packet["review"],
                    "packet_sha256": review_digest,
                }
            context = evidence["context"] or "(no context provided)"
            last_error: Exception | None = None
            for _ in range(3):
                try:
                    result = generation_function(
                        question=question,
                        context=context,
                        model=model,
                        temperature=config["generation"]["temperature"],
                        max_tokens=config["generation"]["max_tokens"],
                        timeout=300,
                        prompt_template=CAMPAIGN_RAG_PROMPT,
                    )
                    if not result.answer.strip():
                        raise RuntimeError("generator returned an empty answer")
                    break
                except (ConnectionError, RuntimeError, TimeoutError) as exc:
                    last_error = exc
            else:
                raise RuntimeError(
                    f"generation failed three times for {example_id}: {last_error}"
                )
            timestamp = _now()
            start = row["answer_start"]
            question_language, data_origin = LANGUAGE_METADATA[language]
            record = {
                "schema_version": "1.1.0",
                "example_id": example_id,
                "document_id": row["document_id"],
                "split_group_id": _split_group_id(row),
                "source_id": "BanglaQuAD-v1.0",
                "question": question,
                "question_language": question_language,
                "language_condition": language,
                "data_origin": data_origin,
                "source_text": row["source_text"],
                "source_span": {"start_char": start, "end_char": start + len(row["answer"]), "text": row["answer"]},
                "intended_answer": row["answer"],
                "retrieved_context": evidence["context"],
                "retrieval_documents": evidence["documents"],
                "raw_retrieval_documents": [_document_dict(item) for item in raw_retrieval.documents],
                "raw_retrieval_configuration": raw_retrieval.configuration,
                "retrieval_configuration": retrieval.configuration,
                "evidence_condition": evidence_condition,
                "evidence_condition_notes": evidence["notes"],
                "evidence_topical_relevance": evidence["topical_relevance"],
                "evidence_answerability": evidence["answerability"],
                "generated_answer": result.answer,
                "raw_generated_output": result.raw_response,
                "answer_claims": [],
                "faithfulness_category": None,
                "factual_correctness_category": None,
                "response_behavior": None,
                "hallucination_type": None,
                "annotations": [],
                "adjudication": None,
                "construction_review": construction_review,
                "construction_review_status": (
                    "approved" if construction_review is not None else "unreviewed"
                ),
                "generator_model": result.model,
                "generator_version": result.model,
                "generation_settings": {
                    "temperature": result.temperature,
                    "seed": 42,
                    "max_tokens": result.max_tokens,
                    "token_count": result.eval_count,
                    "finish_reason": result.finish_reason,
                    "truncated": result.truncated,
                    "prompt_template": config["generation"]["prompt_version"],
                },
                "context_language": "none" if not evidence["context"] else "bn",
                "answer_language": _detect_answer_language(result.answer),
                "embedding_model": retrievers[method].embedding_model,
                "retriever": retrievers[method].name,
                "reranker": retrievers[method].reranker,
                "prompt_version": config["generation"]["prompt_version"],
                "question_generation": _question_generation_metadata(row, language),
                "corruption_metadata": evidence["corruption"],
                "evaluator_outputs": [],
                "dataset_version": config["dataset_version"],
                "timestamp": timestamp,
                "random_seed": construction_seed,
                "latency": result.latency_seconds,
                "cost": 0.0,
                "notes": (
                    f"Source: {row['source_url']} | BanglaQuAD MIT | "
                    + (
                        "construction review required."
                        if config.get("review_gate", {}).get("required")
                        else "protocol deviation recorded in campaign manifest."
                    )
                ),
            }
            validation = validate_record(record)
            if not validation.is_valid:
                raise ValueError(f"invalid record {example_id}: {validation.errors}")
            return record, language, evidence_condition, method

        if generation_concurrency == 1:
            results = map(generate_record, pending)
            executor = None
        else:
            executor = ThreadPoolExecutor(max_workers=generation_concurrency)
            results = executor.map(generate_record, pending)

        try:
            for record, language, evidence_condition, method in results:
                example_id = record["example_id"]
                # executor.map yields in input order, preserving deterministic JSONL order.
                if example_id in completed:
                    continue
                _append_jsonl(output, record)
                completed.add(example_id)
                generated += 1
                if generated % 25 == 0:
                    print(f"[campaign {generated}/{total}] {language}/{evidence_condition}/{method}/{model}")
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)

    missing = expected_ids - completed
    return {"expected": total, "completed": len(completed), "missing": len(missing), "output": str(output)}


def _validate_campaign_integrity(
    records: list[dict[str, Any]], config: dict[str, Any]
) -> None:
    """Enforce the frozen factorial, paired contexts, and evidence contracts."""
    base_count = config["base_examples"]
    languages = config["language_conditions"]
    conditions = config["evidence_conditions"]
    models = config["generator_models"]
    blocked = config.get("experimental_design", "full_factorial") == "blocked_language"
    language_base = base_count // len(languages) if blocked else base_count
    expected_counts = {
        "language_condition": language_base * len(conditions) * len(models),
        "evidence_condition": base_count * (1 if blocked else len(languages)) * len(models),
        "generator_model": base_count * (1 if blocked else len(languages)) * len(conditions),
        "document_id": (1 if blocked else len(languages)) * len(conditions) * len(models),
    }
    expected_values = {
        "language_condition": set(languages),
        "evidence_condition": set(conditions),
        "generator_model": set(models),
    }
    for field, expected_per_value in expected_counts.items():
        counts = Counter(str(record[field]) for record in records)
        if field in expected_values and set(counts) != expected_values[field]:
            raise ValueError(f"unexpected {field} categories: {sorted(counts)}")
        if any(count != expected_per_value for count in counts.values()):
            raise ValueError(f"unbalanced {field} counts: {dict(sorted(counts.items()))}")
    if len({record["document_id"] for record in records}) != base_count:
        raise ValueError("campaign does not contain the configured number of base documents")

    retriever_counts = Counter(str(record["retriever"]) for record in records)
    expected_retriever_count = config["target_records"] // len(config["retrieval_methods"])
    if len(retriever_counts) != len(config["retrieval_methods"]) or any(
        count != expected_retriever_count for count in retriever_counts.values()
    ):
        raise ValueError(f"unbalanced retriever counts: {dict(sorted(retriever_counts.items()))}")

    paired: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    raw_inputs: dict[tuple[str, str], set[str]] = defaultdict(set)
    shown_counts: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    for record in records:
        answer = record["intended_answer"]
        context = record["retrieved_context"]
        condition = record["evidence_condition"]
        if condition == "correct" and answer not in context:
            raise ValueError(f"correct evidence misses answer: {record['example_id']}")
        if condition in {"partially_relevant", "irrelevant", "contradictory"} and answer in context:
            raise ValueError(f"corrupted evidence leaks answer: {record['example_id']}")
        if condition == "missing" and (context or record["retrieval_documents"]):
            raise ValueError(f"missing evidence is not empty: {record['example_id']}")
        if record["annotations"] or record["adjudication"] is not None:
            raise ValueError(f"pre-annotation record contains human labels: {record['example_id']}")
        if config.get("review_gate", {}).get("required") and not record.get("construction_review"):
            raise ValueError(f"record lacks required construction review: {record['example_id']}")
        raw_inputs[(record["document_id"], record["language_condition"])].add(
            json.dumps(record.get("raw_retrieval_documents", []), ensure_ascii=False, sort_keys=True)
        )
        shown_counts[(record["document_id"], record["language_condition"], condition)].add(
            len(record["retrieval_documents"])
        )
        paired[(record["document_id"], record["language_condition"], condition)].append(record)

    if config.get("fixed_retrieval_per_question") and any(
        len(values) != 1 for values in raw_inputs.values()
    ):
        raise ValueError("evidence variants do not share one frozen raw retrieval result")
    for (document_id, language), _ in raw_inputs.items():
        nonmissing = {
            next(iter(shown_counts[(document_id, language, condition)]))
            for condition in conditions
            if condition != "missing"
        }
        if len(nonmissing) != 1:
            raise ValueError(
                f"passage count changed across non-missing conditions for {document_id}/{language}"
            )

    expected_pairs = base_count * (1 if blocked else len(languages)) * len(conditions)
    if len(paired) != expected_pairs:
        raise ValueError(f"expected {expected_pairs} paired scenarios, found {len(paired)}")
    for key, pair in paired.items():
        if len(pair) != len(models) or {item["generator_model"] for item in pair} != set(models):
            raise ValueError(f"incomplete generator pair for scenario {key}")
        shared = {
            json.dumps(
                {
                    "question": item["question"],
                    "retrieved_context": item["retrieved_context"],
                    "retrieval_documents": item["retrieval_documents"],
                    "retriever": item["retriever"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            for item in pair
        }
        if len(shared) != 1:
            raise ValueError(f"paired generators received different inputs for scenario {key}")


def finalize_campaign(
    working_path: str | Path,
    final_path: str | Path,
    categories_path: str | Path,
    manifest_path: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    """Validate, sort, and publish one canonical JSONL plus category index."""
    config = load_campaign_config(config_path)
    records = _read_jsonl(Path(working_path))
    ids = [record["example_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("working campaign contains duplicate example_id values")
    if len(records) != config["target_records"]:
        raise ValueError(f"campaign incomplete: {len(records)}/{config['target_records']} records")
    invalid = []
    for record in records:
        result = validate_record(record)
        if not result.is_valid:
            invalid.append({"example_id": record["example_id"], "errors": result.errors})
    if invalid:
        raise ValueError(f"campaign has {len(invalid)} invalid records; first={invalid[0]}")
    _validate_campaign_integrity(records, config)

    records.sort(key=lambda item: item["example_id"])
    final = Path(final_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    with open(final, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(_serialize_json_line(record))
    validate_jsonl_export(final)

    dimensions = {
        "language_condition": "language_condition",
        "evidence_condition": "evidence_condition",
        "generator_model": "generator_model",
        "retriever": "retriever",
    }
    categories: list[dict[str, Any]] = []
    for dimension, field in dimensions.items():
        grouped: dict[str, list[str]] = defaultdict(list)
        for record in records:
            grouped[str(record[field])].append(record["example_id"])
        for category, example_ids in sorted(grouped.items()):
            categories.append(
                {
                    "dimension": dimension,
                    "category": category,
                    "count": len(example_ids),
                    "example_ids": example_ids,
                }
            )
    categories_file = Path(categories_path)
    categories_file.parent.mkdir(parents=True, exist_ok=True)
    with open(categories_file, "w", encoding="utf-8") as handle:
        for category in categories:
            handle.write(_serialize_json_line(category))
    validate_jsonl_export(categories_file)

    digest = hashlib.sha256(final.read_bytes()).hexdigest()
    manifest = {
        "dataset_version": config["dataset_version"],
        "record_count": len(records),
        "unique_example_ids": len(set(ids)),
        "schema_valid_records": len(records),
        "human_annotations_present": 0,
        "sha256": digest,
        "source": config["source"],
        "question_preparation": config.get("question_preparation", {}),
        "generator_artifacts": config.get("model_artifacts", {}),
        "generation": config["generation"],
        "retrieval": {
            "methods": config["retrieval_methods"],
            "top_k": config["retrieval_top_k"],
            "semfuse": config.get("semfuse"),
            "fixed_per_question": config.get("fixed_retrieval_per_question", False),
            "oracle_kept_separate": True,
        },
        "split_grouping": {
            "field": "split_group_id",
            "unit": "source_article",
            "requirement": "All QAs and evidence/language variants from one article stay in one split.",
        },
        "protocol_deviation": config["protocol_deviation"],
        "counts": {
            field: dict(sorted(Counter(str(record[field]) for record in records).items()))
            for field in dimensions.values()
        },
        "files": {"records": str(final), "categories": str(categories_file)},
        "created_at": _now(),
    }
    manifest_file = Path(manifest_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
