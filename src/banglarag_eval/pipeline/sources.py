"""Source document interface for the pilot pipeline.

Loads local Bangla/English documents and provides a uniform interface
for question generation and retrieval. RAGTruth (MIT-licensed) can be
used as an English reference subset — but only after license verification
and only for the English baseline condition, not for Bangla.

The supervisor was explicit:
  "But don't download/translate it into data/ yet. First we need to
   verify the license and decide exactly which subset we can adapt
   for Bangla."

RAGTruth license: MIT (confirmed 2026-08-30).
We use it as a structural reference and for the English baseline only.
Bangla source documents are authored/curated locally.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceDocument:
    """A single source document with provenance."""

    document_id: str
    text: str
    language: str  # "bn", "en", "mixed"
    source_collection: str  # "local_curated", "ragtruth_english", "bangla_wikipedia"
    source_url: str | None = None
    license: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # NFC-normalize Bangla text for consistent downstream processing.
        object.__setattr__(self, "text", unicodedata.normalize("NFC", self.text))

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "text": self.text,
            "language": self.language,
            "source_collection": self.source_collection,
            "source_url": self.source_url,
            "license": self.license,
            "metadata": self.metadata,
        }


def load_local_documents(directory: str | Path) -> list[SourceDocument]:
    """Load documents from a local directory.

    Supports .txt (one document per file) and .jsonl (one document per line
    with fields: document_id, text, language, source_collection).

    Args:
        directory: Path to a directory containing source documents.

    Returns:
        List of SourceDocument instances.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"source directory not found: {dir_path}")

    documents: list[SourceDocument] = []

    for filepath in sorted(dir_path.iterdir()):
        if filepath.suffix == ".txt":
            text = filepath.read_text(encoding="utf-8").strip()
            if not text:
                continue
            doc_id = filepath.stem
            documents.append(SourceDocument(
                document_id=doc_id,
                text=text,
                language="bn",  # default for our curated Bangla docs
                source_collection="local_curated",
                license="local_authored",
            ))
        elif filepath.suffix == ".jsonl":
            with open(filepath, encoding="utf-8") as fh:
                for line_num, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"{filepath}:{line_num}: invalid JSON: {exc}"
                        ) from exc
                    documents.append(SourceDocument(
                        document_id=data["document_id"],
                        text=data["text"],
                        language=data.get("language", "bn"),
                        source_collection=data.get("source_collection", "local_curated"),
                        source_url=data.get("source_url"),
                        license=data.get("license"),
                        metadata=data.get("metadata", {}),
                    ))

    return documents


def load_ragtruth_english(
    source_info_path: str | Path,
    max_documents: int = 50,
) -> list[SourceDocument]:
    """Load English source documents from RAGTruth (MIT licensed).

    This is used ONLY for the English baseline condition. Bangla conditions
    use locally curated documents. The supervisor approved RAGTruth as
    "the best place to start, not necessarily the final dataset."

    Args:
        source_info_path: Path to RAGTruth source_info.jsonl.
        max_documents: Maximum number of documents to load.

    Returns:
        List of SourceDocument instances with RAGTruth provenance.
    """
    path = Path(source_info_path)
    if not path.exists():
        raise FileNotFoundError(f"RAGTruth source_info not found: {path}")

    documents: list[SourceDocument] = []
    seen_ids: set[str] = set()

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            source_id = data["source_id"]
            if source_id in seen_ids:
                continue
            seen_ids.add(source_id)

            documents.append(SourceDocument(
                document_id=f"ragtruth-{source_id}",
                text=data["source_info"],
                language="en",
                source_collection="ragtruth_english",
                source_url="https://github.com/ParticleMedia/RAGTruth",
                license="MIT",
                metadata={
                    "task_type": data.get("task_type"),
                    "source": data.get("source"),
                    "original_prompt": data.get("prompt"),
                    "ragtruth_source_id": source_id,
                },
            ))

            if len(documents) >= max_documents:
                break

    return documents


def split_into_passages(text: str, max_chars: int = 500) -> list[str]:
    """Split a long text into passages of approximately max_chars.

    Tries to split on sentence boundaries. For Bangla, uses danda (।)
    and double-newline as sentence delimiters.

    Args:
        text: The source text to split.
        max_chars: Target maximum characters per passage.

    Returns:
        List of passage strings.
    """
    if len(text) <= max_chars:
        return [text]

    # Split on danda (Bangla full stop), English period+space, or double newline.
    import re
    sentences = re.split(r'(?<=[।.])\s+|\n\n+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    passages: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) > max_chars and current:
            passages.append(" ".join(current))
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len += len(sentence) + 1

    if current:
        passages.append(" ".join(current))

    return passages
