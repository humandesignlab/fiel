"""Corpus loading. The corpus is an INPUT — a JSONL file of `Passage`s.

Swapping corpora is just pointing at a different JSONL. Nothing here is SEC-specific.
"""

from __future__ import annotations

import json
from pathlib import Path

from fiel.schema import Passage


def with_provenance(text: str, *, company: str, filing: str, section: str) -> str:
    """Prepend a one-line provenance header to a passage.

    Real RAG chunks carry source metadata; including it makes entity references
    (e.g. the company name) legitimately grounded, so the judge isn't penalized for
    an attribution it cannot verify from bare passage text. (See the t04 finding.)
    """
    return f"From {company} {filing}, {section}:\n\n{text}"


def load_corpus(path: str | Path) -> dict[str, Passage]:
    """Load passages from a JSONL file, keyed by passage_id for fast lookup."""
    path = Path(path)
    passages: dict[str, Passage] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                passage = Passage.model_validate_json(line)
            except Exception as e:  # noqa: BLE001 - surface the offending line
                raise ValueError(f"{path}:{line_no}: invalid Passage: {e}") from e
            if passage.passage_id in passages:
                raise ValueError(f"{path}:{line_no}: duplicate passage_id {passage.passage_id!r}")
            passages[passage.passage_id] = passage
    return passages


def save_corpus(passages: list[Passage], path: str | Path) -> None:
    """Write passages to JSONL (used by the corpus-build scripts)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in passages:
            f.write(p.model_dump_json() + "\n")
