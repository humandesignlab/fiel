"""Labeled eval set loading + conversion to Inspect samples.

Key design decision (the crux of Fiel): the JUDGE is the system under test, not a
RAG pipeline. So each Inspect `Sample` carries:
  - input   : the question (human-readable in the log)
  - target  : the GROUND-TRUTH label (grounded/ungrounded) we control
  - metadata: passage text + candidate answer (the judge solver builds its prompt
              from these) plus held-out fields (failure_mode, difficulty) used only
              for validation breakdowns — never shown to the judge.
"""

from __future__ import annotations

import json
from pathlib import Path

from inspect_ai.dataset import MemoryDataset, Sample

from fiel.schema import EvalItem, Passage

# The Inspect CLI chdir's into the task file's directory (src/fiel) when running a
# task by path, so relative data paths given at the repo root won't resolve. Anchor
# relative paths to the project root, falling back to the CWD, so both `inspect eval`
# and direct script use work regardless of where they're launched from.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_data_path(path: str | Path) -> Path:
    """Resolve a possibly-relative data path robustly to the caller's intent."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    root_relative = _PROJECT_ROOT / p
    if root_relative.exists():
        return root_relative
    return p  # let the loader raise a clear FileNotFoundError on the original path


def load_eval_set(path: str | Path) -> list[EvalItem]:
    """Load labeled items from a JSONL file, validating each against the schema."""
    path = Path(path)
    items: list[EvalItem] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = EvalItem.model_validate_json(line)
            except Exception as e:  # noqa: BLE001 - surface the offending line
                raise ValueError(f"{path}:{line_no}: invalid EvalItem: {e}") from e
            if item.item_id in seen:
                raise ValueError(f"{path}:{line_no}: duplicate item_id {item.item_id!r}")
            seen.add(item.item_id)
            items.append(item)
    return items


def save_eval_set(items: list[EvalItem], path: str | Path) -> None:
    """Write labeled items to JSONL (used by the eval-set-build scripts)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")


def to_samples(items: list[EvalItem], corpus: dict[str, Passage]) -> list[Sample]:
    """Join items with their source passages and build Inspect samples."""
    samples: list[Sample] = []
    for item in items:
        passage = corpus.get(item.passage_id)
        if passage is None:
            raise ValueError(f"{item.item_id}: passage_id {item.passage_id!r} not in corpus")
        samples.append(
            Sample(
                id=item.item_id,
                input=item.question,
                target=item.label.value,
                metadata={
                    "passage_id": item.passage_id,
                    "passage_text": passage.text,
                    "question": item.question,
                    "answer": item.answer,
                    # Held out from the judge; used for validation breakdowns.
                    "failure_mode": item.failure_mode.value,
                    "difficulty": item.difficulty.value,
                    "complexity": item.complexity.value,
                    "rationale": item.rationale,
                    **{f"src_{k}": v for k, v in passage.metadata.items()},
                },
            )
        )
    return samples


def build_dataset(eval_set_path: str | Path, corpus_path: str | Path) -> MemoryDataset:
    """Convenience: load corpus + eval set from disk and return an Inspect dataset."""
    from fiel.corpus import load_corpus

    corpus = load_corpus(resolve_data_path(corpus_path))
    items = load_eval_set(resolve_data_path(eval_set_path))
    return MemoryDataset(to_samples(items, corpus))
