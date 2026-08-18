"""Print selected eval items joined with their source passage, for human review.

Run: python scripts/show_items.py f005 f045 f051 ...
Defaults to a curated representative sample of hard adversarial items.
"""

from __future__ import annotations

import sys

from fiel.corpus import load_corpus
from fiel.dataset import load_eval_set

DEFAULT_SAMPLE = [
    "f005", "f045", "f051", "f050", "f067", "f066", "f046", "f077", "f008", "f088",
    "f022", "f023", "f004",
]


def main() -> None:
    ids = sys.argv[1:] or DEFAULT_SAMPLE
    corpus = load_corpus("data/sec/corpus.jsonl")
    items = {i.item_id: i for i in load_eval_set("data/sec/eval_set.jsonl")}

    for iid in ids:
        it = items[iid]
        p = corpus[it.passage_id]
        print("=" * 88)
        print(f"{it.item_id}  [{it.label.value.upper()} / {it.failure_mode.value} / {it.difficulty.value}]")
        print(f"PASSAGE ({it.passage_id}):\n  {p.text}")
        print(f"Q: {it.question}")
        print(f"A: {it.answer}")
        print(f"WHY: {it.rationale}")
        print()


if __name__ == "__main__":
    main()
