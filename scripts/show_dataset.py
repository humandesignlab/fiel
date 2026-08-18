"""Load a corpus + labeled eval set, validate, and print balance stats.

Run: python scripts/show_dataset.py [eval_set.jsonl] [corpus.jsonl]
Defaults to the tiny bootstrap set.
"""

from __future__ import annotations

import sys
from collections import Counter

from fiel.corpus import load_corpus
from fiel.dataset import load_eval_set, to_samples


def main() -> None:
    eval_path = sys.argv[1] if len(sys.argv) > 1 else "data/sec/eval_set_tiny.jsonl"
    corpus_path = sys.argv[2] if len(sys.argv) > 2 else "data/sec/corpus_tiny.jsonl"

    corpus = load_corpus(corpus_path)
    items = load_eval_set(eval_path)
    samples = to_samples(items, corpus)  # exercises the join + Inspect conversion

    print(f"corpus:    {corpus_path}  ({len(corpus)} passages)")
    print(f"eval set:  {eval_path}  ({len(items)} items -> {len(samples)} samples)\n")

    by_label = Counter(i.label.value for i in items)
    by_mode = Counter(i.failure_mode.value for i in items)
    by_diff = Counter(i.difficulty.value for i in items)
    by_cplx = Counter(i.complexity.value for i in items)

    print("label:      " + ", ".join(f"{k}={v}" for k, v in sorted(by_label.items())))
    print("difficulty: " + ", ".join(f"{k}={v}" for k, v in sorted(by_diff.items())))
    print("complexity: " + ", ".join(f"{k}={v}" for k, v in sorted(by_cplx.items())))
    print("failure_mode:")
    for k, v in sorted(by_mode.items()):
        print(f"  {k:24s} {v}")

    print("\nitems:")
    for i in items:
        print(f"  {i.item_id}  {i.label.value:10s} {i.failure_mode.value:22s} {i.difficulty.value:5s} {i.complexity.value:12s} {i.passage_id}")


if __name__ == "__main__":
    main()
