"""Build the tiny (bootstrap) corpus from a real SEC filing — verbatim slices.

Run: python scripts/build_tiny_corpus.py
Writes data/sec/corpus_tiny.jsonl. Passages are exact substrings of the fetched
filing text (no paraphrasing), so ground-truth labeling stays honest.
"""

from __future__ import annotations

from fiel import edgar
from fiel.corpus import save_corpus
from fiel.schema import Passage

AAPL_10K = "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm"
AAPL_META = {
    "company": "Apple Inc.",
    "ticker": "AAPL",
    "filing": "10-K",
    "fiscal_year_end": "2025-09-27",
    "filed": "2025-10-31",
    "url": AAPL_10K,
}


def _slice(text: str, start_anchor: str, end_anchor: str, *, include_end: bool = True) -> str:
    start = text.find(start_anchor)
    if start < 0:
        raise ValueError(f"start anchor not found: {start_anchor!r}")
    end = text.find(end_anchor, start)
    if end < 0:
        raise ValueError(f"end anchor not found: {end_anchor!r}")
    if include_end:
        end += len(end_anchor)
    return text[start:end].strip()


def main() -> None:
    text = edgar.html_to_text(edgar.fetch(AAPL_10K))

    passages = [
        Passage(
            passage_id="aapl-2025-netsales-by-category",
            text=_slice(
                text,
                "The following table shows net sales by category",
                "Total net sales $ 416,161 6 % $ 391,035 2 % $ 383,285",
            ),
            metadata={**AAPL_META, "section": "MD&A — Products and Services Performance"},
        ),
        Passage(
            passage_id="aapl-2025-segment-commentary",
            text=_slice(
                text,
                "Americas Americas net sales increased during 2025",
                "Rest of Asia Pacific net sales",
            ),
            metadata={**AAPL_META, "section": "MD&A — Segment Operating Performance"},
        ),
        Passage(
            passage_id="aapl-2025-inventory-risk",
            text=_slice(
                text,
                "The Company is exposed to the risk of write-downs on the value of its inventory",
                "cost exceeds net realizable value.",
            ),
            metadata={**AAPL_META, "section": "Risk Factors — Inventory and purchase commitments"},
        ),
    ]

    save_corpus(passages, "data/sec/corpus_tiny.jsonl")
    for p in passages:
        print(f"\n=== {p.passage_id} ({len(p.text)} chars) ===\n{p.text}")


if __name__ == "__main__":
    main()
