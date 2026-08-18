"""Build the full corpus from real SEC filings — verbatim slices + provenance headers.

Run: python scripts/build_corpus.py
Writes data/sec/corpus.jsonl.

Passages are exact substrings of each filing's extracted text (whitespace collapsed
to single spaces for clean, robust slicing). Each passage gets a one-line provenance
header (company / filing / section) so entity references are legitimately grounded
(the t04 fix). Anchors are matched quote-insensitively (curly vs. straight) but the
stored text is the filing's original characters.
"""

from __future__ import annotations

from dataclasses import dataclass

from fiel import edgar
from fiel.corpus import save_corpus, with_provenance
from fiel.schema import Passage

# Quote/dash normalization used ONLY for locating anchors (1:1 char map preserves
# indices, so we still slice the original text).
_QUOTE_MAP = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"'})


@dataclass
class Filing:
    key: str
    company: str
    filing: str
    url: str


@dataclass
class Spec:
    passage_id: str
    filing_key: str
    section: str
    start: str
    end: str


FILINGS = [
    Filing("aapl", "Apple Inc.", "FY2025 Form 10-K",
           "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm"),
    Filing("msft", "Microsoft Corporation", "FY2026 Form 10-K",
           "https://www.sec.gov/Archives/edgar/data/789019/000119312526323660/msft-20260630.htm"),
    Filing("nvda", "NVIDIA Corporation", "FY2026 Form 10-K",
           "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm"),
    Filing("tsla", "Tesla, Inc.", "FY2025 Form 10-K",
           "https://www.sec.gov/Archives/edgar/data/1318605/000162828026003952/tsla-20251231.htm"),
]

SPECS = [
    # --- Apple ---
    Spec("aapl-netsales-category", "aapl", "Item 7 MD&A (Products and Services Performance)",
         "The following table shows net sales by category",
         "Total net sales $ 416,161 6 % $ 391,035 2 % $ 383,285"),
    Spec("aapl-segment", "aapl", "Item 7 MD&A (Segment Operating Performance)",
         "Americas Americas net sales increased during 2025",
         "Japan net sales increased during 2025 compared to 2024 primarily due to higher net sales of iPhone, Services and iPad."),
    Spec("aapl-inventory-risk", "aapl", "Item 1A Risk Factors (Inventory and purchase commitments)",
         "The Company is exposed to the risk of write-downs on the value of its inventory",
         "cost exceeds net realizable value."),
    Spec("aapl-fx-gross-margin", "aapl", "Item 1A Risk Factors (Foreign exchange)",
         "Gross margins on the Company",
         "foreign exchange rate fluctuations."),
    # --- Microsoft ---
    Spec("msft-revenue-by-product", "msft", "Item 8 Notes (Revenue by product and service)",
         "Revenue, classified by significant product and service offerings",
         "Search advertising 15,176 13,878 12,306"),
    Spec("msft-cloud-metrics", "msft", "Item 7 MD&A (Fiscal year highlights)",
         "Microsoft Cloud revenue increased 27% to $214.4 billion",
         "XBOX content and services revenue decreased 5%"),
    Spec("msft-server-commentary", "msft", "Item 7 MD&A (Segment results of operations)",
         "Server products and cloud services revenue increased $31.0 billion",
         "continued customer shift to cloud."),
    Spec("msft-gross-margin", "msft", "Item 7 MD&A (Gross margin)",
         "Gross margin increased $31.6 billion or 16%",
         "efficiency gains in Azure and Microsoft 365 Commercial cloud."),
    # --- NVIDIA ---
    Spec("nvda-fy26-summary", "nvda", "Item 7 MD&A (Fiscal Year 2026 Summary)",
         "Fiscal Year 2026 Summary",
         "Data Center revenue for fiscal year 2026 was up 68% from a year ago."),
    Spec("nvda-h20-china", "nvda", "Item 7 MD&A (H20 products and China)",
         "We generated approximately $60 million in H20 revenue",
         "licensed sales of our products"),
    Spec("nvda-china-export-risk", "nvda", "Item 1A Risk Factors (China and export controls)",
         "The Chinese government has encouraged customers to purchase from our China-based competitors",
         "designed to comply with U.S. export controls."),
    Spec("nvda-competition-risk", "nvda", "Item 1A Risk Factors (Competition)",
         "Some of our competitors operate their own fabrication facilities",
         "greater financial, sales, marketing and distribution resources than we do."),
    # --- Tesla ---
    Spec("tsla-revenue-table", "tsla", "Item 7 MD&A (Results of Operations — Revenues)",
         "Automotive sales $ 65,821",
         "Total revenues $ 94,827 $ 97,690 $ 96,773 $ (2,863) (3) % $ 917 1 %"),
    Spec("tsla-auto-decrease", "tsla", "Item 7 MD&A (Automotive revenue)",
         "Automotive sales revenue decreased $6.66 billion",
         "higher customer incentives such as attractive financing options."),
    Spec("tsla-gross-margin", "tsla", "Item 7 MD&A (Gross margin)",
         "Total gross margin 18.0 % 17.9 % 18.2 %",
         "Total gross margin 18.0 % 17.9 % 18.2 %"),
    Spec("tsla-demand-risk", "tsla", "Item 1A Risk Factors (Demand for EVs)",
         "Our future growth and success are dependent upon demand for our electric vehicles",
         "adoption of autonomous driving solutions."),
    Spec("tsla-factory-ramp-risk", "tsla", "Item 1A Risk Factors (Factory ramp)",
         "We may be unable to meet our projected construction timelines",
         "difficulties in generating and maintaining demand for products manufactured there and related services."),
]


def _slice(flat: str, start_anchor: str, end_anchor: str) -> str:
    norm = flat.translate(_QUOTE_MAP)
    start = norm.find(start_anchor.translate(_QUOTE_MAP))
    if start < 0:
        raise ValueError(f"start anchor not found: {start_anchor!r}")
    end = norm.find(end_anchor.translate(_QUOTE_MAP), start)
    if end < 0:
        raise ValueError(f"end anchor not found: {end_anchor!r}")
    return flat[start:end + len(end_anchor)].strip()


def main() -> None:
    flats: dict[str, str] = {}
    meta: dict[str, Filing] = {}
    for f in FILINGS:
        flats[f.key] = " ".join(edgar.html_to_text(edgar.fetch(f.url)).split())
        meta[f.key] = f

    passages: list[Passage] = []
    for spec in SPECS:
        f = meta[spec.filing_key]
        body = _slice(flats[spec.filing_key], spec.start, spec.end)
        text = with_provenance(body, company=f.company, filing=f.filing, section=spec.section)
        passages.append(
            Passage(
                passage_id=spec.passage_id,
                text=text,
                metadata={
                    "company": f.company,
                    "filing": f.filing,
                    "section": spec.section,
                    "url": f.url,
                },
            )
        )

    save_corpus(passages, "data/sec/corpus.jsonl")
    print(f"wrote {len(passages)} passages to data/sec/corpus.jsonl\n")
    for p in passages:
        print(f"=== {p.passage_id}  ({len(p.text)} chars) ===")
        print(p.text)
        print()


if __name__ == "__main__":
    main()
