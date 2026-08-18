"""One-off: author the multi-claim + computation adversarial items (Increment 3e).

The hardness probe showed frontier judges are near-perfect on single-claim lookups.
These 40 items target the two documented weaknesses:
  - multi_claim: a long, fluent answer where every claim is correct except one buried
    subtle number/percentage error (judges over-approve fluent answers).
  - computation: correctness hinges on a derived quantity (difference, share, growth
    rate) that a lookup passes but only arithmetic catches.

All figures below are copied from data/sec/corpus.jsonl; every derived value is chosen
so the correct answer is unambiguous and the wrong variant is off by a wide margin
(~10 pts or ~$10B) — no rounding gray zones, so the ground-truth label is not itself
contestable. Idempotent: re-running skips items whose ids already exist.

Run: python scripts/add_hard_items.py
"""

from __future__ import annotations

from fiel.dataset import load_eval_set, resolve_data_path, save_eval_set
from fiel.schema import Complexity, Difficulty, EvalItem, FailureMode, Label

HARD = Difficulty.HARD
MC = Complexity.MULTI_CLAIM
COMP = Complexity.COMPUTATION
GROUNDED = Label.GROUNDED
UNGROUNDED = Label.UNGROUNDED
NONE = FailureMode.NONE
NUM = FailureMode.SUBTLE_NUMBER_ERROR


def g_mc(item_id, passage_id, question, answer, rationale):
    return EvalItem(
        item_id=item_id, passage_id=passage_id, question=question, answer=answer,
        label=GROUNDED, failure_mode=NONE, difficulty=HARD, complexity=MC,
        rationale=rationale,
    )


def u_mc(item_id, passage_id, question, answer, rationale):
    return EvalItem(
        item_id=item_id, passage_id=passage_id, question=question, answer=answer,
        label=UNGROUNDED, failure_mode=NUM, difficulty=HARD, complexity=MC,
        rationale=rationale,
    )


def g_comp(item_id, passage_id, question, answer, rationale):
    return EvalItem(
        item_id=item_id, passage_id=passage_id, question=question, answer=answer,
        label=GROUNDED, failure_mode=NONE, difficulty=HARD, complexity=COMP,
        rationale=rationale,
    )


def u_comp(item_id, passage_id, question, answer, rationale):
    return EvalItem(
        item_id=item_id, passage_id=passage_id, question=question, answer=answer,
        label=UNGROUNDED, failure_mode=NUM, difficulty=HARD, complexity=COMP,
        rationale=rationale,
    )


NEW_ITEMS: list[EvalItem] = [
    # ---- multi-claim, grounded (every claim exact) --------------------------------
    g_mc(
        "fm01", "msft-revenue-by-product",
        "Summarize Microsoft's fiscal 2026 revenue by major product and service offering.",
        "In fiscal 2026, Microsoft's Server products and cloud services revenue was "
        "$129,425 million, Microsoft 365 Commercial products and cloud services was "
        "$101,997 million, XBOX was $21,790 million, LinkedIn was $19,817 million, and "
        "Search advertising was $15,176 million.",
        "All five figures match the FY2026 column exactly.",
    ),
    g_mc(
        "fm02", "aapl-netsales-category",
        "Give Apple's fiscal 2025 net sales for its major categories and the total.",
        "For fiscal 2025, iPhone net sales were $209,586 million, Services $109,158 "
        "million, Mac $33,708 million, Wearables, Home and Accessories $35,686 million, "
        "and total net sales were $416,161 million.",
        "All five figures match the 2025 column exactly.",
    ),
    g_mc(
        "fm03", "nvda-fy26-summary",
        "Summarize NVIDIA's key fiscal 2026 results.",
        "For fiscal 2026, NVIDIA reported revenue of $215,938 million, operating income "
        "of $130,387 million, net income of $120,067 million, a gross margin of 71.1%, "
        "and diluted EPS of $4.90.",
        "All five figures match the FY2026 summary exactly.",
    ),
    g_mc(
        "fm04", "tsla-revenue-table",
        "Break down Tesla's 2025 revenue lines.",
        "In 2025, Tesla's automotive sales revenue was $65,821 million, automotive "
        "regulatory credits $1,993 million, automotive leasing $1,712 million, services "
        "and other $12,530 million, and energy generation and storage segment revenue "
        "$12,771 million.",
        "All five figures match the 2025 column exactly.",
    ),
    g_mc(
        "fm05", "msft-cloud-metrics",
        "What were Microsoft's fiscal 2026 cloud growth highlights?",
        "In fiscal 2026, Microsoft Cloud revenue grew 27% to $214.4 billion, Azure and "
        "other cloud services grew 41%, Microsoft 365 Commercial cloud grew 17%, Dynamics "
        "365 grew 18%, and LinkedIn grew 11%.",
        "All five growth figures match the highlights bullets exactly.",
    ),
    g_mc(
        "fm06", "aapl-netsales-category",
        "How did Apple's key categories change from fiscal 2024 to 2025?",
        "iPhone net sales rose from $201,183 million in 2024 to $209,586 million in 2025, "
        "Mac from $29,984 million to $33,708 million, and Services from $96,169 million to "
        "$109,158 million, lifting total net sales from $391,035 million to $416,161 million.",
        "All eight endpoint figures match the 2024 and 2025 columns exactly.",
    ),
    g_mc(
        "fm07", "tsla-revenue-table",
        "How did Tesla's revenue lines change year over year in 2025?",
        "In 2025, Tesla's automotive sales declined 9%, automotive leasing declined 6%, "
        "services and other rose 19%, and energy generation and storage rose 27%.",
        "All four percentage changes match the year-over-year change column exactly.",
    ),
    g_mc(
        "fm08", "nvda-fy26-summary",
        "How did NVIDIA's key metrics change in fiscal 2026?",
        "For fiscal 2026, NVIDIA's revenue rose 65%, operating expenses rose 41%, "
        "operating income rose 60%, and net income rose 65%.",
        "All four change figures match the 'Change' column exactly.",
    ),
    g_mc(
        "fm09", "msft-revenue-by-product",
        "How did selected Microsoft product revenues move from fiscal 2025 to 2026?",
        "Server products and cloud services revenue rose from $98,435 million to $129,425 "
        "million, XBOX declined from $23,455 million to $21,790 million, and Windows and "
        "Devices edged down from $17,314 million to $17,084 million.",
        "All six endpoint figures match the 2025 and 2026 columns exactly.",
    ),
    g_mc(
        "fm10", "tsla-revenue-table",
        "Give Tesla's 2025 revenue subtotals and total.",
        "In 2025, Tesla's total automotive revenues were $69,526 million, total automotive "
        "& services and other segment revenue was $82,056 million, and total revenues were "
        "$94,827 million.",
        "All three subtotal/total figures match the 2025 column exactly.",
    ),
    # ---- multi-claim, ungrounded (exactly one buried wrong figure) -----------------
    u_mc(
        "fn01", "msft-revenue-by-product",
        "Summarize Microsoft's fiscal 2026 revenue by major product and service offering.",
        "In fiscal 2026, Microsoft's Server products and cloud services revenue was "
        "$129,425 million, Microsoft 365 Commercial products and cloud services was "
        "$101,997 million, XBOX was $21,790 million, LinkedIn was $18,917 million, and "
        "Search advertising was $15,176 million.",
        "Buried error: LinkedIn was $19,817 million, not $18,917 million (digit "
        "transposition); all four other figures are exact.",
    ),
    u_mc(
        "fn02", "aapl-netsales-category",
        "Give Apple's fiscal 2025 net sales for its major categories and the total.",
        "For fiscal 2025, iPhone net sales were $209,586 million, Services $109,158 "
        "million, Mac $33,708 million, Wearables, Home and Accessories $36,586 million, "
        "and total net sales were $416,161 million.",
        "Buried error: Wearables was $35,686 million, not $36,586 million; all four other "
        "figures are exact.",
    ),
    u_mc(
        "fn03", "nvda-fy26-summary",
        "Summarize NVIDIA's key fiscal 2026 results.",
        "For fiscal 2026, NVIDIA reported revenue of $215,938 million, operating income "
        "of $103,387 million, net income of $120,067 million, a gross margin of 71.1%, "
        "and diluted EPS of $4.90.",
        "Buried error: operating income was $130,387 million, not $103,387 million; all "
        "four other figures are exact.",
    ),
    u_mc(
        "fn04", "tsla-revenue-table",
        "Break down Tesla's 2025 revenue lines.",
        "In 2025, Tesla's automotive sales revenue was $65,821 million, automotive "
        "regulatory credits $1,993 million, automotive leasing $1,172 million, services "
        "and other $12,530 million, and energy generation and storage segment revenue "
        "$12,771 million.",
        "Buried error: automotive leasing was $1,712 million, not $1,172 million (digit "
        "transposition); all four other figures are exact.",
    ),
    u_mc(
        "fn05", "msft-cloud-metrics",
        "What were Microsoft's fiscal 2026 cloud growth highlights?",
        "In fiscal 2026, Microsoft Cloud revenue grew 27% to $214.4 billion, Azure and "
        "other cloud services grew 41%, Microsoft 365 Commercial cloud grew 17%, Dynamics "
        "365 grew 15%, and LinkedIn grew 11%.",
        "Buried error: Dynamics 365 grew 18%, not 15%; all four other figures are exact.",
    ),
    u_mc(
        "fn06", "aapl-netsales-category",
        "How did Apple's key categories change from fiscal 2024 to 2025?",
        "iPhone net sales rose from $201,183 million in 2024 to $209,586 million in 2025, "
        "Mac from $29,984 million to $34,708 million, and Services from $96,169 million to "
        "$109,158 million, lifting total net sales from $391,035 million to $416,161 million.",
        "Buried error: Mac 2025 net sales were $33,708 million, not $34,708 million; all "
        "seven other endpoints are exact.",
    ),
    u_mc(
        "fn07", "tsla-revenue-table",
        "How did Tesla's revenue lines change year over year in 2025?",
        "In 2025, Tesla's automotive sales declined 9%, automotive leasing declined 16%, "
        "services and other rose 19%, and energy generation and storage rose 27%.",
        "Buried error: automotive leasing declined 6%, not 16%; all three other changes "
        "are exact.",
    ),
    u_mc(
        "fn08", "nvda-fy26-summary",
        "How did NVIDIA's key metrics change in fiscal 2026?",
        "For fiscal 2026, NVIDIA's revenue rose 65%, operating expenses rose 41%, "
        "operating income rose 50%, and net income rose 65%.",
        "Buried error: operating income rose 60%, not 50%; all three other changes are exact.",
    ),
    u_mc(
        "fn09", "msft-revenue-by-product",
        "How did selected Microsoft product revenues move from fiscal 2025 to 2026?",
        "Server products and cloud services revenue rose from $98,435 million to $129,425 "
        "million, XBOX declined from $23,455 million to $22,790 million, and Windows and "
        "Devices edged down from $17,314 million to $17,084 million.",
        "Buried error: XBOX FY2026 revenue was $21,790 million, not $22,790 million; all "
        "five other endpoints are exact.",
    ),
    u_mc(
        "fn10", "tsla-revenue-table",
        "Give Tesla's 2025 revenue subtotals and total.",
        "In 2025, Tesla's total automotive revenues were $69,526 million, total automotive "
        "& services and other segment revenue was $83,056 million, and total revenues were "
        "$94,827 million.",
        "Buried error: total automotive & services and other segment revenue was $82,056 "
        "million, not $83,056 million; the other two figures are exact.",
    ),
    # ---- computation, grounded (derived value is correct) -------------------------
    g_comp(
        "fc01", "nvda-fy26-summary",
        "By how much did NVIDIA's revenue grow in absolute dollars in fiscal 2026?",
        "NVIDIA's revenue rose from $130,497 million to $215,938 million in fiscal 2026, "
        "an increase of about $85 billion.",
        "215,938 - 130,497 = 85,441M ~ $85B. Correct derivation.",
    ),
    g_comp(
        "fc02", "nvda-fy26-summary",
        "How much did NVIDIA's operating income increase in dollars in fiscal 2026?",
        "NVIDIA's operating income rose from $81,453 million to $130,387 million, an "
        "increase of roughly $49 billion.",
        "130,387 - 81,453 = 48,934M ~ $49B. Correct derivation.",
    ),
    g_comp(
        "fc03", "nvda-fy26-summary",
        "How much did NVIDIA's net income increase in dollars in fiscal 2026?",
        "NVIDIA's net income increased by about $47 billion in fiscal 2026, from $72,880 "
        "million to $120,067 million.",
        "120,067 - 72,880 = 47,187M ~ $47B. Correct derivation.",
    ),
    g_comp(
        "fc04", "aapl-netsales-category",
        "By how many dollars did Apple's total net sales grow in fiscal 2025?",
        "Apple's total net sales grew by about $25 billion in fiscal 2025, from $391,035 "
        "million to $416,161 million.",
        "416,161 - 391,035 = 25,126M ~ $25B. Correct derivation.",
    ),
    g_comp(
        "fc05", "aapl-netsales-category",
        "How much did Apple's Services net sales grow in dollars year over year?",
        "Apple's Services net sales rose about $13 billion year over year, from $96,169 "
        "million to $109,158 million.",
        "109,158 - 96,169 = 12,989M ~ $13B. Correct derivation.",
    ),
    g_comp(
        "fc06", "aapl-netsales-category",
        "What share of Apple's fiscal 2025 total net sales came from Services?",
        "Services net sales of $109,158 million were about 26% of Apple's $416,161 million "
        "total net sales in fiscal 2025.",
        "109,158 / 416,161 = 26.2% ~ 26%. Correct derivation.",
    ),
    g_comp(
        "fc07", "aapl-netsales-category",
        "How large was iPhone relative to Apple's total net sales in fiscal 2025?",
        "iPhone net sales of $209,586 million were just over half of Apple's $416,161 "
        "million total net sales in fiscal 2025.",
        "209,586 / 416,161 = 50.4%, i.e. just over half. Correct derivation.",
    ),
    g_comp(
        "fc08", "msft-revenue-by-product",
        "By how much did Microsoft's Server products and cloud services revenue grow in dollars?",
        "Microsoft's Server products and cloud services revenue grew by about $31 billion, "
        "from $98,435 million to $129,425 million.",
        "129,425 - 98,435 = 30,990M ~ $31B. Correct derivation (delta not stated in this "
        "passage).",
    ),
    g_comp(
        "fc09", "msft-revenue-by-product",
        "By what percentage did Microsoft's XBOX revenue change in fiscal 2026?",
        "XBOX revenue fell about 7% in fiscal 2026, from $23,455 million to $21,790 million.",
        "(21,790 - 23,455) / 23,455 = -7.1% ~ -7%. Correct derivation.",
    ),
    g_comp(
        "fc10", "tsla-revenue-table",
        "What share of Tesla's 2025 total revenues came from automotive sales?",
        "Tesla's automotive sales revenue of $65,821 million was roughly 70% of its "
        "$94,827 million total revenues in 2025.",
        "65,821 / 94,827 = 69.4% ~ 70%. Correct derivation.",
    ),
    # ---- computation, ungrounded (derived value wrong; endpoints correct) ----------
    u_comp(
        "fd01", "nvda-fy26-summary",
        "By how much did NVIDIA's revenue grow in absolute dollars in fiscal 2026?",
        "NVIDIA's revenue rose from $130,497 million to $215,938 million in fiscal 2026, "
        "an increase of about $95 billion.",
        "Actual increase is 85,441M ~ $85B, not $95B. Endpoints correct; derived value wrong.",
    ),
    u_comp(
        "fd02", "nvda-fy26-summary",
        "How much did NVIDIA's operating income increase in dollars in fiscal 2026?",
        "NVIDIA's operating income rose from $81,453 million to $130,387 million, an "
        "increase of roughly $39 billion.",
        "Actual increase is 48,934M ~ $49B, not $39B. Endpoints correct; derived value wrong.",
    ),
    u_comp(
        "fd03", "nvda-fy26-summary",
        "How much did NVIDIA's net income increase in dollars in fiscal 2026?",
        "NVIDIA's net income increased by about $57 billion in fiscal 2026, from $72,880 "
        "million to $120,067 million.",
        "Actual increase is 47,187M ~ $47B, not $57B. Endpoints correct; derived value wrong.",
    ),
    u_comp(
        "fd04", "aapl-netsales-category",
        "By how many dollars did Apple's total net sales grow in fiscal 2025?",
        "Apple's total net sales grew by about $35 billion in fiscal 2025, from $391,035 "
        "million to $416,161 million.",
        "Actual increase is 25,126M ~ $25B, not $35B. Endpoints correct; derived value wrong.",
    ),
    u_comp(
        "fd05", "aapl-netsales-category",
        "How much did Apple's Services net sales grow in dollars year over year?",
        "Apple's Services net sales rose about $19 billion year over year, from $96,169 "
        "million to $109,158 million.",
        "Actual increase is 12,989M ~ $13B, not $19B. Endpoints correct; derived value wrong.",
    ),
    u_comp(
        "fd06", "aapl-netsales-category",
        "What share of Apple's fiscal 2025 total net sales came from Services?",
        "Services net sales of $109,158 million were about 36% of Apple's $416,161 million "
        "total net sales in fiscal 2025.",
        "Actual share is 26.2% ~ 26%, not 36%. Endpoints correct; derived value wrong.",
    ),
    u_comp(
        "fd07", "aapl-netsales-category",
        "How large was iPhone relative to Apple's total net sales in fiscal 2025?",
        "iPhone net sales of $209,586 million were about 40% of Apple's $416,161 million "
        "total net sales in fiscal 2025.",
        "Actual share is 50.4% (just over half), not 40%. Endpoints correct; derived value wrong.",
    ),
    u_comp(
        "fd08", "msft-revenue-by-product",
        "By how much did Microsoft's Server products and cloud services revenue grow in dollars?",
        "Microsoft's Server products and cloud services revenue grew by about $21 billion, "
        "from $98,435 million to $129,425 million.",
        "Actual increase is 30,990M ~ $31B, not $21B. Endpoints correct; derived value wrong.",
    ),
    u_comp(
        "fd09", "msft-revenue-by-product",
        "By what percentage did Microsoft's XBOX revenue change in fiscal 2026?",
        "XBOX revenue fell about 17% in fiscal 2026, from $23,455 million to $21,790 million.",
        "Actual change is -7.1% ~ -7%, not -17%. Endpoints correct; derived value wrong.",
    ),
    u_comp(
        "fd10", "tsla-revenue-table",
        "What share of Tesla's 2025 total revenues came from automotive sales?",
        "Tesla's automotive sales revenue of $65,821 million was roughly 60% of its "
        "$94,827 million total revenues in 2025.",
        "Actual share is 69.4% ~ 70%, not 60%. Endpoints correct; derived value wrong.",
    ),
]


def main() -> None:
    path = resolve_data_path("data/sec/eval_set.jsonl")
    existing = load_eval_set(path)
    have = {i.item_id for i in existing}
    to_add = [it for it in NEW_ITEMS if it.item_id not in have]
    if not to_add:
        print(f"Nothing to add; all {len(NEW_ITEMS)} new ids already present in {path}.")
        return
    combined = existing + to_add
    save_eval_set(combined, path)
    print(f"Added {len(to_add)} items -> {path} (now {len(combined)} total).")


if __name__ == "__main__":
    main()
