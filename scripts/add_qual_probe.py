"""One-off: author the buried-QUALITATIVE-error probe (Increment 3f).

The numeric levers didn't dent Opus (40/40) because numbers give a frontier judge a
crisp check it owns. Our only real Opus miss so far (f077) was a dropped quantifier —
a semantic error with no crisp check. This set presses exactly that axis: long, fluent,
otherwise-impeccable multi-claim answers with ONE buried QUALITATIVE deviation
(scope/quantifier drift, hedge removal, or unsupported causation), embedded mid-answer.

Two rails (per the plan):
  1. Subtle: the wrong claim must read as plausible and fluent, not flag itself.
  2. Defensible label: it must still be a CLEAR deviation from the source on a careful
     read, so the ground truth is not contestable (no t04-style artifact).

Written to a SEPARATE probe file so the main eval set stays clean until we decide
whether/how to fold these in. Corpus is the shared data/sec/corpus.jsonl.

Run: python scripts/add_qual_probe.py
Then probe Opus:
    uv run inspect eval src/fiel/task.py \\
        -T eval_set=data/sec/eval_set_probe_qual.jsonl -T judge_style=generic \\
        --model anthropic/claude-opus-5 --log-dir logs/probe/qual
    uv run python scripts/show_results.py logs/probe/qual
"""

from __future__ import annotations

from fiel.dataset import save_eval_set
from fiel.schema import Complexity, Difficulty, EvalItem, FailureMode, Label

HARD = Difficulty.HARD
MC = Complexity.MULTI_CLAIM


def grounded(item_id, passage_id, question, answer, rationale):
    return EvalItem(
        item_id=item_id, passage_id=passage_id, question=question, answer=answer,
        label=Label.GROUNDED, failure_mode=FailureMode.NONE, difficulty=HARD,
        complexity=MC, rationale=rationale,
    )


def ungrounded(item_id, passage_id, question, answer, mode, rationale):
    return EvalItem(
        item_id=item_id, passage_id=passage_id, question=question, answer=answer,
        label=Label.UNGROUNDED, failure_mode=mode, difficulty=HARD, complexity=MC,
        rationale=rationale,
    )


DRIFT = FailureMode.TEMPORAL_SCOPE_DRIFT
CAUSE = FailureMode.UNSUPPORTED_CAUSATION
HEDGE = FailureMode.HEDGE_REMOVAL
OVER = FailureMode.OVERGENERALIZATION


ITEMS: list[EvalItem] = [
    # ---- grounded controls: long, fluent, fully faithful --------------------------
    grounded(
        "fqg01", "aapl-segment",
        "How did Apple's geographic segments perform in fiscal 2025 and why?",
        "Across Apple's geographic segments in fiscal 2025, Americas net sales increased, "
        "primarily due to higher net sales of iPhone and Services, though weakness in "
        "foreign currencies relative to the U.S. dollar had an unfavorable year-over-year "
        "impact on the region. Europe net sales also increased, primarily due to higher "
        "net sales of Services, iPhone and Mac. Japan net sales increased as well, "
        "primarily due to higher net sales of iPhone, Services and iPad. Greater China was "
        "the exception, with net sales decreasing primarily due to lower net sales of "
        "iPhone, partially offset by higher net sales of Mac.",
        "Every segment driver and the FX note match the source exactly.",
    ),
    grounded(
        "fqg02", "msft-server-commentary",
        "Explain the drivers behind Microsoft's Server products and cloud services growth.",
        "In fiscal 2026, Microsoft's Server products and cloud services revenue increased "
        "$31.0 billion, or 31%, driven by Azure and other cloud services. Azure and other "
        "cloud services revenue grew 41%, driven by demand for services across the platform "
        "with continued growth across all workloads. Server products revenue, by contrast, "
        "increased just 1%, primarily driven by higher purchases of licenses running in "
        "multi-cloud environments, offset in part by a continued customer shift to cloud.",
        "All figures and causal attributions match the source exactly.",
    ),
    grounded(
        "fqg03", "nvda-h20-china",
        "What did NVIDIA disclose about H20 revenue and the U.S. government?",
        "NVIDIA disclosed that it generated approximately $60 million in H20 revenue under "
        "the applicable licenses. It further noted that U.S. government officials expressed "
        "an expectation that the U.S. government will receive 15% or more of the revenue "
        "generated from licensed sales of its products.",
        "Amount and the hedged expectation ('expressed an expectation') are both faithful.",
    ),
    grounded(
        "fqg04", "tsla-auto-decrease",
        "Why did Tesla's automotive sales revenue decline in 2025?",
        "Tesla's automotive sales revenue decreased $6.66 billion, or 9%, in 2025 compared "
        "to 2024. The company attributed this to a decrease of approximately 8% in cash "
        "deliveries and to a lower average selling price per unit, which was itself driven "
        "by sales mix and by higher customer incentives such as attractive financing "
        "options.",
        "Magnitude and both stated causes match the source exactly.",
    ),
    grounded(
        "fqg05", "nvda-competition-risk",
        "How does NVIDIA characterize its competitors in its risk factors?",
        "NVIDIA states that some of its competitors operate their own fabrication "
        "facilities. It adds that these competitors have longer operating histories, larger "
        "customer bases, more comprehensive IP portfolios and patent protections, more "
        "design wins, and greater financial, sales, marketing and distribution resources "
        "than NVIDIA does.",
        "The quantifier 'some of' is preserved and every attribute matches the source.",
    ),
    grounded(
        "fqg06", "aapl-fx-gross-margin",
        "What does Apple say about foreign exchange and gross margins?",
        "Apple discloses that gross margins on its products in foreign countries, and on "
        "products that include components obtained from foreign suppliers, have in the past "
        "been adversely affected by foreign exchange rate fluctuations, and could in the "
        "future be materially adversely affected by them.",
        "Both the past-tense fact and the hedged future ('could') are preserved faithfully.",
    ),
    # ---- ungrounded: one buried qualitative deviation, rest impeccable -------------
    ungrounded(
        "fqt01", "aapl-segment",
        "How did Apple's geographic segments perform in fiscal 2025 and why?",
        "Across Apple's geographic segments in fiscal 2025, Japan net sales increased, "
        "primarily due to higher net sales of iPhone, Services and iPad. Europe net sales "
        "increased, primarily due to higher net sales of Services, iPhone and Mac. Americas "
        "net sales increased, primarily due to higher net sales of iPhone, Services and "
        "Mac, though weakness in foreign currencies had an unfavorable year-over-year "
        "impact on the region. Greater China net sales decreased, primarily due to lower "
        "net sales of iPhone, partially offset by higher net sales of Mac.",
        DRIFT,
        "Buried scope drift: the source lists Americas drivers as iPhone and Services only; "
        "Mac is a Europe driver, not an Americas one. Every other segment claim is exact.",
    ),
    ungrounded(
        "fqt02", "msft-server-commentary",
        "Explain the drivers behind Microsoft's Server products and cloud services growth.",
        "In fiscal 2026, Microsoft's Server products and cloud services revenue increased "
        "$31.0 billion, or 31%, driven by Azure and other cloud services. Within this, "
        "Azure and other cloud services revenue grew 41%, primarily driven by higher "
        "purchases of licenses running in multi-cloud environments. Server products revenue "
        "increased 1%, offset in part by a continued customer shift to cloud.",
        DRIFT,
        "Buried scope drift: the source attributes 'higher purchases of licenses running in "
        "multi-cloud environments' to Server products (up 1%), not to Azure; Azure's growth "
        "is driven by demand across the platform. The driver is misattributed to Azure.",
    ),
    ungrounded(
        "fqc01", "tsla-auto-decrease",
        "Why did Tesla's automotive sales revenue decline in 2025?",
        "Tesla's automotive sales revenue decreased $6.66 billion, or 9%, in 2025 compared "
        "to 2024. The company attributed this to a decrease of approximately 8% in cash "
        "deliveries and to a lower average selling price per unit, driven by sales mix, "
        "higher customer incentives such as attractive financing options, and intensifying "
        "competition from lower-priced electric vehicles.",
        CAUSE,
        "Buried unsupported causation: the source cites only sales mix and higher customer "
        "incentives for the lower ASP; 'intensifying competition from lower-priced EVs' is "
        "not a cause the source states. Magnitude and the two real causes are exact.",
    ),
    ungrounded(
        "fqc02", "msft-cloud-metrics",
        "Summarize Microsoft's fiscal 2026 cloud growth highlights.",
        "Microsoft's fiscal 2026 cloud highlights were broad-based: Microsoft Cloud revenue "
        "increased 27% to $214.4 billion, Azure and other cloud services revenue increased "
        "41%, and Microsoft 365 Commercial cloud revenue increased 17%, driven by seat "
        "growth and higher average revenue per user. Dynamics 365 revenue increased 18%, "
        "and LinkedIn revenue increased 11%.",
        CAUSE,
        "Buried unsupported causation: the source states the 17% M365 Commercial cloud "
        "growth with no driver; 'driven by seat growth and higher average revenue per user' "
        "is invented. All five growth figures are exact.",
    ),
    ungrounded(
        "fqh01", "nvda-h20-china",
        "What did NVIDIA disclose about H20 revenue and the U.S. government?",
        "NVIDIA disclosed that it generated approximately $60 million in H20 revenue under "
        "the applicable licenses, and that under those arrangements the U.S. government "
        "will receive 15% or more of the revenue generated from licensed sales of its "
        "products.",
        HEDGE,
        "Buried hedge removal: the source says U.S. government officials 'expressed an "
        "expectation that' the USG will receive 15% or more — an expectation, not a fact. "
        "Stating it 'will receive' asserts as settled what the source only anticipates.",
    ),
    ungrounded(
        "fqh02", "aapl-fx-gross-margin",
        "What does Apple say about foreign exchange and gross margins?",
        "Apple discloses that gross margins on its products in foreign countries, and on "
        "products that include components obtained from foreign suppliers, have in the past "
        "been adversely affected by foreign exchange rate fluctuations, and are materially "
        "adversely affected by them going forward.",
        HEDGE,
        "Buried hedge removal: the source says margins 'could in the future be materially "
        "adversely affected'; 'are materially adversely affected going forward' converts a "
        "possibility into a stated fact. The past-tense clause is faithful.",
    ),
    ungrounded(
        "fqo01", "nvda-competition-risk",
        "How does NVIDIA characterize its competitors in its risk factors?",
        "NVIDIA states that its competitors operate their own fabrication facilities. It "
        "adds that these competitors have longer operating histories, larger customer "
        "bases, more comprehensive IP portfolios and patent protections, more design wins, "
        "and greater financial, sales, marketing and distribution resources than NVIDIA "
        "does.",
        OVER,
        "Buried overgeneralization: the source says 'SOME of our competitors' operate their "
        "own fabrication facilities; dropping 'some' asserts this of all competitors. The "
        "remaining attributes match.",
    ),
    ungrounded(
        "fqo02", "nvda-china-export-risk",
        "What does NVIDIA say about the Chinese government and its customers?",
        "NVIDIA states that the Chinese government has encouraged customers to purchase "
        "from its China-based competitors and has discouraged customers from purchasing, "
        "importing, or using NVIDIA's products, including any China-specific product "
        "designed to comply with U.S. export controls.",
        OVER,
        "Buried overgeneralization: the source scopes the discouragement to NVIDIA's 'data "
        "center products'; broadening it to 'NVIDIA's products' overstates the scope. The "
        "competitor clause and the export-control clause are faithful.",
    ),
]


def main() -> None:
    from fiel.dataset import load_eval_set, resolve_data_path

    path = "data/sec/eval_set_probe_qual.jsonl"
    save_eval_set(ITEMS, path)
    n_g = sum(1 for i in ITEMS if i.label is Label.GROUNDED)
    n_u = len(ITEMS) - n_g
    print(f"Wrote {len(ITEMS)} items ({n_g} grounded, {n_u} ungrounded) -> {path}")

    # Opus aced these (74/74 across all three probes), so they are not boundary cases —
    # but they are subtle enough to separate weaker judges/prompts in the full A/B, so
    # fold them into the authoritative eval set. Idempotent: skip ids already present.
    main_path = resolve_data_path("data/sec/eval_set.jsonl")
    existing = load_eval_set(main_path)
    have = {i.item_id for i in existing}
    to_add = [it for it in ITEMS if it.item_id not in have]
    if to_add:
        save_eval_set(existing + to_add, main_path)
        print(f"Folded {len(to_add)} items into {main_path} (now {len(existing) + len(to_add)} total).")
    else:
        print(f"All {len(ITEMS)} items already present in {main_path}; nothing folded.")


if __name__ == "__main__":
    main()
