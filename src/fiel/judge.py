"""The LLM-as-judge, implemented as an Inspect solver.

Design note: in Fiel the judge is the SYSTEM UNDER TEST. The solver takes a
(question, source passage, candidate answer) from the sample metadata, prompts the
judge model, and leaves the judge's response in `state.output` for the scorer to
compare against the controlled ground-truth label.

Three prompt styles are provided so we can show that prompt design changes judge
reliability (spec §3), and to keep the A/B honest:
  - `naive`:      a one-liner with no guidance.
  - `generic`:    a thorough, corpus-agnostic faithfulness prompt (the sensible
                  default any careful team would write) — the fair baseline.
  - `enumerated`: additionally spells out the six failure modes THIS eval injects.
                  Comparing enumerated vs generic quantifies "teaching-to-the-test"
                  inflation: how much of a judge's score comes from being handed the
                  exact taxonomy of tricks the eval uses.
The verdict must end with a machine-parseable line.
"""

from __future__ import annotations

from typing import Literal

from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, solver

JudgeStyle = Literal["naive", "generic", "enumerated"]

# The judge must end with exactly this so the scorer can parse it deterministically.
_VERDICT_INSTRUCTION = (
    "End your response with a single final line in exactly this format:\n"
    "VERDICT: grounded\n"
    "or\n"
    "VERDICT: ungrounded"
)

_NAIVE_SYSTEM = (
    "You are checking whether an answer is grounded in a source passage. "
    "An answer is 'grounded' if it is supported by the source, and 'ungrounded' if it "
    "is not.\n\n" + _VERDICT_INSTRUCTION
)

# Generic prompt: a thorough, corpus-agnostic faithfulness instruction. It tells the
# judge to verify every claim and to do the arithmetic, but it does NOT hand over the
# specific taxonomy of tricks this eval injects. This is the fair default baseline.
_GENERIC_SYSTEM = (
    "You are a meticulous faithfulness auditor. You are given a SOURCE passage, a "
    "QUESTION, and a candidate ANSWER. Decide whether EVERY claim in the ANSWER is "
    "fully supported by the SOURCE.\n\n"
    "Check each claim independently and carefully:\n"
    "- Verify every name, entity, figure, percentage, unit, date, and period against "
    "the source. A single mismatched detail makes the whole answer ungrounded.\n"
    "- When the answer states a derived quantity (a difference, a total, a growth rate, "
    "a share/proportion), compute it yourself from the source figures and confirm it is "
    "correct. Do not accept a derived number just because the inputs appear in the source.\n"
    "- Watch for claims the source never makes, and for statements that are stronger, "
    "broader, or more certain than the source supports.\n\n"
    "Only label 'grounded' if the answer makes no claim beyond what the source supports. "
    "Do not use outside knowledge; judge solely against the SOURCE.\n\n"
    "Think briefly, then " + _VERDICT_INSTRUCTION[0].lower() + _VERDICT_INSTRUCTION[1:]
)

# Enumerated prompt: the generic instruction PLUS the exact failure-mode taxonomy this
# eval injects. Kept only as an A/B contrast — the generic-vs-enumerated gap measures
# how much score comes from teaching the judge the test.
_ENUMERATED_SYSTEM = (
    "You are a meticulous faithfulness auditor for high-stakes financial-document QA. "
    "You are given a SOURCE passage, a QUESTION, and a candidate ANSWER. Decide whether "
    "EVERY claim in the ANSWER is fully supported by the SOURCE.\n\n"
    "Label the answer 'ungrounded' if it contains ANY unsupported claim, including subtle ones:\n"
    "- Numeric errors: a figure, percentage, or unit that does not match the source "
    "(even if the surrounding structure is correct), including a wrong derived value "
    "(difference, total, growth rate, or share) — compute these yourself to check.\n"
    "- Unsupported additions: a fluent extra claim the source never makes.\n"
    "- Overgeneralization: dropping qualifiers the source states (period, geography, segment).\n"
    "- Unsupported causation: asserting a cause ('because', 'due to', 'driven by') the source "
    "does not state.\n"
    "- Temporal/scope drift: applying a figure to the wrong period, segment, or entity.\n"
    "- Hedge removal: stating as fact something the source only presents as a risk, "
    "possibility, or expectation ('may', 'could', 'expects', 'exposed to the risk of').\n\n"
    "In multi-claim answers, most claims may be correct while a single one is wrong; "
    "verify each claim independently and do not let overall fluency excuse one bad claim.\n"
    "Only label 'grounded' if the answer makes no claim beyond what the source supports. "
    "Be strict: a single unsupported claim makes the whole answer ungrounded. "
    "Do not use outside knowledge; judge solely against the SOURCE.\n\n"
    "Think briefly, then " + _VERDICT_INSTRUCTION[0].lower() + _VERDICT_INSTRUCTION[1:]
)

_SYSTEM_BY_STYLE: dict[str, str] = {
    "naive": _NAIVE_SYSTEM,
    "generic": _GENERIC_SYSTEM,
    "enumerated": _ENUMERATED_SYSTEM,
}

_USER_TEMPLATE = (
    "SOURCE PASSAGE:\n{source}\n\n"
    "QUESTION:\n{question}\n\n"
    "ANSWER TO EVALUATE:\n{answer}"
)


@solver
def grounding_judge(style: JudgeStyle = "generic") -> Solver:
    """Prompt the judge model to rule on whether the answer is grounded in the source."""
    try:
        system_prompt = _SYSTEM_BY_STYLE[style]
    except KeyError:
        raise ValueError(
            f"unknown judge_style {style!r}; expected one of {sorted(_SYSTEM_BY_STYLE)}"
        ) from None

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        meta = state.metadata
        user_prompt = _USER_TEMPLATE.format(
            source=meta["passage_text"],
            question=meta["question"],
            answer=meta["answer"],
        )
        state.messages = [
            ChatMessageSystem(content=system_prompt),
            ChatMessageUser(content=user_prompt),
        ]
        return await generate(state)

    return solve
