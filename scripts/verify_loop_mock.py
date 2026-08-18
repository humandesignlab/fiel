"""Deterministic end-to-end plumbing check with no API key and no cost.

Uses Inspect's mockllm provider forced to always answer "VERDICT: ungrounded".
Expectation: the 4 grounded items score INCORRECT and the 4 ungrounded items score
CORRECT -> accuracy 0.5. This proves dataset -> judge solver -> validation scorer ->
metrics wiring works before we spend anything on real models.
"""

from __future__ import annotations

from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, get_model

from fiel.task import grounding_reliability

FIXED = "The answer adds a claim not in the source.\nVERDICT: ungrounded"


def main() -> None:
    # mockllm consumes one custom output per generate() call (no cycling), so supply
    # one per sample in the tiny set.
    mock = get_model(
        "mockllm/model",
        custom_outputs=[ModelOutput.from_content("mockllm/model", FIXED) for _ in range(8)],
    )

    logs = inspect_eval(grounding_reliability(), model=mock, log_dir="logs/_mock")
    log = logs[0]

    print("status:", log.status)
    print("samples:", len(log.samples))
    correct = sum(
        1 for s in log.samples if s.scores["grounding_agreement"].value == "C"
    )
    print(f"correct: {correct}/{len(log.samples)} (expected 4/8)")
    print("\nper-item:")
    for s in log.samples:
        sc = s.scores["grounding_agreement"]
        md = sc.metadata
        print(
            f"  {s.id}  gold={md['gold_label']:10s} verdict={str(md['judge_verdict']):10s} "
            f"score={sc.value}  mode={md['failure_mode']}"
        )


if __name__ == "__main__":
    main()
