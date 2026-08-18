"""The Inspect task: wire dataset + judge (solver) + validation (scorer).

Run with, e.g.:
    inspect eval src/fiel/task.py --model anthropic/claude-haiku-4-5 --log-dir logs
    inspect eval src/fiel/task.py -T judge_style=naive --model anthropic/claude-haiku-4-5
    inspect eval src/fiel/task.py -T difficulty=hard --model anthropic/claude-opus-5
    inspect eval src/fiel/task.py -T complexity=computation --model anthropic/claude-opus-5

Task params (override with -T):
    eval_set    path to the labeled eval set JSONL   (default: full SEC set)
    corpus      path to the corpus JSONL             (default: full SEC corpus)
    judge_style "generic" (default), "naive", or "enumerated"
    difficulty  optional filter: "easy" or "hard"                       (default: all)
    complexity  optional filter: "single", "multi_claim", "computation" (default: all)
"""

from __future__ import annotations

from inspect_ai import Task, task

from fiel.dataset import build_dataset
from fiel.judge import JudgeStyle, grounding_judge
from fiel.scorer import grounding_agreement


@task
def grounding_reliability(
    eval_set: str = "data/sec/eval_set.jsonl",
    corpus: str = "data/sec/corpus.jsonl",
    judge_style: JudgeStyle = "generic",
    difficulty: str | None = None,
    complexity: str | None = None,
) -> Task:
    dataset = build_dataset(eval_set, corpus)
    if difficulty is not None:
        # Probe a single difficulty band (e.g. the hard adversarial cases only).
        dataset = dataset.filter(lambda s: s.metadata.get("difficulty") == difficulty)
    if complexity is not None:
        # Probe a single structure (e.g. only the computation-dependent items).
        dataset = dataset.filter(lambda s: s.metadata.get("complexity") == complexity)
    return Task(
        dataset=dataset,
        solver=grounding_judge(judge_style),
        scorer=grounding_agreement(),
    )
