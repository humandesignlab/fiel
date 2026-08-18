"""Run the multi-model x prompt-variant judge A/B (Increment 4).

Default scope is the approved LEAN plan: the full model x prompt grid on the two
cheaper judges, plus a single Opus reference pass on the fair (generic) prompt --
because Opus is saturated (74/74 across probes), paying frontier prices for its
naive/enumerated cells buys almost nothing.

    Haiku 4.5  x {naive, generic, enumerated}
    Sonnet 5   x {naive, generic, enumerated}
    Opus 5     x {generic}                       <- single expensive reference, runs LAST

Cells run cheapest-first so you can watch spend and stop before the Opus pass if the
cap is threatened. Each cell logs to logs/ab/<model>__<style>/.

Usage:
    uv run python scripts/run_ab.py --dry-run           # print the plan, run nothing
    uv run python scripts/run_ab.py                      # run the full lean grid
    uv run python scripts/run_ab.py --models haiku sonnet
    uv run python scripts/run_ab.py --only opus          # just the Opus reference pass

Requires ANTHROPIC_API_KEY in the environment (Inspect loads .env automatically).
"""

from __future__ import annotations

import argparse
import sys

MODELS = {
    "haiku": "anthropic/claude-haiku-4-5",
    "sonnet": "anthropic/claude-sonnet-5",
    "opus": "anthropic/claude-opus-5",
}

# Cost-ordered lean grid: (model_key, judge_style). Opus generic runs last.
LEAN_CELLS: list[tuple[str, str]] = [
    ("haiku", "naive"),
    ("haiku", "generic"),
    ("haiku", "enumerated"),
    ("sonnet", "naive"),
    ("sonnet", "generic"),
    ("sonnet", "enumerated"),
    ("opus", "generic"),
]

LOG_ROOT = "logs/ab"


def _plan(models: list[str] | None, prompts: list[str] | None) -> list[tuple[str, str]]:
    cells = LEAN_CELLS
    if models:
        cells = [c for c in cells if c[0] in models]
    if prompts:
        cells = [c for c in cells if c[1] in prompts]
    return cells


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", choices=list(MODELS), help="restrict models")
    ap.add_argument("--prompts", nargs="*",
                    choices=["naive", "generic", "enumerated"], help="restrict prompts")
    ap.add_argument("--only", choices=list(MODELS), help="shortcut for --models X")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    ap.add_argument("--eval-set", default="data/sec/eval_set.jsonl")
    ap.add_argument("--corpus", default="data/sec/corpus.jsonl")
    args = ap.parse_args()

    models = [args.only] if args.only else args.models
    cells = _plan(models, args.prompts)
    if not cells:
        raise SystemExit("No cells selected.")

    print(f"A/B plan ({len(cells)} cells, cheapest-first):")
    for i, (mk, style) in enumerate(cells, 1):
        print(f"  {i}. {MODELS[mk]:32s}  judge_style={style:11s}  -> {LOG_ROOT}/{mk}__{style}")
    print(f"eval_set={args.eval_set}  corpus={args.corpus}")

    if args.dry_run:
        print("\n[dry run] nothing executed.")
        return

    # Import Inspect lazily so --dry-run stays instant and dependency-free.
    from inspect_ai import eval as inspect_eval

    from fiel.task import grounding_reliability

    for i, (mk, style) in enumerate(cells, 1):
        model = MODELS[mk]
        log_dir = f"{LOG_ROOT}/{mk}__{style}"
        if mk == "opus":
            print("\n" + "=" * 70)
            print("NEXT IS THE OPUS REFERENCE PASS (frontier pricing). Ctrl-C now to skip.")
            print("=" * 70)
        print(f"\n>>> [{i}/{len(cells)}] {model}  judge_style={style}  -> {log_dir}")
        inspect_eval(
            grounding_reliability,
            model=model,
            task_args={
                "judge_style": style,
                "eval_set": args.eval_set,
                "corpus": args.corpus,
            },
            log_dir=log_dir,
        )

    print("\nDone. Summarize with:  uv run python scripts/show_ab.py")


if __name__ == "__main__":
    main()
