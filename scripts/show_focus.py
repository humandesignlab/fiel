"""Tighten the finding: the overgeneralization localization, with honest uncertainty.

The whole story now rests on one axis — residual judge error concentrates in
overgeneralization / dropped quantifiers — so this script quantifies exactly that,
with confidence bands, plus:
  - per-failure-mode catch rate across judges (Wilson CIs) — the overgeneralization row
    is the one that separates models,
  - bootstrap CIs on whole-cell accuracy and recall for the generic-prompt cells (a
    cross-check on the analytic Wilson intervals),
  - the required-N power calc: how big the eval would need to be to resolve the
    directional model gaps we see at N=181.

Run: uv run python scripts/show_focus.py [logs/ab]
"""

from __future__ import annotations

import math
import sys

from fiel.analysis.stats import (
    accuracy_flags,
    bootstrap_ci,
    cell_metrics,
    discover_cells,
    mode_recall,
    recall_flags,
    required_n_two_prop,
    wilson_ci,
)

MODEL_ORDER = ["haiku", "sonnet", "opus"]


def _short(model: str) -> str:
    m = model.split("/")[-1]
    for k in MODEL_ORDER:
        if k in m:
            return k
    return m


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "logs/ab"
    cells = discover_cells(root)
    if not cells:
        raise SystemExit(f"No A/B cells under {root!r}. Run scripts/run_ab.py first.")

    generic = {_short(c.model): c for c in cells if c.judge_style == "generic"}

    # 1) Per-failure-mode catch rate across the generic-prompt judges, with Wilson CIs.
    print("=" * 78)
    print("PER-FAILURE-MODE CATCH RATE (generic prompt), caught/total with 95% Wilson CI")
    print("=" * 78)
    modes: list[str] = []
    for c in generic.values():
        for m in mode_recall(c):
            if m not in modes:
                modes.append(m)
    modes.sort()
    order = [k for k in MODEL_ORDER if k in generic]
    header = f"{'failure_mode':24s}" + "".join(f"{k:>22s}" for k in order)
    print(header)
    print("-" * len(header))
    for mode in modes:
        row = f"{mode:24s}"
        for k in order:
            mr = mode_recall(generic[k]).get(mode)
            if mr is None:
                row += f"{'-':>22s}"
            else:
                caught, tot = mr
                lo, hi = wilson_ci(caught, tot)
                row += f"{f'{caught}/{tot} [{lo:.0%},{hi:.0%}]':>22s}"
        print(row)
    print("\nNote: with ~12 items per mode the CIs are wide and overlap heavily — the "
          "overgeneralization gap\nis a localized, directional signal, not a resolved "
          "difference. That honesty is the point.")

    # 2) Bootstrap cross-check on whole-cell accuracy + recall (generic cells).
    print("\n" + "=" * 78)
    print("BOOTSTRAP vs WILSON (generic prompt) — should agree closely")
    print("=" * 78)
    print(f"{'judge':8s} {'accuracy':>9s} {'Wilson 95%':>16s} {'bootstrap 95%':>18s}"
          f" {'recall':>8s} {'boot recall 95%':>18s}")
    for k in order:
        c = generic[k]
        m = cell_metrics(c)
        acc_boot = bootstrap_ci(accuracy_flags(c))
        rec_boot = bootstrap_ci(recall_flags(c))
        print(
            f"{k:8s} {m['accuracy']:>9.1%} "
            f"{f'[{m['accuracy_ci'][0]:.0%},{m['accuracy_ci'][1]:.0%}]':>16s} "
            f"{f'[{acc_boot[0]:.0%},{acc_boot[1]:.0%}]':>18s} "
            f"{m['recall']:>8.1%} "
            f"{f'[{rec_boot[0]:.0%},{rec_boot[1]:.0%}]':>18s}"
        )

    # 3) Power: required N per group to resolve the directional model gaps at generic.
    print("\n" + "=" * 78)
    print("POWER — N PER GROUP to detect the observed gap (alpha=0.05, power=0.80)")
    print("=" * 78)
    pairs = [("haiku", "sonnet"), ("sonnet", "opus")]
    for a, b in pairs:
        if a not in generic or b not in generic:
            continue
        pa = cell_metrics(generic[a])["accuracy"]
        pb = cell_metrics(generic[b])["accuracy"]
        need = required_n_two_prop(pa, pb)
        need_s = "n/a (no observed gap)" if need is None else f"~{math.ceil(need)} per group"
        print(f"  {a} ({pa:.1%}) vs {b} ({pb:.1%}): {need_s}   [we have 181]")
    print("\nTakeaway: at N=181 these model gaps are directional, not significant. The "
          "harness localizes\nWHERE trust erodes (overgeneralization, cheapest judge); "
          "resolving the model ranking would\nrequire a much larger labeled set.")


if __name__ == "__main__":
    main()
