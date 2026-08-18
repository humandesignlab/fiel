"""Summarize the judge A/B across all cells (Increment 5 preview).

Reads every cell under logs/ab/ (or a dir you pass) and prints:
  - a comparison table: accuracy (95% CI), recall on catching ungrounded (95% CI),
    over-flag rate on grounded answers, Cohen's kappa, and parse failures,
  - the headline significance tests (two-proportion z-tests) for the model gradient
    at the fair prompt and for prompt sensitivity within each model,
  - per-cell breakdowns by complexity / difficulty / failure mode.

Run: uv run python scripts/show_ab.py [logs/ab]
"""

from __future__ import annotations

import sys

from fiel.analysis.stats import (
    breakdown,
    cell_metrics,
    discover_cells,
    two_proportion_ztest,
)


def _pct(x: float) -> str:
    return "  n/a" if x != x else f"{x:5.1%}"  # NaN check


def _ci(lo_hi: tuple[float, float]) -> str:
    return f"[{lo_hi[0]:.0%},{lo_hi[1]:.0%}]"


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "logs/ab"
    cells = discover_cells(root)
    if not cells:
        raise SystemExit(f"No A/B cells found under {root!r}. Run scripts/run_ab.py first.")

    mets = [cell_metrics(c) for c in cells]
    by_name = {m["name"]: m for m in mets}

    print(f"A/B cells under {root!r}: {len(cells)}\n")
    hdr = f"{'cell':28s} {'n':>4s} {'acc':>7s} {'acc 95% CI':>12s} {'recall':>7s} {'rec 95% CI':>12s} {'overflag':>9s} {'kappa':>6s} {'pf':>3s}"
    print(hdr)
    print("-" * len(hdr))
    for m in sorted(mets, key=lambda x: (x["model"], x["judge_style"])):
        kappa = m["kappa"]
        print(
            f"{m['name']:28s} {m['n']:>4d} {_pct(m['accuracy']):>7s} {_ci(m['accuracy_ci']):>12s} "
            f"{_pct(m['recall']):>7s} {_ci(m['recall_ci']):>12s} {_pct(m['over_flag_rate']):>9s} "
            f"{(f'{kappa:.2f}' if kappa is not None else ' n/a'):>6s} {m['parse_failures']:>3d}"
        )

    # Recall counts drive the safety story; show them explicitly.
    print("\ncatch rate on ungrounded (TP / total ungrounded), over-flag (FP / total grounded):")
    for m in sorted(mets, key=lambda x: (x["model"], x["judge_style"])):
        tp, ung = m["recall_counts"]
        fp, gr = m["over_flag_counts"]
        print(f"  {m['name']:28s} caught {tp}/{ung} ungrounded,  over-flagged {fp}/{gr} grounded")

    _sig_tests(by_name)
    _breakdowns(cells)


def _z(a: dict, b: dict, field: str) -> str:
    """Two-proportion z-test on a count/total field between two cells."""
    if field == "accuracy":
        k1, n1 = a["n_correct"], a["n"]
        k2, n2 = b["n_correct"], b["n"]
    else:  # recall
        k1, n1 = a["recall_counts"]
        k2, n2 = b["recall_counts"]
    z, p = two_proportion_ztest(k1, n1, k2, n2)
    star = "  <- significant (p<0.05)" if p < 0.05 else ""
    return f"z={z:+.2f}, p={p:.3f}{star}"


def _sig_tests(by_name: dict[str, dict]) -> None:
    def find(model_sub: str, style: str) -> dict | None:
        for name, m in by_name.items():
            if model_sub in name and m["judge_style"] == style:
                return m
        return None

    print("\n--- significance: model gradient at the fair (generic) prompt ---")
    order = ["haiku", "sonnet", "opus"]
    generic = [(k, find(k, "generic")) for k in order]
    generic = [(k, m) for k, m in generic if m]
    for i in range(len(generic) - 1):
        (ka, a), (kb, b) = generic[i], generic[i + 1]
        print(f"  {ka} vs {kb}  accuracy: {_z(a, b, 'accuracy')}")
        print(f"  {ka} vs {kb}  recall:   {_z(a, b, 'recall')}")

    print("\n--- significance: prompt sensitivity within each model ---")
    for k in order:
        naive, generic_m, enum = find(k, "naive"), find(k, "generic"), find(k, "enumerated")
        if naive and generic_m:
            print(f"  {k}: naive vs generic     accuracy: {_z(naive, generic_m, 'accuracy')}")
        if generic_m and enum:
            print(f"  {k}: generic vs enumerated accuracy: {_z(generic_m, enum, 'accuracy')}  "
                  "(gap = teaching-to-the-test)")


def _breakdowns(cells: list) -> None:
    print("\n--- per-cell breakdowns (correct/total) ---")
    for c in sorted(cells, key=lambda x: (x.model, x.judge_style)):
        print(f"\n{c.name}")
        for key in ("complexity", "difficulty", "failure_mode"):
            parts = [f"{g}={k}/{n}" for g, (k, n) in breakdown(c, key).items()]
            print(f"  {key:12s} " + "  ".join(parts))


if __name__ == "__main__":
    main()
