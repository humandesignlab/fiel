"""Summarize a judge-validation run from its Inspect log.

Run: python scripts/show_results.py [path]
  path may be a specific .eval log, or a directory (uses the latest log within it).
  With no arg, uses the most recent log under logs/ (excluding the mock dir).

Previews the real deliverable: overall agreement, a confusion matrix (judge verdict
vs. ground truth), recall on CATCHING ungrounded answers (the safety-critical metric),
and breakdowns by difficulty and failure mode. Full statistics land in Increment 5.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log


def _latest_in(log_dir: str, *, exclude_mock: bool = False) -> str:
    logs = list_eval_logs(log_dir)
    if exclude_mock:
        # Ignore scratch dirs (convention: leading underscore, e.g. logs/_mock).
        logs = [lg for lg in logs if not any(p.startswith("_") for p in Path(lg.name).parts)]
    if not logs:
        raise SystemExit(f"No logs found under {log_dir!r}. Run an eval first.")
    return sorted(logs, key=lambda x: x.name)[-1].name


def _resolve_log(arg: str | None) -> str:
    if arg is None:
        return _latest_in("logs", exclude_mock=True)
    if Path(arg).is_dir():
        return _latest_in(arg)
    return arg


def main() -> None:
    log_path = _resolve_log(sys.argv[1] if len(sys.argv) > 1 else None)
    log = read_eval_log(log_path)

    model = log.eval.model
    print(f"log:   {log_path}")
    print(f"model: {model}")
    print(f"status: {log.status}, samples: {len(log.samples)}\n")

    # Confusion: rows = gold, cols = judge verdict.
    conf: dict[tuple[str, str], int] = defaultdict(int)
    n_correct = n_parse_fail = 0
    by_diff: dict[str, list[bool]] = defaultdict(list)
    by_mode: dict[str, list[bool]] = defaultdict(list)
    by_cplx: dict[str, list[bool]] = defaultdict(list)
    rows = []

    for s in log.samples:
        sc = s.scores["grounding_agreement"]
        md = sc.metadata
        gold = md["gold_label"]
        verdict = md["judge_verdict"] or "PARSE_FAIL"
        correct = sc.value == "C"
        n_correct += int(correct)
        n_parse_fail += int(md["parse_failed"])
        conf[(gold, verdict)] += 1
        by_diff[md["difficulty"]].append(correct)
        by_mode[md["failure_mode"]].append(correct)
        by_cplx[md.get("complexity") or "single"].append(correct)
        rows.append((s.id, gold, verdict, md["difficulty"], md["failure_mode"], correct))

    n = len(log.samples)
    print(f"OVERALL agreement: {n_correct}/{n} = {n_correct / n:.1%}"
          + (f"   (parse failures: {n_parse_fail})" if n_parse_fail else ""))

    # Recall on catching ungrounded = of truly ungrounded items, how many judge flagged.
    ung_total = sum(v for (g, _), v in conf.items() if g == "ungrounded")
    ung_caught = conf.get(("ungrounded", "ungrounded"), 0)
    if ung_total:
        print(f"RECALL on catching ungrounded: {ung_caught}/{ung_total} = {ung_caught / ung_total:.1%}"
              "   <- the safety-critical metric")

    print("\nconfusion (gold \\ judge):")
    labels = ["grounded", "ungrounded", "PARSE_FAIL"]
    present = [c for c in labels if any(k[1] == c for k in conf)]
    print("            " + "".join(f"{c:>12s}" for c in present))
    for g in ["grounded", "ungrounded"]:
        print(f"  {g:10s}" + "".join(f"{conf.get((g, c), 0):>12d}" for c in present))

    print("\nby difficulty:")
    for k, v in sorted(by_diff.items()):
        print(f"  {k:6s} {sum(v)}/{len(v)} = {sum(v) / len(v):.0%}")

    print("\nby complexity:")
    for k, v in sorted(by_cplx.items()):
        print(f"  {k:12s} {sum(v)}/{len(v)} = {sum(v) / len(v):.0%}")

    print("\nby failure mode:")
    for k, v in sorted(by_mode.items()):
        print(f"  {k:24s} {sum(v)}/{len(v)} = {sum(v) / len(v):.0%}")

    print("\nper-item:")
    for item_id, gold, verdict, diff, mode, correct in rows:
        mark = "ok " if correct else "MISS"
        print(f"  [{mark}] {item_id}  gold={gold:10s} judge={verdict:11s} {diff:5s} {mode}")


if __name__ == "__main__":
    main()
