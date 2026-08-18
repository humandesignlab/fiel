"""Reliability statistics for the judge A/B — the actual deliverable.

Inspect's built-in metrics stop at accuracy/stderr. The interesting, hireable numbers
live here and are computed from the logged per-item scores:

  - accuracy with a Wilson confidence interval,
  - recall on CATCHING ungrounded answers (the safety-critical metric) + its CI,
  - the opposite error: the over-flag rate on grounded answers (alert fatigue),
  - Cohen's kappa of judge-vs-gold (chance-corrected agreement),
  - two-proportion z-tests between judges/prompts (is a gap real or noise?),
  - breakdowns by difficulty / complexity / failure mode.

Everything is corpus-agnostic: it reads whatever labels the eval logged.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log

# Positive class = "ungrounded": catching a hallucination is the event we care about.
UNGROUNDED = "ungrounded"
GROUNDED = "grounded"


@dataclass(frozen=True)
class ItemResult:
    item_id: str
    gold: str
    verdict: str | None  # None => parse failure
    correct: bool
    parse_failed: bool
    difficulty: str | None
    complexity: str | None
    failure_mode: str | None


@dataclass
class Cell:
    """One (model, prompt) condition of the A/B."""

    model: str
    judge_style: str
    log_path: str
    results: list[ItemResult]

    @property
    def name(self) -> str:
        short = self.model.split("/")[-1]
        return f"{short} | {self.judge_style}"


def _latest_log(log_dir: str) -> str | None:
    logs = list_eval_logs(log_dir)
    if not logs:
        return None
    return sorted(logs, key=lambda x: x.name)[-1].name


def load_cell(log_dir_or_file: str) -> Cell:
    """Load one A/B cell from a specific .eval file or a directory (latest within)."""
    path = log_dir_or_file
    if Path(path).is_dir() or not path.endswith(".eval"):
        latest = _latest_log(path)
        if latest is None:
            raise SystemExit(f"No .eval logs under {path!r}.")
        path = latest
    log = read_eval_log(path)
    style = (log.eval.task_args or {}).get("judge_style", "generic")
    results: list[ItemResult] = []
    for s in log.samples or []:
        sc = s.scores["grounding_agreement"]
        md = sc.metadata
        results.append(
            ItemResult(
                item_id=str(s.id),
                gold=md["gold_label"],
                verdict=md.get("judge_verdict"),
                correct=sc.value == "C",
                parse_failed=bool(md.get("parse_failed")),
                difficulty=md.get("difficulty"),
                complexity=md.get("complexity"),
                failure_mode=md.get("failure_mode"),
            )
        )
    return Cell(model=log.eval.model, judge_style=style, log_path=path, results=results)


def discover_cells(root: str) -> list[Cell]:
    """Load every cell under a root dir (each immediate subdir is one cell)."""
    root_path = Path(root)
    if not root_path.exists():
        return []
    cells: list[Cell] = []
    for sub in sorted(p for p in root_path.iterdir() if p.is_dir()):
        if _latest_log(str(sub)) is not None:
            cells.append(load_cell(str(sub)))
    return cells


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (robust at small n / extremes)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _confusion(results: list[ItemResult]) -> dict[str, int]:
    """TP/FP/FN/TN with ungrounded as the positive class. Parse fails count as the
    judge NOT flagging (conservative: a hallucination the judge failed to surface)."""
    tp = fp = fn = tn = 0
    for r in results:
        said_ung = r.verdict == UNGROUNDED  # parse-fail (None) => not flagged
        if r.gold == UNGROUNDED:
            tp += int(said_ung)
            fn += int(not said_ung)
        else:  # gold grounded
            fp += int(said_ung)
            tn += int(not said_ung)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def cohen_kappa(results: list[ItemResult]) -> float | None:
    """Chance-corrected agreement between judge verdict and gold (binary).

    Computed only over items with a parsed verdict; returns None if undefined.
    """
    pairs = [(r.gold, r.verdict) for r in results if r.verdict is not None]
    n = len(pairs)
    if n == 0:
        return None
    labels = [GROUNDED, UNGROUNDED]
    po = sum(g == v for g, v in pairs) / n
    kappa_num = po
    pe = 0.0
    for lab in labels:
        p_gold = sum(g == lab for g, _ in pairs) / n
        p_judge = sum(v == lab for _, v in pairs) / n
        pe += p_gold * p_judge
    if pe == 1.0:
        return 1.0 if po == 1.0 else None
    return (kappa_num - pe) / (1 - pe)


def two_proportion_ztest(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Pooled two-proportion z-test. Returns (z, two-sided p). (0,1) if degenerate."""
    if n1 == 0 or n2 == 0:
        return (0.0, 1.0)
    p1, p2 = k1 / n1, k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return (0.0, 1.0)
    z = (p1 - p2) / se
    # Two-sided p via the standard normal survival function.
    p = 2 * (1 - _norm_cdf(abs(z)))
    return (z, p)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def cell_metrics(cell: Cell) -> dict[str, object]:
    """The headline numbers for one A/B cell."""
    r = cell.results
    n = len(r)
    n_correct = sum(x.correct for x in r)
    n_parse_fail = sum(x.parse_failed for x in r)
    c = _confusion(r)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]

    acc = n_correct / n if n else 0.0
    recall = tp / (tp + fn) if (tp + fn) else float("nan")  # catch rate on ungrounded
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    over_flag = fp / (fp + tn) if (fp + tn) else float("nan")  # alert-fatigue rate

    return {
        "name": cell.name,
        "model": cell.model,
        "judge_style": cell.judge_style,
        "n": n,
        "n_correct": n_correct,
        "accuracy": acc,
        "accuracy_ci": wilson_ci(n_correct, n),
        "recall": recall,
        "recall_ci": wilson_ci(tp, tp + fn),
        "recall_counts": (tp, tp + fn),
        "precision": precision,
        "over_flag_rate": over_flag,
        "over_flag_counts": (fp, fp + tn),
        "kappa": cohen_kappa(r),
        "parse_failures": n_parse_fail,
        "confusion": c,
    }


def breakdown(cell: Cell, key: str) -> dict[str, tuple[int, int]]:
    """Per-group (correct, total) for key in {difficulty, complexity, failure_mode}."""
    groups: dict[str, list[bool]] = defaultdict(list)
    for r in cell.results:
        groups[getattr(r, key) or "?"].append(r.correct)
    return {g: (sum(v), len(v)) for g, v in sorted(groups.items())}


def mode_recall(cell: Cell) -> dict[str, tuple[int, int]]:
    """Catch rate per ungrounding failure mode: (caught, total) over ungrounded items.

    'Caught' = judge said ungrounded. This is recall restricted to a single failure
    mode, and it's where the model differences actually live (overgeneralization).
    """
    groups: dict[str, list[bool]] = defaultdict(list)
    for r in cell.results:
        if r.gold != UNGROUNDED:
            continue
        groups[r.failure_mode or "?"].append(r.verdict == UNGROUNDED)
    return {g: (sum(v), len(v)) for g, v in sorted(groups.items())}


def bootstrap_ci(
    flags: list[bool], n_boot: int = 20000, alpha: float = 0.05, seed: int = 12345
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of a 0/1 sample.

    Complements the Wilson interval: Wilson is the analytic small-n interval; the
    bootstrap resamples this specific item set, so agreement between the two is a
    reassuring cross-check rather than new information.
    """
    n = len(flags)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    xs = [1.0 if f else 0.0 for f in flags]
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += xs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


def accuracy_flags(cell: Cell) -> list[bool]:
    return [r.correct for r in cell.results]


def recall_flags(cell: Cell) -> list[bool]:
    """0/1 caught-flags over the ungrounded items only (basis for recall)."""
    return [r.verdict == UNGROUNDED for r in cell.results if r.gold == UNGROUNDED]


def required_n_two_prop(
    p1: float, p2: float, alpha: float = 0.05, power: float = 0.80
) -> float | None:
    """Approx N PER GROUP to detect a p1-vs-p2 gap at given alpha/power (two-sided).

    Uses the arcsine (Cohen's h) effect size. Returns None if the effect is ~0.
    Owning this number is the point: it says exactly how underpowered N=181 is.
    """
    h = abs(2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2)))
    if h < 1e-9:
        return None
    z_alpha = 1.959963985  # two-sided 0.05
    z_beta = 0.841621234  # power 0.80
    return ((z_alpha + z_beta) / h) ** 2
