# Fiel: Findings

*How reliably do current LLM judges catch ungrounded answers, and where do they fail?*

This is the writeup behind the one-line answer in the [README](README.md). It reports
the full A/B results with confidence intervals and significance tests, the adversarial
probe history that makes the finding credible, the honest limits, and (the part that
matters most) a specific, bounded **[deploy verdict](#deploy-verdict)**.

---

## 1. The headline answer

**For single-passage grounding, today's LLM judges are reliable graders, cheap ones
included.** On 181 controlled items, Claude Haiku 4.5, Sonnet 5, and Opus 5 each agreed
with ground-truth labels ≥97% of the time (Cohen's κ ≥ 0.94) and each caught ≥97% of
ungrounded answers. Three things sharpen this:

1. **The residual error is localized, not diffuse.** Of six injected failure modes,
   five are *saturated*: every judge catches ~all of them. The **only** lossy mode is
   **overgeneralization / dropped quantifiers** ("some of our competitors" rendered as
   "our competitors"). That single semantic mode carries the entire model gradient.
2. **Prompt taxonomy doesn't inflate scores.** A generic faithfulness prompt performed
   as well as one that spells out the exact failure taxonomy this eval injects, i.e.
   **no teaching-to-the-test effect**. A capable judge doesn't need to be handed the
   answer key.
3. **The cheapest judge fails in *both* directions.** Haiku is the only model that both
   slightly under-catches ungrounded answers *and* over-flags faithful ones, the
   alert-fatigue direction that Sonnet and Opus never exhibit here.

The model ranking (Haiku < Sonnet < Opus) is **directional but not statistically
significant at N=181**, and we report that as a finding about statistical power, not as
a shortcoming (see [§5](#5-power-owning-the-underpowering)).

---

## 2. Setup

**Task.** The judge is the *system under test*. It sees a source passage, a question,
and a candidate answer, and must return `grounded` / `ungrounded`. We compare that
verdict against a **controlled** ground-truth label and score agreement. This is
implemented in Inspect AI: the judge is a *solver*, the label is the *target*, and a
custom *scorer* records the verdict plus held-out metadata for breakdowns.

**Dataset (181 items).** Hand-labeled over 17 passages from four 10-K filings: Apple
FY2025, Microsoft FY2026, NVIDIA FY2026, Tesla FY2025 (MD&A, revenue notes, risk
factors).

| Axis | Distribution |
|---|---|
| Label | 79 grounded / 102 ungrounded |
| Difficulty | 35 easy / 146 hard |
| Complexity | 127 single-claim / 34 multi-claim / 20 computation |
| Failure mode (ungrounded) | subtle_number 42 · unsupported_causation 13 · hedge_removal 12 · overgeneralization 12 · temporal_scope_drift 12 · unsupported_addition 11 |

Ungrounded items are *deliberately subtle*: right-structure/wrong-figure numbers, one
buried error inside an otherwise-correct multi-claim answer, wrong derived values that
only arithmetic catches, dropped qualifiers, removed hedges, and invented causes.

**Judges.** `anthropic/claude-haiku-4-5`, `claude-sonnet-5`, `claude-opus-5`.

**Prompt variants.** `naive` (one line), `generic` (thorough but corpus-agnostic, the
fair default), `enumerated` (generic + this eval's exact failure taxonomy).

**Design (approved "lean" grid).** Haiku and Sonnet × all three prompts, plus a single
Opus × generic reference pass. Opus was saturated in probing, so paying frontier prices
for its `naive`/`enumerated` cells buys nothing.

**Metrics.** Accuracy (agreement with gold); **recall = catch rate on ungrounded** (the
safety-critical metric); **over-flag rate** = false positives on grounded answers (alert
fatigue); Cohen's κ (chance-corrected agreement); Wilson + bootstrap CIs;
two-proportion z-tests; a required-N power calc. These live in
[`src/fiel/analysis/stats.py`](src/fiel/analysis/stats.py).

---

## 3. Results

### 3.1 The full grid

| Judge | Prompt | Accuracy (95% CI) | Catch rate (ungrounded, 95% CI) | Over-flag (grounded) | κ |
|---|---|---|---|---|---|
| Haiku 4.5 | naive | 96.7% [93, 98] | 98/102 = 96.1% [90, 98] | 2/79 | 0.93 |
| Haiku 4.5 | generic | 97.2% [94, 99] | 99/102 = 97.1% [92, 99] | 2/79 | 0.94 |
| Haiku 4.5 | enumerated | 97.8% [94, 99] | 99/102 = 97.1% [92, 99] | 1/79 | 0.96 |
| Sonnet 5 | naive | 98.3% [95, 99] | 99/102 = 97.1% [92, 99] | 0/79 | 0.97 |
| Sonnet 5 | generic | 99.4% [97, 100] | 101/102 = 99.0% [95, 100] | 0/79 | 0.99 |
| Sonnet 5 | enumerated | 99.4% [97, 100] | 101/102 = 99.0% [95, 100] | 0/79 | 0.99 |
| Opus 5 | generic | 100.0% [98, 100] | 102/102 = 100% [96, 100] | 0/79 | 1.00 |

No parse failures in any cell (verdict format was 100% machine-parseable).

*(These saturation rates aren't because the cases were easy; see [§4](#4-the-probe-arc-why-this-finding-is-credible), where we escalated difficulty specifically to break the frontier judge across numeric, computational, and qualitative axes, and documented that it held.)*

### 3.2 Significance: the model gradient is directional, not resolved

Two-proportion z-tests at the fair (generic) prompt:

- Haiku vs Sonnet: accuracy z = −1.65, **p = 0.10**; recall z = −1.01, p = 0.31
- Sonnet vs Opus: accuracy z = −1.00, **p = 0.32**; recall p = 0.32

None reach p < 0.05. At this N the ranking is a trend, not a distinguishable difference.

### 3.3 Prompt sensitivity: no teaching-to-the-test

- Haiku: naive→generic p = 0.76; **generic→enumerated p = 0.74**
- Sonnet: naive→generic p = 0.32; **generic→enumerated p = 1.00** (identical: 99.4%)

Handing the judge the exact taxonomy of tricks did **not** improve it. The generic
faithfulness prompt is sufficient: a clean fairness result, and the *opposite* of the
inflation we hedged might appear.

### 3.4 The localization: one mode carries everything

Per-failure-mode catch rate at the generic prompt, with 95% Wilson intervals:

| Failure mode | Haiku | Sonnet | Opus |
|---|---|---|---|
| subtle_number_error | 42/42 [92, 100] | 42/42 [92, 100] | 42/42 [92, 100] |
| hedge_removal | 12/12 [76, 100] | 12/12 [76, 100] | 12/12 [76, 100] |
| temporal_scope_drift | 12/12 [76, 100] | 12/12 [76, 100] | 12/12 [76, 100] |
| unsupported_causation | 13/13 [77, 100] | 13/13 [77, 100] | 13/13 [77, 100] |
| unsupported_addition | 11/11 [74, 100] | 11/11 [74, 100] | 11/11 [74, 100] |
| **overgeneralization** | **9/12 [47, 91]** | **11/12 [65, 99]** | **12/12 [76, 100]** |

Every mode but one is saturated across all judges. **Overgeneralization** is the sole
axis where models separate, and it's the same mode responsible for the single miss in
early probing (a dropped "some of"). The honest caveat: at ~12 items/mode the intervals
are wide and overlap heavily (Haiku 75% [47, 91] vs Opus 100% [76, 100]), so this is a
**localized, directional signal, not a resolved model difference.** But it is a
*specific, actionable* place to distrust the judge.

### 3.5 Bootstrap cross-check

A 20k-resample percentile bootstrap agrees tightly with the analytic Wilson intervals
(generic cells): Haiku accuracy Wilson [94, 99] vs bootstrap [94, 99]; Sonnet recall
bootstrap [97, 100]; Opus [100, 100]. The intervals hold up under resampling.

---

## 4. The probe arc: why this finding is credible

The result "frontier judges are reliable here" is only trustworthy if we genuinely
*tried to break them*. We did, in escalating rounds, and greenlit the full run only
after the ceiling held:

1. **Single-claim hard set (92 items).** Opus 5 scored 91/92; its one miss an
   overgeneralization (dropped quantifier). First sign of the boundary.
2. **A naive-prompt diagnostic** confirmed the rubric wasn't a crutch (Δ ≈ 1 item), so
   difficulty is *intrinsic*: case construction, not prompt phrasing, is what matters.
3. **Numeric multi-claim (20) + computation (20).** Long fluent answers with one buried
   wrong figure, and answers whose error only surfaces if you do the arithmetic. Opus:
   **40/40.** Diagnosis: numbers give a frontier model a *crisp check it owns*.
4. **Buried qualitative errors (14).** Scope/quantifier/hedge/causation deviations
   embedded mid-paragraph in otherwise-impeccable fluent answers: the documented
   "fluent-answer over-approval" weakness, on non-memorizable semantic content. Opus:
   **14/14**, including 6/6 precision on the faithful controls.

Across all three probes Opus went **74/74**. We deliberately **did not** keep authoring
ever-more-contrived items to manufacture a failure; that would overfit the set to a
predetermined conclusion. The credibility of this demo is precisely that we stopped when
the honest boundary appeared and reported it as-is.

---

## 5. Power: owning the underpowering

The model gaps are small, so N=181 cannot resolve them. Using the observed proportions,
the sample size **per group** required to detect each gap at α=0.05, 80% power:

| Comparison (generic) | Observed | N needed / group | We have |
|---|---|---|---|
| Haiku vs Sonnet | 97.2% vs 99.4% | ~229 | 181 |
| Sonnet vs Opus | 99.4% vs 100% | ~355 | 181 |

This is the point, not an apology: at this scale the ranking is indistinguishable from
noise, and here is exactly how much larger a labeled set would need to be to certify one
judge over another. Knowing *when your N can't support a claim* is the job.

---

## 6. Honest limits

- **Single-passage only.** Every item grounds an answer in *one* retrieved chunk.
  Multi-passage synthesis and long-context grounding are untested and the results do not
  transfer to them.
- **Underpowered ranking.** See [§5](#5-power-owning-the-underpowering). Treat the
  model ordering as directional.
- **Small per-mode N.** The overgeneralization signal rests on 12 items/judge; wide CIs.
- **Possible recall confound.** The passages are real, recent filings, so a judge's
  numeric "catches" could in principle draw on recalled figures rather than the passage.
  This is partly why the *qualitative* (non-memorizable) probe matters, and the judges
  handled those too.
- **Narrow corpus.** Four issuers, one domain (financial filings), English.
- **Judge non-determinism.** Single run per cell; verdicts can vary slightly across runs.

---

## Deploy verdict

**Would I put one of these judges into production to police single-passage grounding in
financial-document QA?** Yes, with explicit bounds:

- **A cheap judge is a viable first-line filter.** Even Haiku 4.5 agreed with controlled
  gold on 97.2% of items (κ = 0.94) and caught 99/102 ungrounded answers. You do not
  need a frontier model to get most of the safety value.
- **Add a human (or stronger-model) check on scope/quantifier claims.**
  Overgeneralization / dropped-qualifier errors are the one mode every judge missed some
  of (Haiku 9/12, Sonnet 11/12, and even Opus only 12/12 with a wide band). That is the
  specific place I would not trust the judge unattended.
- **Don't lean on the cheapest model where false alarms are expensive.** Haiku is the
  only judge that also over-flagged faithful answers (2/79). If alert fatigue is costly
  in your setting, the incremental spend on Sonnet buys you a cleaner precision profile.
- **This does not transfer to multi-passage / long-context grounding.** That regime is
  untested; I would re-run Fiel on a labeled set from that regime before trusting a judge
  there.
- **I would not certify one model over another on this evidence.** The Haiku < Sonnet <
  Opus ranking is directional but not significant at N=181; certifying it would require
  the ~230-355 items/group in [§5](#5-power-owning-the-underpowering).

**In practice:** deploy a cheap judge as an automated first-line faithfulness filter,
route scope/quantifier-sensitive and high-stakes cases to a stronger model or a human,
and don't treat any of this as validated beyond single-passage retrieval.

The reusable takeaway is the method, not the leaderboard: **Fiel localizes *where* a
judge's trust erodes and quantifies it with confidence intervals**. Point it at any
judge, on any corpus, with a labeled set, and get the same bounded verdict for your
setting.

---

## Reproduce

```bash
uv sync && cp .env.example .env      # add ANTHROPIC_API_KEY
uv run python scripts/run_ab.py      # 7-cell lean grid (Opus runs last, behind a guard)
uv run python scripts/show_ab.py     # grid + significance
uv run python scripts/show_focus.py  # per-mode CIs, bootstrap, power
```

Logs are written per cell under `logs/ab/<model>__<prompt>/`. The shipped labeled set is
`data/sec/eval_set.jsonl` (181 items) over `data/sec/corpus.jsonl`.
