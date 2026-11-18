# Fiel: audit the judge

*"Quis custodiet ipsos custodes?" Who audits the judge?*

LLM-as-judge is everywhere in RAG evals, but the judge is rarely itself validated.
**Fiel is a corpus-agnostic harness that measures whether an LLM judge reliably
catches *ungrounded* answers**, comparing its verdicts against controlled
ground-truth labels and reporting *exactly where* the judge can and cannot be trusted.
The corpus and labeled set are inputs; **SEC 10-K filings are only the shipped example.**

## The headline: when can you trust an LLM judge?

**For single-passage grounding of financial filings, yes: even a cheap judge is a
reliable grader, within specific limits.** Across 181 labeled items (79 grounded / 102
ungrounded) from four companies' 10-Ks, every current Anthropic judge agreed with
controlled gold ≥97% of the time (κ ≥ 0.94) and caught ≥97% of ungrounded answers.
**But the residual risk is narrow and specific, and inseparable from that claim:**
subtle **scope/quantifier broadening** ("some of" → "all"), and the **cheapest model**,
which is the only one that *also* raises false alarms on faithful text. This is scoped
to single-passage grounding; it does **not** extend to multi-passage or long-context.

**A/B results (fair "generic" faithfulness prompt, N=181):**

| Judge | Accuracy | Catch rate (ungrounded) | Over-flag (grounded) | Cohen's κ |
|---|---|---|---|---|
| Claude Haiku 4.5 | 97.2% | 99 / 102 | 2 / 79 | 0.94 |
| Claude Sonnet 5 | 99.4% | 101 / 102 | 0 / 79 | 0.99 |
| Claude Opus 5 | 100.0% | 102 / 102 | 0 / 79 | 1.00 |

The model gradient is real in *direction* but **not statistically significant at
N=181**. Naming the failure taxonomy in the prompt gave **no** measurable boost (no
teaching-to-the-test inflation). See **[FINDINGS.md](FINDINGS.md)** for CIs,
significance tests, the per-failure-mode localization, the adversarial probe history,
and the power analysis.

## Deploy verdict (one line)

**Would I ship an LLM judge for single-passage financial grounding?** Yes, even a
cheap one, but with a human check on scope/quantifier claims, and not the cheapest
model where false alarms are costly; and *none of this transfers to multi-passage /
long-context grounding, which is untested.* Full, bounded verdict is in
**[FINDINGS.md § Deploy verdict](FINDINGS.md#deploy-verdict)**.

## Run it

```bash
uv sync
cp .env.example .env        # add ANTHROPIC_API_KEY (and OPENAI_API_KEY if desired)

# One judge, one prompt, on the shipped SEC set:
uv run inspect eval src/fiel/task.py --model anthropic/claude-haiku-4-5 -T judge_style=generic

# The full multi-model A/B grid, then the analysis:
uv run python scripts/run_ab.py            # cheapest-first; Opus runs last behind a guard
uv run python scripts/show_ab.py           # comparison table + significance tests
uv run python scripts/show_focus.py        # per-mode CIs, bootstrap cross-check, power calc
```

Task knobs (override with `-T`): `judge_style` = `naive` | `generic` | `enumerated`;
`difficulty` = `easy` | `hard`; `complexity` = `single` | `multi_claim` | `computation`.

## Run it on *your* judge and *your* corpus

Fiel is corpus-agnostic; nothing about SEC is hardcoded. Provide two JSONL files:

- **corpus**: `Passage` records: `{passage_id, text, metadata}`.
- **eval set**: `EvalItem` records: `{item_id, passage_id, question, answer, label,
  failure_mode, difficulty, complexity, rationale}`, where `label` ∈ {grounded,
  ungrounded} is *your controlled ground truth*.

Then point the task at them and swap in any Inspect-supported judge:

```bash
uv run inspect eval src/fiel/task.py \
  --model openai/gpt-... \
  -T eval_set=path/to/your_eval_set.jsonl -T corpus=path/to/your_corpus.jsonl
```

The schema is defined in [`src/fiel/schema.py`](src/fiel/schema.py).

## How it works (design in one breath)

The **judge is the system under test**, modeled as an Inspect *solver*; the controlled
label is the *target*. A custom *scorer* ([`src/fiel/scorer.py`](src/fiel/scorer.py))
compares verdict vs. gold and records failure-mode / difficulty / complexity for the
breakdowns. Reliability statistics that Inspect doesn't ship (recall CIs, Cohen's κ,
two-proportion z-tests, bootstrap, power) live in
[`src/fiel/analysis/stats.py`](src/fiel/analysis/stats.py). No vector DB, no retrieval
pipeline, no web UI. The point is judge validation, not RAG plumbing.

## Repo layout

```
src/fiel/
  schema.py        # Passage / EvalItem / Verdict: the corpus-agnostic contract
  corpus.py        # load/save passages (+ provenance headers)
  dataset.py       # join items+passages -> Inspect samples (judge sees Q+passage+answer)
  judge.py         # the LLM-as-judge solver: naive / generic / enumerated prompts
  scorer.py        # verdict-vs-gold validation scorer
  task.py          # wires dataset + judge + scorer into an Inspect task
  analysis/stats.py# recall CIs, kappa, z-tests, bootstrap, power
scripts/           # build corpus/items, run the A/B, summarize results
data/sec/          # shipped example: corpus.jsonl + eval_set.jsonl (181 labeled items)
FINDINGS.md        # the writeup: results, CIs, probe arc, limits, deploy verdict
```

## Limits (up front)

Single-passage grounding only; 181 items across 4 issuers in one domain; the model
ranking is underpowered at this N; and because the passages are real recent filings, a
judge's numeric "catches" could in principle lean on recalled figures rather than the
passage. All of this is characterized honestly in [FINDINGS.md](FINDINGS.md).

---
MIT licensed. Built with [Inspect AI](https://inspect.aisi.org.uk/). Data: public SEC
EDGAR filings.

Built by Humberto Mangino. **[LinkedIn:](https://www.linkedin.com/in/humbertomangino)  linkedin.com/in/humbertomangino**
