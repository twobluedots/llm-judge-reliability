# Scope: LLM judge reliability

## Question
How far can you trust an LLM judge, and when does it fail?

## Sub-questions
1. Agreement: how often do judges agree with humans? (accuracy, Cohen's kappa)
2. Position bias: does swapping answer order change the verdict? (flip rate + direction)
3. Verbosity bias: does padding an answer shift the verdict? (preference shift)

## Data
- MT-Bench human judgments (lmsys/mt_bench_human_judgments), 100 pairs, pairwise
  - turn 1 only
  - spread across all 8 categories; don't let lopsided matchups dominate
  - keep ties; include some pairs with several expert votes (human–human ceiling)
- Second Brain, 50 answers, pointwise pass/fail, labelled by me with a short rubric
  - aim for ~30–40% fails (otherwise kappa is unstable)

## Setup (locked for the week)
- Judge models: claude-haiku-5-5 (thinking off) + gpt-4o-mini-2024-07-18; 3rd optional
- Judge prompts: TBD (2 per dataset)
- Temperature 0 where the API allows it (gpt-4o-mini); Claude rejects temperature,
  so default sampling, recorded per call. All 100 MT-Bench pairs re-judged as a noise baseline
  (first 20, extended 9 Nov, see Decisions).
- Exact model versions recorded
- Every call cached in runs/ keyed by (item, prompt, model, order, repeat)

## Pipeline
Inputs (written by me, in git): data/, prompts/, config.yaml
Outputs (written by scripts): runs/, results/

    src/prepare_data.py  ->  data/          sample MT-Bench, load my labels
    src/run_judges.py    ->  runs/*.jsonl   API calls: slow, costs money, cached
    src/analyze.py       ->  results/       metrics, CIs, tables: fast, free

Analysis only reads runs/, so re-running it never costs an API call.

## Decisions
- Judges: Haiku 5.5 + gpt-4o-mini, same price tier, different families.
  Why: no spend on a strong judge this week; clean family/self-preference comparison
  (gpt-4o-mini generated the Second Brain answers). Third slot stays open.
- Haiku thinking off.
  Why: gpt-4o-mini doesn't reason, so family isn't confounded with reasoning. Thinking-on is a later variant.
- Mixed protocol: pairwise for MT-Bench, pointwise for Second Brain.
  Why: matches how each dataset is labelled; covers both real-world judge uses.
- Agreement reported with and without ties.
  Why: reproduces the paper's two headline numbers as a sanity check.
- RAGAS comparison -> future work.
  Why: lost Monday; generic-vs-rubric comparison is outside the core question.
- Every pair judged in both orders from the start.
  Why: agreement and position bias come from the same runs, recovering the lost day.
- Same-order repeat run on all 100 MT-Bench pairs, not 20 (9 Nov; ab order only, 320 extra calls).
  Why: 20 was a budget pick with no stated reason; 4/20 has a 95% CI of ~6–44%, too wide to
  tell Haiku's 4–8% flip rate from noise, and noise varies by pair, so it must be measured on
  the same pairs as the flips. gpt-4o-mini at temperature 0 also disagreed with itself, so both judges need it.
- Repeat run in ab order only.
  Why: noise is a property of a single call, so one order is enough to measure it, and it halves
  the extra calls.
- Position bias reported as order-flip rate minus same-order noise on the same pairs, plus flip direction.
  Why: a raw flip rate mixes the order effect with plain call-to-call noise (gpt-4o-mini disagrees
  with itself even at temperature 0); direction tells a lean toward one position from random flips.
- Exact McNemar p reported next to the bootstrap CI (10,000 resamples of pairs, seed 42).
  Why: flips and noise are paired yes/no flags on the same pairs, with only ~15 discordant pairs,
  so an exact test fits; the CI shows how big the effect is, and the fixed seed makes reruns identical.
- Position bias broken down by model-rank gap in three groups: 1 / 2 / 3+.
  Why: tests whether bias hits close pairs; three groups give about 33 pairs each, finer would be too small.

## Plan (Tue 27 – Fri 30 Oct)
| Day | Done when this exists |
|-----|-----------------------|
| Tue | MT-Bench 100-pair sample, rubric, 50 labelled Second Brain answers |
| Wed | All judge runs in both orders -> agreement table + flip rate |
| Thu | Verbosity test, both-orders fix measured, bootstrap CIs on every number |
| Fri | README results table, "where judges fail" |

Status (9 Nov):
- Done: all judge runs in both orders; MT-Bench agreement (with/without ties, paper-style) and
  human–human ceiling; position bias vs same-order noise on all 100 pairs, with a breakdown by
  model-rank gap; both-orders fix measured (no clear agreement gain, no CIs yet); Second Brain
  agreement with bootstrap CIs and paired rubric-vs-generic diffs; Opus 5 added as a strong
  judge on Second Brain only.
- Next: bootstrap CIs for MT-Bench agreement, verbosity test, finish the README.

## Non-goals
Turn-2 questions, RAGAS, fine-tuning judges, >3 models, other bias types, UI.

## Cut order if late
1. Third judge model
2. Verbosity variant on Second Brain items
