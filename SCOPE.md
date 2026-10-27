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
- Judge models: TBD (2 required, 3rd optional)
- Judge prompts: TBD (2)
- Temperature 0, exact model versions recorded
- Every call cached in runs/ keyed by (item, prompt, model, order)

## Pipeline
Inputs (written by me, in git): data/, prompts/, config.yaml
Outputs (written by scripts): runs/, results/

    src/prepare_data.py  ->  data/          sample MT-Bench, load my labels
    src/run_judges.py    ->  runs/*.jsonl   API calls: slow, costs money, cached
    src/analyze.py       ->  results/       metrics, CIs, tables: fast, free

Analysis only reads runs/, so re-running it never costs an API call.

## Decisions
- Mixed protocol: pairwise for MT-Bench, pointwise for Second Brain.
  Why: matches how each dataset is labelled; covers both real-world judge uses.
- Agreement reported with and without ties.
  Why: reproduces the paper's two headline numbers as a sanity check.
- RAGAS comparison -> future work.
  Why: lost Monday; generic-vs-rubric comparison is outside the core question.
- Every pair judged in both orders from the start.
  Why: agreement and position bias come from the same runs, recovering the lost day.

## Plan (Tue 27 – Fri 30 Oct)
| Day | Done when this exists |
|-----|-----------------------|
| Tue | MT-Bench 100-pair sample, rubric, 50 labelled Second Brain answers |
| Wed | All judge runs in both orders -> agreement table + flip rate |
| Thu | Verbosity test, both-orders fix measured, bootstrap CIs on every number |
| Fri | README results table, "where judges fail" |

## Non-goals
Turn-2 questions, RAGAS, fine-tuning judges, >3 models, other bias types, UI.

## Cut order if late
1. Third judge model
2. Verbosity variant on Second Brain items
