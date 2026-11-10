# LLM judge reliability

How far can you trust an LLM judge, and when does it fail?

## Results

_Work in progress: MT-Bench agreement CIs and the verbosity test are still to come. The full
tables are in `results/` (written by `src/analyze.py`)._

Judges: `claude-haiku-5-5` (thinking off, default sampling) and `gpt-4o-mini-2024-07-18`
(temperature 0); `claude-opus-5` as a strong judge on Second Brain only.

### Agreement with human labels

MT-Bench: `mtb_paper` prompt (FastChat pair-v2), combined verdict (each pair judged in both orders;
if the two orders disagree, the verdict is a tie), scored against the human majority label.
Second Brain: `sb_rubric` prompt, n = 50, 64% of human labels are pass.

| Dataset | Judge model | Prompt | Accuracy (95% CI) | Cohen's kappa (95% CI) | Missed-fail rate (95% CI) |
|---------|-------------|--------|-------------------|------------------------|---------------------------|
| MT-Bench (with ties, n = 100) | gpt-4o-mini | mtb_paper | 0.66 (CI pending) | 0.48 (CI pending) | – |
| MT-Bench (with ties, n = 100) | Haiku 5.5 | mtb_paper | 0.70 (CI pending) | 0.54 (CI pending) | – |
| MT-Bench (no ties, n = 73) | gpt-4o-mini | mtb_paper | 0.78 (CI pending) | 0.60 (CI pending) | – |
| MT-Bench (no ties, n = 73) | Haiku 5.5 | mtb_paper | 0.81 (CI pending) | 0.65 (CI pending) | – |
| Second Brain (pass/fail) | gpt-4o-mini | sb_rubric | 0.74 [0.62, 0.86] | 0.37 [0.10, 0.63] | 0.61 [0.38, 0.83] |
| Second Brain (pass/fail) | Haiku 5.5 | sb_rubric | 0.74 [0.62, 0.86] | 0.48 [0.23, 0.72] | 0.17 [0.00, 0.36] |
| Second Brain (pass/fail) | Opus 5 | sb_rubric | 0.80 [0.68, 0.90] | 0.58 [0.32, 0.79] | 0.22 [0.05, 0.43] |

Missed-fail rate = share of human fails the judge passed.

Paper comparison (combined verdict vs each expert vote, pooled over pairs as in FastChat):
gpt-4o-mini 0.69 with ties / 0.84 without, Haiku 0.68 / 0.85; GPT-4 in the paper 0.66 / 0.85.

Human–human agreement (MT-Bench ceiling): experts agree with each other 0.73 with ties
(73 comparisons) / 0.84 without (57); the paper reports 0.63 / 0.81. Caveat: measured on our
82 pairs with 2+ expert votes, not the paper's set, and those pairs are not a random subset.

### Bias

Position bias: every MT-Bench pair is judged in both orders, and also twice in the same order.
Bias is the order-flip rate minus that same-order noise, on the same 100 pairs.

| Judge model | Prompt | Same-order noise | Order flip rate | Bias = flip − noise (95% CI) | McNemar p | Flips toward first / second |
|-------------|--------|------------------|-----------------|------------------------------|-----------|-----------------------------|
| gpt-4o-mini | mtb_paper | 0.05 | 0.14 | +0.09 [+0.02, +0.17] | 0.035 | 10 / 4 |
| gpt-4o-mini | mtb_nodebias | 0.04 | 0.12 | +0.08 [0.00, +0.16] | 0.077 | 7 / 5 |
| Haiku 5.5 | mtb_paper | 0.08 | 0.08 | 0.00 [−0.06, +0.06] | 1.000 | 7 / 1 (p = 0.07, unresolved) |
| Haiku 5.5 | mtb_nodebias | 0.04 | 0.04 | 0.00 [−0.05, +0.05] | 1.000 | 2 / 2 |

CI: bootstrap over pairs, 10,000 resamples, seed 42. McNemar: exact test of flip-only vs
noise-only pairs. Haiku's lean toward the first answer on `mtb_paper` is still being looked into.

By model-rank gap, gpt-4o-mini's flips concentrate on gap-1 pairs (the closest models): 8 of its
14 flips with `mtb_paper` and 7 of 12 with `mtb_nodebias` come from the 31 gap-1 pairs (flip rate
0.26 / 0.23, against 0.06 on gap 3+). Groups are about 33 pairs each, so this is a pattern, not a test.

Verbosity bias: not run yet.

## Where judges fail

- **gpt-4o-mini is swayed by answer order, mostly when the answers are close.** Bias +0.09 above
  noise with the paper prompt, concentrated on the closest pairs.
- **The same call doesn't always give the same verdict.** Re-running the identical call changes the
  verdict on 4–8% of MT-Bench pairs, including gpt-4o-mini at temperature 0 (4–5%).
- **The debiasing sentences have no measurable effect.** With and without them: bias +0.09 vs +0.08,
  agreement 0.66 vs 0.65 (gpt-4o-mini).
- **gpt-4o-mini lets most bad Second Brain answers through.** It passes 11 of 18 human fails even with
  the rubric (judge pass rate 0.82 vs human 0.64), mostly `unsupported` and `should-abstain` answers.
  It also generated those answers, so self-preference is possible; not tested.
- **A rubric mainly helps catch fails.** Rubric vs generic prompt cuts the missed-fail rate by 0.33
  [0.12, 0.56] for Haiku and 0.39 [0.17, 0.63] for Opus; accuracy and kappa gains have CIs that include 0.
  The cost is more false fails (Haiku 0.22 → 0.31).

## Reproducibility notes

- `gpt-4o-mini-2024-07-18` is deprecated and leaves OpenAI's API on 14 Jan 2027; after that
  its runs can only be reproduced from the cached judge calls. The MT-Bench ones are in git
  (`runs/*__mtbench.jsonl`, an exception in `.gitignore`); the Second Brain ones quote personal
  notes and stay local.
- Claude models reject `temperature`, so Haiku runs use default sampling. A same-order repeat of
  all 100 MT-Bench pairs, for both judges, measures how much of each flip rate is call-to-call noise.

## How to run

API keys go in `.env` (or the shell): `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`.

```bash
uv run src/prepare_data.py           # re-samples data/mtbench_100.jsonl (same seed, same file)
uv run src/run_judges.py             # dry run: prints the prompts and the number of calls
uv run src/run_judges.py --run       # makes the API calls, cached in runs/
uv run src/analyze.py                # metrics from runs/ -> results/, no API calls
uv run src/error_analysis.py         # judge-human disagreements; --show misses / false-fails / all
```

`run_judges.py` takes `--dataset`, `--judge` and `--limit` to run a subset. Without the private
Second Brain data only the MT-Bench half runs: `analyze.py` reproduces the MT-Bench tables from the
cached calls in `runs/` and skips Second Brain, and `error_analysis.py` (Second Brain only) exits.

`error_analysis.py` writes a judge–human disagreement review page (every answer, each judge's
verdict and reasoning) to `reports/`. It quotes personal notes, so `reports/` is gitignored.

## Data and licenses

Code in this repository is MIT-licensed (see `LICENSE`). Two inputs come from elsewhere and keep
their own licenses:

- `data/mtbench_100.jsonl`: a 100-pair, turn-1 sample of
  [lmsys/mt_bench_human_judgments](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments)
  (CC BY 4.0), reshaped by `src/prepare_data.py`. From Zheng et al. 2023,
  "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" ([arXiv:2306.05685](https://arxiv.org/abs/2306.05685)).
- `prompts/mtb_paper.yaml`: the "pair-v2" judge prompt copied verbatim from
  [FastChat](https://github.com/lm-sys/FastChat) (Apache-2.0); `prompts/mtb_nodebias.yaml` is a modified copy.

The Second Brain data (personal notes and their labels) is not included.
