# LLM judge reliability

How far can you trust an LLM judge, and when does it fail?

## Results

_Work in progress: the tables below are filled in as results are written up. The latest
numbers are in `results/` (written by `src/analyze.py`)._

### Agreement with human labels

| Dataset | Judge model | Prompt | Accuracy | Cohen's kappa (95% CI) |
|---------|-------------|--------|----------|------------------------|
| MT-Bench (with ties) | | | | |
| MT-Bench (no ties) | | | | |
| Second Brain (pass/fail) | | | | |

Human–human agreement (MT-Bench ceiling): _TBD_

### Bias

| Judge model | Position flip rate (95% CI) | Flips toward first answer | Verbosity preference shift (95% CI) |
|-------------|-----------------------------|---------------------------|-------------------------------------|
| | | | |

## Where judges fail

_TBD_

## Reproducibility notes

- `gpt-4o-mini-2024-07-18` is deprecated and leaves OpenAI's API on 14 Jan 2027; after that
  its runs can only be reproduced from cached judge calls (`runs/*.jsonl`, not in the repo yet).
- Claude models reject `temperature`, so Haiku runs use default sampling. A 20-pair repeat run
  measures how much of its flip rate is sampling noise.

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
Second Brain data only the MT-Bench half can be run.

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
