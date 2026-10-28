# LLM judge reliability

How far can you trust an LLM judge, and when does it fail?

## Results

_Filled in Fri 30 Oct._

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
  its runs can only be reproduced from the cached lines in `runs/`.
- Claude models reject `temperature`, so Haiku runs use default sampling. A 20-pair repeat run
  measures how much of its flip rate is sampling noise.

## How to run

```bash
uv run src/prepare_data.py
uv run src/run_judges.py
uv run src/analyze.py
```
