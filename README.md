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

## How to run

```bash
uv run src/prepare_data.py
uv run src/run_judges.py
uv run src/analyze.py
```
