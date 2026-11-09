"""Run every judge x prompt x item from config.yaml and cache each call in runs/.

MT-Bench pairs are judged in both orders: in `ab` assistant A is model_x, in `ba`
assistant A is model_y. The parsed verdict is mapped back to x / y / tie, so the
two orders can be compared directly. Second Brain items are judged once (order "-").

One JSONL line per call in runs/<judge>__<dataset>.jsonl, keyed by
(item, prompt, judge, order, repeat). Cached keys are skipped, so the script can
be stopped and re-run. API errors are not cached (retried on the next run);
unparseable outputs are cached with verdict null.

API keys come from .env or the shell (ANTHROPIC_API_KEY, OPENAI_API_KEY); the SDKs
read them. Without --run nothing is called: the script prints the plan and prompts.
It stops before any call if a judge's key is missing, and stops on non-retryable API
errors (400/401/403/404) or after MAX_CONSECUTIVE_ERRORS errors in a row.

    uv run src/run_judges.py --limit 1                         # dry run: plan + prompts
    uv run src/run_judges.py --dataset mtbench --judge haiku --limit 5 --run
    uv run src/run_judges.py --run                             # everything
"""

import argparse
import hashlib
import json
import os
import random
import sys
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config.yaml"
PROMPTS = ROOT / "prompts"
RUNS = ROOT / "runs"

MAX_TOKENS = 1024
NO_ORDER = "-"  # Second Brain items are pointwise
KEY_VARS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
FATAL_STATUS = {400, 401, 403, 404}  # same for every call (bad request, auth, unknown model)
MAX_CONSECUTIVE_ERRORS = 3


def load_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open() if l.strip()]


def load_prompt(prompt_id):
    path = PROMPTS / f"{prompt_id}.yaml"
    prompt = yaml.safe_load(path.read_text())
    # Hash of the file, so a prompt edited after a run shows up in the cache
    prompt["sha"] = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return prompt


def render_notes(context):
    """Same layout as the labelling tool showed (src/label_secondbrain.py)."""
    lines = []
    for c in context:
        date = (c.get("created_at") or "")[:16]
        lines.append(f"[{c['n']}] {date} | {c.get('category') or 'uncategorized'} | {c.get('content_type') or 'text'}")
        lines.append(c["content"])
    return "\n".join(lines)


def build_tasks(config, datasets, judges, limit):
    """One task per call: (judge, prompt, item, order, repeat) plus the rendered user message."""
    items = {name: load_jsonl(ROOT / config["data"][name]) for name in datasets}
    rep = config.get("repeats")
    repeat_ids = set()
    if rep and rep["dataset"] in items:
        all_ids = [it["pair_id"] for it in load_jsonl(ROOT / config["data"][rep["dataset"]])]
        repeat_ids = set(random.Random(rep["seed"]).sample(all_ids, rep["n_items"]))

    tasks = []
    for judge in judges:
        for name in datasets:
            if name not in judge.get("datasets", datasets):  # judge limited to some datasets
                continue
            for p in config["prompts"][name]:
                prompt = load_prompt(p["id"])
                for it in items[name][:limit]:
                    if name == "mtbench":
                        for order in config["orders"]:
                            tasks.append(mtbench_task(judge, prompt, it, order, 0))
                        if it["pair_id"] in repeat_ids:
                            tasks.append(mtbench_task(judge, prompt, it, rep["order"], 1))
                    else:
                        user = prompt["template"].format(
                            question=it["question"],
                            notes=render_notes(it["retrieved_context"]),
                            answer=it["answer"],
                        )
                        tasks.append(dict(judge=judge, prompt=prompt, dataset=name, item_id=it["item_id"],
                                          order=NO_ORDER, repeat=0, positions=None, user=user))
    return tasks


def mtbench_task(judge, prompt, it, order, repeat):
    a, b = ("x", "y") if order == "ab" else ("y", "x")
    user = prompt["template"].format(question=it["question"], answer_a=it[f"answer_{a}"], answer_b=it[f"answer_{b}"])
    positions = {"A": a, "B": b, "A_model": it[f"model_{a}"], "B_model": it[f"model_{b}"]}
    return dict(judge=judge, prompt=prompt, dataset="mtbench", item_id=it["pair_id"],
                order=order, repeat=repeat, positions=positions, user=user)


def cache_key(item_id, prompt_id, judge_id, order, repeat):
    return (item_id, prompt_id, judge_id, order, repeat)


def parse_verdict(prompt, text, positions):
    """Last [[...]] match wins (the verdict follows the explanation). Returns (raw, mapped)."""
    matches = re.findall(prompt["verdict_regex"], text or "")
    if not matches:
        return None, None
    raw = matches[-1]
    mapped = prompt["verdicts"][raw]
    if positions is not None and mapped in ("a", "b"):
        mapped = positions[mapped.upper()]  # display position -> model identity x / y
    return raw, mapped


def call_anthropic(client, judge, system, user):
    kwargs = {}
    if judge.get("thinking"):
        kwargs["thinking"] = {"type": judge["thinking"]}
    if judge.get("effort"):
        kwargs["output_config"] = {"effort": judge["effort"]}
    r = client.messages.create(
        model=judge["model"], max_tokens=MAX_TOKENS, system=system,
        messages=[{"role": "user", "content": user}], **kwargs,
    )
    text = "".join(b.text for b in r.content if b.type == "text")
    sent = {"temperature": None, "max_tokens": MAX_TOKENS} | kwargs
    return dict(text=text, model_version=r.model, stop_reason=r.stop_reason, request_id=r._request_id,
                usage={"input": r.usage.input_tokens, "output": r.usage.output_tokens}, sent=sent)


def call_openai(client, judge, system, user):
    r = client.chat.completions.create(
        model=judge["model"], max_tokens=MAX_TOKENS, temperature=judge["temperature"],
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    choice = r.choices[0]
    return dict(text=choice.message.content, model_version=r.model, stop_reason=choice.finish_reason,
                request_id=r.id, system_fingerprint=r.system_fingerprint,
                usage={"input": r.usage.prompt_tokens, "output": r.usage.completion_tokens},
                sent={"temperature": judge["temperature"], "max_tokens": MAX_TOKENS})


def make_clients(judges):
    clients = {}
    for j in judges:
        if j["provider"] == "anthropic" and "anthropic" not in clients:
            import anthropic
            clients["anthropic"] = (anthropic.Anthropic(), call_anthropic)
        elif j["provider"] == "openai" and "openai" not in clients:
            import openai
            clients["openai"] = (openai.OpenAI(), call_openai)
    return clients


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mtbench", "secondbrain"], help="default: both")
    ap.add_argument("--judge", help="judge id from config.yaml; default: all")
    ap.add_argument("--limit", type=int, help="first N items per dataset (test runs)")
    ap.add_argument("--run", action="store_true", help="make the API calls (default: dry run, no calls)")
    args = ap.parse_args()

    config = yaml.safe_load(CONFIG.read_text())
    datasets = [args.dataset] if args.dataset else list(config["prompts"])
    judges = [j for j in config["judges"] if args.judge in (None, j["id"])]
    if not judges:
        ap.error(f"no judge with id {args.judge!r} in config.yaml")

    tasks = build_tasks(config, datasets, judges, args.limit)
    cached = {}
    for t in tasks:
        path = RUNS / f"{t['judge']['id']}__{t['dataset']}.jsonl"
        if path not in cached:
            cached[path] = {cache_key(r["item_id"], r["prompt_id"], r["judge_id"], r["order"], r["repeat"])
                            for r in load_jsonl(path)}
    todo = [t for t in tasks if cache_key(t["item_id"], t["prompt"]["id"], t["judge"]["id"], t["order"], t["repeat"])
            not in cached[RUNS / f"{t['judge']['id']}__{t['dataset']}.jsonl"]]
    print(f"{len(tasks)} calls planned, {len(tasks) - len(todo)} cached, {len(todo)} to run")

    if not args.run:
        for t in todo:
            print(f"\n=== {t['judge']['id']} | {t['prompt']['id']} | {t['item_id']} | {t['order']} | r{t['repeat']}")
            print("--- system\n" + t["prompt"]["system"] + "\n--- user\n" + t["user"])
        print(f"\nDry run: no calls made. Add --run to make the {len(todo)} calls.")
        return
    if not todo:
        return

    load_dotenv(ROOT / ".env")
    # Checks only that each needed key is set, never its value
    missing = sorted({KEY_VARS[t["judge"]["provider"]] for t in todo} - {k for k in KEY_VARS.values() if os.environ.get(k)})
    if missing:
        sys.exit(f"Missing {', '.join(missing)} (add to .env). No calls made.")
    clients = make_clients(judges)
    counts = {"ok": 0, "unparsed": 0, "error": 0}
    consecutive_errors = 0
    for i, t in enumerate(todo, 1):
        judge, prompt = t["judge"], t["prompt"]
        client, call = clients[judge["provider"]]
        label = f"[{i}/{len(todo)}] {judge['id']} {prompt['id']} {t['item_id']} {t['order']} r{t['repeat']}"
        try:
            out = call(client, judge, prompt["system"], t["user"])
        except Exception as e:  # SDKs already retried 429/5xx; leave uncached for the next run
            counts["error"] += 1
            consecutive_errors += 1
            print(f"{label} ERROR {type(e).__name__}: {e}")
            if getattr(e, "status_code", None) in FATAL_STATUS or consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                sys.exit(f"Stopping: {counts}. Fix the error and re-run; cached calls are kept.")
            continue
        consecutive_errors = 0
        raw, verdict = parse_verdict(prompt, out["text"], t["positions"])
        line = {
            "item_id": t["item_id"], "dataset": t["dataset"], "prompt_id": prompt["id"], "prompt_sha": prompt["sha"],
            "judge_id": judge["id"], "provider": judge["provider"], "model": judge["model"],
            "model_version": out["model_version"], "order": t["order"], "repeat": t["repeat"],
            "positions": t["positions"], "sent": out["sent"],
            "raw_output": out["text"], "verdict_raw": raw, "verdict": verdict, "parse_ok": verdict is not None,
            "stop_reason": out["stop_reason"], "usage": out["usage"], "request_id": out["request_id"],
            "system_fingerprint": out.get("system_fingerprint"),
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        RUNS.mkdir(exist_ok=True)
        with (RUNS / f"{judge['id']}__{t['dataset']}.jsonl").open("a") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        counts["ok" if verdict else "unparsed"] += 1
        print(f"{label} -> {raw} = {verdict}")
    print(f"Done: {counts}")


if __name__ == "__main__":
    main()
