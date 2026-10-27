"""Select 50 Second Brain Ask answers for pointwise labelling -> data/secondbrain_50.jsonl.

Two groups, picked at random within fixed quotas (no judge scores, no reading answers):
  A. ask_log (real use): one row per distinct question, quotas by intent.
  B. abstention runs (adversarial set): one record per question, any prompt version / repeat.
Questions in the abstention set only go to group B, so no question appears twice.
Each item carries the retrieved notes numbered as the generator saw them: [n] = note n.
The first 10 items in the file (batch 1) are labelled first to test the rubric.

Picks found unusable during labelling are swapped afterwards, keeping their slot, using a
separate RNG per round so the other picks don't move:
  round 1: no model answer (pipeline fallback list)
  round 2: EXCLUDE (privacy, or same answer as a reworded twin)
Rewordings that stay in the set share a question_group, for a cluster bootstrap later.
"""

import json
import random
import sqlite3
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

SEED = 42
SECOND_BRAIN = Path.home() / "second-brain"
ASK_DB = SECOND_BRAIN / "data" / "database" / "entries.db"
RECORDS = SECOND_BRAIN / "experiments" / "artifacts" / "ask_eval" / "records"
ABSTENTION_QUERIES = SECOND_BRAIN / "experiments" / "ask_eval" / "queries" / "abstention.json"

LOG_QUOTAS = {"qa": 21, "pattern": 5, "factual": 4}  # group A, 30; browse logs no generated answer
N_ABSTENTION = 20  # group B, out of 22 questions
BATCH1 = {"ask_log": 6, "abstention_run": 4}

# Decided while labelling, from the question only (never the label)
EXCLUDE = {
    "abs_abs-008_20261026T172921_r1": "privacy",
    "log_0fa6404e": "privacy",
    "log_376fcdd0": "same-answer-twin",  # of log_e7641ecd (favourite recipes lately)
    "log_4335aaaa": "same-answer-twin",  # of log_2993a576 (three things I do regularly)
    "abs_abs-011_20261026T144046_r1": "same-answer-twin",  # of log_4f9bda8d (watch)
    "abs_ans-007_20261022T161819_r2": "same-answer-twin",  # of log_fd42a747 (GoPro)
    "log_08e64c99": "privacy",  # drawn as a replacement above, then excluded
}
# Reworded questions kept in the set (different answers); one cluster each
QUESTION_GROUPS = {
    "favourite-recipes": ["log_e7641ecd", "log_31fbc0b4", "abs_abs-001_20261026T172919_r0"],
    "before-turkey": ["log_aab43829", "abs_ans-002_20261022T160826_r2"],
    "eye-irritation": ["log_a2ad5b92", "log_9953608f"],
    "wake-up-late": ["log_bc736913", "log_df5ddb6e"],
}
REWORDING_SIMILARITY = 0.8  # replacements this close to a used question are skipped

OUT = Path(__file__).resolve().parent.parent / "data" / "secondbrain_50.jsonl"


def norm(q):
    return " ".join(q.lower().split())


def context_entry(n, note_id, created_at, category, content_type, content):
    return {"n": n, "id": note_id, "created_at": created_at, "category": category,
            "content_type": content_type, "content": content}


def load_log_candidates(abstention_questions):
    """ask_log rows with an answer whose notes all exist and were not edited after the answer."""
    conn = sqlite3.connect(ASK_DB)
    notes = {r[0]: r[1:] for r in conn.execute(
        "SELECT id, content, description, created_at, modified_at, category, content_type FROM entries")}
    rows = conn.execute(
        "SELECT id, query, intent, retrieved_note_ids, answer, generator_model, created_at "
        "FROM ask_log WHERE answer IS NOT NULL AND answer != '' ORDER BY created_at").fetchall()

    dropped = Counter()
    candidates = []
    for log_id, query, intent, ids_json, answer, model, asked_at in rows:
        if norm(query) in abstention_questions:
            dropped["in abstention set"] += 1
            continue
        ids = json.loads(ids_json or "[]")
        if not ids or any(i not in notes for i in ids):
            dropped["missing notes"] += 1
            continue
        if any((notes[i][3] or "") > asked_at for i in ids):
            dropped["note edited after answer"] += 1
            continue
        context = []
        for n, i in enumerate(ids, 1):
            content, description, created_at, _, category, content_type = notes[i]
            context.append(context_entry(n, i, created_at, category, content_type, content or description or ""))
        candidates.append({
            "item_id": f"log_{log_id[:8]}",
            "source": "ask_log",
            "question": query,
            "intent": intent,
            "retrieved_context": context,
            "answer": answer,
            "meta": {"ask_log_id": log_id, "created_at": asked_at, "generator_model": model},
        })
    return candidates, dropped


def load_abstention_candidates():
    """All records from abstention runs that recorded a prompt version (run_meta)."""
    by_question = defaultdict(list)
    for path in sorted(RECORDS.glob("run_*.jsonl")):
        lines = [json.loads(l) for l in path.open()]
        meta = next((l for l in lines if l.get("kind") == "run_meta"), None)
        if not meta or meta["query_set"] != "abstention":
            continue
        for r in lines:
            if r.get("kind") == "run_meta":
                continue
            context = [context_entry(n, c["id"], c.get("created_at"), c.get("category"),
                                     c.get("content_type"), c.get("content", ""))
                       for n, c in enumerate(r["retrieved_contexts"], 1)]
            by_question[r["query_id"]].append({
                "item_id": f"abs_{r['query_id']}_{r['run_id']}_r{r['repeat']}",
                "source": "abstention_run",
                "question": r["user_input"],
                "intent": r["intent"],
                "retrieved_context": context,
                "answer": r["response"],
                "meta": {"query_id": r["query_id"], "expected": r["expected"], "types": r["types"],
                         "run_id": r["run_id"], "repeat": r["repeat"],
                         "prompt_version": meta["prompt_version"], "generator_model": meta["model"]},
            })
    return by_question


def sample_log(candidates, rng):
    by_question = defaultdict(list)
    for c in candidates:
        by_question[norm(c["question"])].append(c)
    # One row per distinct question; the intent of the chosen row decides its quota
    one_per_question = [rng.choice(rows) for _, rows in sorted(by_question.items())]
    by_intent = defaultdict(list)
    for c in one_per_question:
        by_intent[c["intent"]].append(c)

    chosen = []
    for intent, quota in LOG_QUOTAS.items():
        pool = by_intent[intent]
        if len(pool) < quota:
            print(f"WARNING: intent {intent} has {len(pool)} questions, quota {quota}")
        chosen.extend(rng.sample(pool, min(quota, len(pool))))
    return chosen, {i: len(p) for i, p in by_intent.items()}


def sample_abstention(by_question, rng):
    question_ids = rng.sample(sorted(by_question), N_ABSTENTION)
    return [rng.choice(by_question[q]) for q in question_ids]


def is_model_answer(item):
    # generator_model is None when every LLM call failed and the pipeline listed the notes instead
    return item["meta"]["generator_model"] is not None


def replace_invalid(items, log_candidates):
    """Swap invalid picks in place, keeping their slot (and batch). Added after labelling began."""
    rng = random.Random(SEED + 1)
    used = {norm(i["question"]) for i in items}
    replaced = []
    for pos, it in enumerate(items):
        if is_model_answer(it):
            continue
        pool = sorted({norm(c["question"]) for c in log_candidates
                       if is_model_answer(c) and c["intent"] == it["intent"] and norm(c["question"]) not in used})
        question = rng.choice(pool)
        new = rng.choice([c for c in log_candidates if norm(c["question"]) == question and is_model_answer(c)])
        used.add(question)
        items[pos] = new | {"batch": it["batch"], "replaces": it["item_id"]}
        replaced.append((pos + 1, it["item_id"], new["item_id"], "no model answer"))
    return replaced


def is_rewording(question, used):
    return any(SequenceMatcher(None, question, u).ratio() >= REWORDING_SIMILARITY for u in used)


def replace_excluded(items, log_candidates, abs_pool):
    """Round 2: swap EXCLUDE items in place. Abstention slots take an unused abstention
    question while any are left, then fall back to ask_log with the same intent."""
    rng = random.Random(SEED + 2)
    used = {norm(i["question"]) for i in items}
    replaced = []
    for pos in range(len(items)):
        # A replacement can itself be excluded, so keep drawing until the slot is clean
        while items[pos]["item_id"] in EXCLUDE:
            it = items[pos]
            new = draw_replacement(it, used, log_candidates, abs_pool, rng)
            used.add(norm(new["question"]))
            items[pos] = new | {"batch": it["batch"], "replaces": it["item_id"]}
            replaced.append((pos + 1, it["item_id"], new["item_id"], EXCLUDE[it["item_id"]]))
    return replaced


def draw_replacement(it, used, log_candidates, abs_pool, rng):
    abs_left = sorted(q for q, recs in abs_pool.items()
                      if norm(recs[0]["question"]) not in used and not is_rewording(norm(recs[0]["question"]), used))
    if it["source"] == "abstention_run" and abs_left:
        return rng.choice(abs_pool[rng.choice(abs_left)])
    free = [c for c in log_candidates if is_model_answer(c)
            and norm(c["question"]) not in used and not is_rewording(norm(c["question"]), used)]
    # Same intent if any is left, else any intent (ask_log has only 5 factual questions)
    same_intent = [c for c in free if c["intent"] == it["intent"]]
    question = rng.choice(sorted({norm(c["question"]) for c in same_intent or free}))
    return rng.choice([c for c in log_candidates if norm(c["question"]) == question and is_model_answer(c)])


def add_question_groups(items):
    group_of = {item_id: g for g, ids in QUESTION_GROUPS.items() for item_id in ids}
    for it in items:
        it["question_group"] = group_of.get(it["item_id"], it["item_id"])


def assign_batches(items, rng):
    rng.shuffle(items)
    first, rest = [], []
    taken = Counter()
    for it in items:
        if taken[it["source"]] < BATCH1[it["source"]]:
            taken[it["source"]] += 1
            first.append(it | {"batch": 1})
        else:
            rest.append(it | {"batch": 2})
    return first + rest


def write_jsonl(items):
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def print_summary(dropped, log_pool, abs_pool, items):
    print(f"ask_log dropped: {dict(dropped)}")
    print(f"ask_log questions per intent (pool): {log_pool}")
    print(f"abstention questions: {len(abs_pool)}, records: {sum(len(v) for v in abs_pool.values())}")
    print(f"Wrote {len(items)} items -> {OUT.name}")
    print("Per source:", dict(Counter(i["source"] for i in items)))
    print("Per intent:", dict(Counter(i["intent"] for i in items)))
    abs_items = [i for i in items if i["source"] == "abstention_run"]
    print("Abstention expected:", dict(Counter(i["meta"]["expected"] for i in abs_items)))
    print("Abstention prompt versions:", dict(Counter(i["meta"]["prompt_version"] for i in abs_items)))
    print("Batch 1:", dict(Counter(i["source"] for i in items if i["batch"] == 1)))
    print("Distinct questions:", len({norm(i["question"]) for i in items}),
          " question groups:", len({i["question_group"] for i in items}))


def main():
    rng = random.Random(SEED)
    abstention_questions = {norm(q["query"]) for q in json.loads(ABSTENTION_QUERIES.read_text())}
    log_candidates, dropped = load_log_candidates(abstention_questions)
    abs_pool = load_abstention_candidates()

    log_items, log_pool = sample_log(log_candidates, rng)
    abs_items = sample_abstention(abs_pool, rng)
    items = assign_batches(log_items + abs_items, rng)
    replaced = replace_invalid(items, log_candidates)
    replaced += replace_excluded(items, log_candidates, abs_pool)
    add_question_groups(items)
    write_jsonl(items)
    print_summary(dropped, log_pool, abs_pool, items)
    for pos, old, new, reason in replaced:
        print(f"Replaced item {pos}: {old} -> {new} ({reason})")


if __name__ == "__main__":
    main()
