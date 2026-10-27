"""Sample 100 MT-Bench turn-1 answer pairs with human labels -> data/mtbench_100.jsonl.

A pair is (question, model_x, model_y) with model names sorted, so votes cast in
either order are merged. The human label is the majority vote, or tie when there
is no majority. All votes are kept for human-human agreement later.
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

SEED = 42
N_PAIRS = 100
MAX_LOPSIDED = 0.20  # share of pairs with rank gap >= LOPSIDED_GAP
LOPSIDED_GAP = 4
MAX_PAIRS_PER_QUESTION = 2
TARGET_MULTI_VOTE = 30  # pairs with >= 2 votes, for the human-human ceiling

# Strength order from the MT-Bench paper, strongest first
RANK = ["gpt-4", "claude-v1", "gpt-3.5-turbo", "vicuna-13b-v1.2", "alpaca-13b", "llama-13b"]

# Question id blocks, checked against FastChat's mt_bench/question.jsonl
CATEGORIES = ["writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities"]

OUT = Path(__file__).resolve().parent.parent / "data" / "mtbench_100.jsonl"


def category(question_id):
    return CATEGORIES[(question_id - 81) // 10]


def load_turn1_votes():
    ds = load_dataset("lmsys/mt_bench_human_judgments", split="human")
    votes = []
    for r in ds:
        if r["turn"] != 1:
            continue
        # Conversations hold both turns; turn 1 is the first user message and first reply
        votes.append({
            "question_id": r["question_id"],
            "model_a": r["model_a"],
            "model_b": r["model_b"],
            "winner": r["winner"],
            "judge": r["judge"],
            "question": r["conversation_a"][0]["content"],
            "answer_a": r["conversation_a"][1]["content"],
            "answer_b": r["conversation_b"][1]["content"],
        })
    return votes


def group_into_pairs(votes):
    pairs = {}
    for v in votes:
        x, y = sorted([v["model_a"], v["model_b"]])
        flipped = v["model_a"] != x
        key = (v["question_id"], x, y)
        if key not in pairs:
            pairs[key] = {
                "pair_id": f"q{v['question_id']}_{x}_{y}",
                "question_id": v["question_id"],
                "model_x": x,
                "model_y": y,
                "question": v["question"],
                "answer_x": v["answer_b"] if flipped else v["answer_a"],
                "answer_y": v["answer_a"] if flipped else v["answer_b"],
                "votes": [],
            }
        if v["winner"] == "tie":
            vote = "tie"
        else:
            winner_is_a = v["winner"] == "model_a"
            vote = "y" if winner_is_a == flipped else "x"
        pairs[key]["votes"].append({"judge": v["judge"], "vote": vote})

    for p in pairs.values():
        counts = Counter(v["vote"] for v in p["votes"]).most_common()
        top_is_unique = len(counts) == 1 or counts[0][1] > counts[1][1]
        p["human_label"] = counts[0][0] if top_is_unique else "tie"
        p["n_votes"] = len(p["votes"])
    return list(pairs.values())


def add_fields(pairs):
    for p in pairs:
        p["category"] = category(p["question_id"])
        p["gap"] = abs(RANK.index(p["model_x"]) - RANK.index(p["model_y"]))
        p["lopsided"] = p["gap"] >= LOPSIDED_GAP
    return pairs


def sample(pairs):
    rng = random.Random(SEED)
    by_category = defaultdict(list)
    for p in sorted(pairs, key=lambda p: p["pair_id"]):
        by_category[p["category"]].append(p)

    base, extra = divmod(N_PAIRS, len(CATEGORIES))
    chosen = []
    for i, cat in enumerate(CATEGORIES):
        quota = base + (1 if i < extra else 0)
        max_lopsided = int(quota * MAX_LOPSIDED)
        multi_target = round(quota * TARGET_MULTI_VOTE / N_PAIRS)

        candidates = by_category[cat][:]
        rng.shuffle(candidates)
        multi = [p for p in candidates if p["n_votes"] >= 2]
        single = [p for p in candidates if p["n_votes"] < 2]

        picked, per_question, n_lopsided = [], Counter(), 0

        def try_add(p):
            nonlocal n_lopsided
            if len(picked) >= quota or per_question[p["question_id"]] >= MAX_PAIRS_PER_QUESTION:
                return
            if p["lopsided"] and n_lopsided >= max_lopsided:
                return
            picked.append(p)
            per_question[p["question_id"]] += 1
            n_lopsided += p["lopsided"]

        for p in multi:
            if sum(q["n_votes"] >= 2 for q in picked) >= multi_target:
                break
            try_add(p)
        for p in single + multi:
            if p not in picked:
                try_add(p)

        if len(picked) < quota:
            print(f"WARNING: {cat} filled {len(picked)}/{quota}")
        chosen.extend(picked)
    return chosen


def write_jsonl(chosen):
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w") as f:
        for p in chosen:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def print_summary(all_pairs, chosen):
    n = len(chosen)
    print(f"Turn-1 pairs available: {len(all_pairs)} "
          f"({sum(p['n_votes'] >= 2 for p in all_pairs)} with >= 2 votes)")
    print(f"Sampled: {n} pairs, {len({p['question_id'] for p in chosen})} distinct questions -> {OUT.name}")
    print(f"Multi-vote: {sum(p['n_votes'] >= 2 for p in chosen)}   "
          f"Lopsided: {sum(p['lopsided'] for p in chosen)}")
    print("Labels:", dict(Counter(p["human_label"] for p in chosen)))
    print("Per category:", dict(Counter(p["category"] for p in chosen)))
    models = Counter(m for p in chosen for m in (p["model_x"], p["model_y"]))
    print("Model share of pairs:", {m: f"{c / n:.0%}" for m, c in models.most_common()})
    print("Gap:", dict(sorted(Counter(p["gap"] for p in chosen).items())))


def main():
    pairs = add_fields(group_into_pairs(load_turn1_votes()))
    chosen = sample(pairs)
    write_jsonl(chosen)
    print_summary(pairs, chosen)


if __name__ == "__main__":
    main()
