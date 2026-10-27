"""Label Second Brain answers pass/fail in the terminal -> data/secondbrain_50_labeled.jsonl.

Shows only the question, the numbered retrieved notes and the answer. Source, item id
and meta (e.g. whether a question was designed to be declined) stay hidden so they
can't steer the label. Saves after every item; rerun to resume.

    uv run python src/label_secondbrain.py               # batch 1 (first 10)
    uv run python src/label_secondbrain.py --batch 2     # the other 40
    uv run python src/label_secondbrain.py --redo 3      # relabel item 3 of the file
"""

import argparse
import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
ITEMS = DATA / "secondbrain_50.jsonl"
LABELED = DATA / "secondbrain_50_labeled.jsonl"
RUBRIC_VERSION = "v1.1"

FAIL_TAGS = ["unsupported", "should-abstain", "false-abstain", "wrong-citation", "off-topic"]
WIDTH = 100


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open() if l.strip()]


def save_labels(items, labels):
    """Rewrite the whole file in item order, so a relabel replaces the old line."""
    tmp = LABELED.with_suffix(".tmp")
    with tmp.open("w") as f:
        for it in items:
            if it["item_id"] in labels:
                f.write(json.dumps(it | labels[it["item_id"]], ensure_ascii=False) + "\n")
    tmp.replace(LABELED)


def wrap(text, indent="    "):
    return "\n".join(textwrap.fill(p, WIDTH, initial_indent=indent, subsequent_indent=indent)
                     for p in text.splitlines() if p.strip())


def show(item, position, total):
    print("\n" + "=" * WIDTH)
    print(f"Item {position}/{total}\n")
    print("QUESTION")
    print(wrap(item["question"]))
    print(f"\nRETRIEVED NOTES ({len(item['retrieved_context'])})")
    for c in item["retrieved_context"]:
        date = (c.get("created_at") or "")[:16]
        print(f"  [{c['n']}] {date} | {c.get('category') or 'uncategorized'} | {c.get('content_type') or 'text'}")
        print(wrap(c["content"], indent="      "))
    print("\nANSWER")
    print(wrap(item["answer"]))
    print("-" * WIDTH)


def ask_label():
    while True:
        key = input("[p]ass  [f]ail  [s]kip  [q]uit > ").strip().lower()
        if key in ("p", "f", "s", "q"):
            return key


def ask_note(label):
    if label == "fail":
        print("Tags: " + ", ".join(FAIL_TAGS) + "  (start the note with one)")
    while True:
        note = input("Note > ").strip()
        if label == "pass" or note.split(" ")[0].rstrip(":,") in FAIL_TAGS:
            return note
        print("A fail note must start with a tag.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--redo", type=int, help="relabel the item at this position (1-based, file order)")
    args = parser.parse_args()

    items = read_jsonl(ITEMS)
    labels = {r["item_id"]: {k: r[k] for k in ("label", "note", "rubric_version", "labeled_at")}
              for r in read_jsonl(LABELED)}

    if args.redo:
        queue = [(args.redo, items[args.redo - 1])]
    else:
        queue = [(i, it) for i, it in enumerate(items, 1)
                 if it["batch"] == args.batch and it["item_id"] not in labels]
    in_batch = sum(it["batch"] == args.batch for it in items)
    print(f"Batch {args.batch}: {in_batch - len(queue)}/{in_batch} labelled, {len(queue)} to go. "
          f"Rubric: prompts/secondbrain_rubric.md ({RUBRIC_VERSION})")

    for position, item in queue:
        show(item, position, len(items))
        key = ask_label()
        if key == "q":
            break
        if key == "s":
            continue
        label = "pass" if key == "p" else "fail"
        labels[item["item_id"]] = {
            "label": label,
            "note": ask_note(label),
            "rubric_version": RUBRIC_VERSION,
            "labeled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        save_labels(items, labels)

    print(f"\nSaved {len(labels)} labels -> {LABELED.name}")


if __name__ == "__main__":
    main()
