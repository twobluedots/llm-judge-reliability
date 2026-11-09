"""Error analysis page for Second Brain: each answer next to every judge's verdict and reasoning.

Judge-human disagreement review: a verdict matrix (green = judge agrees with the human
label, red = disagrees), then one card per answer with the question, answer, retrieved
notes and each judge's full output. Items are sorted by how many judge runs disagree.

    --show disagreements  answers where at least one judge run disagrees with you  [default]
    --show misses         answers labelled fail (did the judges catch them?)
    --show false-fails    answers labelled pass (did any judge fail them?)
    --show all            all 50 answers

Chips above the table filter rows and cards by fail tag (or "pass").
Reads only runs/ and data/ (no API calls). The page quotes personal notes, so it is
written to reports/, which is gitignored; never commit or publish it.

    uv run src/error_analysis.py
"""

import argparse
from html import escape

import pandas as pd
import yaml

from analyze import CONFIG, ROOT, join_labels, judges_for, load_jsonl, load_runs

REPORTS = ROOT / "reports"
SHOW_LABEL = {"disagreements": None, "misses": "fail", "false-fails": "pass", "all": None}

CSS = """
:root{--bg:#fbfaf8;--fg:#1d1d1b;--muted:#6b6862;--card:#fff;--line:#e4e1db;--ok:#dff1e3;--okfg:#1d5e2c;--bad:#fbe1dc;--badfg:#8a2a17;--null:#ecebe7;--tag:#ece8f7;--tagfg:#4b3b8f}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161614;--fg:#ecebe7;--muted:#a19e97;--card:#1f1f1c;--line:#34332f;--ok:#1d3a24;--okfg:#9fd8ac;--bad:#43211b;--badfg:#f2a596;--null:#2a2a27;--tag:#2c2740;--tagfg:#c6bcf2}}
:root[data-theme="dark"]{--bg:#161614;--fg:#ecebe7;--muted:#a19e97;--card:#1f1f1c;--line:#34332f;--ok:#1d3a24;--okfg:#9fd8ac;--bad:#43211b;--badfg:#f2a596;--null:#2a2a27;--tag:#2c2740;--tagfg:#c6bcf2}
body{background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,sans-serif;margin:0 auto;padding:24px 16px;max-width:1100px}
a{color:inherit} small{color:var(--muted)}
.wrap{overflow-x:auto} table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
td.agree,td.disagree,td.null{text-align:center} td.agree{background:var(--ok);color:var(--okfg)}
td.disagree{background:var(--bad);color:var(--badfg);font-weight:600} td.null{background:var(--null);color:var(--muted)}
td.num{text-align:center} .tag{background:var(--tag);color:var(--tagfg);border-radius:4px;padding:1px 6px;font-size:12px;white-space:nowrap}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 20px;margin:20px 0}
.card h2{font-size:17px;margin:0 0 6px} .card h3{font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin:16px 0 6px}
.pre{white-space:pre-wrap;overflow-wrap:anywhere} .answer{border-left:3px solid var(--line);padding-left:12px}
details{margin:6px 0} summary{cursor:pointer} .note{margin:8px 0 8px 12px;font-size:13px}
details.judge{border-radius:6px;padding:6px 10px} details.judge.agree{background:var(--ok)} details.judge.disagree{background:var(--bad)} details.judge.null{background:var(--null)}
details.judge .pre{font-size:13px;margin-top:6px}
.chips{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.chip{font:inherit;font-size:13px;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:999px;padding:3px 12px;cursor:pointer}
.chip.on{background:var(--tag);color:var(--tagfg);border-color:var(--tagfg)}
"""

# Chip click: show only rows and cards with that data-tag ("all" shows everything)
FILTER_JS = """
document.querySelectorAll('.chip').forEach(chip => chip.addEventListener('click', () => {
  const tag = chip.dataset.filter;
  document.querySelectorAll('.chip').forEach(c => c.classList.toggle('on', c === chip));
  let shown = 0;
  document.querySelectorAll('[data-tag]').forEach(el => {
    el.hidden = tag !== 'all' && el.dataset.tag !== tag;
    if (!el.hidden && el.tagName === 'TR') shown++;
  });
  document.getElementById('shown').textContent = shown;
}));
"""


def item_tag(item):
    """Fail tag (note prefix) for fails; passes show the label only."""
    return item["note"].split(" - ")[0].strip() if item["label"] == "fail" else "pass"


def status(verdict, label):
    if verdict is None or pd.isna(verdict):
        return "null"
    return "agree" if verdict == label else "disagree"


def count_disagreements(it, runs, columns):
    """Judge runs whose verdict differs from the human label (unparsed and missing runs not counted)."""
    return sum(status(runs[k]["verdict"], it["label"]) == "disagree"
               for k in ((it["item_id"], j, p) for j, p, _ in columns) if k in runs)


def build_page(items, runs, columns, show):
    """columns: (judge_id, prompt_id, header) in display order; runs: {(item, judge, prompt): run line}."""
    def disagreements(it):
        return count_disagreements(it, runs, columns)

    items = sorted(items, key=lambda it: (-disagreements(it), item_tag(it)))
    tags = pd.Series([item_tag(it) for it in items]).value_counts()
    chips = "" if len(tags) < 2 else (
        '<p class="chips"><button class="chip on" data-filter="all">All '
        f'({len(items)})</button>'
        + "".join(f'<button class="chip" data-filter="{escape(t)}">{escape(t)} ({n})</button>'
                  for t, n in tags.items())
        + f' <small>showing <span id="shown">{len(items)}</span> of {len(items)}</small></p>')
    head = "".join(f"<th>{h}</th>" for _, _, h in columns)
    rows, cards = [], []
    for i, it in enumerate(items):
        cells, judges = [], []
        for j, p, h in columns:
            r = runs.get((it["item_id"], j, p))
            if r is None:  # judge not run on this item (yet)
                cells.append('<td class="null">—</td>')
                continue
            s = status(r["verdict"], it["label"])
            v = "UNPARSED" if s == "null" else r["verdict"].upper()
            cells.append(f'<td class="{s}">{v}</td>')
            judges.append(f'<details class="judge {s}"><summary>{h} → <b>{v}</b></summary>'
                          f'<div class="pre">{escape(r["raw_output"] if isinstance(r["raw_output"], str) else "")}</div></details>')
        rows.append(f'<tr data-tag="{escape(item_tag(it))}"><td><a href="#i{i}">{escape(it["question"][:70])}</a></td>'
                    f'<td>{it["label"].upper()}</td><td><span class="tag">{escape(item_tag(it))}</span></td>'
                    f'<td class="num">{disagreements(it)}/{len(columns)}</td>{"".join(cells)}</tr>')
        notes = "".join(
            f'<div class="note"><b>[{c["n"]}]</b> <small>{escape((c.get("created_at") or "")[:10])} · '
            f'{escape(c.get("category") or "uncategorized")}</small><div class="pre">{escape(c["content"])}</div></div>'
            for c in it["retrieved_context"])
        cards.append(f"""<section id="i{i}" class="card" data-tag="{escape(item_tag(it))}">
<h2>{escape(it["question"])}</h2>
<p><span class="tag">{escape(item_tag(it))}</span> <b>Your label:</b> {it["label"].upper()} · <b>Your note:</b> {escape(it["note"])}
· <b>{disagreements(it)}/{len(columns)}</b> judge runs disagree</p>
<h3>Answer</h3><div class="pre answer">{escape(it["answer"])}</div>
<details><summary>Retrieved notes ({len(it["retrieved_context"])})</summary>{notes}</details>
<h3>Judges</h3>{"".join(judges)}
<p><a href="#top">↑ back to table</a></p></section>""")

    what = {"disagreements": "where at least one judge run disagrees with your label "
                             "(a missed fail or a false fail): red marks the disagreeing runs",
            "misses": "you labelled <b>fail</b>: red means the judge passed it (a missed fail)",
            "false-fails": "you labelled <b>pass</b>: red means the judge failed it (a false fail)",
            "all": "all answers: red means the judge disagrees with your label"}[show]
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Judge Error Analysis</title>
<style>{CSS}</style></head><body>
<h1 id="top">Second Brain error analysis: {show}</h1>
<p>{len(items)} answers {what}; green means the judge agrees with you. Sorted by number of disagreeing
judge runs. Answers were written by gpt-4o-mini. Click a question for the answer, notes and each judge's reasoning.</p>
{chips}
<div class="wrap"><table><thead><tr><th>Question</th><th>You</th><th>Tag</th><th>Disagree</th>{head}</tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>
{"".join(cards)}
<script>{FILTER_JS}</script>
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", choices=list(SHOW_LABEL), default="disagreements")
    args = ap.parse_args()

    config = yaml.safe_load(CONFIG.read_text())
    items = load_jsonl(ROOT / config["data"]["secondbrain"])
    labels = pd.DataFrame(items)
    df = join_labels(load_runs("secondbrain"), labels)  # same checks as analyze.py
    runs = {(r["item_id"], r["judge_id"], r["prompt_id"]): r for r in df.to_dict("records")}

    columns = [(j["id"], p["id"], f'{escape(j["model"])}<br><small>{escape(p["id"])}</small>')
               for j in judges_for(config, "secondbrain") for p in config["prompts"]["secondbrain"]]
    if SHOW_LABEL[args.show]:
        items = [it for it in items if it["label"] == SHOW_LABEL[args.show]]
    if args.show == "disagreements":
        items = [it for it in items if count_disagreements(it, runs, columns)]

    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"secondbrain_{args.show.replace('-', '_')}.html"
    out.write_text(build_page(items, runs, columns, args.show))
    print(f"Wrote {out.relative_to(ROOT)} ({len(items)} answers, {len(columns)} judge runs each). "
          "Quotes personal notes: keep it local.")


if __name__ == "__main__":
    main()
