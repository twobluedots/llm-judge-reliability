"""Compute judge reliability metrics from runs/ and data/; write tables to results/.

Reads only cached judge calls, so re-running never makes an API call. Parse
failures (verdict null) are counted in every table and left out of the metrics,
never dropped silently.

    uv run src/analyze.py
"""

import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config.yaml"
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
CSV_FLOAT = "%.3f"  # results/ tables are for reading; metrics are recomputed from runs/

KEY = ["item_id", "prompt_id", "judge_id", "order", "repeat"]  # same cache key as run_judges.py
FAIL_TAGS = ("should-abstain", "unsupported", "off-topic")  # rubric v1.1 fail reasons


def load_jsonl(path):
    """A run may still be appending, so an incomplete last line is skipped with a warning."""
    lines = [l for l in path.open() if l.strip()]
    rows = []
    for i, l in enumerate(lines):
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError:
            if i < len(lines) - 1:
                raise
            print(f"warning: skipped incomplete last line in {path.name}")
    return rows


def load_runs(dataset):
    """All runs/<judge>__<dataset>.jsonl in one DataFrame; stops if the cache is inconsistent."""
    rows = [r for p in sorted(RUNS.glob(f"*__{dataset}.jsonl")) for r in load_jsonl(p)]
    if not rows:
        sys.exit(f"No cached {dataset} runs in runs/: run `uv run src/run_judges.py --run` first")
    runs = pd.DataFrame(rows)
    shas = runs.groupby("prompt_id")["prompt_sha"].nunique()
    if (shas > 1).any():
        sys.exit(f"Prompt edited between runs (several prompt_sha): {list(shas[shas > 1].index)}")
    dups = runs[runs.duplicated(KEY, keep=False)]
    if len(dups):
        sys.exit(f"Duplicate cache keys in runs/:\n{dups[KEY].to_string()}")
    return runs


def judges_for(config, dataset):
    """Judges planned for a dataset: those without a `datasets` field, or whose field lists it."""
    return [j for j in config["judges"] if dataset in j.get("datasets", [dataset])]


def coverage(runs, config, dataset, expected):
    """Calls cached per judge x prompt, against the number planned (runs may be in progress)."""
    counts = runs.groupby(["judge_id", "prompt_id"]).size()
    return [(j["id"], p["id"], int(counts.get((j["id"], p["id"]), 0)), expected)
            for j in judges_for(config, dataset) for p in config["prompts"][dataset]]


def accuracy(truth, pred):
    return float((pd.Series(truth).values == pd.Series(pred).values).mean())


def cohen_kappa(truth, pred):
    """(observed - chance agreement) / (1 - chance); chance from each rater's label frequencies.

    Undefined (NaN) when chance agreement is 1, i.e. both raters always give the same single label.
    """
    truth, pred = pd.Series(truth).reset_index(drop=True), pd.Series(pred).reset_index(drop=True)
    observed = (truth == pred).mean()
    p_truth, p_pred = truth.value_counts(normalize=True), pred.value_counts(normalize=True)
    chance = sum(p_truth.get(l, 0) * p_pred.get(l, 0) for l in sorted(set(truth) | set(pred)))  # fixed order: same float sum every run
    if chance == 1:
        return float("nan")
    return float((observed - chance) / (1 - chance))


# Row-wise metrics for the bootstrap: truth and pred are (resamples, items) arrays,
# one value per resample. Same definitions as accuracy / cohen_kappa above.
def accuracy_rows(truth, pred):
    return (truth == pred).mean(axis=1)


def kappa_rows(truth, pred):
    observed = (truth == pred).mean(axis=1)
    chance = sum((truth == c).mean(axis=1) * (pred == c).mean(axis=1) for c in np.union1d(truth, pred))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(chance == 1, np.nan, (observed - chance) / (1 - chance))


def missed_fail_rows(truth, pred):
    """Share of human fails the judge passed; NaN for a resample with no fails."""
    fails = (truth == "fail").sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(fails == 0, np.nan, ((truth == "fail") & (pred == "pass")).sum(axis=1) / fails)


BOOT_METRICS = {"accuracy": accuracy_rows, "kappa": kappa_rows, "missed_fail_rate": missed_fail_rows}
BOOT_N, BOOT_SEED = 10_000, 42


def bootstrap_ci(n_items, stat_fn, n=BOOT_N, seed=BOOT_SEED):
    """Percentile 95% CI: resample the items with replacement n times and recompute the statistic.

    stat_fn maps an (n, n_items) array of resampled item indices to n values. The fixed seed means
    every call with the same n_items draws the same resamples, so statistics on the same aligned
    items are paired. Resamples where the statistic is undefined (NaN, e.g. kappa when every label
    is one class) are skipped and counted, never scored as 0. Returns (lo, hi, n_undefined).
    """
    idx = np.random.default_rng(seed).integers(0, n_items, size=(n, n_items))
    stats = np.asarray(stat_fn(idx), dtype=float)
    ok = stats[~np.isnan(stats)]
    lo, hi = np.percentile(ok, [2.5, 97.5]) if len(ok) else (np.nan, np.nan)
    return float(lo), float(hi), int(n - len(ok))


def fail_tag(note):
    """Labelling notes on fails start with the rubric tag, e.g. "unsupported - wrong claim from note 1"."""
    tag = note.split(" - ")[0].strip()
    if tag not in FAIL_TAGS:
        sys.exit(f"Unknown fail tag {tag!r}; expected one of {FAIL_TAGS}")
    return tag


def join_labels(runs, labels):
    """Attach the human label (and fail tag) to every Second Brain run line."""
    labels = labels.assign(fail_tag=[fail_tag(n) if l == "fail" else None
                                     for l, n in zip(labels["label"], labels["note"])])
    df = runs.merge(labels[["item_id", "label", "fail_tag"]], on="item_id", how="left", validate="many_to_one")
    if df["label"].isna().any():
        sys.exit(f"Run items missing from the labelled data: {sorted(df.loc[df['label'].isna(), 'item_id'])}")
    return df


def secondbrain_agreement(runs, labels):
    """One row per judge x prompt. A human fail is the case the judge should catch."""
    df = join_labels(runs, labels)
    rows = []
    for (judge, prompt), g in df.groupby(["judge_id", "prompt_id"]):
        ok = g[g["verdict"].notna()]
        human, judged = ok["label"], ok["verdict"]
        fails, passes = human == "fail", human == "pass"
        rows.append({
            "judge": judge, "prompt": prompt, "n": len(ok), "n_null": len(g) - len(ok),
            "accuracy": accuracy(human, judged), "kappa": cohen_kappa(human, judged),
            "human_pass_rate": passes.mean(), "judge_pass_rate": (judged == "pass").mean(),
            # Share of human fails the judge let through / human passes it failed
            "missed_fail_rate": (judged[fails] == "pass").mean() if fails.any() else float("nan"),
            "false_fail_rate": (judged[passes] == "fail").mean() if passes.any() else float("nan"),
            "fail_caught": int((fails & (judged == "fail")).sum()),
            "fail_missed": int((fails & (judged == "pass")).sum()),
            "pass_kept": int((passes & (judged == "pass")).sum()),
            "pass_flagged": int((passes & (judged == "fail")).sum()),
        })
    return pd.DataFrame(rows)


def secondbrain_by_tag(runs, labels):
    """How many human fails of each kind the judge caught: one row per judge x prompt x fail tag."""
    df = join_labels(runs, labels)
    df = df[df["label"] == "fail"]
    rows = []
    for (judge, prompt, tag), g in df.groupby(["judge_id", "prompt_id", "fail_tag"]):
        ok = g[g["verdict"].notna()]
        caught = int((ok["verdict"] == "fail").sum())
        rows.append({"judge": judge, "prompt": prompt, "fail_tag": tag, "n": len(ok), "n_null": len(g) - len(ok),
                     "caught": caught, "missed": len(ok) - caught,
                     "missed_fail_rate": (len(ok) - caught) / len(ok) if len(ok) else float("nan")})
    return pd.DataFrame(rows)


def secondbrain_verdicts(runs, labels):
    """Wide table: one row per item (sorted), the human label plus one verdict column per (judge, prompt)."""
    df = join_labels(runs, labels)
    wide = df.pivot(index="item_id", columns=["judge_id", "prompt_id"], values="verdict").sort_index()
    return df.groupby("item_id")["label"].first().sort_index(), wide


def secondbrain_cis(runs, labels):
    """Bootstrap 95% CIs per judge x prompt, resampling the items that judge parsed."""
    human, wide = secondbrain_verdicts(runs, labels)
    rows = []
    for (judge, prompt), verdicts in wide.items():
        ok = verdicts.notna()
        t, p = human[ok].to_numpy(), verdicts[ok].to_numpy()
        row, undefined = {"judge": judge, "prompt": prompt}, 0
        for name, metric in BOOT_METRICS.items():
            lo, hi, und = bootstrap_ci(len(t), lambda idx: metric(t[idx], p[idx]))
            row |= {f"{name}_lo": lo, f"{name}_hi": hi}
            undefined += und
        rows.append(row | {"boot_undefined": undefined})
    return pd.DataFrame(rows)


def secondbrain_diffs(runs, labels, prompts):
    """Paired bootstrap on differences: each resample draws the same items for both sides.

    Comparisons: later prompt minus earlier prompt for each judge (rubric - generic), and every
    judge pair within each prompt. Only items both sides parsed are used.
    """
    human, wide = secondbrain_verdicts(runs, labels)
    judges = sorted(wide.columns.get_level_values(0).unique())
    pairs = [((j, prompts[1]), (j, prompts[0])) for j in judges]
    pairs += [((a, pr), (b, pr)) for pr in prompts for a, b in combinations(judges, 2)]
    rows = []
    for a, b in pairs:
        if a not in wide or b not in wide:
            continue
        ok = wide[a].notna() & wide[b].notna()
        t, va, vb = human[ok].to_numpy(), wide[a][ok].to_numpy(), wide[b][ok].to_numpy()
        for name, metric in BOOT_METRICS.items():
            def diff(idx):
                return metric(t[idx], va[idx]) - metric(t[idx], vb[idx])
            estimate = float(diff(np.arange(len(t))[None, :])[0])
            lo, hi, undefined = bootstrap_ci(len(t), diff)
            rows.append({"metric": name, "a": " ".join(a), "b": " ".join(b), "diff_a_minus_b": estimate,
                         "lo": lo, "hi": hi, "excludes_0": bool(lo > 0 or hi < 0),
                         "n": len(t), "boot_undefined": undefined})
    return pd.DataFrame(rows)


def combine_orders(runs):
    """One verdict per pair from its ab and ba calls (repeat 0): agree -> that verdict, disagree -> tie.

    A null in either order (or a missing order) makes the combined verdict null.
    """
    calls = runs[runs["repeat"] == 0].pivot(index=["judge_id", "prompt_id", "item_id"],
                                            columns="order", values="verdict")
    calls = calls.reindex(columns=["ab", "ba"])
    ab, ba = calls["ab"], calls["ba"]
    verdict = ab.where(ab == ba, "tie").where(ab.notna() & ba.notna(), None)
    return verdict.rename("verdict").reset_index()


def mtbench_agreement(runs, pairs):
    """Judge vs human majority label, per judge x prompt x view x tie setting.

    view "one_call": each order's verdict scored on its own; "combined": see combine_orders.
    "without" ties drops pairs the human labelled tie; a judge tie on the rest counts as a miss.
    """
    views = {"one_call": runs[runs["repeat"] == 0][["judge_id", "prompt_id", "item_id", "verdict"]],
             "combined": combine_orders(runs)}
    human = pairs[["pair_id", "human_label"]].rename(columns={"pair_id": "item_id"})
    rows = []
    for view, v in views.items():
        df = v.merge(human, on="item_id", how="left", validate="many_to_one")
        if df["human_label"].isna().any():
            sys.exit(f"Run items missing from MT-Bench data: {sorted(df.loc[df['human_label'].isna(), 'item_id'])}")
        for (judge, prompt), g in df.groupby(["judge_id", "prompt_id"]):
            for ties, h in (("with", g), ("without", g[g["human_label"] != "tie"])):
                ok = h[h["verdict"].notna()]
                rows.append({"judge": judge, "prompt": prompt, "view": view, "ties": ties,
                             "n": len(ok), "n_null": len(h) - len(ok),
                             "accuracy": accuracy(ok["human_label"], ok["verdict"]),
                             "kappa": cohen_kappa(ok["human_label"], ok["verdict"])})
    return pd.DataFrame(rows).sort_values(["judge", "prompt", "view", "ties"], ignore_index=True)


PAPER_GPT4_VS_EXPERT = {"with": 0.66, "without": 0.85}  # Zheng et al. 2023, Table 5, turn 1 (S1 / S2)


def mtbench_vs_votes(runs, pairs):
    """Paper-style agreement: the combined verdict vs each expert vote, comparisons pooled over pairs.

    As in FastChat's compute_agreement.py, a pair with k votes gives k comparisons, and "without"
    ties skips a comparison if either side is a tie. A null verdict's comparisons go to n_null
    (without ties: only those against a non-tie vote, since the judge's side is unknown).
    """
    votes = pd.DataFrame([{"item_id": p["pair_id"], "vote": v["vote"]}
                          for _, p in pairs.iterrows() for v in p["votes"]])
    df = combine_orders(runs).merge(votes, on="item_id", how="left")
    if df["vote"].isna().any():
        sys.exit(f"Run items missing from MT-Bench data: {sorted(df.loc[df['vote'].isna(), 'item_id'])}")
    rows = []
    for (judge, prompt), g in df.groupby(["judge_id", "prompt_id"]):
        for ties, h in (("with", g), ("without", g[(g["vote"] != "tie") & (g["verdict"] != "tie")])):
            ok = h[h["verdict"].notna()]
            rows.append({"judge": judge, "prompt": prompt, "ties": ties, "n": len(ok), "n_null": len(h) - len(ok),
                         "accuracy": accuracy(ok["vote"], ok["verdict"]), "paper": PAPER_GPT4_VS_EXPERT[ties]})
    return pd.DataFrame(rows)


PAPER_EXPERT_VS_EXPERT = {"with": 0.63, "without": 0.81}  # Zheng et al. 2023, Table 5, turn 1 (S1 / S2)


def mtbench_human_ceiling(runs, pairs):
    """Expert vs expert on pairs with 2+ votes, next to judge vs each expert on those same pairs.

    Every two-expert combination on a pair is one comparison, pooled over pairs (as in FastChat);
    "without" ties skips a comparison if either expert said tie.
    """
    multi = pairs[pairs["votes"].str.len() >= 2]
    rows = []
    for ties in ("with", "without"):
        n = agree = 0
        for votes in multi["votes"]:
            v = [x["vote"] for x in votes]
            for i in range(len(v)):
                for j in range(i + 1, len(v)):
                    if ties == "without" and "tie" in (v[i], v[j]):
                        continue
                    n, agree = n + 1, agree + (v[i] == v[j])
        rows.append({"rater": "expert", "ties": ties, "n": n, "n_null": 0,
                     "accuracy": agree / n if n else float("nan"), "paper": PAPER_EXPERT_VS_EXPERT[ties]})
    judges = mtbench_vs_votes(runs[runs["item_id"].isin(multi["pair_id"])], multi)
    judges.insert(0, "rater", judges.pop("judge") + " " + judges.pop("prompt"))
    return pd.concat([pd.DataFrame(rows), judges], ignore_index=True)


def winning_position(verdict, positions):
    """Which slot the verdict picked: "first" (shown as A), "second" (B), "tie", or None if unparsed."""
    if verdict is None or (isinstance(verdict, float) and math.isnan(verdict)):
        return None
    if verdict == "tie":
        return "tie"
    return "first" if verdict == positions["A"] else "second"


def position_flags(runs, repeat_order="ab"):
    """One row per judge x prompt x pair, for pairs with a same-order repeat call.

    Three calls per pair: both orders (repeat 0) and the repeat (repeat 1, in repeat_order only,
    so noise is measured in that order).
      flip:      the two orders give different verdicts (noise + any position bias)
      noise:     the repeat gives a different verdict from the same order's first call
      direction: for a flip, the slot it favoured. A flip always has one slot winning in at least one
                 order and the other slot in neither, so "first" or "second" is always defined.
      hard:      a flip where the same slot won outright in both orders (soft: one order said tie)
    A pair with any of the three verdicts unparsed is kept with null=True and no flags.
    """
    swapped_order = "ba" if repeat_order == "ab" else "ab"
    calls = runs[(runs["repeat"] == 0) | (runs["order"] == repeat_order)]
    calls = calls.assign(pos=[winning_position(v, p) for v, p in zip(calls["verdict"], calls["positions"])],
                         present=True)
    wide = calls.pivot(index=["judge_id", "prompt_id", "item_id"], columns=["order", "repeat"],
                       values=["verdict", "pos", "present"])
    first, swapped, again = (repeat_order, 0), (swapped_order, 0), (repeat_order, 1)
    wide = wide.reindex(columns=pd.MultiIndex.from_tuples(
        [(v, *c) for v in ("verdict", "pos", "present") for c in (first, swapped, again)]))
    wide = wide[wide[("present", *again)].notna()]  # pairs without a repeat call are not in the sample

    v1, v2, v3 = (wide[("verdict", *c)] for c in (first, swapped, again))
    p1, p2 = wide[("pos", *first)], wide[("pos", *swapped)]
    null = v1.isna() | v2.isna() | v3.isna()
    flip, noise = (v1 != v2) & ~null, (v1 != v3) & ~null
    favours_second = (p1 == "second") | (p2 == "second")
    direction = pd.Series(np.where(favours_second, "second", "first"), index=wide.index).where(flip, None)
    hard = flip & (p1 == p2) & (p1 != "tie")
    out = pd.DataFrame({"null": null, "flip": flip.where(~null, None), "noise": noise.where(~null, None),
                        "direction": direction, "hard": hard.where(~null, None)})
    return out.reset_index()


def exact_binom_p(b, c):
    """Two-sided exact test of b vs c against 50/50 (exact McNemar on discordant pairs; sign test)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def mtbench_position_bias(flags):
    """Flip rate vs same-order noise per judge x prompt, on the same pairs.

    gap = flip - noise: the part of the flip rate noise can't explain. Its CI is a bootstrap over
    pairs (paired: each resample draws the same pairs for both rates); mcnemar_p tests flip_only vs
    noise_only pairs exactly. direction_p tests flips toward first vs second against 50/50, which is
    what noise alone would give.
    """
    rows = []
    for (judge, prompt), g in flags.groupby(["judge_id", "prompt_id"]):
        ok = g[~g["null"]]
        f, z = ok["flip"].to_numpy(dtype=int), ok["noise"].to_numpy(dtype=int)
        d = f - z
        lo, hi, _ = bootstrap_ci(len(d), lambda idx: d[idx].mean(axis=1)) if len(d) else (np.nan, np.nan, 0)
        flip_only, noise_only = int(((f == 1) & (z == 0)).sum()), int(((f == 0) & (z == 1)).sum())
        to_first, to_second = int((ok["direction"] == "first").sum()), int((ok["direction"] == "second").sum())
        hard = int(ok["hard"].astype(bool).sum())
        rows.append({"judge": judge, "prompt": prompt, "n": len(ok), "n_null": int(g["null"].sum()),
                     "noise": z.mean() if len(z) else np.nan, "flip": f.mean() if len(f) else np.nan,
                     "gap": d.mean() if len(d) else np.nan, "gap_lo": lo, "gap_hi": hi,
                     "flip_only": flip_only, "noise_only": noise_only,
                     "mcnemar_p": exact_binom_p(flip_only, noise_only),
                     "to_first": to_first, "to_second": to_second,
                     "direction_p": exact_binom_p(to_first, to_second),
                     "hard": hard, "soft": int(f.sum()) - hard})
    return pd.DataFrame(rows)


def main():
    config = yaml.safe_load(CONFIG.read_text())
    RESULTS.mkdir(exist_ok=True)

    pairs = pd.DataFrame(load_jsonl(ROOT / config["data"]["mtbench"]))
    mt = load_runs("mtbench")
    print("MT-Bench coverage (cached / planned):")
    planned = 2 * len(pairs) + config["repeats"]["n_items"]  # both orders + the repeated subset
    for judge, prompt, got, expected in coverage(mt, config, "mtbench", planned):
        print(f"  {judge} x {prompt}: {got}/{expected}" + ("" if got == expected else "  (incomplete)"))

    agree = mtbench_agreement(mt, pairs)
    if agree["n_null"].sum():
        print("warning: unparsed verdicts left out (see n_null)")
    agree.to_csv(RESULTS / "mtbench_agreement.csv", index=False, float_format=CSV_FLOAT)
    print("\nMT-Bench agreement with the human majority label (repeat 0):")
    print(agree.to_markdown(index=False, floatfmt=".2f"))
    print(f"\nWrote {(RESULTS / 'mtbench_agreement.csv').relative_to(ROOT)}")

    paper = mtbench_vs_votes(mt, pairs)
    paper.to_csv(RESULTS / "mtbench_paper_style.csv", index=False, float_format=CSV_FLOAT)
    print("\nMT-Bench paper-style agreement: combined verdict vs each expert vote "
          "(n = comparisons; paper = GPT-4, turn 1):")
    print(paper.to_markdown(index=False, floatfmt=".2f"))
    print(f"\nWrote {(RESULTS / 'mtbench_paper_style.csv').relative_to(ROOT)}")

    ceiling = mtbench_human_ceiling(mt, pairs)
    ceiling.to_csv(RESULTS / "mtbench_human_ceiling.csv", index=False, float_format=CSV_FLOAT)
    print(f"\nMT-Bench human-human ceiling on the {(pairs['votes'].str.len() >= 2).sum()} pairs with 2+ expert votes "
          "(n = comparisons; paper = turn 1):")
    print(ceiling.to_markdown(index=False, floatfmt=".2f"))
    print("Caveat: multi-vote pairs are not a random subset, so judges are scored on these same pairs only.")
    print(f"\nWrote {(RESULTS / 'mtbench_human_ceiling.csv').relative_to(ROOT)}")

    bias = mtbench_position_bias(position_flags(mt, config["repeats"]["order"]))
    bias.to_csv(RESULTS / "mtbench_position_bias.csv", index=False, float_format=CSV_FLOAT)
    print(f"\nMT-Bench position bias: order flip rate vs same-order repeat noise on the same pairs "
          f"(gap CI: bootstrap, {BOOT_N:,} resamples of pairs, seed {BOOT_SEED}; p-values exact):")
    print(bias.to_markdown(index=False, floatfmt=".3f"))
    if bias["n_null"].sum():
        print("warning: pairs with an unparsed verdict left out (see n_null)")
    print(f"\nWrote {(RESULTS / 'mtbench_position_bias.csv').relative_to(ROOT)}\n")

    sb_path = ROOT / config["data"]["secondbrain"]
    if not sb_path.exists():  # personal notes, not in the repo
        print(f"Second Brain skipped: {sb_path.relative_to(ROOT)} not found (private data, not in the repo).")
        return
    labels = pd.DataFrame(load_jsonl(sb_path))
    runs = load_runs("secondbrain")
    print("Second Brain coverage (cached / planned):")
    for judge, prompt, got, expected in coverage(runs, config, "secondbrain", len(labels)):
        print(f"  {judge} x {prompt}: {got}/{expected}" + ("" if got == expected else "  (incomplete)"))

    sb = secondbrain_agreement(runs, labels).merge(secondbrain_cis(runs, labels), on=["judge", "prompt"])
    first = ["judge", "prompt", "n", "n_null"] + [c for m in BOOT_METRICS for c in (m, f"{m}_lo", f"{m}_hi")]
    sb = sb[first + [c for c in sb.columns if c not in first]]
    if sb["n_null"].sum():
        print(f"warning: {sb['n_null'].sum()} unparsed verdicts left out (see n_null)")
    sb.to_csv(RESULTS / "secondbrain_agreement.csv", index=False, float_format=CSV_FLOAT)
    print(f"\nSecond Brain agreement with human pass/fail (fail = the case to catch), "
          f"bootstrap 95% CIs ({BOOT_N:,} resamples of items, seed {BOOT_SEED}):")
    print(sb.to_markdown(index=False, floatfmt=".2f"))
    if sb["boot_undefined"].sum():
        print(f"note: {sb['boot_undefined'].sum()} resamples had an undefined statistic and were skipped")
    print(f"With n={len(labels)} items the kappa CIs are wide (about ±0.2): small gaps between judges are noise.")
    print(f"\nWrote {(RESULTS / 'secondbrain_agreement.csv').relative_to(ROOT)}")

    diffs = secondbrain_diffs(runs, labels, [p["id"] for p in config["prompts"]["secondbrain"]])
    diffs.to_csv(RESULTS / "secondbrain_diffs.csv", index=False, float_format=CSV_FLOAT)
    print("\nSecond Brain paired differences (a - b, same resampled items for both sides; "
          "excludes_0 = CI does not include 0):")
    print(diffs.to_markdown(index=False, floatfmt=".2f"))
    print(f"\nWrote {(RESULTS / 'secondbrain_diffs.csv').relative_to(ROOT)}")

    tags = secondbrain_by_tag(runs, labels)
    tags.to_csv(RESULTS / "secondbrain_by_tag.csv", index=False, float_format=CSV_FLOAT)
    caught = tags.assign(cell=tags["caught"].astype(str) + "/" + tags["n"].astype(str),
                         col=tags["judge"] + " " + tags["prompt"])
    print("\nSecond Brain human fails caught, by fail tag (caught / fails scored):")
    print(caught.pivot(index="fail_tag", columns="col", values="cell").to_markdown())
    print(f"\nWrote {(RESULTS / 'secondbrain_by_tag.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
