"""Metrics checked against small examples worked out by hand.

    uv run pytest
"""

import math

import numpy as np
import pandas as pd
import pytest

import analyze
from analyze import (accuracy, bootstrap_ci, cohen_kappa, combine_orders, fail_tag, judges_for, kappa_rows,
                     load_runs, missed_fail_rows, mtbench_agreement, mtbench_human_ceiling, mtbench_vs_votes,
                     secondbrain_agreement, secondbrain_by_tag, secondbrain_diffs)


def test_kappa_two_labels():
    # observed 8/10 = 0.80; both raters 60% pass / 40% fail -> chance 0.6*0.6 + 0.4*0.4 = 0.52
    # kappa = (0.80 - 0.52) / (1 - 0.52) = 0.28 / 0.48
    truth = ["pass"] * 6 + ["fail"] * 4
    pred = ["pass"] * 5 + ["fail"] * 4 + ["pass"]
    assert accuracy(truth, pred) == pytest.approx(0.8)
    assert cohen_kappa(truth, pred) == pytest.approx(0.28 / 0.48)


def test_kappa_three_labels():
    # observed 3/4; truth x .50 y .25 tie .25, pred x .75 y .25 -> chance .375 + .0625 = .4375
    # kappa = (.75 - .4375) / (1 - .4375) = 5/9
    assert cohen_kappa(["x", "y", "tie", "x"], ["x", "y", "x", "x"]) == pytest.approx(5 / 9)


def test_kappa_edge_cases():
    assert cohen_kappa(["pass", "fail"], ["pass", "fail"]) == pytest.approx(1.0)
    assert cohen_kappa(["pass", "fail"], ["fail", "pass"]) == pytest.approx(-1.0)
    assert math.isnan(cohen_kappa(["pass", "pass"], ["pass", "pass"]))  # chance agreement is 1


def test_secondbrain_agreement_counts_and_nulls():
    labels = pd.DataFrame({"item_id": ["a", "b", "c", "d", "e"],
                           "label": ["fail", "fail", "pass", "pass", "pass"],
                           "note": ["unsupported", "off-topic", "ok", "ok", "ok"]})
    runs = pd.DataFrame({"item_id": ["a", "b", "c", "d", "e"], "judge_id": "j", "prompt_id": "p",
                         "verdict": ["fail", "pass", "pass", "fail", None]})
    row = secondbrain_agreement(runs, labels).iloc[0]
    assert (row["n"], row["n_null"]) == (4, 1)  # the null is counted, not scored
    assert (row["fail_caught"], row["fail_missed"], row["pass_kept"], row["pass_flagged"]) == (1, 1, 1, 1)
    assert row["accuracy"] == pytest.approx(0.5)
    assert row["missed_fail_rate"] == pytest.approx(0.5)
    assert row["human_pass_rate"] == pytest.approx(0.5)  # among the 4 scored items


def test_secondbrain_by_tag():
    # Tag = note text before " - "; passes are ignored; a null verdict is counted, not scored
    labels = pd.DataFrame({"item_id": ["a", "b", "c", "d", "e"],
                           "label": ["fail", "fail", "fail", "fail", "pass"],
                           "note": ["should-abstain - near-miss", "should-abstain", "should-abstain - x",
                                    "unsupported - wrong claim", "abstain - false-premise"]})
    runs = pd.DataFrame({"item_id": ["a", "b", "c", "d", "e"], "judge_id": "j", "prompt_id": "p",
                         "verdict": ["fail", "pass", None, "pass", "pass"]})
    tags = secondbrain_by_tag(runs, labels).set_index("fail_tag")
    assert list(tags.index) == ["should-abstain", "unsupported"]
    assert tuple(tags.loc["should-abstain", ["n", "n_null", "caught", "missed"]]) == (2, 1, 1, 1)
    assert tuple(tags.loc["unsupported", ["n", "caught", "missed"]]) == (1, 0, 1)


def test_unknown_fail_tag_stops():
    with pytest.raises(SystemExit):
        fail_tag("hallucinated - typo in tag")


def test_no_cached_runs_stops_with_hint(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "RUNS", tmp_path)  # an empty runs/
    with pytest.raises(SystemExit, match="run_judges.py --run"):
        load_runs("mtbench")


def test_judges_for_respects_datasets_field():
    config = {"judges": [{"id": "a"}, {"id": "b", "datasets": ["secondbrain"]}]}
    assert [j["id"] for j in judges_for(config, "mtbench")] == ["a"]
    assert [j["id"] for j in judges_for(config, "secondbrain")] == ["a", "b"]


def mt_runs(calls):
    """calls: (item, order, repeat, verdict) for one judge x prompt."""
    return pd.DataFrame([{"item_id": i, "order": o, "repeat": r, "verdict": v, "judge_id": "j", "prompt_id": "p"}
                         for i, o, r, v in calls])


def test_combine_orders():
    # agree -> that verdict; x vs y or x vs tie -> tie; a null in either order -> null; repeat 1 ignored
    runs = mt_runs([("a", "ab", 0, "x"), ("a", "ba", 0, "x"), ("a", "ab", 1, "y"),
                    ("b", "ab", 0, "x"), ("b", "ba", 0, "y"),
                    ("c", "ab", 0, "tie"), ("c", "ba", 0, "y"),
                    ("d", "ab", 0, None), ("d", "ba", 0, "y")])
    got = combine_orders(runs).set_index("item_id")["verdict"]
    assert got[["a", "b", "c"]].tolist() == ["x", "tie", "tie"]
    assert pd.isna(got["d"])


def test_mtbench_agreement():
    # Human: a=x, b=y, c=tie. Calls: a x/x, b y/tie, c tie/None.
    # one_call with ties: 5 scored (a,a,b,b,c-ab), 1 null; correct a,a,b-ab,c-ab -> 4/5
    # one_call without: drop c -> 4 scored, 0 null; the judge tie on b-ba is a miss -> 3/4
    # combined: a=x, b=tie, c=null. with: 2 scored, 1 null, correct a -> 1/2; without: 2 scored, 1/2
    pairs = pd.DataFrame({"pair_id": ["a", "b", "c"], "human_label": ["x", "y", "tie"]})
    runs = mt_runs([("a", "ab", 0, "x"), ("a", "ba", 0, "x"), ("b", "ab", 0, "y"), ("b", "ba", 0, "tie"),
                    ("c", "ab", 0, "tie"), ("c", "ba", 0, None), ("a", "ab", 1, "y")])
    t = mtbench_agreement(runs, pairs).set_index(["view", "ties"])
    expected = {("one_call", "with"): (5, 1, 4 / 5), ("one_call", "without"): (4, 0, 3 / 4),
                ("combined", "with"): (2, 1, 1 / 2), ("combined", "without"): (2, 0, 1 / 2)}
    for key, (n, n_null, acc) in expected.items():
        assert (t.loc[key, "n"], t.loc[key, "n_null"]) == (n, n_null), key
        assert t.loc[key, "accuracy"] == pytest.approx(acc), key


def test_mtbench_vs_votes():
    # Combined: a=x (x/x), b=tie (x/y), c=null. Votes: a [x, x, y], b [tie, x], c [y].
    # with: a gives 3 comparisons (2 agree), b gives 2 (tie==tie agrees) -> 3/5; c's 1 vote -> n_null 1
    # without: a's 3 kept (2/3); b dropped (judge tie); c's non-tie vote -> n_null 1
    pairs = pd.DataFrame({"pair_id": ["a", "b", "c"],
                          "votes": [[{"vote": "x"}, {"vote": "x"}, {"vote": "y"}],
                                    [{"vote": "tie"}, {"vote": "x"}], [{"vote": "y"}]]})
    runs = mt_runs([("a", "ab", 0, "x"), ("a", "ba", 0, "x"), ("b", "ab", 0, "x"), ("b", "ba", 0, "y"),
                    ("c", "ab", 0, None), ("c", "ba", 0, "y")])
    t = mtbench_vs_votes(runs, pairs).set_index("ties")
    assert (t.loc["with", "n"], t.loc["with", "n_null"]) == (5, 1)
    assert t.loc["with", "accuracy"] == pytest.approx(3 / 5)
    assert (t.loc["without", "n"], t.loc["without", "n_null"]) == (3, 1)
    assert t.loc["without", "accuracy"] == pytest.approx(2 / 3)


def test_mtbench_human_ceiling():
    # Multi-vote pairs: a [x, x, tie] -> 3 comparisons (x=x agree, x/tie, x/tie); b [y, x] -> 1 (disagree).
    # c has one vote and is left out everywhere.
    # expert with ties: 1/4; without: skip the 2 with a tie -> 1/2
    # judge (combined a=x, b=y) vs votes on a, b: with 5 comparisons, agree x,x,y -> 3/5; without 4, 3/4
    pairs = pd.DataFrame({"pair_id": ["a", "b", "c"],
                          "votes": [[{"vote": "x"}, {"vote": "x"}, {"vote": "tie"}],
                                    [{"vote": "y"}, {"vote": "x"}], [{"vote": "y"}]]})
    runs = mt_runs([("a", "ab", 0, "x"), ("a", "ba", 0, "x"), ("b", "ab", 0, "y"), ("b", "ba", 0, "y"),
                    ("c", "ab", 0, "x"), ("c", "ba", 0, "x")])
    t = mtbench_human_ceiling(runs, pairs).set_index(["rater", "ties"])
    expected = {("expert", "with"): (4, 1 / 4), ("expert", "without"): (2, 1 / 2),
                ("j p", "with"): (5, 3 / 5), ("j p", "without"): (4, 3 / 4)}
    for key, (n, acc) in expected.items():
        assert t.loc[key, "n"] == n, key
        assert t.loc[key, "accuracy"] == pytest.approx(acc), key


def test_row_metrics_match_scalar_versions():
    # The bootstrap's vectorised metrics must equal the tested scalar ones on the full sample
    truth = np.array(["pass"] * 6 + ["fail"] * 4)
    pred = np.array(["pass"] * 5 + ["fail"] * 4 + ["pass"])
    full = np.arange(10)[None, :]
    assert kappa_rows(truth[full], pred[full])[0] == pytest.approx(cohen_kappa(truth, pred))
    assert missed_fail_rows(truth[full], pred[full])[0] == pytest.approx(1 / 4)  # 1 of 4 fails passed
    same = np.array([["pass", "pass"]])
    assert math.isnan(kappa_rows(same, same)[0])  # one class: undefined, not 0


def test_bootstrap_ci_reproducible_and_bounds():
    truth = np.array(["pass"] * 6 + ["fail"] * 4)
    pred = np.array(["pass"] * 5 + ["fail"] * 4 + ["pass"])
    stat = lambda idx: kappa_rows(truth[idx], pred[idx])
    first = bootstrap_ci(10, stat, n=2000)
    assert first == bootstrap_ci(10, stat, n=2000)  # fixed seed -> same output
    lo, hi, undefined = first
    assert lo < cohen_kappa(truth, pred) < hi
    # A perfect judge has no uncertainty in accuracy
    assert bootstrap_ci(10, lambda idx: (truth[idx] == truth[idx]).mean(axis=1), n=500)[:2] == (1.0, 1.0)


def test_bootstrap_ci_counts_undefined_resamples():
    # Every resample is single-class, so kappa is undefined throughout: skipped and counted, never 0
    labels = np.array(["pass"] * 5)
    lo, hi, undefined = bootstrap_ci(5, lambda idx: kappa_rows(labels[idx], labels[idx]), n=300)
    assert undefined == 300 and math.isnan(lo) and math.isnan(hi)


def test_secondbrain_diffs_paired():
    # Judges j and k give identical verdicts -> every paired difference is exactly 0
    labels = pd.DataFrame({"item_id": list("abcdef"), "label": ["fail", "fail", "pass", "pass", "pass", "fail"],
                           "note": ["unsupported", "off-topic", "ok", "ok", "ok", "unsupported"]})
    verdicts = ["fail", "pass", "pass", "fail", "pass", "fail"]
    runs = pd.DataFrame([{"item_id": i, "judge_id": j, "prompt_id": p, "verdict": v}
                         for j in ("j", "k") for p in ("g", "r") for i, v in zip("abcdef", verdicts)])
    d = secondbrain_diffs(runs, labels, ["g", "r"])
    assert len(d) == 4 * 3  # j r-g, k r-g, j-k on g, j-k on r; three metrics each
    assert (d[["diff_a_minus_b", "lo", "hi"]] == 0).all().all()
    assert not d["excludes_0"].any()
