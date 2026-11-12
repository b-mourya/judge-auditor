"""Each audit should detect the bias it targets, and stay quiet when the bias is absent."""
from pathlib import Path

import pytest

from judge_auditor import audits
from judge_auditor.data import load_pairs
from judge_auditor.judges import SimulatedJudge, parse_verdict

DATA = Path(__file__).resolve().parents[1] / "data" / "sample_pairs.jsonl"


@pytest.fixture(scope="module")
def pairs():
    return load_pairs(DATA)


def run(judge, pairs):
    js = audits.judge_both_orders(judge, pairs)
    return (
        audits.agreement_audit(js, iters=300),
        audits.position_audit(js, iters=300),
        audits.verbosity_audit(judge, pairs, js, iters=300),
    )


def test_unbiased_judge_looks_clean(pairs):
    agreement, position, verbosity = run(SimulatedJudge(pairs, noise=0.0), pairs)
    assert agreement["kappa"] == 1.0
    assert position["consistency"] == 1.0
    assert verbosity["mean_delta"] == 0.0


def test_position_bias_is_detected(pairs):
    _, position, _ = run(SimulatedJudge(pairs, noise=0.0, position_bias=0.4), pairs)
    assert position["consistency"] < 0.9
    assert position["first_position_win_rate"] > 0.5


def test_verbosity_bias_is_detected(pairs):
    _, _, verbosity = run(SimulatedJudge(pairs, noise=0.0, length_bias=0.8), pairs)
    assert verbosity["mean_delta"] > 0.1
    assert verbosity["mean_delta_ci"][0] > 0


def test_to_label_round_trip():
    assert audits.to_label("1", a_first=True) == "A"
    assert audits.to_label("1", a_first=False) == "B"
    assert audits.to_label("2", a_first=False) == "A"
    assert audits.to_label("tie", a_first=False) == "tie"


@pytest.mark.parametrize("text,expected", [
    ("Answer 1 is correct.\n[[1]]", "1"),
    ("Both fine. [[TIE]]", "tie"),
    ("I first thought [[1]] but actually [[2]]", "2"),
    ("[[ tie ]]", "tie"),
])
def test_parse_verdict(text, expected):
    assert parse_verdict(text) == expected


def test_parse_verdict_missing():
    with pytest.raises(ValueError):
        parse_verdict("no verdict here")
