"""The three audits: human agreement, position bias, and verbosity bias."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from .data import Pair
from .judges import Judge
from .stats import bootstrap_ci, cohens_kappa, mean

# Filler that makes an answer longer without making it more correct.
FILLER_PREFIX = "That's a great question, and it deserves a careful and thorough answer."
FILLER_SUFFIX = (
    " There are many factors that can matter here, and the best approach often depends "
    "on the specific context and goals involved. In summary, the explanation above covers "
    "the key considerations you should keep in mind."
)


def pad(answer: str) -> str:
    return f"{FILLER_PREFIX} {answer}{FILLER_SUFFIX}"


def to_label(verdict: str, a_first: bool) -> str:
    """Turn a positional verdict ("1"/"2"/"tie") into an answer identity ("A"/"B"/"tie")."""
    if verdict == "tie":
        return "tie"
    picked_first = verdict == "1"
    if a_first:
        return "A" if picked_first else "B"
    return "B" if picked_first else "A"


@dataclass
class Judgment:
    pair_id: str
    human: str
    ab: str  # judge's label when shown A first
    ba: str  # judge's label when shown B first


def judge_both_orders(
    judge: Judge, pairs: list[Pair], progress: Callable[[int, int], None] | None = None
) -> list[Judgment]:
    out = []
    for i, p in enumerate(pairs, start=1):
        ab = to_label(judge.compare(p.question, p.answer_a, p.answer_b), a_first=True)
        ba = to_label(judge.compare(p.question, p.answer_b, p.answer_a), a_first=False)
        out.append(Judgment(p.id, p.human, ab, ba))
        if progress:
            progress(i, len(pairs))
    return out


# ------------------------------------------------------------------ 1. agreement


def agreement_audit(js: list[Judgment], iters: int = 2000, seed: int = 0) -> dict:
    """How often does the judge agree with humans? (Uses the A-first ordering.)"""
    acc = lambda items: mean([float(j.ab == j.human) for j in items])  # noqa: E731
    kap = lambda items: cohens_kappa([j.ab for j in items], [j.human for j in items])  # noqa: E731
    return {
        "n": len(js),
        "accuracy": acc(js),
        "accuracy_ci": bootstrap_ci(js, acc, iters, seed),
        "kappa": kap(js),
        "kappa_ci": bootstrap_ci(js, kap, iters, seed),
        "disagreements": [
            {"id": j.pair_id, "human": j.human, "judge": j.ab} for j in js if j.ab != j.human
        ],
    }


# ------------------------------------------------------------------ 2. position bias


def position_audit(js: list[Judgment], iters: int = 2000, seed: int = 0) -> dict:
    """Does the verdict change when we swap the order of the two answers?"""
    consistency = lambda items: mean([float(j.ab == j.ba) for j in items])  # noqa: E731

    # Count decisive verdicts that went to whichever answer was shown first.
    picks_first = sum(j.ab == "A" for j in js) + sum(j.ba == "B" for j in js)
    decisive = sum(j.ab != "tie" for j in js) + sum(j.ba != "tie" for j in js)
    return {
        "n": len(js),
        "consistency": consistency(js),
        "consistency_ci": bootstrap_ci(js, consistency, iters, seed),
        "first_position_win_rate": picks_first / decisive if decisive else None,
        "inconsistent": [
            {"id": j.pair_id, "a_first": j.ab, "b_first": j.ba} for j in js if j.ab != j.ba
        ],
    }


# ------------------------------------------------------------------ 3. verbosity bias


def _loser_score(label: str, loser: str) -> float:
    return 1.0 if label == loser else 0.5 if label == "tie" else 0.0


def verbosity_audit(
    judge: Judge, pairs: list[Pair], js: list[Judgment], iters: int = 2000, seed: int = 0
) -> dict:
    """Pad the *worse* answer with content-free filler. Does the judge start liking it more?

    Each answer is judged in both orders, so position bias cancels out of this measurement.
    """
    by_id = {j.pair_id: j for j in js}
    rows = []
    for p in pairs:
        if p.human == "tie":
            continue
        loser = "B" if p.human == "A" else "A"
        before_j = by_id[p.id]
        before = (_loser_score(before_j.ab, loser) + _loser_score(before_j.ba, loser)) / 2

        padded = replace(p, **{f"answer_{loser.lower()}": pad(getattr(p, f"answer_{loser.lower()}"))})
        after_j = judge_both_orders(judge, [padded])[0]
        after = (_loser_score(after_j.ab, loser) + _loser_score(after_j.ba, loser)) / 2
        rows.append({"id": p.id, "before": before, "after": after, "delta": after - before})

    if not rows:
        return {"n": 0, "mean_delta": None, "mean_delta_ci": None, "flipped": []}
    delta = lambda items: mean([r["delta"] for r in items])  # noqa: E731
    return {
        "n": len(rows),
        "loser_score_before": mean([r["before"] for r in rows]),
        "loser_score_after": mean([r["after"] for r in rows]),
        "mean_delta": delta(rows),
        "mean_delta_ci": bootstrap_ci(rows, delta, iters, seed),
        "flipped": [r["id"] for r in rows if r["before"] == 0.0 and r["after"] > 0.0],
    }
