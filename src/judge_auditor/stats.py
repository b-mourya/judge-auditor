"""Small statistics helpers (stdlib only, so the math is easy to read and explain)."""
from __future__ import annotations

import random
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")


def mean(xs: Sequence[float]) -> float:
    if not xs:
        raise ValueError("mean() of empty sequence")
    return sum(xs) / len(xs)


def cohens_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Agreement between two raters, corrected for agreement expected by chance.

    1.0 = perfect agreement, 0.0 = no better than chance, < 0 = worse than chance.
    """
    if len(a) != len(b) or not a:
        raise ValueError("cohens_kappa needs two equal-length, non-empty sequences")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    labels = set(a) | set(b)
    expected = sum((list(a).count(l) / n) * (list(b).count(l) / n) for l in labels)
    if expected == 1.0:  # both raters used one identical label for everything
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def bootstrap_ci(
    items: Sequence[T],
    stat: Callable[[list[T]], float],
    iters: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for any statistic over a list of items."""
    if not items:
        raise ValueError("bootstrap_ci needs at least one item")
    rng = random.Random(seed)
    n = len(items)
    samples = sorted(stat([items[rng.randrange(n)] for _ in range(n)]) for _ in range(iters))
    lo = samples[int((alpha / 2) * iters)]
    hi = samples[int((1 - alpha / 2) * iters) - 1]
    return lo, hi
