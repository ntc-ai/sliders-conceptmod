"""Small-n statistics for the paired comparison. Stdlib + numpy only.

The env's scipy is ABI-broken against numpy 2 (see blindspots.md) — nothing
here may import it. Spearman uses average-rank ties: the naive numbered-rank
version produced a fake rho=1.0 on an all-zero column during the blind-spot
audit, so ties handling is load-bearing, and the selftest pins it.
"""

from __future__ import annotations

import itertools
import math
import statistics as st
from math import comb


def rank_avg(values: list[float]) -> list[float]:
    """Average ranks (1-based) with tie averaging."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 3:
        raise ValueError("spearman needs >= 3 paired values")
    rx, ry = rank_avg(x), rank_avg(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    if den == 0:
        return 0.0  # a constant series correlates with nothing (post-audit rule)
    return num / den


def sign_test_p(n_pos: int, n: int) -> float:
    """Two-sided exact binomial p for n_pos successes out of n at p=0.5."""
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(0, min(n_pos, n - n_pos) + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def paired_summary(diffs: list[float]) -> dict[str, float]:
    """Mean / median / sd / sign-consistency of per-seed paired differences."""
    n = len(diffs)
    n_pos = sum(1 for d in diffs if d > 0)
    n_neg = sum(1 for d in diffs if d < 0)
    return {
        "n": n,
        "mean": st.mean(diffs) if n else float("nan"),
        "median": st.median(diffs) if n else float("nan"),
        "sd": st.stdev(diffs) if n > 1 else float("nan"),
        "sign_agree": max(n_pos, n_neg) / n if n else float("nan"),
        "sign_p": sign_test_p(n_pos, n_pos + n_neg),
    }


def perm_sign_flip_p(per_seed_spans: list[float], n_perm: int | None = None) -> float:
    """Sign-flip permutation test: is the pooled span reliably one-signed?

    Exact enumeration when 2^n is small (it always is here: 3-10 seeds).
    p = fraction of sign assignments whose |mean| >= |observed mean|.
    Used by pair onboarding (G0): a concept with no certifiable direction has
    spans whose mean is unremarkable under sign flips.
    """
    n = len(per_seed_spans)
    if n == 0:
        return 1.0
    obs = abs(st.mean(per_seed_spans))
    count = 0
    total = 0
    for signs in itertools.product((1.0, -1.0), repeat=n):
        m = abs(st.mean([s * v for s, v in zip(signs, per_seed_spans)]))
        total += 1
        if m >= obs - 1e-12:
            count += 1
    return count / total


def percentile(values: list[float], q: float) -> float:
    """Inclusive linear-interpolation percentile (numpy 'linear'), stdlib impl."""
    if not values:
        raise ValueError("percentile of empty list")
    v = sorted(values)
    if len(v) == 1:
        return v[0]
    pos = (len(v) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    frac = pos - lo
    return v[lo] * (1 - frac) + v[hi] * frac


def selftest() -> int:
    # ties: the audit's fake-rho bug — constant column must give rho 0, not 1
    assert spearman([0.0, 0.0, 0.0, 0.0], [1, 2, 3, 4]) == 0.0
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-12
    assert abs(spearman([1, 2, 3, 4], [40, 30, 20, 10]) + 1.0) < 1e-12
    # exact binomial: 9/11 -> p=.0654 two-sided ... one-sided .0327 (protocol quotes .033 one-sided)
    assert abs(sign_test_p(9, 11) - 2 * 0.03271) < 2e-3
    assert sign_test_p(2, 4) == 1.0
    # permutation: strongly one-signed spans are significant, mixed are not
    assert perm_sign_flip_p([1.0, 1.1, 0.9, 1.2, 1.05]) <= 2 / 32 + 1e-9
    assert perm_sign_flip_p([1.0, -1.0, 0.9, -0.9]) > 0.5
    assert percentile([1, 2, 3, 4, 5], 0.95) == 4.8
    s = paired_summary([0.5, 0.4, 0.6])
    assert s["sign_agree"] == 1.0 and abs(s["mean"] - 0.5) < 1e-12
    print("stats selftest OK")
    return 0
