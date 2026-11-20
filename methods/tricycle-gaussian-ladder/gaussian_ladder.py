"""TriCycLe triangle target using Ladder scores and integer Gaussian comparisons.

Only the returned target is covered by the edge-zCDP claim. The raw count,
profile and all comparisons stay inside the trusted process.
"""

import argparse
from bisect import bisect_left
from fractions import Fraction
from math import comb, isqrt
from random import SystemRandom


def _exp_coin(x, rng):
    """Exact Bernoulli(exp(-x)) for a nonnegative Fraction (CKS Algorithms 1–2)."""
    if x > 1:
        for _ in range(x.numerator // x.denominator):
            if not _exp_coin(Fraction(1), rng):
                return False
        x %= 1
    k = 1
    while rng.randrange(x.denominator * k) < x.numerator:
        k += 1
    return k % 2 == 1


def _discrete_gaussian(sigma2, rng):
    """Exact integer Gaussian rejection sampling (CKS Algorithm 3)."""
    t = isqrt(sigma2.numerator // sigma2.denominator) + 1
    center = sigma2 / t
    while True:
        while True:
            u = rng.randrange(t)
            if _exp_coin(Fraction(u, t), rng):
                break
        v = 0
        while _exp_coin(Fraction(1), rng):
            v += 1
        magnitude = u + t * v
        negative = rng.randrange(2)
        if negative and not magnitude:
            continue
        if _exp_coin((magnitude - center) ** 2 / (2 * sigma2), rng):
            return -magnitude if negative else magnitude


class _Ladder:
    def __init__(self, n, edges):
        self.cap = max(0, n - 2)
        self.triangles = 0
        self.prefix = [0]
        if not self.cap:
            return
        adjacency = [set() for _ in range(n)]
        for u, v in edges:
            if not (0 <= u < n and 0 <= v < n) or u == v:
                raise ValueError('expected simple undirected edges on public IDs 0..n-1')
            adjacency[u].add(v)
            adjacency[v].add(u)

        # Zhang et al., Lemma 5.1: keep the largest number of common
        # neighbors a for each number b of exclusive neighbors.
        best = [None] * (self.cap + 1)
        triple_count = 0
        for u, nu in enumerate(adjacency):
            for v in range(u + 1, n):
                nv = adjacency[v]
                a = len(nu & nv)
                adjacent = v in nu
                b = len(nu) + len(nv) - 2 * (a + adjacent)
                if best[b] is None or a > best[b]:
                    best[b] = a
                if adjacent:
                    triple_count += a
        self.triangles = triple_count // 3

        # b >= t gives a+t. b < t gives floor((2a+b+t)/2).
        suffix = [None] * (self.cap + 2)
        for b in range(self.cap, -1, -1):
            candidates = (x for x in (best[b], suffix[b + 1]) if x is not None)
            suffix[b] = max(candidates, default=None)
        lower = None
        for t in range(2 * self.cap + 1):
            if 0 < t <= len(best) and best[t - 1] is not None:
                candidate = 2 * best[t - 1] + t - 1
                lower = candidate if lower is None else max(lower, candidate)
            upper = suffix[t] if t < len(suffix) else None
            width = min(self.cap, max(t + upper if upper is not None else 0,
                                      (t + lower) // 2 if lower is not None else 0))
            if width == self.cap:
                break  # From this rung onward every width is the public cap.
            self.prefix.append(self.prefix[-1] + width)

    def rung(self, target):
        distance = abs(target - self.triangles)
        if distance <= self.prefix[-1]:
            return bisect_left(self.prefix, distance)
        extra = distance - self.prefix[-1]
        return len(self.prefix) - 1 + (extra + self.cap - 1) // self.cap

    def interval_min(self, low, high):
        return self.rung(min(max(self.triangles, low), high))


def _select(ladder, upper, rho, rng):
    lo, hi = 0, upper
    rounds = upper.bit_length()  # ceil(log2(upper + 1))
    if not rounds:
        return 0
    sigma2 = Fraction(2 * rounds) / rho  # sensitivity 2, per-round rho/rounds
    while lo < hi:
        middle = (lo + hi) // 2
        difference = ladder.interval_min(lo, middle) - ladder.interval_min(middle + 1, hi)
        if difference + _discrete_gaussian(sigma2, rng) <= 0:
            hi = middle
        else:
            lo = middle + 1
    return lo


def triangle_target(n, edges, rho_triangle):
    """Return one private integer target for a public n; rho accepts strings/Fractions."""
    if n < 0:
        raise ValueError('public node count must be nonnegative')
    rho = Fraction(rho_triangle)
    if rho <= 0:
        raise ValueError('rho_triangle must be positive')
    if n < 3:
        return 0
    return _select(_Ladder(n, edges), comb(n, 3), rho, SystemRandom())


def _read_edges(path):
    with open(path, encoding='utf-8') as source:
        for line in source:
            fields = line.split('#', 1)[0].split()
            if fields:
                u, v = map(int, fields)
                yield u, v


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, required=True, help='public vertex count including isolates')
    parser.add_argument('--edges', required=True, help='trusted whitespace-separated edge list')
    parser.add_argument('--rho-triangle', required=True, help='positive decimal or rational budget')
    args = parser.parse_args()
    print(triangle_target(args.n, _read_edges(args.edges), args.rho_triangle))
