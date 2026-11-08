# Weighted swap search for the minimum dominating set (Furones).
# Author: Frank Vega

"""
Weighted swap search for the minimum dominating set.

The search keeps a set ``S`` of a fixed size ``k`` that may leave vertices undominated and minimises the weight of the
undominated vertices:

    cost(S) = sum of w[v] over the undominated vertices v.

``S`` is a dominating set exactly when the cost is zero; it is then recorded, and one vertex is removed to try ``k - 1``.
This is the scheme of the strongest local searches for vertex cover and dominating set (NuMVC, Cai, Su, Luo and Sattar,
JAIR 2013; the configuration-checking dominating-set searches).

Quantities (maintained incrementally):

* ``w[v] >= 1`` -- weights; after every step the weight of every undominated vertex grows by one, so violations that
  persist become more and more important (escape from local minima);
* ``dom[v] = |N[v] & S|``; ``v`` is undominated when ``dom[v] = 0``;
* ``score[z]`` -- the decrease of ``cost`` when ``z`` changes side: for ``z`` outside ``S``, the weight of the
  undominated vertices of ``N[z]``; for ``z`` in ``S``, minus the weight of the vertices of ``N[z]`` dominated by ``z``
  alone.

A step: (1) remove the vertex of ``S`` of largest ``score`` among ``SAMPLE`` sampled vertices of ``S`` (best from multiple
selection), never the vertex added in the previous step; (2) if some vertex is undominated, pick one at random and add
the vertex of largest ``score`` of its closed neighbourhood allowed by configuration checking (a removed vertex may come
back only after one of its neighbours changed side), ties by age; (3) add one to the weight of every undominated vertex.

Every update costs the degrees of the two-hop neighbourhood of the changed vertex; all of it is charged to a work counter
and the search stops at ``max(budget * (n + m), floor)`` units, so it runs in ``O(n + m)`` time for a fixed budget.
Weights are halved (and the scores recomputed, ``O(n + m)``) when their mean exceeds a threshold. The search is split
into :data:`SEGMENTS` segments that cycle through :data:`WEIGHT_LIMITS`, each restarted with fresh weights from the best
set found so far. The search never returns a set larger than its input, which must be a dominating set.
"""

from __future__ import annotations

import random

#: Vertices of S sampled for the removal (best from multiple selection).
SAMPLE: int = 50
#: Mean weights above which the weights are halved; the segments of a search cycle through them.
WEIGHT_LIMITS: tuple[float, ...] = (10.0, 100.0, 30.0, 300.0)
#: Segments of one search (each with its own weight cap, restarted from the best set).
SEGMENTS: int = 4


class _IndexedSet:
    """Set of integers with O(1) insert, delete and random choice."""

    __slots__ = ("items", "pos")

    def __init__(self, n: int):
        self.items: list[int] = []
        self.pos = [-1] * n

    def add(self, v: int) -> None:
        if self.pos[v] == -1:
            self.pos[v] = len(self.items)
            self.items.append(v)

    def discard(self, v: int) -> None:
        p = self.pos[v]
        if p != -1:
            last = self.items.pop()
            if last != v:
                self.items[p] = last
                self.pos[last] = p
            self.pos[v] = -1

    def __len__(self) -> int:
        return len(self.items)


def _search(adj: list[list[int]], in_s: list[bool], budget: float, seed: int,
            floor: int, weight_limit: float, target: int = 0) -> list[bool]:
    """Improve the dominating set ``in_s`` by the weighted swap search. Returns a dominating set no larger than
    ``in_s``."""
    n = len(adj)
    m = sum(len(a) for a in adj) // 2
    best = list(in_s)
    if m == 0:
        return best
    rng = random.Random(seed)
    limit = max(budget * (n + m), floor)
    work = 0

    S = list(in_s)
    wv = [1] * n
    dom = [0] * n
    score = [0] * n
    conf = [True] * n
    age = [0] * n
    sset = _IndexedSet(n)
    undom = _IndexedSet(n)
    isolated = [not a for a in adj]

    for v in range(n):
        if S[v]:
            sset.add(v)
            dom[v] += 1
            for y in adj[v]:
                dom[y] += 1

    def recompute() -> None:
        for v in range(n):
            score[v] = 0
        for v in range(n):
            if dom[v] == 0:
                score[v] += wv[v]
                for z in adj[v]:
                    score[z] += wv[v]
            elif dom[v] == 1:
                if S[v]:
                    score[v] -= wv[v]
                else:
                    for s in adj[v]:
                        if S[s]:
                            score[s] -= wv[v]
                            break

    for v in range(n):
        if dom[v] == 0:
            undom.add(v)
    recompute()
    work += 3 * (n + m)

    def own_score(z: int) -> int:
        """score[z] from its definition, O(deg z)."""
        sc = 0
        if S[z]:
            if dom[z] == 1:
                sc -= wv[z]
            for y in adj[z]:
                if dom[y] == 1 and not S[y]:
                    sc -= wv[y]
        else:
            if dom[z] == 0:
                sc += wv[z]
            for y in adj[z]:
                if dom[y] == 0:
                    sc += wv[y]
        return sc

    def flip(z: int) -> None:
        """Move ``z`` to the other side, updating dom, the undominated set and the scores."""
        nonlocal work
        entering = not S[z]
        S[z] = entering
        if entering:
            sset.add(z)
        else:
            sset.discard(z)
        az = adj[z]
        for y in az:
            conf[y] = True
        for y in (z, *az):
            d = dom[y]
            ay = adj[y]
            w = wv[y]
            work += len(ay) + 1
            if entering:
                dom[y] = d + 1
                if d == 0:
                    undom.discard(y)
                    for t in (y, *ay):
                        if t != z:
                            score[t] -= w
                elif d == 1:
                    for s in (y, *ay):
                        if s != z and S[s]:
                            score[s] += w
                            break
            else:
                dom[y] = d - 1
                if d == 1:
                    undom.add(y)
                    for t in (y, *ay):
                        if t != z:
                            score[t] += w
                elif d == 2:
                    for s in (y, *ay):
                        if S[s]:
                            score[s] -= w
                            break
        score[z] = own_score(z)
        work += 2 * len(az) + 2

    best_size = sum(best)
    step = 0
    last_added = -1
    while work <= limit and len(sset) > 0 and best_size > target:
        if len(undom) == 0:
            if len(sset) < best_size:
                best_size = len(sset)
                best = list(S)
                work += n
            # Try one vertex fewer: remove the vertex whose removal costs least.
            x = max((v for v in sset.items if not isolated[v]), key=lambda v: score[v], default=-1)
            work += len(sset)
            if x == -1:
                break
            flip(x)
            conf[x] = False
            continue
        step += 1
        items = sset.items
        cand = items if len(items) <= SAMPLE else [items[rng.randrange(len(items))] for _ in range(SAMPLE)]
        x = -1
        for c in cand:
            if c == last_added or isolated[c]:
                continue
            if x == -1 or score[c] > score[x] or (score[c] == score[x] and age[c] < age[x]):
                x = c
        work += len(cand)
        if x != -1:
            flip(x)
            conf[x] = False
            age[x] = step
        if len(undom):
            v = undom.items[rng.randrange(len(undom))]
            u = -1
            fallback = -1
            for z in [v, *adj[v]]:
                if fallback == -1 or score[z] > score[fallback] or (score[z] == score[fallback] and age[z] < age[fallback]):
                    fallback = z
                if conf[z] and (u == -1 or score[z] > score[u] or (score[z] == score[u] and age[z] < age[u])):
                    u = z
            work += len(adj[v]) + 1
            if u == -1:
                u = fallback
            flip(u)
            age[u] = step
            last_added = u
        # Weights of the undominated vertices.
        for y in undom.items:
            wv[y] += 1
            score[y] += 1
            for z in adj[y]:
                score[z] += 1
            work += len(adj[y]) + 1
        if step % 64 == 0:
            work += n
        if step % 64 == 0 and sum(wv) > weight_limit * n:
            for y in range(n):
                wv[y] = max(1, wv[y] // 2)
            recompute()
            work += 3 * (n + m)
    return best


def weighted_swap_search(adj: list[list[int]], in_s: list[bool], budget: float, seed: int,
                         floor: int, target: int = 0) -> list[bool]:
    """Improve the dominating set ``in_s`` by the weighted swap search, in segments: the work
    ``max(budget * (n + m), floor)`` is split into :data:`SEGMENTS` equal parts, segment ``i`` uses the weight cap
    ``WEIGHT_LIMITS[(seed + i) % len(WEIGHT_LIMITS)]`` and starts, with fresh weights, from the best set found so far.
    Returns a dominating set no larger than ``in_s``. The search stops as soon as a set of at most ``target`` vertices
    is found."""
    n = len(adj)
    m = sum(len(a) for a in adj) // 2
    total = max(budget * (n + m), floor)
    part = total / SEGMENTS
    best = list(in_s)
    for i in range(SEGMENTS):
        if sum(best) <= target:
            break
        cap = WEIGHT_LIMITS[(seed + i) % len(WEIGHT_LIMITS)]
        res = _search(adj, best, 0.0, seed + 7919 * i, int(part), cap, target)
        if sum(res) < sum(best):
            best = res
    return best
