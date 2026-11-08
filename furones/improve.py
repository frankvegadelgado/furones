# Improvement phase of Furones' dominating sets.
# Author: Frank Vega

"""
Improvement of the candidates of the Furones ensemble by the strategies that are not base strategies:

* *union re-prune*: the vertices chosen by at least half of the candidates, completed greedily into a dominating set
  and pruned;
* *budgeted weighted swap search* (:mod:`furones.weighted_search`) from the smallest candidate;
* *consensus forcing*: the vertices chosen by at least two thirds of the candidates (the candidates, the union re-prune
  and the swap search) are forced, the rest is re-solved by the greedy completion, the pruning and the swap search;
* *final refinement*: the swap search, with another seed, from the smallest set found.

Every step works on the index graph ``adj`` (``adj[v]`` the neighbours of ``v``) and returns a dominating set; the
result is the smallest set found, never larger than the smallest candidate.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, List, Set, Tuple

import networkx as nx

from .weighted_search import weighted_swap_search

#: Work budget of each swap search, in units of (n + m), and the constant floor of the work.
IMPROVE_BUDGET: float = 100.0
IMPROVE_FLOOR: int = 300_000
#: Seed of the swap searches.
SEED: int = 20261031


def _dominates(adj: List[List[int]], members: Iterable[int]) -> bool:
    covered = [False] * len(adj)
    for v in members:
        covered[v] = True
        for y in adj[v]:
            covered[y] = True
    return all(covered)


def _complete(adj: List[List[int]], seed: Iterable[int]) -> Set[int]:
    """The set ``seed`` completed into a dominating set: while some vertex is undominated, the vertex whose closed
    neighbourhood holds most undominated vertices is added. Gains are maintained incrementally and kept in buckets
    (lazy entries), so the whole completion costs ``O(n + m)``."""
    n = len(adj)
    chosen = set(seed)
    dominated = [False] * n
    gain = [len(adj[v]) + 1 for v in range(n)]
    remaining = n

    def dominate(y: int) -> None:
        nonlocal remaining
        dominated[y] = True
        remaining -= 1
        gain[y] -= 1
        for z in adj[y]:
            gain[z] -= 1

    for v in chosen:
        for y in (v, *adj[v]):
            if not dominated[y]:
                dominate(y)
    if remaining == 0:
        return chosen
    top = max(gain) if n else 0
    buckets: List[List[int]] = [[] for _ in range(top + 1)]
    for v in range(n):
        if gain[v] > 0:
            buckets[gain[v]].append(v)
    level = top
    while remaining and level > 0:
        if not buckets[level]:
            level -= 1
            continue
        v = buckets[level].pop()
        if gain[v] != level:
            if gain[v] > 0:
                buckets[gain[v]].append(v)
            continue
        chosen.add(v)
        for y in (v, *adj[v]):
            if not dominated[y]:
                dominate(y)
    return chosen


def _prune(adj: List[List[int]], members: Iterable[int]) -> Set[int]:
    """Removes redundant vertices (smallest degree first, by counting sort) while the set stays dominating. ``O(n + m)``."""
    chosen = set(members)
    count = [0] * len(adj)
    for v in chosen:
        count[v] += 1
        for y in adj[v]:
            count[y] += 1
    top = max((len(adj[v]) for v in chosen), default=0)
    buckets: List[List[int]] = [[] for _ in range(top + 1)]
    for v in chosen:
        buckets[len(adj[v])].append(v)
    for bucket in buckets:
        for v in bucket:
            if count[v] >= 2 and all(count[y] >= 2 for y in adj[v]):
                chosen.discard(v)
                count[v] -= 1
                for y in adj[v]:
                    count[y] -= 1
    return chosen


def _search(adj: List[List[int]], members: Set[int], seed: int) -> Set[int]:
    flags = [False] * len(adj)
    for v in members:
        flags[v] = True
    result = weighted_swap_search(adj, flags, IMPROVE_BUDGET, seed, IMPROVE_FLOOR)
    found = {v for v, flag in enumerate(result) if flag}
    return found if _dominates(adj, found) and len(found) <= len(members) else set(members)


def union_reprune(adj: List[List[int]], candidates: List[Set[int]]) -> Set[int]:
    """Vertices chosen by at least half of the candidates, completed and pruned."""
    votes = Counter(v for c in candidates for v in c)
    threshold = (len(candidates) + 1) // 2
    return _prune(adj, _complete(adj, [v for v, k in votes.items() if k >= threshold]))


def consensus_forcing(adj: List[List[int]], candidates: List[Set[int]], seed: int) -> Set[int]:
    """Vertices chosen by at least two thirds of the candidates are forced; the rest is completed greedily, pruned and
    improved by the swap search."""
    votes = Counter(v for c in candidates for v in c)
    threshold = -(-2 * len(candidates) // 3)
    completed = _prune(adj, _complete(adj, [v for v, k in votes.items() if k >= threshold]))
    return _search(adj, completed, seed)


def improvement_steps(graph: nx.Graph, candidates: List[Set[Any]]) -> List[Tuple[str, Set[Any]]]:
    """The dominating sets produced by the improvement strategies, in the order they run, as ``(name, set)`` pairs:
    ``union_reprune``, ``swap_search``, ``consensus_forcing`` and ``final_refinement``. Candidates that do not dominate
    ``graph`` are ignored; without a valid candidate the result is empty. Each step receives the sets found before it."""
    nodes = list(graph.nodes())
    index = {v: i for i, v in enumerate(nodes)}
    adj = [[index[u] for u in graph.adj[v] if u != v] for v in nodes]
    valid = []
    for c in candidates:
        members = {index[v] for v in c if v in index}
        if _dominates(adj, members):
            valid.append(members)
    if not valid:
        return []
    best = min(valid, key=len)
    pool = list(valid)
    steps = []

    def record(name: str, found: Set[int]) -> None:
        pool.append(found)
        steps.append((name, {nodes[i] for i in found}))

    record("union_reprune", union_reprune(adj, valid))
    record("swap_search", _search(adj, best, SEED))
    record("consensus_forcing", consensus_forcing(adj, pool, SEED + 1))
    record("final_refinement", _search(adj, min(pool, key=len), SEED + 2))
    return steps


def improve_dominating_set(graph: nx.Graph, candidates: List[Set[Any]]) -> Set[Any]:
    """The smallest dominating set found from the candidate dominating sets of ``graph`` (see the module docstring).
    Candidates that do not dominate ``graph`` are ignored; without a valid candidate the result is empty."""
    steps = improvement_steps(graph, candidates)
    if not steps:
        return set()
    nodes = set(graph.nodes())
    valid = [set(c) for c in candidates if nx.is_dominating_set(graph, set(c) & nodes)]
    return set(min(valid + [s for _, s in steps], key=len))
