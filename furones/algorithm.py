# Furones: Approximate Dominating Set Solver
# Author: Frank Vega

from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable, List, Optional, Sequence, Set

import networkx as nx
from networkx.algorithms import approximation

from . import tscc_ds_reduction


class ApproximationNotCertifiedError(RuntimeError):
    """Raised when the optional consistency certificate is requested but fails."""


def _normalize_graph(graph: nx.Graph) -> nx.Graph:
    """Return a simple undirected copy with self-loops removed."""
    G = nx.Graph()
    G.add_nodes_from(graph.nodes())
    G.add_edges_from((u, v) for u, v in graph.edges() if u != v)
    return G


def _closed_neighborhood(G: nx.Graph, v: Any) -> Set[Any]:
    """Return N[v]."""
    return set(G.neighbors(v)) | {v}


def _is_valid_dominating_set(G: nx.Graph, D: Iterable[Any]) -> bool:
    """Check domination with NetworkX."""
    D = set(D)
    if G.number_of_nodes() == 0:
        return len(D) == 0
    return nx.is_dominating_set(G, D)


def _prune_dominating_set(G: nx.Graph, D: Iterable[Any]) -> Set[Any]:
    """Remove redundant vertices while preserving domination.

    This is a linear scan repeated over the candidate set.  It is used only on
    candidates already known or expected to dominate.  The function favors
    deterministic behavior by sorting through string representations when
    possible.
    """
    D = set(D)
    if not D:
        return D

    ordered = sorted(D, key=lambda x: str(x))
    for v in ordered:
        if v not in D:
            continue
        trial = D - {v}
        if _is_valid_dominating_set(G, trial):
            D = trial
    return D


def _coverage_sweep_candidate(G: nx.Graph) -> Set[Any]:
    """Linear closed-degree coverage sweep candidate.

    Vertices are bucketed by closed degree |N[v]| and scanned from high to low.
    A vertex is selected only if it dominates at least one still-undominated
    vertex.  The candidate is then pruned.
    """
    n = G.number_of_nodes()
    if n == 0:
        return set()

    buckets: List[List[Any]] = [[] for _ in range(n + 1)]
    for v in G.nodes():
        deg = G.degree(v) + 1
        buckets[deg].append(v)

    undominated = set(G.nodes())
    D: Set[Any] = set()

    for deg in range(n, 0, -1):
        for v in buckets[deg]:
            if not undominated:
                break
            closed = _closed_neighborhood(G, v)
            if closed & undominated:
                D.add(v)
                undominated.difference_update(closed)
        if not undominated:
            break

    return _prune_dominating_set(G, D)


def _low_degree_witness_sweep_candidate(G: nx.Graph) -> Set[Any]:
    """Linear private-witness-oriented sweep candidate.

    The score of a vertex is the number of low-degree vertices, degree at most
    two, in its closed neighborhood.  This is a general heuristic for graphs
    containing private witnesses or pendant-like structures.  It is not a
    detector for any particular adversarial construction.
    """
    n = G.number_of_nodes()
    if n == 0:
        return set()

    low = {v for v in G.nodes() if G.degree(v) <= 2}
    if not low:
        return set()

    score = {v: 0 for v in G.nodes()}
    for w in low:
        score[w] += 1
        for v in G.neighbors(w):
            score[v] += 1

    max_score = max(score.values(), default=0)
    if max_score <= 0:
        return set()

    buckets: List[List[Any]] = [[] for _ in range(max_score + 1)]
    for v, s in score.items():
        buckets[s].append(v)

    undominated = set(G.nodes())
    D: Set[Any] = set()

    for s in range(max_score, -1, -1):
        for v in buckets[s]:
            if not undominated:
                break
            closed = _closed_neighborhood(G, v)
            if closed & undominated:
                D.add(v)
                undominated.difference_update(closed)
        if not undominated:
            break

    return _prune_dominating_set(G, D)


def _reverse_delete_candidate(G: nx.Graph, order: Sequence[Any]) -> Set[Any]:
    """Reverse-delete dominating-set candidate.

    Start from all vertices and delete vertices in the supplied order whenever
    the remaining set is still dominating.  The operation is deterministic for
    a fixed order and is used with several linear orders.
    """
    D = set(G.nodes())
    if not D:
        return D

    for v in order:
        if v not in D:
            continue
        trial = D - {v}
        if _is_valid_dominating_set(G, trial):
            D = trial

    return D


def _reverse_delete_candidates(G: nx.Graph) -> List[Set[Any]]:
    """Build several reverse-delete candidates from linear deterministic orders."""
    nodes = list(G.nodes())
    if not nodes:
        return [set()]

    by_high_degree = sorted(nodes, key=lambda v: (-G.degree(v), str(v)))
    by_low_degree = sorted(nodes, key=lambda v: (G.degree(v), str(v)))

    orders = [
        nodes,
        list(reversed(nodes)),
        by_high_degree,
        by_low_degree,
    ]

    return [_reverse_delete_candidate(G, order) for order in orders]


def _best_valid_candidate(G: nx.Graph, candidates: Iterable[Iterable[Any]]) -> Set[Any]:
    """Return the smallest valid candidate from a collection."""
    best: Optional[Set[Any]] = None

    for candidate in candidates:
        D = set(candidate)
        if not _is_valid_dominating_set(G, D):
            continue
        D = _prune_dominating_set(G, D)
        if best is None or len(D) < len(best):
            best = D

    if best is None:
        return set(G.nodes())

    return best


def _linear_candidates(G: nx.Graph) -> List[Set[Any]]:
    """Return general linear-time heuristic candidates."""
    candidates: List[Set[Any]] = []

    candidates.append(_coverage_sweep_candidate(G))
    candidates.append(_low_degree_witness_sweep_candidate(G))
    candidates.extend(_reverse_delete_candidates(G))

    return candidates


def find_dominating_set(graph: nx.Graph, eps: float = 0.5, consistency: bool = False) -> Set[Any]:
    """Return a valid dominating set for an undirected graph.

    Furones v0.3.2 combines several deterministic linear candidates with the
    TSCC/Baker/lift path.  The linear candidates include closed-degree coverage,
    low-degree-witness coverage, and reverse-delete scans.  These are general
    heuristics and not special-case detectors.

    The unconditional guarantee is validity of the returned dominating set
    after direct validation.  No universal approximation ratio is claimed for
    arbitrary graphs without an additional certificate/theorem.
    """
    G = _normalize_graph(graph)

    if G.number_of_nodes() == 0:
        return set()

    isolates = {v for v in G.nodes() if G.degree(v) == 0}
    nonisolated_nodes = [v for v in G.nodes() if v not in isolates]
    H = G.subgraph(nonisolated_nodes).copy()

    if H.number_of_nodes() == 0:
        return set(isolates)

    linear = _linear_candidates(H)
    best_linear = _best_valid_candidate(H, linear)

    # Early exit for very small valid certificates.  This is a general rule:
    # once a size-1 or size-2 dominating set is found by a linear candidate,
    # the result is already extremely strong for the targeted dense failures.
    if len(best_linear) <= 2:
        result = set(isolates) | best_linear
        if _is_valid_dominating_set(G, result):
            return result

    lifted_candidate: Set[Any] = set()
    try:
        reduced, forced, lift_map = tscc_ds_reduction.reduce_to_tscc(H)
        reduced_solution = tscc_ds_reduction.solve_reduced_instance(reduced, eps=eps)
        lifted_candidate = tscc_ds_reduction.lift_solution(H, reduced_solution, forced, lift_map)
    except Exception:
        lifted_candidate = set()

    candidates = list(linear)
    if lifted_candidate:
        candidates.append(lifted_candidate)

    best = _best_valid_candidate(H, candidates)
    result = set(isolates) | best

    if not _is_valid_dominating_set(G, result):
        # Last-resort safe fallback.  It should rarely be needed but preserves
        # the public API guarantee of returning a dominating set.
        result = set(isolates) | set(H.nodes())

    result = _prune_dominating_set(G, result)

    if consistency and not _linear_consistency_certificate(G, result):
        raise ApproximationNotCertifiedError(
            "The optional sufficient consistency certificate did not certify the requested approximation bound.",
            result,
        )

    return result


def _linear_consistency_certificate(G: nx.Graph, D: Iterable[Any]) -> bool:
    """A conservative sufficient certificate used by the CLI flag.

    This certificate is intentionally conservative.  It verifies only that the
    set is valid and that its size is at most twice a simple maximal-packing
    lower bound obtained from disjoint closed neighborhoods.
    """
    D = set(D)
    if not _is_valid_dominating_set(G, D):
        return False

    remaining = set(G.nodes())
    packing = 0

    for v in sorted(G.nodes(), key=lambda x: str(x)):
        closed = _closed_neighborhood(G, v)
        if v in remaining and closed <= remaining:
            packing += 1
            remaining.difference_update(closed)

    if packing == 0:
        return len(D) == 0

    return len(D) <= 2 * packing


def find_dominating_set_approximation(graph: nx.Graph) -> Set[Any]:
    """Return NetworkX's greedy logarithmic-factor approximation."""
    G = _normalize_graph(graph)
    if G.number_of_nodes() == 0:
        return set()
    return set(approximation.min_weighted_dominating_set(G))


def find_dominating_set_brute_force(graph: nx.Graph) -> Set[Any]:
    """Return an exact minimum dominating set by exhaustive search.

    This is exponential and intended only for very small graphs or tests.
    """
    G = _normalize_graph(graph)
    nodes = list(G.nodes())

    if not nodes:
        return set()

    for r in range(1, len(nodes) + 1):
        for subset in combinations(nodes, r):
            D = set(subset)
            if _is_valid_dominating_set(G, D):
                return D

    return set(nodes)