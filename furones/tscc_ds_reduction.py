# Furones TSCC-style reduction utilities
# Author: Frank Vega

from __future__ import annotations

from typing import Any, Dict, Iterable, Set, Tuple

import networkx as nx


def _normalize_graph(graph: nx.Graph) -> nx.Graph:
    """Return a simple undirected graph with self-loops removed."""
    G = nx.Graph()
    G.add_nodes_from(graph.nodes())
    G.add_edges_from((u, v) for u, v in graph.edges() if u != v)
    return G


def _closed_neighborhood(G: nx.Graph, v: Any) -> Set[Any]:
    """Return the closed neighborhood N[v]."""
    return set(G.neighbors(v)) | {v}


def _prune_dominating_set(G: nx.Graph, D: Iterable[Any]) -> Set[Any]:
    """Remove redundant vertices while preserving domination."""
    D = set(D)
    for v in sorted(list(D), key=lambda x: str(x)):
        if v not in D:
            continue
        trial = D - {v}
        if nx.is_dominating_set(G, trial):
            D = trial
    return D


def _pendant_cascade(G: nx.Graph) -> Tuple[nx.Graph, Set[Any], Dict[Any, Set[Any]]]:
    """Apply a simple pendant/isolated cascade.

    If a vertex x has degree zero, it is forced into the solution.
    If a vertex x has degree one with neighbor y, then y is forced because y
    dominates x and is at least as useful locally as x for the remaining graph.

    The function returns the reduced graph, the forced vertices, and a lift map.
    The lift map records vertices removed because they are already dominated by
    forced choices.
    """
    H = G.copy()
    forced: Set[Any] = set()
    lift_map: Dict[Any, Set[Any]] = {}

    changed = True
    while changed:
        changed = False

        isolates = [v for v in list(H.nodes()) if H.degree(v) == 0]
        if isolates:
            for v in isolates:
                forced.add(v)
                lift_map.setdefault(v, set()).add(v)
            H.remove_nodes_from(isolates)
            changed = True
            continue

        pendants = [v for v in list(H.nodes()) if H.degree(v) == 1]
        if pendants:
            remove: Set[Any] = set()
            for x in pendants:
                if x not in H:
                    continue
                nbrs = list(H.neighbors(x))
                if not nbrs:
                    continue
                y = nbrs[0]
                forced.add(y)
                dominated = _closed_neighborhood(H, y)
                lift_map.setdefault(y, set()).update(dominated)
                remove.update(dominated)
            H.remove_nodes_from(remove)
            changed = True

    return H, forced, lift_map


def _forest_projection(G: nx.Graph) -> nx.Graph:
    """Return G if planar; otherwise return a DFS spanning forest.

    The projection is used as a fast safe reduced instance.  It can discard
    domination-relevant dense edges, so the paper and code do not claim that
    approximation ratios transfer automatically from this projection to the
    original graph.
    """
    if G.number_of_nodes() == 0:
        return G.copy()

    try:
        planar, _ = nx.check_planarity(G)
    except Exception:
        planar = False

    if planar:
        return G.copy()

    F = nx.Graph()
    F.add_nodes_from(G.nodes())

    for component in nx.connected_components(G):
        root = next(iter(component))
        tree_edges = list(nx.dfs_edges(G.subgraph(component), source=root))
        F.add_edges_from(tree_edges)

    return F


def reduce_to_tscc(graph: nx.Graph) -> Tuple[nx.Graph, Set[Any], Dict[Any, Set[Any]]]:
    """Reduce a graph to a simplified core instance.

    The current v0.3.2 reduction is intentionally conservative.  It removes
    self-loops, applies pendant/isolated cascades, projects non-planar residuals
    to a DFS spanning forest when needed, and cascades once more.

    Returns:
        reduced graph, forced vertices, lift map.

    The reduction is validity-preserving for the lifted dominating set.  A
    universal approximation transfer from the reduced graph to the original
    graph is not claimed without an additional certificate/theorem.
    """
    G = _normalize_graph(graph)

    core1, forced1, lift1 = _pendant_cascade(G)
    projected = _forest_projection(core1)
    core2, forced2, lift2 = _pendant_cascade(projected)

    forced = set(forced1) | set(forced2)
    lift_map: Dict[Any, Set[Any]] = {}
    for key, values in lift1.items():
        lift_map.setdefault(key, set()).update(values)
    for key, values in lift2.items():
        lift_map.setdefault(key, set()).update(values)

    return core2, forced, lift_map


def solve_reduced_instance(reduced: nx.Graph, eps: float = 0.5) -> Set[Any]:
    """Solve the reduced instance by a lightweight Baker-style layer heuristic.

    For a planar graph, Baker's paradigm removes layers modulo k and solves
    bounded-treewidth pieces.  This implementation is a lightweight practical
    surrogate: it builds BFS layers, tries every layer residue as a deletion
    class, greedily dominates the remaining graph, lifts deleted residues when
    needed, and prunes.

    The function is deterministic and returns a valid dominating set for the
    reduced graph.  It is not presented as a full Baker PTAS implementation.
    """
    G = _normalize_graph(reduced)

    if G.number_of_nodes() == 0:
        return set()

    if G.number_of_nodes() <= 2:
        return {next(iter(G.nodes()))}

    k = max(2, int(round(1.0 / max(eps, 1e-9))) + 1)

    layers: Dict[Any, int] = {}
    for component in nx.connected_components(G):
        root = min(component, key=lambda x: str(x))
        lengths = nx.single_source_shortest_path_length(G.subgraph(component), root)
        for v, dist in lengths.items():
            layers[v] = dist

    candidates = []

    for residue in range(k):
        deleted = {v for v, dist in layers.items() if dist % k == residue}
        kept = [v for v in G.nodes() if v not in deleted]
        H = G.subgraph(kept).copy()

        D = _greedy_dominate(H)

        undominated = set(G.nodes())
        for v in D:
            undominated.difference_update(_closed_neighborhood(G, v))

        for v in sorted(deleted, key=lambda x: str(x)):
            if v in undominated:
                D.add(v)
                undominated.difference_update(_closed_neighborhood(G, v))

        candidates.append(_prune_dominating_set(G, D))

    candidates.append(_greedy_dominate(G))

    best = min(candidates, key=lambda D: (len(D), sorted(str(v) for v in D)))
    return _prune_dominating_set(G, best)


def _greedy_dominate(G: nx.Graph) -> Set[Any]:
    """Greedy dominating set used inside the reduced-instance heuristic."""
    if G.number_of_nodes() == 0:
        return set()

    undominated = set(G.nodes())
    D: Set[Any] = set()

    while undominated:
        best = None
        best_gain = -1
        for v in G.nodes():
            gain = len(_closed_neighborhood(G, v) & undominated)
            if gain > best_gain or (gain == best_gain and str(v) < str(best)):
                best = v
                best_gain = gain

        if best is None:
            break

        D.add(best)
        undominated.difference_update(_closed_neighborhood(G, best))

    return _prune_dominating_set(G, D)


def lift_solution(
    original: nx.Graph,
    reduced_solution: Iterable[Any],
    forced: Iterable[Any],
    lift_map: Dict[Any, Set[Any]] | None = None,
) -> Set[Any]:
    """Lift a reduced solution back to the original graph.

    The lifted solution is the union of forced choices and reduced choices,
    followed by direct validation and pruning.  If the union fails to dominate
    the original graph, remaining undominated vertices are greedily repaired.
    """
    G = _normalize_graph(original)
    D = set(forced) | set(reduced_solution)

    if not nx.is_dominating_set(G, D):
        undominated = set(G.nodes())
        for v in D:
            if v in G:
                undominated.difference_update(_closed_neighborhood(G, v))

        while undominated:
            best = None
            best_gain = -1
            for v in G.nodes():
                gain = len(_closed_neighborhood(G, v) & undominated)
                if gain > best_gain or (gain == best_gain and str(v) < str(best)):
                    best = v
                    best_gain = gain
            if best is None:
                break
            D.add(best)
            undominated.difference_update(_closed_neighborhood(G, best))

    return _prune_dominating_set(G, D)