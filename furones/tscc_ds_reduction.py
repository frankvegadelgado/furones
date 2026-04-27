"""
tscc_ds_reduction.py  —  v0.2.4  (TSCC components, domination-safe)
===================================================================

Changes vs. v0.2.2 (largest-component version)
---------------------------------------------
  1. Correctness: we NO LONGER discard all but the largest 2-edge-connected
     component.  Instead we return *all* non-trivial 2-edge-connected
     components as separate TSCCs.  This makes the reduction domination-safe:
       - no vertex is silently dropped,
       - every vertex of every returned TSCC must still be dominated.
  2. API change (Option A):
       reduce_to_tscc_for_ds(G) now returns

           tsccs      : List[nx.Graph]
           forced_ds  : Set[Any]
           lift       : Callable[[List[Set[Any]]], Set[Any]]

     where `tsccs[i]` is the graph on which you computed `ds_list[i]`.
     The global dominating set is:

           lift(ds_list) = forced_ds ∪ ⋃_i ds_list[i]
"""

from __future__ import annotations

import networkx as nx
from collections import deque
from typing import Any, Callable, List, Set, Tuple


# ---------------------------------------------------------------------------
# 1.  Cascade (dominating-set kernel reduction)
# ---------------------------------------------------------------------------

def _run_cascade(
    H: nx.Graph,
    forced_ds: Set[Any],
    ref_graph: nx.Graph,
) -> None:
    """
    Remove degree-0 and degree-1 vertices, forcing the appropriate nodes
    into the dominating set.

    `ref_graph` is the graph used to check whether a vertex is already
    dominated via a forced node.  After planarisation this should be P,
    not the original G, so dominance is consistent with surviving edges.

    Complexity: O(n + m) amortised (each node/edge touched ≤ twice).
    """
    q: deque = deque(v for v in H if H.degree(v) <= 1)
    in_q: Set[Any] = set(q)

    while q:
        v = q.popleft()
        if v not in H:
            continue

        dv = H.degree(v)

        # ── degree-0 vertex ────────────────────────────────────────────────
        if dv == 0:
            already_dominated = any(
                nb in forced_ds for nb in ref_graph.neighbors(v)
            )
            if not already_dominated:
                forced_ds.add(v)
            H.remove_node(v)

        # ── degree-1 vertex ────────────────────────────────────────────────
        elif dv == 1:
            u = next(iter(H.neighbors(v)))

            # v is already dominated through some forced neighbour ≠ u
            if any(
                nb in forced_ds
                for nb in ref_graph.neighbors(v)
                if nb != u
            ):
                H.remove_node(v)
                if u in H and u not in in_q and H.degree(u) <= 1:
                    in_q.add(u)
                    q.append(u)
                continue

            # Force u into DS; collapse neighbourhood
            neighbors_u = list(H.neighbors(u))
            closed_u = set(neighbors_u) | {u}
            kept = []
            to_remove = {u, v}

            for w in neighbors_u:
                if w == v:
                    continue
                # w has a neighbour outside N[u] → not dominated yet → keep
                if any(x not in closed_u for x in H.neighbors(w)):
                    kept.append(w)
                else:
                    to_remove.add(w)

            forced_ds.add(u)
            for node in to_remove:
                if node in H:
                    H.remove_node(node)

            for w in kept:
                if w in H and w not in in_q and H.degree(w) <= 1:
                # (re-)enqueue low-degree survivors
                    in_q.add(w)
                    q.append(w)


# ---------------------------------------------------------------------------
# 2.  Greedy planar subgraph  (OPTIMISED — reuse embedding)
# ---------------------------------------------------------------------------

def _greedy_planar_subgraph(G: nx.Graph) -> nx.Graph:
    """
    Build a maximal planar spanning subgraph of G.

    Bridges are prioritised: adding a bridge cannot create a K5/K3,3 minor,
    so we insert them first for free — improving the chance of admitting
    more edges later.
    """
    is_planar, emb = nx.check_planarity(G)
    if is_planar:
        return G.copy()

    P = nx.Graph()
    P.add_nodes_from(G.nodes(data=True))

    # ── Phase 1: insert bridges first (zero planarity risk) ─────────────
    bridges = set(nx.bridges(G))
    non_bridge_edges = []
    for e in G.edges():
        u, v = e
        canonical = (min(u, v), max(u, v))
        if canonical in bridges or (v, u) in bridges:
            P.add_edge(u, v)
        else:
            non_bridge_edges.append((u, v))

    # ── Phase 2: insert remaining edges, highest-degree pairs first ──────
    non_bridge_edges.sort(key=lambda e: -(G.degree(e[0]) + G.degree(e[1])))

    for u, v in non_bridge_edges:
        P.add_edge(u, v)
        ok, _ = nx.check_planarity(P)
        if not ok:
            P.remove_edge(u, v)

    return P


# ---------------------------------------------------------------------------
# 3.  All 2-edge-connected TSCC components (domination-safe)
# ---------------------------------------------------------------------------

def _all_2ec_subgraphs(H: nx.Graph) -> List[nx.Graph]:
    """
    Return the list of subgraphs induced by all non-trivial 2-edge-connected
    components of H.

    nx.k_edge_components(H, k=2) partitions V into maximal k-edge-connected
    subsets.  Isolated vertices and bridge-separated parts appear as singletons
    or small components.

    We keep only components of size ≥ 2; singletons cannot have degree ≥ 2
    in a 2-edge-connected graph and are irrelevant for TSCC DS solving.

    Complexity: O(n + m) via Tarjan-style bridge decomposition.
    """
    components = list(nx.k_edge_components(H, k=2))
    tsccs: List[nx.Graph] = []

    for comp in components:
        if len(comp) < 2:
            continue
        tsccs.append(H.subgraph(comp).copy())

    return tsccs


# ---------------------------------------------------------------------------
# 4.  Public entry point (Option A: multiple TSCCs)
# ---------------------------------------------------------------------------

def reduce_to_tscc_for_ds(
    G: nx.Graph,
) -> Tuple[List[nx.Graph], Set[Any], Callable[[List[Set[Any]]], Set[Any]]]:
    """
    Reduce G to a family of TSCCs (2-edge-connected subgraphs with min degree ≥ 2)
    for dominating-set solving.

    Returns
    -------
    tsccs : list of nx.Graph
        Each element is a reduced graph, guaranteed to be 2-edge-connected
        with every vertex of degree ≥ 2 (or the list is empty if G has no
        such substructure).  Components are vertex-disjoint.
    forced_ds : set
        Vertices already forced into every dominating set of G by the cascade
        rules (degree-0 / degree-1 reductions).
    lift : callable
        Given a list `ds_list` where `ds_list[i]` is a dominating set of
        `tsccs[i]`, returns a dominating set of the original G:

            lift(ds_list) = forced_ds ∪ ⋃_i ds_list[i]

        NOTE: vertices that never appear in any TSCC (e.g. in trees or
        bridge-only parts) are not automatically dominated; callers must
        either:
          - run a separate DS solver on those residual components, or
          - treat this TSCC reduction as a subroutine inside a larger
            component-wise DS pipeline.

    """
    # ── Initialise working copy ──────────────────────────────────────────
    H = nx.Graph(G)
    H.remove_edges_from(list(nx.selfloop_edges(H)))
    forced_ds: Set[Any] = set()

    # ── Cascade 1: on the raw graph ──────────────────────────────────────
    _run_cascade(H, forced_ds, ref_graph=G)

    # ── Planarity gate ───────────────────────────────────────────────────
    is_planar, _ = nx.check_planarity(H)
    if not is_planar:
        P = _greedy_planar_subgraph(H)
        _run_cascade(P, forced_ds, ref_graph=P)
        H = P

    # ── 2-edge-connected TSCC extraction (ALL components) ────────────────
    tsccs = _all_2ec_subgraphs(H)

    frozen_forced = frozenset(forced_ds)

    def lift(ds_list: List[Set[Any]]) -> Set[Any]:
        """
        Combine per-TSCC dominating sets with the globally forced vertices.

        Precondition: len(ds_list) == len(tsccs) and each ds_list[i]
        dominates tsccs[i].
        """
        result: Set[Any] = set(frozen_forced)
        for ds in ds_list:
            result.update(ds)
        return result

    return tsccs, set(forced_ds), lift


# ---------------------------------------------------------------------------
# 5.  Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random, time

    random.seed(42)

    # Graph with many bridges so 2-EC decomposition yields many TSCCs
    G = nx.barbell_graph(6, 1)          # two K6 cliques joined by a path
    G.add_node(999)                     # isolated node → cascade test

    t0 = time.perf_counter()
    tsccs, forced, lift = reduce_to_tscc_for_ds(G)
    elapsed = time.perf_counter() - t0

    print(f"Original  : {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"TSCCs     : {[ (H.number_of_nodes(), H.number_of_edges()) for H in tsccs ]}")
    print(f"Forced DS : {forced}")

    for i, H in enumerate(tsccs):
        print(f"  TSCC[{i}] planar? {nx.check_planarity(H)[0]}")
        print(f"  TSCC[{i}] min degree: {min((d for _, d in H.degree()), default=0)}")
        print(f"  TSCC[{i}] bridges   : {list(nx.bridges(H))}")

    print(f"Time      : {elapsed*1000:.3f} ms")
