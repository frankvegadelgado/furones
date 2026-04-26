# Created on 26/07/2025
# Author: Frank Vega

import itertools

import networkx as nx
from collections import defaultdict
from typing import FrozenSet, Dict, Set, Tuple

# ────────────────────────────────────────────────────────────────────────────
#  Public API
# ────────────────────────────────────────────────────────────────────────────

def weighted_dominating_set_max_deg4(
    G: nx.Graph,
    weight: str = "weight",
) -> Tuple[FrozenSet, float]:
    """
    Weighted minimum dominating set for graphs with maximum degree ≤ 4.

    A *dominating set* D satisfies: every vertex v ∉ D has ≥ 1 neighbour
    in D.  We minimise Σ_{v ∈ D} w[v].

    Three phases
    ────────────
    1. Primal-dual  [O(n + m)]
       LP dual: max Σ y[v]  s.t.  Σ_{u ∈ N[v]} y[u] ≤ w[v]  ∀ v, y ≥ 0
       slack[v] = w[v] − Σ_{u ∈ N[v]} y[u] tracks residual budget.
       For each undominated v, raise y[v] by ε = min_{u ∈ N[v]} slack[u].
       Every u in N[v] loses ε of slack; tight vertices (slack ≈ 0) join D
       and cover their closed neighbourhoods.
       → Feasible dominating set, ≤ (Δ+1)-approximation.

    2. Greedy swap  [O(n · Δ²) = O(n)]
       For each v ∈ D scan N(v) for a lighter u ∉ D that can replace v
       without stranding any neighbour (checked in O(Δ) = O(1)).
       → Reduces solution weight without losing feasibility.

    3. Redundancy prune  [O(n + m)]
       Remove every v ∈ D that has a D-neighbour (so v stays dominated)
       AND whose N(v) all remain dominated via their other D-neighbours.
       → Equivalent to the companion prune_redundant_vertices_dominating().

    Parameters
    ----------
    G      : nx.Graph — node attribute `weight` must be set via
                 nx.set_node_attributes(G, weights, 'weight')
    weight : str — node-weight attribute name (default 'weight')

    Returns
    -------
    D            : frozenset — nodes forming the weighted dominating set
    total_weight : float    — Σ w[v] for v ∈ D
    """
    if G.number_of_nodes() == 0:
        return frozenset(), 0.0

    w: Dict = {v: float(G.nodes[v].get(weight, 1.0)) for v in G}

    D, dom_count = _phase1_primal_dual(G, w)
    D, dom_count = _phase2_greedy_swap(G, w, D, dom_count)
    D            = _phase3_prune(G, D)

    return D

# ────────────────────────────────────────────────────────────────────────────
#  Phase 1 — Primal-Dual Construction
# ────────────────────────────────────────────────────────────────────────────

def _phase1_primal_dual(
    G: nx.Graph,
    w: Dict,
) -> Tuple[Set, Dict]:
    """
    O(n + m).  Builds a feasible dominating set via LP duality.

    Dual program for min-weight dominating set (set-cover formulation):
        max  Σ_v y[v]
        s.t. Σ_{u ∈ N[v]} y[u] ≤ w[v]   for all v   (v's "budget")
             y[v] ≥ 0

    slack[v] = w[v] − Σ_{u ∈ N[v]} y[u]  is the remaining budget of v.
    When we raise y[v] by ε = min_{u ∈ N[v]} slack[u]:
      • every u ∈ N[v] loses ε (since v ∈ N[u] for all u ∈ N[v])
      • the tightest u hits slack = 0 → it enters D and covers N[u]

    Correctness: every undominated v gets processed → some u ∈ N[v] enters D
    → v is covered (u = v ∈ D, or u ∈ N(v) and v ∈ N(u) → v covered).
    """
    slack:     Dict = {v: w[v] for v in G}
    dominated: Set  = set()
    D:         Set  = set()

    for v in G.nodes():
        if v in dominated:
            continue

        # Closed neighbourhood — at most Δ + 1 ≤ 5 vertices
        Nv  = [v] + list(G.neighbors(v))

        # Maximum legal raise: the tightest slack in N[v]
        eps = min(slack[u] for u in Nv)   # O(Δ) = O(1)

        # Raise y[v] by eps; every u ∈ N[v] loses eps of slack
        for u in Nv:
            slack[u] -= eps

        # Tight vertices become dominators and cover their neighbourhoods
        for u in Nv:
            if slack[u] < 1e-9 and u not in D:
                D.add(u)
                dominated.add(u)
                dominated.update(G.neighbors(u))  # O(Δ) = O(1)

    # Build dom_count for Phase 2: dom_count[u] = |N(u) ∩ D|
    dom_count: Dict = defaultdict(int)
    for v in D:
        for u in G.neighbors(v):
            dom_count[u] += 1

    return D, dom_count

# ────────────────────────────────────────────────────────────────────────────
#  Phase 2 — Greedy Weight-Reducing Swap
# ────────────────────────────────────────────────────────────────────────────

def _phase2_greedy_swap(
    G:         nx.Graph,
    w:         Dict,
    D:         Set,
    dom_count: Dict,
) -> Tuple[Set, Dict]:
    """
    O(n · Δ²) = O(n) for Δ ≤ 4.  Single-pass weight-reducing swap.

    For each v ∈ D, find the cheapest neighbour u ∉ D that can replace v
    while keeping every vertex dominated.

    Swap v → u is valid iff for every t ∈ N(v):
        t ∈ D           → self-dominated, fine regardless
      OR dom_count[t] ≥ 2  → t has another D-neighbour after v leaves
      OR t ∈ N(u)       → u (the incoming dominator) will cover t

    v itself is always dominated after the swap because u ∈ N(v) joins D.
    """
    dom_count = defaultdict(int, dom_count)   # local copy

    for v in list(D):        # snapshot: D changes in-place during loop
        if v not in D:
            continue          # was swapped out by a prior iteration

        best_u: object = None
        best_w: float  = w[v]   # only accept strictly lighter replacements

        for u in G.neighbors(v):
            if u in D or w[u] >= best_w:
                continue

            # O(Δ) validity check
            if all(
                (t in D) or (dom_count[t] >= 2) or (t in G[u])
                for t in G.neighbors(v)
            ):
                best_u, best_w = u, w[u]

        if best_u is not None:
            D.discard(v)
            D.add(best_u)
            for t in G.neighbors(v):
                dom_count[t] -= 1
            for t in G.neighbors(best_u):
                dom_count[t] += 1

    return D, dom_count

# ────────────────────────────────────────────────────────────────────────────
#  Phase 3 — Redundancy Pruning
# ────────────────────────────────────────────────────────────────────────────

def _phase3_prune(G: nx.Graph, D: Set) -> Set:
    """
    O(n + m).  Mirrors prune_redundant_vertices_dominating().

    v ∈ D is *redundant* (safely removable) iff:
      (a) dom_count[v] ≥ 1  — v has a D-neighbour → v stays dominated
      (b) ∀ u ∈ N(v):  u ∈ D  OR  dom_count[u] ≥ 2
          — every neighbour of v retains ≥ 1 D-neighbour after v leaves

    dom_count is updated immediately on each removal so later checks
    see the tighter, already-pruned state.
    """
    D = set(D)
    dom_count: Dict = defaultdict(int)
    for v in D:
        for u in G.neighbors(v):
            dom_count[u] += 1

    for v in list(D):
        # (a) v must remain dominated
        if dom_count[v] < 1:
            continue
        # (b) every neighbour of v must remain dominated
        if all(u in D or dom_count[u] >= 2 for u in G.neighbors(v)):
            D.discard(v)
            for u in G.neighbors(v):
                dom_count[u] -= 1

    return D


def dominating_via_reduction_max_degree_4(graph):
        # Create a working copy to avoid modifying the original graph
        G = graph.copy()
        weights = {}

        # Reduction step: Replace each vertex with auxiliary vertices
        # This transforms the problem into a maximum degree 1 case
        for u in list(graph.nodes()):  # Use list to avoid modification during iteration
            neighbors = list(G.neighbors(u))  # Get neighbors before removing node
            G.remove_node(u)  # Remove original vertex
            k = len(neighbors)  # Degree of original vertex

            # Create auxiliary vertices and connect each to one neighbor
            previous = None
            for i, v in enumerate(neighbors):
                aux_vertex = (u, i)  # Auxiliary vertex naming: (original_vertex, index)
                G.add_edge(aux_vertex, v)
                if previous is not None:
                    G.add_edge(previous, v)
                previous = aux_vertex    
                # Weight 1/k balances Cauchy-Schwarz bounds for <2 approximation
                weights[aux_vertex] = 1 / k**2  # k >= 1 post-isolate removal

        # Verify the reduction was successful (max degree should be 1)
        max_degree = max(dict(G.degree()).values()) if G.number_of_nodes() > 0 else 0
        if max_degree > 4:
            raise RuntimeError(f"Polynomial-time reduction failed: max degree is {max_degree}, expected ≤ 4")
        nx.set_node_attributes(G, weights, 'weight')
        
        dominating_set = weighted_dominating_set_max_deg4(G)
        # Extract original vertices from auxiliary vertex pairs
        greedy_solution = {u for u, _ in dominating_set}

        return greedy_solution

def prune_redundant_vertices_dominating(adj, D):
    """
    Linear-time single-pass removal of redundant vertices from a dominating set.

    A vertex v ∈ D is redundant iff:
      (a) v itself stays dominated: has ≥ 1 neighbor remaining in D
      (b) every neighbor u of v stays dominated:
            u ∈ D (self-dominated), or dom_count[u] ≥ 2 (another D-neighbor survives)

    Precompute dom_count in O(n + m), then check each v in O(deg(v)).
    Updates are propagated immediately so later checks see the reduced D.
    Total time: O(n + m).
    """
    D = set(D)

    # dom_count[u] = number of neighbours of u currently in D
    dom_count = {}
    for v in D:
        for u in adj.get(v, []):
            dom_count[u] = dom_count.get(u, 0) + 1

    for v in list(D):          # list() snapshot guards against mid-loop mutation
        # (a) v must stay dominated once it leaves D
        v_still_dominated = dom_count.get(v, 0) >= 1

        # (b) every neighbour of v must stay dominated after v's removal
        neighbors_still_dominated = all(
            u in D or dom_count.get(u, 0) >= 2
            for u in adj.get(v, [])
        )

        if v_still_dominated and neighbors_still_dominated:
            D.remove(v)
            # Propagate: v no longer contributes to neighbours' dom_counts
            for u in adj.get(v, []):
                dom_count[u] -= 1

    return D

def find_dominating_set(graph):
    """
    Approximate minimum dominating set for an undirected graph by transforming it into a bounded max-4 degree graph.

    Args:
        graph (nx.Graph): A NetworkX Graph object representing the input graph.

    Returns:
        set: A set of vertex indices representing the approximate minimum dominating set.
             Returns an empty set if the graph is empty or has no edges.
    Raises:
        ValueError: If input is not a NetworkX Graph object.
    """
    
    if not isinstance(graph, nx.Graph):
        raise ValueError("Input must be an undirected NetworkX Graph.")

    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return set()

    working_graph = graph.copy()
    working_graph.remove_edges_from(list(nx.selfloop_edges(working_graph)))
    working_graph.remove_nodes_from(list(nx.isolates(working_graph)))

    if working_graph.number_of_nodes() == 0:
        return set()

    approximate_dominating_set = set()

    
    for component in nx.connected_components(working_graph):
        G = working_graph.subgraph(component)

        # Reduction-based solution
        D = dominating_via_reduction_max_degree_4(G)

        adj = {v: set(G[v]) for v in G}

        solution = prune_redundant_vertices_dominating(adj, D)

        approximate_dominating_set.update(solution)

    return approximate_dominating_set

def find_dominating_set_brute_force(graph):
    """
    Computes an exact minimum dominating set in exponential time.

    Args:
        graph: A NetworkX Graph.

    Returns:
        A set of vertex indices representing the exact dominating set, or None if the graph is empty.
    """

    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return None

    n_vertices = len(graph.nodes())

    for k in range(1, n_vertices + 1): # Iterate through all possible sizes of the dominating set
        for candidate in itertools.combinations(graph.nodes(), k):
            dominating_candidate = set(candidate)
            if nx.dominating.is_dominating_set(graph, dominating_candidate):
                return dominating_candidate
                
    return None



def find_dominating_set_approximation(graph):
    """
    Computes an approximate dominating set in polynomial time with a logarithmic approximation ratio for undirected graphs.

    Args:
        graph: A NetworkX Graph.

    Returns:
        A set of vertex indices representing the approximate dominating set, or None if the graph is empty.
    """

    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return None

    #networkx doesn't have a guaranteed minimum dominating set function, so we use approximation
    dominating_set = nx.approximation.min_weighted_dominating_set(graph)
    return dominating_set