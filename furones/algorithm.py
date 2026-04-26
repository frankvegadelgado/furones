# Created on 26/07/2025
# Author: Frank Vega

import itertools

import networkx as nx
from . import tscc_ds_reduction
from . import baker_algo
from collections import defaultdict
from typing import Dict, Set

def prune_redundant_vertices_dominating(G: nx.Graph, D: Set) -> Set:
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

def find_dominating_set(graph, eps=0.5):
    """
    Compute an approximate minimum dominating set (MDS) of an undirected graph.

    The algorithm combines structural reductions with Baker's PTAS for planar graphs.
    Specifically, it reduces the input to a planar 2-connected core (TSCC form),
    applies a (1 + ε)-approximation scheme on the reduced instance, and lifts the
    solution back to the original graph.

    Guarantees:
        • For general graphs, the algorithm returns a valid dominating set.
        • If the reduced instance is planar, the solution achieves a
          (1 + ε)-approximation with respect to the reduced graph.
        • The overall approximation factor depends on the reduction and lifting
          steps and is typically small in practice.
        • The running time is O(n + m) · f(1/ε) for fixed ε.

    Args:
        graph (nx.Graph):
            An undirected NetworkX graph.
        eps (float):
            Approximation parameter ε ∈ (0, 1].

    Returns:
        set:
            A dominating set of the input graph.

    Raises:
        ValueError:
            If the input is invalid or ε ∉ (0, 1].
        RuntimeError:
            If a required structural assumption is violated.
    """

    # --- Parameter validation ---
    # Ensure ε is within the admissible PTAS range
    if eps <= 0 or eps > 1:
        raise ValueError("epsilon must be in this interval (0, 1].")

    # Ensure the input is an undirected NetworkX graph
    if not isinstance(graph, nx.Graph):
        raise ValueError("Input must be an undirected NetworkX Graph.")

    # --- Trivial cases ---
    # If the graph has no vertices or no edges, domination is trivial
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return set()

    # --- Preprocessing ---
    # Work on a cleaned copy of the graph:
    #   • remove self-loops (irrelevant for domination)
    #   • remove isolated vertices (handled separately)
    working_graph = graph.copy()
    working_graph.remove_edges_from(list(nx.selfloop_edges(working_graph)))

    # Isolated vertices must belong to every dominating set
    isolates = set(nx.isolates(working_graph))
    working_graph.remove_nodes_from(isolates)

    # If all vertices were isolated, they form the unique dominating set
    if working_graph.number_of_nodes() == 0:
        return isolates

    # --- Reduction phase ---
    # Reduce to a planar TSCC instance:
    #   • G_reduced: reduced planar core
    #   • forced_ds: vertices forced into any dominating set
    #   • lift: maps solutions of G_reduced back to the original graph
    G_reduced, forced_ds, lift = tscc_ds_reduction.reduce_to_tscc_for_ds(
        working_graph
    )

    # Baker's PTAS requires planarity of the reduced instance
    if not nx.is_planar(G_reduced):
        raise RuntimeError("2-connected edge graph is not planar.")

    # --- PTAS phase (on reduced graph) ---
    if G_reduced:
        # Relabel vertices to consecutive integers [0, ..., n-1]
        # (required by the PTAS implementation)
        mapping = {u: k for k, u in enumerate(G_reduced.nodes())}
        unmapping = {k: u for u, k in mapping.items()}

        # Build the internal graph representation expected by Baker's algorithm
        G = baker_algo.Graph(G_reduced.number_of_nodes())
        for u, v in G_reduced.edges():
            G.add_edge(mapping[u], mapping[v])

        # Compute a (1 + ε)-approximate dominating set on the reduced graph
        ptas_sol = baker_algo.baker_ptas(G, eps, verbose=False)

        # Translate the solution back to original vertex labels of G_reduced
        md_reduced = {unmapping[u] for u in ptas_sol}

        # --- Lifting phase ---
        # Extend the reduced solution to a valid dominating set of the original graph
        D = lift(md_reduced)

        # --- Postprocessing ---
        # Remove redundant vertices while preserving domination
        # (greedy pruning step)
        approximate_dominating_set = prune_redundant_vertices_dominating(
            working_graph, D
        )

    else:
        # Degenerate case: reduction collapses completely
        # Use the forced vertices directly
        approximate_dominating_set = forced_ds

    # --- Reintegration of isolated vertices ---
    # All isolated vertices must be included to ensure domination
    approximate_dominating_set.update(isolates)

    # --- Verification (safety check) ---
    # Validate that the constructed set is indeed dominating
    # (runs in O(n + m))
    if not nx.is_dominating_set(graph, approximate_dominating_set):
        raise RuntimeError(
            "Invalid solution: the computed set is not a dominating set."
        )

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