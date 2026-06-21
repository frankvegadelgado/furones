# Created on 26/07/2025
# Author: Frank Vega

import itertools

import networkx as nx
from . import tscc_ds_reduction
from . import baker_algo
from collections import defaultdict
from typing import Any, Dict, Set


class ApproximationNotCertifiedError(RuntimeError):
    """Raised when linear-time consistency checks cannot certify a 2-bound."""


def prune_redundant_vertices_dominating(G: nx.Graph, D: Set[Any]) -> Set[Any]:
    """
    Remove redundant vertices while preserving domination.

    Runs in O(n + m) for one pass over the current dominating set.

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


def is_two_approximation_certified(
    graph: nx.Graph,
    reduced_graph: nx.Graph,
    forced_ds: Set[Any],
    dominating_set: Set[Any],
) -> bool:
    """
    Linear-time sufficient check for the proved 2-approximation cases.

    The certificate is intentionally conservative.  It accepts the tight case
    and the non-tight case covered by the proved inequality |F| >= 2|F_R|.
    It does not use the post-pruning boundary alone, and it does not
    claim a universal 2-approximation for general graphs.
    """
    if not nx.is_dominating_set(graph, dominating_set):
        return False

    reduced_nodes = set(reduced_graph.nodes())
    forced_boundary = {
        v for v in reduced_nodes
        if any(u in forced_ds for u in graph.neighbors(v))
    }
    return not forced_boundary or len(forced_ds) >= 2 * len(forced_boundary)


def find_dominating_set(graph, eps=1, consistency=False):
    """
    Compute a Furones v0.3.0 dominating set of an undirected graph.

    The algorithm combines structural reductions with Baker's PTAS for planar graphs.
    Specifically, it reduces the input to a planar 2-connected core (TSCC form),
    applies a (1 + ε)-approximation scheme on the reduced instance, and lifts the
    solution back to the original graph.

    Guarantees:
        • For general graphs, the algorithm returns a valid dominating set.
        • If the reduced instance is planar, the solution achieves a
          (1 + ε)-approximation with respect to the reduced graph.
        • The overall approximation factor depends on the reduction, lifting,
          and optional consistency certificate; no exponential fallback is used
          by this routine.

    Args:
        graph (nx.Graph):
            An undirected NetworkX graph.
        eps (float):
            Approximation parameter ε ∈ (0, 1].

        consistency (bool):
            If True, require a linear-time certificate for the proved
            2-approximation cases. No exponential fallback is used.

    Returns:
        set:
            A dominating set of the input graph.

    Raises:
        ValueError:
            If the input is invalid or ε ∉ (0, 1].
        RuntimeError:
            If a required structural assumption is violated.
        ApproximationNotCertifiedError:
            If consistency=True and the current linear-time proof conditions do
            not certify a 2-approximation for this instance.
    """

    # --- Parameter validation ---
    # Ensure ε is within the admissible PTAS range
    if eps <= 0 or eps > 1:
        raise ValueError("epsilon must be in this interval (0, 1].")

    # Ensure the input is an undirected NetworkX graph
    if not isinstance(graph, nx.Graph):
        raise ValueError("Input must be an undirected NetworkX Graph.")

    # --- Trivial cases ---
    # If the graph has no vertices, domination is trivial
    if graph.number_of_nodes() == 0:
        return set()
    if graph.number_of_edges() == 0:
        return set(graph.nodes())

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

    else:
        # Degenerate case: reduction collapses completely
        # Use the forced vertices directly
        D = forced_ds

    # --- Postprocessing ---
    # Remove redundant vertices while preserving domination
    # (greedy pruning step)
    approximate_dominating_set = prune_redundant_vertices_dominating(
        working_graph, D
    )

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

    if consistency:
        certified = is_two_approximation_certified(
            working_graph,
            G_reduced,
            forced_ds,
            approximate_dominating_set - isolates,
        )
        if not certified:
            raise ApproximationNotCertifiedError(
                "Linear-time consistency checks cannot certify a 2-approximation "
                "for this instance. A universal polynomial 2-approximation for "
                "general Dominating Set would imply P = NP.",
                approximate_dominating_set
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

    if graph.number_of_nodes() == 0:
        return set()
    if graph.number_of_edges() == 0:
        return set(graph.nodes())

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
