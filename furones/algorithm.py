# Created on 26/07/2025
# Author: Frank Vega

import itertools

import networkx as nx
from . import tscc_ds_reduction
from . import baker_algo
from collections import defaultdict
from typing import Any, Dict, Iterable, Set


class ApproximationNotCertifiedError(RuntimeError):
    """Raised when linear-time consistency checks cannot certify a 2-bound."""


def _closed_neighborhood(G: nx.Graph, v: Any) -> Iterable[Any]:
    """Yield v and its neighbours without allocating a temporary set."""
    yield v
    yield from G.neighbors(v)


def prune_redundant_vertices_dominating(G: nx.Graph, D: Set[Any]) -> Set[Any]:
    """
    Remove redundant vertices while preserving domination.

    This is a single linear pass over the closed neighbourhoods of the
    current solution.  A vertex v in D is safely removable exactly when every
    vertex in N[v] is still dominated by another selected vertex after v is
    deleted.  Counts are updated immediately, so later removals see the
    already-pruned state.

    Complexity: O(n + m).
    """
    D = set(D)
    dom_count: Dict[Any, int] = defaultdict(int)
    for v in D:
        for u in _closed_neighborhood(G, v):
            dom_count[u] += 1

    for v in list(D):
        # Removing v only changes domination counts on N[v].
        if all(dom_count[u] >= 2 for u in _closed_neighborhood(G, v)):
            D.discard(v)
            for u in _closed_neighborhood(G, v):
                dom_count[u] -= 1

    return D


def greedy_closed_degree_dominating_set(G: nx.Graph) -> Set[Any]:
    """
    Build a dominating-set candidate by a linear coverage sweep.

    Vertices are bucketed by closed degree |N[v]| and scanned from largest
    closed degree to smallest.  A vertex is selected only if its closed
    neighbourhood contains at least one still-undominated vertex.  The method
    is not a universal approximation theorem; it is a deterministic linear-time
    heuristic that preserves high-coverage vertices from the original graph.

    This specifically avoids the failure mode where a planar forest projection
    discards dense domination edges: the sweep is run on the original working
    graph, so a universal or near-universal vertex is naturally considered
    before low-coverage path-like vertices, without adding any special-case
    rule for universal vertices.

    Complexity: O(n + m).
    """
    n = G.number_of_nodes()
    if n == 0:
        return set()

    # Degree values are integers in [0, n-1], so bucket sorting is linear.
    buckets = [[] for _ in range(n + 1)]
    for v in G.nodes():
        buckets[G.degree(v) + 1].append(v)

    dominated: Set[Any] = set()
    D: Set[Any] = set()

    for closed_degree in range(n, 0, -1):
        for v in buckets[closed_degree]:
            if all(u in dominated for u in _closed_neighborhood(G, v)):
                continue

            D.add(v)
            for u in _closed_neighborhood(G, v):
                dominated.add(u)

            if len(dominated) == n:
                return prune_redundant_vertices_dominating(G, D)

    # The loop always dominates a finite graph, but keep a defensive fallback.
    return prune_redundant_vertices_dominating(G, D)


def _choose_best_valid_candidate(G: nx.Graph, *candidates: Set[Any]) -> Set[Any]:
    """Return the smallest candidate that is a valid dominating set of G."""
    valid = []
    for D in candidates:
        D = prune_redundant_vertices_dominating(G, set(D))
        if nx.is_dominating_set(G, D):
            valid.append(D)
    if not valid:
        raise RuntimeError("No valid dominating-set candidate was produced.")
    return min(valid, key=lambda D: (len(D), sorted(map(str, D))))


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
    Compute a Furones v0.3.1 dominating set of an undirected graph.

    The algorithm combines structural reductions with Baker's PTAS for planar graphs
    and a linear closed-degree coverage sweep on the original working graph.
    The sweep is a general high-coverage heuristic, not a special-case rule: it
    considers every vertex by closed degree and keeps a vertex only when it covers
    a still-undominated vertex.  The final answer is the smaller valid candidate
    after pruning.

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

    # --- Postprocessing and linear high-coverage comparison ---
    # Candidate A is the reduced/lifted solution.  Candidate B is an
    # independent linear closed-degree coverage sweep on the original working
    # graph.  This is not a special exception for universal vertices: every
    # vertex participates in the same bucketed coverage scan.  It repairs the
    # dense-edge loss caused by the planar forest projection whenever a high
    # coverage original vertex gives a smaller valid dominating set.
    lifted_candidate = prune_redundant_vertices_dominating(working_graph, D)
    sweep_candidate = greedy_closed_degree_dominating_set(working_graph)

    approximate_dominating_set = _choose_best_valid_candidate(
        working_graph,
        lifted_candidate,
        sweep_candidate,
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
