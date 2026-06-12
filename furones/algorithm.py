# Created on 26/07/2025
# Author: Frank Vega

import itertools
from collections import defaultdict
from typing import Dict, Set

import networkx as nx

from . import approx


def prune_redundant_vertices_dominating(graph: nx.Graph, dominating_set: Set) -> Set:
    """
    Remove vertices from a dominating set while preserving domination.

    A selected vertex v is removable when another selected vertex dominates v,
    and every neighbour of v remains dominated after v is deleted.
    """
    result = set(dominating_set)
    missing = result - set(graph.nodes())
    if missing:
        raise ValueError("Dominating set contains vertices that are not in the graph.")

    selected_neighbour_count: Dict = defaultdict(int)
    for v in result:
        for u in graph.neighbors(v):
            selected_neighbour_count[u] += 1

    for v in list(result):
        if selected_neighbour_count[v] < 1:
            continue
        if all(u in result or selected_neighbour_count[u] >= 2 for u in graph.neighbors(v)):
            result.discard(v)
            for u in graph.neighbors(v):
                selected_neighbour_count[u] -= 1

    return result


def _degree_four_auxiliary_graph(component_graph: nx.Graph) -> nx.Graph:
    """Build and validate the sequential degree-four auxiliary graph."""
    auxiliary = component_graph.copy()

    for u in list(component_graph.nodes()):
        neighbours = list(auxiliary.neighbors(u))
        auxiliary.remove_node(u)

        first_auxiliary = None
        previous_neighbour = None
        for i, v in enumerate(neighbours):
            aux_vertex = (u, i)
            auxiliary.add_edge(aux_vertex, v)
            if previous_neighbour is None:
                first_auxiliary = aux_vertex
            else:
                auxiliary.add_edge(aux_vertex, previous_neighbour)
            previous_neighbour = v

        if len(neighbours) > 1:
            auxiliary.add_edge(first_auxiliary, previous_neighbour)

    max_degree = max(dict(auxiliary.degree()).values()) if auxiliary.number_of_nodes() else 0
    if max_degree > 4:
        raise RuntimeError(
            f"Degree-four reduction failed: max degree is {max_degree}, expected <= 4."
        )

    return auxiliary


def _project_auxiliary_solution(auxiliary_solution: Set, component_graph: nx.Graph) -> Set:
    """Map auxiliary vertices back to original component vertices."""
    component_nodes = set(component_graph.nodes())
    projected = set()

    for vertex in auxiliary_solution:
        if isinstance(vertex, tuple) and len(vertex) == 2 and vertex[0] in component_nodes:
            projected.add(vertex[0])
        elif vertex in component_nodes:
            projected.add(vertex)

    return projected


def _find_component_solution(component_graph: nx.Graph) -> Set:
    """Return the smallest verified Furones candidate for one connected component."""
    auxiliary = _degree_four_auxiliary_graph(component_graph)
    auxiliary_result = approx.mds_lp(auxiliary)
    if not auxiliary_result.verified:
        raise RuntimeError("Auxiliary MDS solver returned an invalid dominating set.")

    projected_solution = _project_auxiliary_solution(
        auxiliary_result.dominating_set,
        component_graph,
    )
    complement_solution = set(component_graph.nodes()) - projected_solution

    for candidate in sorted((projected_solution, complement_solution), key=len):
        if nx.dominating.is_dominating_set(component_graph, candidate):
            return candidate

    raise RuntimeError("Degree-four reduction failed: no verified candidate found.")


def find_dominating_set(graph: nx.Graph) -> Set:
    """
    Compute an approximate minimum dominating set of an undirected graph.

    Returns a verified dominating set. Raises RuntimeError if the auxiliary
    projection stage cannot validate a component candidate.
    """
    if not isinstance(graph, nx.Graph):
        raise ValueError("Input must be an undirected NetworkX Graph.")

    if graph.number_of_nodes() == 0:
        return set()

    cleaned_graph = graph.copy()
    cleaned_graph.remove_edges_from(list(nx.selfloop_edges(cleaned_graph)))

    solution = set(nx.isolates(cleaned_graph))
    working_graph = cleaned_graph.copy()
    working_graph.remove_nodes_from(solution)

    for component in nx.connected_components(working_graph):
        component_graph = working_graph.subgraph(component).copy()
        solution.update(_find_component_solution(component_graph))

    solution = prune_redundant_vertices_dominating(cleaned_graph, solution)
    if not nx.dominating.is_dominating_set(cleaned_graph, solution):
        raise RuntimeError("Invalid solution: the computed set is not a dominating set.")

    return solution


def find_dominating_set_brute_force(graph: nx.Graph) -> Set | None:
    """
    Compute an exact minimum dominating set in exponential time.

    Intended only for small validation instances.
    """
    if not isinstance(graph, nx.Graph):
        raise ValueError("Input must be an undirected NetworkX Graph.")

    if graph.number_of_nodes() == 0:
        return set()
    if graph.number_of_edges() == 0:
        return set(graph.nodes())

    nodes = list(graph.nodes())
    for k in range(1, len(nodes) + 1):
        for candidate in itertools.combinations(nodes, k):
            dominating_candidate = set(candidate)
            if nx.dominating.is_dominating_set(graph, dominating_candidate):
                return dominating_candidate

    return None


def find_dominating_set_approximation(graph: nx.Graph) -> Set:
    """
    Compute NetworkX's logarithmic-factor dominating-set approximation.
    """
    if not isinstance(graph, nx.Graph):
        raise ValueError("Input must be an undirected NetworkX Graph.")

    if graph.number_of_nodes() == 0:
        return set()
    if graph.number_of_edges() == 0:
        return set(graph.nodes())

    return set(nx.approximation.min_weighted_dominating_set(graph))
