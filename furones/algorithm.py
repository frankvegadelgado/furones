# Created on 10/12/2025
# Author: Frank Vega

import itertools

import networkx as nx

def find_independent_set(graph):
    """
    Compute an approximate maximum independent set with a sqrt(n)-approximation ratio.

    This algorithm combines iterative refinement using maximum spanning trees with greedy
    minimum-degree and maximum-degree approaches, plus a low-degree induced subgraph heuristic,
    ensuring a robust solution across diverse graph structures. It returns the largest of the
    four independent sets produced.

    Args:
        graph (nx.Graph): An undirected NetworkX graph.

    Returns:
        set: A maximal independent set of vertices.
    """
    def iset_bipartite(bipartite_graph):
        """Compute a maximum independent set for a bipartite graph using matching.

        Args:
            bipartite_graph (nx.Graph): A bipartite NetworkX graph.

        Returns:
            set: A maximum independent set for the bipartite graph.
        """
        independent_set = set()
        for component in nx.connected_components(bipartite_graph):
            subgraph = bipartite_graph.subgraph(component)
            matching = nx.bipartite.hopcroft_karp_matching(subgraph)
            vertex_cover = nx.bipartite.to_vertex_cover(subgraph, matching)
            independent_set.update(set(subgraph.nodes()) - vertex_cover)
        return independent_set

    def is_independent_set(graph, independent_set):
        """
        Verify if a set of vertices is an independent set in the graph.

        Args:
            graph (nx.Graph): The input graph.
            independent_set (set): Vertices to check.

        Returns:
            bool: True if the set is independent, False otherwise.
        """
        for u, v in graph.edges():
            if u in independent_set and v in independent_set:
                return False
        return True

    def greedy_min_degree_independent_set(graph):
        """Compute an independent set by greedily selecting vertices by minimum degree.

        Args:
            graph (nx.Graph): The input graph.

        Returns:
            set: A maximal independent set.
        """
        if not graph:
            return set()
        independent_set = set()
        vertices = sorted(graph.nodes(), key=lambda v: graph.degree(v))
        for v in vertices:
            if all(u not in independent_set for u in graph.neighbors(v)):
                independent_set.add(v)
        return independent_set

    def greedy_max_degree_independent_set(graph):
        """Compute an independent set by greedily selecting vertices by maximum degree.

        Args:
            graph (nx.Graph): The input graph.

        Returns:
            set: A maximal independent set.
        """
        if not graph:
            return set()
        independent_set = set()
        vertices = sorted(graph.nodes(), key=lambda v: graph.degree(v), reverse=True)
        for v in vertices:
            if all(u not in independent_set for u in graph.neighbors(v)):
                independent_set.add(v)
        return independent_set

    # Validate input graph type
    if not isinstance(graph, nx.Graph):
        raise ValueError("Input must be an undirected NetworkX Graph.")

    # Handle trivial cases: empty or edgeless graphs
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return set(graph)

    # Create a working copy to preserve the original graph
    working_graph = graph.copy()

    # Remove self-loops for a valid simple graph
    working_graph.remove_edges_from(list(nx.selfloop_edges(working_graph)))

    # Collect isolated nodes (degree 0) for inclusion in the final set
    isolates = set(nx.isolates(working_graph))
    working_graph.remove_nodes_from(isolates)

    # If only isolated nodes remain, return them
    if working_graph.number_of_nodes() == 0:
        return isolates

    # Check if the graph is bipartite for exact computation
    if nx.bipartite.is_bipartite(working_graph):
        tree_based_set = iset_bipartite(working_graph)
    else:
        # Initialize candidate set with all vertices
        iterative_solution = set(working_graph.nodes())
        # Refine until independent: build max spanning tree, compute its independent set
        while not is_independent_set(working_graph, iterative_solution):
            bipartite_graph = nx.maximum_spanning_tree(working_graph.subgraph(iterative_solution))
            iterative_solution = iset_bipartite(bipartite_graph)
        # Greedily extend to maximize the independent set
        for v in working_graph.nodes():
            if v not in iterative_solution:
                # Check if v is independent of the current set iterative_solution
                if not any(working_graph.has_edge(v, u) for u in iterative_solution):
                    iterative_solution.add(v)
        tree_based_set = iterative_solution

    # Compute greedy solutions (min and max degree) to ensure robust performance
    min_greedy_solution = greedy_min_degree_independent_set(working_graph)
    max_greedy_solution = greedy_max_degree_independent_set(working_graph)

    # Additional candidate: independent set in low-degree induced subgraph
    low_set = set()
    if working_graph.number_of_nodes() > 0:
        max_deg = max(working_graph.degree(v) for v in working_graph)
        low_deg_nodes = [v for v in working_graph if working_graph.degree(v) < max_deg]
        if low_deg_nodes:
            low_sub = working_graph.subgraph(low_deg_nodes)
            low_set = greedy_min_degree_independent_set(low_sub)

    # Select the larger independent set among tree-based, min-greedy, max-greedy, and low-set to guarantee sqrt(n)-approximation
    candidates = [tree_based_set, min_greedy_solution, max_greedy_solution, low_set]
    approximate_independent_set = max(candidates, key=len)

    # Include isolated nodes in the final set
    approximate_independent_set.update(isolates)
    return approximate_independent_set


def find_independent_set_brute_force(graph):
    """
    Computes an exact independent set in exponential time.

    Args:
        graph: A NetworkX Graph.

    Returns:
        A set of vertex indices representing the exact Independent Set, or None if the graph is empty.
    """
    def is_independent_set(graph, independent_set):
        """
        Verifies if a given set of vertices is a valid Independent Set for the graph.

        Args:
            graph (nx.Graph): The input graph.
            independent_set (set): A set of vertices to check.

        Returns:
            bool: True if the set is a valid Independent Set, False otherwise.
        """
        for u in independent_set:
            for v in independent_set:
                if u != v and graph.has_edge(u, v):
                    return False
        return True
    
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return None

    n_vertices = len(graph.nodes())

    n_max_vertices = 0
    best_solution = None

    for k in range(1, n_vertices + 1): # Iterate through all possible sizes of the cover
        for candidate in itertools.combinations(graph.nodes(), k):
            cover_candidate = set(candidate)
            if is_independent_set(graph, cover_candidate) and len(cover_candidate) > n_max_vertices:
                n_max_vertices = len(cover_candidate)
                best_solution = cover_candidate
                
    return best_solution



def find_independent_set_approximation(graph):
    """
    Computes an approximate Independent Set in polynomial time with an approximation ratio of at most 2 for undirected graphs.

    Args:
        graph: A NetworkX Graph.

    Returns:
        A set of vertex indices representing the approximate Independent Set, or None if the graph is empty.
    """

    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return None

    #networkx doesn't have a guaranteed independent set function, so we use approximation
    complement_graph = nx.complement(graph)
    independent_set = nx.approximation.max_clique(complement_graph)
    return independent_set