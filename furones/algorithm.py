# Created on 21/05/2025
# Author: Frank Vega

import itertools

import networkx as nx

def find_independent_set(graph):
    """
    Compute an approximate independent set for an undirected graph by transforming it into a chordal graph.

    Args:
        graph (nx.Graph): A NetworkX Graph object representing the input graph.

    Returns:
        set: A set of vertex indices representing the independent set.
             Returns an empty set if the graph is empty or has no edges.
    """
    def iset_bipartite(bipartite_graph):
        # Initialize an empty set to store the independent set
        independent_set = set()
        # Iterate over each connected component in the bipartite graph
        for component in nx.connected_components(bipartite_graph):
            # Extract the subgraph for the current component
            bipartite_subgraph = bipartite_graph.subgraph(component)
            # Compute the maximum matching in the bipartite subgraph
            maximum_matching = nx.bipartite.hopcroft_karp_matching(bipartite_subgraph)
            # Derive the vertex cover from the maximum matching
            component_vertex_cover = nx.bipartite.to_vertex_cover(bipartite_subgraph, maximum_matching)
            # Add nodes not in the vertex cover to the independent set
            independent_set.update(set(bipartite_subgraph.nodes()) - component_vertex_cover)
        return independent_set

    def is_independent_set(graph, independent_set):
        """
        Verifies if a given set of vertices is a valid Independent Set for the graph.

        Args:
            graph (nx.Graph): The input graph.
            independent_set (set): A set of vertices to check.

        Returns:
            bool: True if the set is a valid Independent Set, False otherwise.
        """
        # Check if any edge has both endpoints in the independent set
        for u, v in graph.edges():
            if u in independent_set and v in independent_set:
                return False
        return True
    
    # Validate input is a NetworkX graph
    if not isinstance(graph, nx.Graph):
        raise ValueError("Input must be an undirected NetworkX Graph.")
    
    # Handle trivial cases
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return set()  # Empty graph or no edges means empty Independent Set
    
    # Create a working copy of the graph to avoid modifying the original
    working_graph = graph.copy()
    
    # Clean the graph: remove self-loops since they're not valid in a simple graph
    working_graph.remove_edges_from(list(nx.selfloop_edges(working_graph)))
    
    # Initialize the isolates set with nodes of degree 0
    isolates = set(nx.isolates(working_graph))

    # Remove isolated nodes from the working graph
    working_graph.remove_nodes_from(isolates)
    
    # If the cleaned graph is empty, return the set of isolated nodes
    if working_graph.number_of_nodes() == 0:
        return isolates
    
    # Check if the working graph is bipartite
    if nx.bipartite.is_bipartite(working_graph):
        # If bipartite, compute the independent set directly
        approximate_independent_set = iset_bipartite(working_graph)
    
    else:
        # Start with all nodes as a candidate independent set
        approximate_independent_set = set(working_graph.nodes())
        # Iteratively refine the set until it is a valid independent set
        while not is_independent_set(working_graph, approximate_independent_set):
            # Create a maximum spanning tree from the current candidate set
            bipartite_graph = nx.maximum_spanning_tree(working_graph.subgraph(approximate_independent_set))
            # Compute an independent set for the spanning tree
            approximate_independent_set = iset_bipartite(bipartite_graph)

        # Greedily add nodes to maximize the independent set
        for u in working_graph.nodes():
            if is_independent_set(working_graph, approximate_independent_set.union({u})):
                approximate_independent_set.add(u)
    
    # Include isolated nodes in the final independent set
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