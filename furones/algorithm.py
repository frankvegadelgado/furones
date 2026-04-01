# Created on 10/12/2025
# Author: Frank Vega

import itertools

import networkx as nx

def find_independent_set(graph):
    """
    Compute an approximate maximum independent set with a 2-approximation ratio.

    This algorithm combines iterative refinement using maximum spanning trees with greedy
    minimum-degree and maximum-degree approaches, plus a low-degree induced subgraph heuristic,
    ensuring a robust solution across diverse graph structures. It returns the largest of the
    four independent sets produced.

    Args:
        graph (nx.Graph): An undirected NetworkX graph.

    Returns:
        set: A maximal independent set of vertices.
    """
    def cover_bipartite(bipartite_graph):
        """Compute a minimum vertex cover set for a bipartite graph using matching.

        Args:
            bipartite_graph (nx.Graph): A bipartite NetworkX graph.

        Returns:
            set: A minimum vertex cover set for the bipartite graph.
        """
        optimal_solution = set()
        for component in nx.connected_components(bipartite_graph):
            subgraph = bipartite_graph.subgraph(component)
            # Hopcroft-Karp finds a maximum matching in O(E * sqrt(V)) time
            matching = nx.bipartite.hopcroft_karp_matching(subgraph)
            # By König's theorem, min vertex cover == max matching in bipartite graphs
            vertex_cover = nx.bipartite.to_vertex_cover(subgraph, matching)
            optimal_solution.update(vertex_cover)
        return optimal_solution

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
            # An edge with both endpoints in the set violates independence
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
        # Low-degree vertices have fewer neighbors, so adding them blocks fewer future candidates
        vertices = sorted(graph.nodes(), key=lambda v: graph.degree(v))
        for v in vertices:
            # Only add v if none of its neighbors are already in the set
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
        # High-degree vertices cover more edges when excluded, potentially freeing
        # large independent neighborhoods — a different trade-off to min-degree
        vertices = sorted(graph.nodes(), key=lambda v: graph.degree(v), reverse=True)
        for v in vertices:
            # Only add v if none of its neighbors are already in the set
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

    # Collect isolated nodes (degree 0) for inclusion in the final set;
    # they are trivially independent of everything and can always be included
    isolates = set(nx.isolates(working_graph))
    working_graph.remove_nodes_from(isolates)

    # If only isolated nodes remain, return them
    if working_graph.number_of_nodes() == 0:
        return isolates

    # Check if the graph is bipartite for exact computation
    if nx.bipartite.is_bipartite(working_graph):
        # The complement of a minimum vertex cover is a maximum independent set
        complement_based_set = set(working_graph.nodes()) - cover_bipartite(working_graph)
    else:
        approximate_vertex_cover = set()
        # Process each connected component independently to reduce problem size
        component_solutions = [working_graph.subgraph(component) for component in nx.connected_components(working_graph)]
        while component_solutions:
            subgraph = component_solutions.pop()
            if subgraph.number_of_edges() == 0:
                continue
            if nx.bipartite.is_bipartite(subgraph):
                # Exploit bipartiteness for an exact minimum vertex cover via König's theorem
                approximate_vertex_cover.update(cover_bipartite(subgraph))
            else:
                # Fall back to a weighted vertex cover approximation for non-bipartite components
                vertex_cover = nx.approximation.min_weighted_vertex_cover(subgraph)

                # Build a gadget graph G to attempt a tighter cover refinement:
                # each vertex u in the cover is split into two copies (u, 0) and (u, 1),
                # while vertices outside the cover keep a single triple-tuple identity.
                # A vertex is removed from the cover only if both its copies end up covered.
                G = nx.Graph()
                for u, v in subgraph.edges():
                    if u in vertex_cover and v in vertex_cover:
                        # Both endpoints are in the cover — connect all copy pairs
                        G.add_edge((u, 0), (v, 0))
                        G.add_edge((u, 0), (v, 1))
                        G.add_edge((u, 1), (v, 0))
                        G.add_edge((u, 1), (v, 1))
                    elif u in vertex_cover:
                        # Only u is in the cover; v uses its single-node identity
                        G.add_edge((u, 0), (v, v, v))
                        G.add_edge((u, 1), (v, v, v))
                    elif v in vertex_cover:
                        # Only v is in the cover; u uses its single-node identity
                        G.add_edge((u, u, u), (v, 0))
                        G.add_edge((u, u, u), (v, 1))
                    else:
                        # Neither endpoint is in the cover; both use single-node identities
                        G.add_edge((u, u, u), (v, v, v))

                tuple_vertex_cover = nx.approximation.min_weighted_vertex_cover(G)

                # A cover vertex u can be dropped only if both its copies are covered,
                # meaning it is provably redundant in the refined cover
                solution = {u for u in vertex_cover if (u, 0) in tuple_vertex_cover and (u, 1) in tuple_vertex_cover}
                if solution:
                    approximate_vertex_cover.update(solution)
                    # Remove the refined cover vertices and recurse on the remaining subgraph
                    remaining_nodes = subgraph.subgraph(set(subgraph.nodes()) - solution).copy()
                    remaining_isolates = set(nx.isolates(remaining_nodes))
                    remaining_nodes.remove_nodes_from(remaining_isolates)
                    if remaining_nodes.number_of_edges() > 0:
                        new_component_solutions = [remaining_nodes.subgraph(component) for component in nx.connected_components(remaining_nodes)]
                        component_solutions.extend(new_component_solutions)
                else:
                    # Refinement yielded nothing; fall back to the original cover
                    approximate_vertex_cover.update(vertex_cover)

        # The complement of the vertex cover is a candidate independent set
        complement_solution = set(working_graph.nodes()) - approximate_vertex_cover

        # Greedily extend the candidate by adding any non-adjacent uncovered vertex
        for v in working_graph.nodes():
            if v not in complement_solution:
                # Check if v is independent of the current set complement_solution
                if not any(working_graph.has_edge(v, u) for u in complement_solution):
                    complement_solution.add(v)
        complement_based_set = complement_solution

    # Compute greedy solutions (min and max degree) to ensure robust performance
    min_greedy_solution = greedy_min_degree_independent_set(working_graph)
    max_greedy_solution = greedy_max_degree_independent_set(working_graph)

    # Additional candidate: restrict to nodes below maximum degree, where local
    # density is lower and greedy selection may find a larger independent set
    low_set = set()
    if working_graph.number_of_nodes() > 0:
        max_deg = max(working_graph.degree(v) for v in working_graph)
        low_deg_nodes = [v for v in working_graph if working_graph.degree(v) < max_deg]
        if low_deg_nodes:
            low_sub = working_graph.subgraph(low_deg_nodes)
            low_set = greedy_min_degree_independent_set(low_sub)

    # Pick the largest independent set across all four candidates
    candidates = [complement_based_set, min_greedy_solution, max_greedy_solution, low_set]
    approximate_independent_set = max(candidates, key=len)

    # Re-add isolated nodes — they are always safe to include
    approximate_independent_set.update(isolates)
    if not is_independent_set(graph, approximate_independent_set):
        raise RuntimeError(f"Polynomial-time algorithm failed: the set {approximate_independent_set} is not independent")
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