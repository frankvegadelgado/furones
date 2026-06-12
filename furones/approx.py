"""
LP-backed unweighted Minimum Dominating Set (MDS) approximation.

This module targets MDS directly. It does not impose independence. The input
graphs produced by the Furones auxiliary reduction are expected to have maximum
degree at most 4, so every closed neighbourhood has size at most 5. The greedy
set-cover algorithm therefore has the standard H_{Delta+1} approximation bound
on those bounded-degree instances; for Delta <= 4 this is H_5 ~= 2.283.

LP relaxation (MDS-LP):
  min   sum x_v
  s.t.  sum_{u in N[v]} x_u >= 1    for all v  (domination)
        0 <= x_v <= 1
"""

import time
from dataclasses import dataclass
from typing import Dict, Set

import networkx as nx
import numpy as np
from scipy.optimize import linprog


@dataclass
class MDSResult:
    dominating_set: Set
    size: int
    lp_lower_bound: float
    approx_ratio: float
    delta: int
    theoretical_bound: float
    lp_solve_time: float
    greedy_time: float
    lp_success: bool
    verified: bool


def harmonic_number(n: int) -> float:
    """Return H_n = 1 + 1/2 + ... + 1/n."""
    if n <= 0:
        return 0.0
    return sum(1.0 / k for k in range(1, n + 1))


def solve_mds_lp(graph: nx.Graph) -> tuple[np.ndarray, float, bool]:
    """
    Solve the LP relaxation of unweighted MDS.

    Returns (x, objective, success).
    """
    if not isinstance(graph, nx.Graph):
        raise ValueError("Input must be an undirected NetworkX Graph.")

    nodes = list(graph.nodes())
    n = len(nodes)
    if n == 0:
        return np.array([]), 0.0, True

    idx = {v: i for i, v in enumerate(nodes)}
    c = np.ones(n)

    a_ub, b_ub = [], []
    for v in nodes:
        row = np.zeros(n)
        row[idx[v]] = -1.0
        for u in graph.neighbors(v):
            row[idx[u]] = -1.0
        a_ub.append(row)
        b_ub.append(-1.0)

    result = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[(0.0, 1.0)] * n,
        method="highs",
        options={"presolve": True},
    )

    if result.success:
        return result.x, float(result.fun), True

    # Feasible fallback used only for priority tie-breaking and diagnostics.
    # x_v = 1 is always feasible for the domination LP.
    return np.ones(n), float(n), False


def greedy_mds_from_lp_priority(graph: nx.Graph, x: np.ndarray, nodes: list) -> Set:
    """
    Greedy set-cover approximation for MDS.

    At each step choose the vertex whose closed neighbourhood covers the largest
    number of currently uncovered vertices. LP values are used only as a stable
    tie-breaker, so the standard greedy set-cover guarantee is preserved.
    """
    if not isinstance(graph, nx.Graph):
        raise ValueError("Input must be an undirected NetworkX Graph.")
    if len(x) != len(nodes):
        raise ValueError("LP priority vector length must match the node list.")

    idx = {v: i for i, v in enumerate(nodes)}
    closed_neighbourhoods: Dict = {
        v: set(graph.neighbors(v)) | {v}
        for v in nodes
    }
    uncovered = set(nodes)
    dominating_set = set()

    while uncovered:
        best = max(
            nodes,
            key=lambda v: (
                len(closed_neighbourhoods[v] & uncovered),
                x[idx[v]],
                graph.degree(v),
                repr(v),
            ),
        )
        newly_covered = closed_neighbourhoods[best] & uncovered
        if not newly_covered:
            best = next(iter(uncovered))
            newly_covered = {best}
        dominating_set.add(best)
        uncovered.difference_update(newly_covered)

    return dominating_set


def prune_redundant_vertices(graph: nx.Graph, dominating_set: Set) -> Set:
    """Remove vertices whose deletion preserves domination."""
    if not isinstance(graph, nx.Graph):
        raise ValueError("Input must be an undirected NetworkX Graph.")

    result = set(dominating_set)
    for v in list(result):
        candidate = result - {v}
        if nx.dominating.is_dominating_set(graph, candidate):
            result = candidate
    return result


def mds_lp(graph: nx.Graph) -> MDSResult:
    """
    Compute an unweighted MDS approximation.

    For maximum degree Delta, greedy set cover gives an H_{Delta+1} guarantee.
    On Furones degree-four auxiliary graphs this is H_5 ~= 2.283.
    """
    if not isinstance(graph, nx.Graph):
        raise ValueError("Input must be an undirected NetworkX Graph.")

    if len(graph) == 0:
        return MDSResult(set(), 0, 0.0, 1.0, 0, 1.0, 0.0, 0.0, True, True)

    nodes = list(graph.nodes())
    delta = max(dict(graph.degree()).values()) if nodes else 0
    theoretical_bound = harmonic_number(delta + 1)

    start = time.perf_counter()
    x, lp_obj, lp_ok = solve_mds_lp(graph)
    lp_time = time.perf_counter() - start

    start = time.perf_counter()
    dominating_set = greedy_mds_from_lp_priority(graph, x, nodes)
    dominating_set = prune_redundant_vertices(graph, dominating_set)
    greedy_time = time.perf_counter() - start

    verified = nx.dominating.is_dominating_set(graph, dominating_set)
    size = len(dominating_set)
    ratio = size / lp_obj if lp_obj > 1e-9 else float("inf")

    return MDSResult(
        dominating_set=dominating_set,
        size=size,
        lp_lower_bound=lp_obj,
        approx_ratio=ratio,
        delta=delta,
        theoretical_bound=theoretical_bound,
        lp_solve_time=lp_time,
        greedy_time=greedy_time,
        lp_success=lp_ok,
        verified=verified,
    )


def mds_greedy_baseline(graph: nx.Graph) -> MDSResult:
    """Pure greedy MDS baseline without LP tie-breaking."""
    if not isinstance(graph, nx.Graph):
        raise ValueError("Input must be an undirected NetworkX Graph.")

    if len(graph) == 0:
        return MDSResult(set(), 0, 0.0, 1.0, 0, 1.0, 0.0, 0.0, True, True)

    nodes = list(graph.nodes())
    delta = max(dict(graph.degree()).values()) if nodes else 0
    theoretical_bound = harmonic_number(delta + 1)
    x = np.zeros(len(nodes))

    start = time.perf_counter()
    dominating_set = greedy_mds_from_lp_priority(graph, x, nodes)
    dominating_set = prune_redundant_vertices(graph, dominating_set)
    greedy_time = time.perf_counter() - start

    _, lp_obj, lp_ok = solve_mds_lp(graph)
    ratio = len(dominating_set) / lp_obj if lp_obj > 1e-9 else float("inf")

    return MDSResult(
        dominating_set=dominating_set,
        size=len(dominating_set),
        lp_lower_bound=lp_obj,
        approx_ratio=ratio,
        delta=delta,
        theoretical_bound=theoretical_bound,
        lp_solve_time=0.0,
        greedy_time=greedy_time,
        lp_success=lp_ok,
        verified=nx.dominating.is_dominating_set(graph, dominating_set),
    )


def run_tests() -> None:
    test_cases = [
        ("Path P6", nx.path_graph(6)),
        ("Cycle C8", nx.cycle_graph(8)),
        ("Complete K6", nx.complete_graph(6)),
        ("Complete Bip K4,4", nx.complete_bipartite_graph(4, 4)),
        ("Petersen", nx.petersen_graph()),
        ("Star S8", nx.star_graph(8)),
        ("ER n=30 p=0.15", nx.erdos_renyi_graph(30, 0.15, seed=42)),
        ("ER n=30 p=0.40", nx.erdos_renyi_graph(30, 0.40, seed=7)),
        ("BA n=40 m=2", nx.barabasi_albert_graph(40, 2, seed=0)),
        ("BA n=50 m=4", nx.barabasi_albert_graph(50, 4, seed=1)),
        ("Grid 5x5", nx.grid_2d_graph(5, 5)),
        ("Grid 6x6", nx.grid_2d_graph(6, 6)),
        ("Random Tree n=20", nx.random_labeled_tree(20, seed=3)),
        ("Balanced Tree r=3 h=3", nx.balanced_tree(3, 3)),
        ("3-Regular n=20", nx.random_regular_graph(3, 20, seed=5)),
        ("4-Regular n=20", nx.random_regular_graph(4, 20, seed=6)),
    ]

    graph = nx.erdos_renyi_graph(80, 0.10, seed=99)
    if not nx.is_connected(graph):
        graph = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()
    test_cases.append(("ER n=80 p=0.10", graph))

    header = (
        f"{'Graph':<26} {'n':>5} {'m':>6} {'Delta':>5} {'LP lb':>7} "
        f"{'LP sz':>6} {'GR sz':>6} {'LP rat':>7} {'GR rat':>7} "
        f"{'H_d':>7} {'OK?':>5} {'LP time':>8}"
    )
    sep = "-" * len(header)
    print(sep)
    print("  LP-backed Greedy Set Cover - Unweighted MDS Approximation")
    print(sep)
    print(header)
    print(sep)

    all_passed = True
    lp_wins = ties = baseline_wins = 0

    for name, graph in test_cases:
        graph = nx.Graph(graph)
        if len(graph) == 0:
            continue

        r_lp = mds_lp(graph)
        r_baseline = mds_greedy_baseline(graph)
        all_passed = all_passed and r_lp.verified

        if r_lp.size < r_baseline.size:
            lp_wins += 1
        elif r_lp.size == r_baseline.size:
            ties += 1
        else:
            baseline_wins += 1

        print(
            f"{name:<26} {graph.number_of_nodes():>5} {graph.number_of_edges():>6} "
            f"{r_lp.delta:>5} {r_lp.lp_lower_bound:>7.2f} {r_lp.size:>6} "
            f"{r_baseline.size:>6} {r_lp.approx_ratio:>7.3f} "
            f"{r_baseline.approx_ratio:>7.3f} {r_lp.theoretical_bound:>7.3f} "
            f"{'yes' if r_lp.verified else 'no':>5} {r_lp.lp_solve_time * 1000:>8.1f}ms"
        )

    print(sep)
    print(f"\nAll verified: {'YES' if all_passed else 'NO'}")
    print(f"LP tie-break wins: {lp_wins}  |  Ties: {ties}  |  Baseline wins: {baseline_wins}")
    print()
    print("Column legend:")
    print("  LP lb  = LP relaxation lower bound on |OPT_MDS|")
    print("  LP sz  = MDS size from greedy set cover with LP tie-breaking")
    print("  GR sz  = MDS size from pure greedy set cover baseline")
    print("  LP rat = LP sz / LP lb")
    print("  GR rat = GR sz / LP lb")
    print("  H_d    = H_(Delta+1), the greedy set-cover guarantee")
    print("  OK?    = verified dominating set")
    print(sep)


if __name__ == "__main__":
    run_tests()
