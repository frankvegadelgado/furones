"""Run Furones against exact SciPy MILP on the adversarial DIMACS suite."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from furones import algorithm, parser


def exact_mds_milp(graph: nx.Graph) -> tuple[set[int], float]:
    nodes = sorted(graph.nodes())
    index = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    for v in nodes:
        row = np.zeros(n)
        row[index[v]] = 1.0
        for u in graph.neighbors(v):
            row[index[u]] = 1.0
        rows.append(row)
        lower.append(1.0)
        upper.append(np.inf)

    start = time.perf_counter()
    result = milp(
        c=np.ones(n),
        integrality=np.ones(n),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(np.vstack(rows), np.array(lower), np.array(upper)),
        options={"time_limit": 60},
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if not result.success:
        raise RuntimeError(f"MILP failed on {graph}: {result.message}")

    x = np.rint(result.x).astype(int)
    solution = {nodes[i] for i, bit in enumerate(x) if bit == 1}
    if not nx.dominating.is_dominating_set(graph, solution):
        raise RuntimeError("MILP solution failed domination verification")
    return solution, elapsed_ms


def main() -> None:
    base = Path(__file__).resolve().parent
    for path in sorted(base.glob("adv_*.dimacs")):
        graph = parser.read(path)
        delta = max(dict(graph.degree()).values()) if graph.number_of_nodes() else 0

        start = time.perf_counter()
        furones = algorithm.find_dominating_set(graph)
        furones_ms = (time.perf_counter() - start) * 1000.0
        if not nx.dominating.is_dominating_set(graph, furones):
            raise RuntimeError(f"Furones solution failed verification on {path.name}")

        optimum, milp_ms = exact_mds_milp(graph)
        ratio = len(furones) / len(optimum)
        print(
            f"{path.name}: n={graph.number_of_nodes()} m={graph.number_of_edges()} "
            f"Delta={delta} Furones={len(furones)} Opt={len(optimum)} "
            f"ratio={ratio:.3f} Furones_ms={furones_ms:.3f} MILP_ms={milp_ms:.3f}"
        )


if __name__ == "__main__":
    main()
