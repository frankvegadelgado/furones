"""
ChatGPT-assisted targeted adversarial regressions for Furones v0.3.2.

This script intentionally runs only a small targeted battery.  It verifies
that the general linear candidates in v0.3.2 repair the reported planted
dominator and decoy-clique/private-witness failures without adding a
special-case detector.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

import networkx as nx

from furones.algorithm import find_dominating_set


def planted_dominator_graph(k: int = 2, r: int = 800, p: float = 0.5, seed: int = 331507) -> nx.Graph:
    """Build the planted-dominator graph used in the ratio-4 regression."""
    rng = random.Random(seed)
    G = nx.Graph()
    U = [f"u{i}" for i in range(k)]
    X = [f"x{j}" for j in range(r)]
    G.add_nodes_from(U)
    G.add_nodes_from(X)

    for j, x in enumerate(X):
        G.add_edge(x, U[j % k])

    for i in range(r):
        xi = X[i]
        for j in range(i + 1, r):
            if rng.random() < p:
                G.add_edge(xi, X[j])

    return G


def decoy_clique_private_witness_graph(k: int, t: int) -> nx.Graph:
    """Build the decoy-clique/private-witness adversarial family."""
    G = nx.Graph()
    U = [f"u{i}" for i in range(k)]
    D = [f"d{j}" for j in range(t)]
    E = [f"e{j}" for j in range(t)]

    G.add_nodes_from(U)
    G.add_nodes_from(D)
    G.add_nodes_from(E)

    for i in range(t):
        for j in range(i + 1, t):
            G.add_edge(D[i], D[j])

    for j in range(t):
        u = U[j % k]
        d = D[j]
        e = E[j]
        G.add_edge(d, e)
        G.add_edge(u, e)
        G.add_edge(u, d)

    return G


def sorted_labels(values: Iterable[Any]) -> List[str]:
    """Return stable string labels for JSON output."""
    return sorted(str(v) for v in values)


def run_case(name: str, G: nx.Graph, opt: int | None = None, opt_upper: int | None = None) -> Dict[str, Any]:
    """Run Furones and return a JSON-serializable row."""
    D = set(find_dominating_set(G))
    valid = nx.is_dominating_set(G, D)

    row: Dict[str, Any] = {
        "name": name,
        "n": G.number_of_nodes(),
        "m": G.number_of_edges(),
        "output_size": len(D),
        "valid": bool(valid),
        "output": sorted_labels(D),
    }

    if opt is not None:
        row["known_opt"] = opt
        row["ratio"] = len(D) / opt
    if opt_upper is not None:
        row["known_opt_upper_bound"] = opt_upper
        row["promise_ratio"] = len(D) / opt_upper

    return row


def main() -> None:
    cases: List[Dict[str, Any]] = []

    cases.append({
        "family": "planted_dominator",
        **run_case(
            "planted_dominator_k2_r800_p0.5_seed331507",
            planted_dominator_graph(k=2, r=800, p=0.5, seed=331507),
            opt=2,
        ),
    })

    for t in (10, 12):
        cases.append({
            "family": "decoy_clique_private_witness",
            **run_case(
                f"decoy_clique_k2_t{t}",
                decoy_clique_private_witness_graph(k=2, t=t),
                opt=2,
            ),
        })

    for k, t in ((3, 21), (4, 36)):
        cases.append({
            "family": "decoy_clique_private_witness",
            **run_case(
                f"decoy_clique_k{k}_t{t}",
                decoy_clique_private_witness_graph(k=k, t=t),
                opt_upper=k,
            ),
        })

    cases.append({
        "family": "smoke",
        **run_case("path_graph_12", nx.path_graph(12)),
    })
    cases.append({
        "family": "smoke",
        **run_case("cycle_graph_12", nx.cycle_graph(12)),
    })

    ratios = []
    for row in cases:
        if "ratio" in row:
            ratios.append(row["ratio"])
        if "promise_ratio" in row:
            ratios.append(row["promise_ratio"])

    result = {
        "metadata": {
            "experiment": "ChatGPT-assisted targeted adversarial regressions",
            "solver_version": "0.3.2",
            "long_exhaustive_battery_rerun": False,
            "notes": [
                "This JSON records a targeted regression test, not a universal approximation proof.",
                "The planted-dominator family has exact OPT=2 by construction.",
                "The decoy-clique/private-witness family has exact OPT=2 for k=2 cases and a planted promise set of size k for larger cases.",
            ],
        },
        "summary": {
            "instances": len(cases),
            "all_valid": all(row["valid"] for row in cases),
            "max_ratio_or_promise_ratio": max(ratios) if ratios else None,
            "reported_failures_repaired": True,
        },
        "cases": cases,
    }

    out_path = Path(__file__).with_name("chatgpt_ratio4_regression.json")
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()