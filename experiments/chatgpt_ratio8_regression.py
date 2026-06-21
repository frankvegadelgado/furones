"""
ChatGPT-assisted targeted adversarial regression for Furones v0.3.3.

This short script checks the deterministic planted-pair/decoy-booster family
that previously reached ratios above 4.  It is a targeted regression only, not
an exhaustive benchmark and not a proof of a universal approximation ratio.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import networkx as nx

from furones.algorithm import find_dominating_set


def deterministic_planted_pair_decoy_graph(q: int, g: int = 5) -> nx.Graph:
    """Build the deterministic planted-pair decoy-booster regression graph."""
    boosters = 5 * q - 9
    G = nx.Graph()
    P = ["p0", "p1"]
    D = [f"d{i}" for i in range(q)]
    B = [f"b{i}" for i in range(boosters)]
    selector = P + D

    G.add_nodes_from(selector)
    G.add_nodes_from(B)

    # Selector clique on p0, p1, and all decoys.
    for i, u in enumerate(selector):
        for v in selector[i + 1:]:
            G.add_edge(u, v)

    # Boosters favour p0 and the decoys, but not p1.
    for b in B:
        G.add_edge("p0", b)
        for d in D:
            G.add_edge(d, b)

    # Each decoy has two private cyclic element groups, one shared with p0 and
    # one shared with p1.  The cycles avoid making these degree-1/2 witnesses.
    for i, d in enumerate(D):
        E0 = [f"e0_{i}_{j}" for j in range(g)]
        E1 = [f"e1_{i}_{j}" for j in range(g)]
        G.add_nodes_from(E0 + E1)
        for group, planted in ((E0, "p0"), (E1, "p1")):
            for x in group:
                G.add_edge(planted, x)
                G.add_edge(d, x)
            for j, x in enumerate(group):
                G.add_edge(x, group[(j + 1) % g])

    return G


def sorted_labels(values: Iterable[Any]) -> List[str]:
    return sorted(str(v) for v in values)


def run_case(q: int) -> Dict[str, Any]:
    G = deterministic_planted_pair_decoy_graph(q=q)
    D = set(find_dominating_set(G))
    return {
        "name": f"planted_pair_decoy_booster_q{q}",
        "q": q,
        "n": G.number_of_nodes(),
        "m": G.number_of_edges(),
        "exact_opt": 2,
        "output_size": len(D),
        "ratio": len(D) / 2,
        "valid": bool(nx.is_dominating_set(G, D)),
        "output": sorted_labels(D),
    }


def main() -> None:
    cases = [run_case(q) for q in (8, 9, 10, 12, 16)]
    smoke_graphs = [("path_graph_12", nx.path_graph(12)), ("cycle_graph_12", nx.cycle_graph(12))]
    for name, G in smoke_graphs:
        D = set(find_dominating_set(G))
        cases.append({
            "name": name,
            "family": "smoke",
            "n": G.number_of_nodes(),
            "m": G.number_of_edges(),
            "output_size": len(D),
            "valid": bool(nx.is_dominating_set(G, D)),
            "output": sorted_labels(D),
        })

    ratios = [row["ratio"] for row in cases if "ratio" in row]
    result = {
        "metadata": {
            "experiment": "ChatGPT-assisted targeted ratio-above-4 regression",
            "solver_version": "0.3.3",
            "long_exhaustive_battery_rerun": False,
            "notes": [
                "This is a targeted regression only, not a universal approximation proof.",
                "For the deterministic planted-pair decoy-booster rows, OPT=2 by construction: {p0,p1} dominates, and no single vertex is universal.",
                "The v0.3.3 seed-and-complete candidate is a general constant-seed residual-coverage heuristic, not a detector for this family."
            ],
        },
        "summary": {
            "instances": len(cases),
            "adversarial_instances": 5,
            "all_valid": all(row["valid"] for row in cases),
            "max_adversarial_ratio": max(ratios),
            "reported_ratio_above_4_repaired": all(row.get("output_size") == 2 for row in cases if "ratio" in row),
        },
        "cases": cases,
    }
    out_path = Path(__file__).with_name("chatgpt_ratio8_regression.json")
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
