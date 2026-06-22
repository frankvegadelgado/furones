"""
ChatGPT-assisted targeted random set-cover regression for Furones v0.3.4.

This script checks the reported random planted-pair set-cover stress tests that
previously reached ratios 4.0--5.0. It is intentionally targeted and fast, not
an exhaustive benchmark and not a proof of a universal approximation ratio.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from furones.algorithm import find_dominating_set


def random_set_cover_planted_pair_graph(q: int, b: int, p: float, seed: int) -> nx.Graph:
    """Build the random set-cover-style planted-pair regression graph."""
    rng = random.Random(seed)
    decoys = [f"d{i}" for i in range(q)]
    planted = ["p0", "p1"]
    e0 = [f"e0_{i}" for i in range(b)]
    e1 = [f"e1_{i}" for i in range(b)]

    # Preserve the adversarial node order from the prompt: decoys first, then
    # planted vertices, then elements.
    G = nx.Graph()
    G.add_nodes_from(decoys + planted + e0 + e1)

    selector = planted + decoys
    for i, u in enumerate(selector):
        for v in selector[i + 1:]:
            G.add_edge(u, v)

    for x in e0:
        G.add_edge("p0", x)
    for x in e1:
        G.add_edge("p1", x)

    elements = e0 + e1
    for d in decoys:
        for x in elements:
            if rng.random() < p:
                G.add_edge(d, x)

    return G


def sorted_labels(values: Iterable[Any]) -> List[str]:
    return sorted(str(v) for v in values)


def run_random_case(q: int, b: int, p: float, seed: int) -> Dict[str, Any]:
    G = random_set_cover_planted_pair_graph(q=q, b=b, p=p, seed=seed)
    D = set(find_dominating_set(G))
    return {
        "name": f"random_set_cover_q{q}_b{b}_p{p}_seed{seed}",
        "family": "random_set_cover_planted_pair",
        "q": q,
        "b": b,
        "p": p,
        "seed": seed,
        "n": G.number_of_nodes(),
        "m": G.number_of_edges(),
        "exact_opt": 2,
        "output_size": len(D),
        "ratio": len(D) / 2,
        "valid": bool(nx.is_dominating_set(G, D)),
        "output": sorted_labels(D),
    }


def main() -> None:
    cases = [
        run_random_case(40, 800, 0.52, 0),
        run_random_case(40, 1600, 0.52, 0),
        run_random_case(60, 1600, 0.50, 0),
        run_random_case(80, 1600, 0.50, 0),
        run_random_case(80, 2400, 0.50, 0),
        run_random_case(80, 2400, 0.50, 2),
    ]

    ratios = [row["ratio"] for row in cases]
    result = {
        "metadata": {
            "experiment": "ChatGPT-assisted targeted random set-cover ratio-5 regression",
            "solver_version": "0.3.4",
            "long_exhaustive_battery_rerun": False,
            "notes": [
                "This is a targeted regression only, not a universal approximation proof.",
                "For each row OPT=2 by construction: {p0,p1} dominates, and no single vertex is universal.",
                "The repair increases the constant seed window in the general seed-and-complete heuristic from 32 to 64.",
            ],
        },
        "summary": {
            "instances": len(cases),
            "all_valid": all(row["valid"] for row in cases),
            "max_ratio": max(ratios),
            "reported_ratio_5_repaired": all(row["output_size"] == 2 for row in cases),
        },
        "cases": cases,
    }

    out_path = Path(__file__).with_name("chatgpt_ratio5_regression.json")
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
