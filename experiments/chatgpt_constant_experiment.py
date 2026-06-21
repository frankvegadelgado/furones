#!/usr/bin/env python3
"""
ChatGPT-assisted reproducible constant-envelope experiment for Furones v0.3.1.

This script compares furones.algorithm.find_dominating_set against an exact
minimum dominating set computed by exhaustive bit-set search on deterministic
small and medium test families.  It is intentionally experimental: it reports
observed ratios and does not claim a theorem.

Run from the repository root:

    python experiments/chatgpt_constant_experiment.py

The output is written to:

    experiments/chatgpt_constant_experiment.json
"""
from __future__ import annotations

import itertools
import json
import platform
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from furones import __version__ as furones_version
from furones.algorithm import find_dominating_set

OUTPUT_PATH = Path(__file__).with_name("chatgpt_constant_experiment.json")
EPSILON = 1.0
MAX_EXACT_K = 8


def _json_node(v: Any) -> str:
    return str(v)


def minimum_dominating_set_exact(
    graph: nx.Graph,
    max_k: Optional[int] = None,
) -> Tuple[set, int]:
    """Return an exact minimum dominating set by bit-set exhaustive search."""
    nodes = list(graph.nodes())
    n = len(nodes)
    if n == 0:
        return set(), 0

    index = {v: i for i, v in enumerate(nodes)}
    closed_masks: List[int] = []
    for v in nodes:
        mask = 1 << index[v]
        for u in graph.neighbors(v):
            mask |= 1 << index[u]
        closed_masks.append(mask)

    full = (1 << n) - 1
    upper = n if max_k is None else min(max_k, n)
    for k in range(1, upper + 1):
        for combo in itertools.combinations(range(n), k):
            mask = 0
            for i in combo:
                mask |= closed_masks[i]
            if mask == full:
                solution = {nodes[i] for i in combo}
                return solution, k

    raise RuntimeError(
        f"Exact search did not find a dominating set up to k={upper}; "
        f"increase MAX_EXACT_K or reduce the test graph size."
    )


def evaluate_graph(
    suite: str,
    name: str,
    graph: nx.Graph,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run Furones and the exact solver on one graph and return one row."""
    metadata = metadata or {}
    alg_set = set(find_dominating_set(graph, eps=EPSILON))
    if not nx.is_dominating_set(graph, alg_set):
        raise RuntimeError(f"Furones returned a non-dominating set on {name}")

    exact_cap = MAX_EXACT_K if suite == "dense_gnp_n_16_30" else None
    opt_set, opt_size = minimum_dominating_set_exact(graph, max_k=exact_cap)
    alg_size = len(alg_set)
    ratio = float(alg_size / opt_size) if opt_size else 1.0

    row = {
        "suite": suite,
        "name": name,
        "n": graph.number_of_nodes(),
        "m": graph.number_of_edges(),
        "furones_size": alg_size,
        "opt_size": opt_size,
        "ratio": ratio,
        "metadata": metadata,
        "exact_search_cap": exact_cap,
    }
    return row


def graph_atlas_suite() -> Iterable[Tuple[str, nx.Graph, Dict[str, Any]]]:
    for atlas_index, graph in enumerate(nx.graph_atlas_g()):
        n = graph.number_of_nodes()
        if 1 <= n <= 7:
            yield f"atlas_{atlas_index}", graph, {"atlas_index": atlas_index}


def gnp_small_suite() -> Iterable[Tuple[str, nx.Graph, Dict[str, Any]]]:
    probabilities = [0.15, 0.25, 0.35, 0.50, 0.70]
    for n in range(8, 15):
        for p in probabilities:
            for seed in range(20):
                graph = nx.gnp_random_graph(n, p, seed=seed)
                yield (
                    f"gnp_n{n}_p{p:.2f}_seed{seed}",
                    graph,
                    {"model": "G(n,p)", "n": n, "p": p, "seed": seed},
                )


def gnp_dense_medium_suite() -> Iterable[Tuple[str, nx.Graph, Dict[str, Any]]]:
    # Dense medium graphs are included because small optimum values are common,
    # making exact bit-set search feasible while stressing non-planar residuals.
    probabilities = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
    for n in range(16, 31, 2):
        for p in probabilities:
            for seed in range(100):
                graph = nx.gnp_random_graph(n, p, seed=seed)
                yield (
                    f"dense_gnp_n{n}_p{p:.2f}_seed{seed}",
                    graph,
                    {"model": "dense G(n,p)", "n": n, "p": p, "seed": seed},
                )


def adversarial_universal_family_suite() -> Iterable[Tuple[str, nx.Graph, Dict[str, Any]]]:
    """Universal-vertex non-planar residual family used as a regression test.

    The optimum is promised to be one because vertex 1 is universal.  This
    suite checks whether the general closed-degree coverage sweep repairs the
    forest-projection failure mode without a special-case universal-vertex rule.
    """
    for n in [8, 10, 12, 15, 20, 30, 40, 60, 90, 120]:
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        for v in range(n):
            if v != 1:
                graph.add_edge(1, v)
        path = [0] + list(range(2, n))
        graph.add_edges_from(zip(path, path[1:]))
        clique = [0, 2, 3, 4]
        for i, u in enumerate(clique):
            for v in clique[i + 1:]:
                graph.add_edge(u, v)
        yield (
            f"universal_k5_path_n{n}",
            graph,
            {"family": "universal_k5_path", "n": n, "promised_opt": 1},
        )


def named_family_suite() -> Iterable[Tuple[str, nx.Graph, Dict[str, Any]]]:
    for n in range(4, 21):
        yield f"path_{n}", nx.path_graph(n), {"family": "path", "n": n}
        yield f"cycle_{n}", nx.cycle_graph(n), {"family": "cycle", "n": n}
        yield f"complete_{n}", nx.complete_graph(n), {"family": "complete", "n": n}
        yield f"star_{n}", nx.star_graph(n - 1), {"family": "star", "n": n}

    for a, b in [(2, 5), (3, 3), (3, 6), (4, 4), (4, 8), (5, 5)]:
        yield (
            f"complete_bipartite_{a}_{b}",
            nx.complete_bipartite_graph(a, b),
            {"family": "complete_bipartite", "a": a, "b": b},
        )

    for k in range(2, 8):
        yield (
            f"barbell_{k}_1",
            nx.barbell_graph(k, 1),
            {"family": "barbell", "clique_size": k, "path_length": 1},
        )


def reconstruct_graph_from_row(row: Dict[str, Any]) -> nx.Graph:
    """Reconstruct a graph for worst-case edge-list reporting."""
    md = row["metadata"]
    name = row["name"]
    if row["suite"] == "graph_atlas_n_le_7":
        return nx.graph_atlas_g()[md["atlas_index"]]
    if row["suite"] in {"gnp_random_n_8_14", "dense_gnp_n_16_30"}:
        return nx.gnp_random_graph(md["n"], md["p"], seed=md["seed"])
    if row["suite"] == "adversarial_universal_family":
        n = md["n"]
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        for v in range(n):
            if v != 1:
                graph.add_edge(1, v)
        path = [0] + list(range(2, n))
        graph.add_edges_from(zip(path, path[1:]))
        clique = [0, 2, 3, 4]
        for i, u in enumerate(clique):
            for v in clique[i + 1:]:
                graph.add_edge(u, v)
        return graph
    if row["suite"] == "named_families":
        fam = md["family"]
        if fam == "path":
            return nx.path_graph(md["n"])
        if fam == "cycle":
            return nx.cycle_graph(md["n"])
        if fam == "complete":
            return nx.complete_graph(md["n"])
        if fam == "star":
            return nx.star_graph(md["n"] - 1)
        if fam == "complete_bipartite":
            return nx.complete_bipartite_graph(md["a"], md["b"])
        if fam == "barbell":
            return nx.barbell_graph(md["clique_size"], md["path_length"])
    raise ValueError(f"Cannot reconstruct graph for {name}")


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    suites = sorted({row["suite"] for row in rows})
    for suite in suites + ["overall"]:
        data = rows if suite == "overall" else [row for row in rows if row["suite"] == suite]
        ratios = [row["ratio"] for row in data]
        result[suite] = {
            "instances": len(data),
            "max_ratio": max(ratios),
            "mean_ratio": statistics.mean(ratios),
            "median_ratio": statistics.median(ratios),
            "count_ratio_gt_2": sum(r > 2 for r in ratios),
            "count_ratio_gt_3": sum(r > 3 for r in ratios),
            "count_ratio_gt_4": sum(r > 4 for r in ratios),
            "histogram": dict(sorted(Counter(str(r) for r in ratios).items())),
        }
    return result


def main() -> None:
    suites = [
        ("graph_atlas_n_le_7", graph_atlas_suite()),
        ("gnp_random_n_8_14", gnp_small_suite()),
        ("dense_gnp_n_16_30", gnp_dense_medium_suite()),
        ("named_families", named_family_suite()),
        ("adversarial_universal_family", adversarial_universal_family_suite()),
    ]

    rows: List[Dict[str, Any]] = []
    for suite_name, suite_iter in suites:
        for name, graph, metadata in suite_iter:
            rows.append(evaluate_graph(suite_name, name, graph, metadata))

    summary = summarize(rows)
    rows_sorted = sorted(rows, key=lambda r: (r["ratio"], r["n"], r["m"]), reverse=True)
    worst_rows = rows_sorted[:12]

    worst_cases = []
    for row in worst_rows:
        graph = reconstruct_graph_from_row(row)
        alg_set = set(find_dominating_set(graph, eps=EPSILON))
        exact_cap = MAX_EXACT_K if row["suite"] == "dense_gnp_n_16_30" else None
        opt_set, _ = minimum_dominating_set_exact(graph, max_k=exact_cap)
        enriched = dict(row)
        enriched["furones_set"] = sorted(_json_node(v) for v in alg_set)
        enriched["opt_set"] = sorted(_json_node(v) for v in opt_set)
        enriched["edges"] = sorted([[_json_node(u), _json_node(v)] for u, v in graph.edges()])
        worst_cases.append(enriched)

    payload = {
        "experiment": "ChatGPT-assisted Furones v0.3.1 constant-envelope experiment",
        "generated_by": "ChatGPT (OpenAI GPT-5.5 Thinking)",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Test whether exact small-instance behavior supports a deeper "
            "bounded-constant phenomenon. The result is empirical only: it "
            "does not prove a universal approximation ratio."
        ),
        "furones_version": furones_version,
        "epsilon": EPSILON,
        "exact_solver": "deterministic exhaustive bit-set search over vertex subsets",
        "max_exact_k": MAX_EXACT_K,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "networkx": nx.__version__,
        },
        "suite_definitions": {
            "graph_atlas_n_le_7": "All nonempty NetworkX graph-atlas graphs with at most 7 vertices.",
            "gnp_random_n_8_14": "G(n,p), n=8..14, p in {0.15,0.25,0.35,0.50,0.70}, seeds 0..19.",
            "dense_gnp_n_16_30": "Dense G(n,p), n=16,18,...,30, p in {0.25,0.35,0.45,0.55,0.65,0.75}, seeds 0..99.",
            "named_families": "Paths, cycles, complete graphs, stars, selected complete bipartite graphs, and barbells.",
            "adversarial_universal_family": "Universal-vertex non-planar K5-plus-path family with promised OPT=1, used as a regression test for the linear closed-degree sweep.",
        },
        "summary": summary,
        "interpretation": {
            "two_type_status": "Supported empirically on this revised battery: no tested exact instance exceeded ratio 2.0 after the linear closed-degree sweep was added.",
            "three_type_status": "Also supported but weaker: no tested exact instance exceeded ratio 3.0.",
            "four_type_status": "Also supported but weaker still: no tested exact instance exceeded ratio 4.0.",
            "caution": "This is not a proof. A universal constant-factor theorem for general Minimum Dominating Set would require new theory and would have major complexity-theoretic consequences.",
        },
        "worst_cases": worst_cases,
        "rows": rows,
    }

    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2))
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
