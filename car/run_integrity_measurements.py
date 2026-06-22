"""Generate Furones v0.3.4 CAR-style exact benchmark measurements.

This script imports the current Furones source tree, builds a deterministic
1000-instance benchmark of small graphs, computes the exact domination number
for every instance by exhaustive search, compares Furones against the optimum,
and runs each exposed strategy as a candidate ablation.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

FURONES_ROOT = Path(os.environ.get("FURONES_ROOT", r"D:\Work\NP\furones"))
if str(FURONES_ROOT) not in sys.path:
    sys.path.insert(0, str(FURONES_ROOT))

import networkx as nx

from furones import __version__ as FURONES_VERSION
from furones import baker_algo, tscc_ds_reduction
from furones.algorithm import (
    find_dominating_set,
    greedy_closed_degree_dominating_set,
    low_degree_witness_dominating_set,
    medium_degree_witness_dominating_set,
    order_ownership_witness_dominating_set,
    prune_redundant_vertices_dominating,
    reverse_delete_dominating_set,
    salvador_planar_bipartite_baker_candidate,
    seed_and_complete_dominating_set,
)


OUT_DIR = Path(__file__).resolve().parent
BENCHMARK_SEED = 20260622
BENCHMARK_INSTANCE_COUNT = 1000
MAX_EXACT_N = 14


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    family: str
    graph: nx.Graph
    reference_size: int
    reference_type: str
    reference_note: str
    exact_solution: frozenset[Any]


def stable_node_list(nodes: set[Any] | frozenset[Any]) -> list[str]:
    return [str(v) for v in sorted(nodes, key=lambda value: repr(value))]


def exact_minimum_dominating_set(G: nx.Graph) -> set[Any]:
    """Return an exact minimum dominating set for a small graph."""
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return set()
    if n > MAX_EXACT_N:
        raise ValueError(f"exact benchmark graph has {n} nodes; limit is {MAX_EXACT_N}")

    index = {v: i for i, v in enumerate(nodes)}
    neighborhoods: list[int] = []
    for v in nodes:
        mask = 1 << index[v]
        for u in G.neighbors(v):
            mask |= 1 << index[u]
        neighborhoods.append(mask)

    full = (1 << n) - 1
    for k in range(1, n + 1):
        for combo in itertools.combinations(range(n), k):
            dominated = 0
            for idx in combo:
                dominated |= neighborhoods[idx]
                if dominated == full:
                    return {nodes[i] for i in combo}
    return set(nodes)


def random_tree_graph(n: int, rng: random.Random) -> nx.Graph:
    """Build a random labelled tree from a Prufer sequence."""
    if n <= 1:
        G = nx.Graph()
        G.add_node(0)
        return G
    sequence = [rng.randrange(n) for _ in range(n - 2)]
    degree = [1] * n
    for v in sequence:
        degree[v] += 1
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for v in sequence:
        leaf = next(i for i, d in enumerate(degree) if d == 1)
        G.add_edge(leaf, v)
        degree[leaf] -= 1
        degree[v] -= 1
    remaining = [i for i, d in enumerate(degree) if d == 1]
    G.add_edge(remaining[0], remaining[1])
    return G


def random_bipartite_graph(a: int, b: int, p: float, rng: random.Random) -> nx.Graph:
    G = nx.Graph()
    left = [f"l{i}" for i in range(a)]
    right = [f"r{i}" for i in range(b)]
    G.add_nodes_from(left + right)
    for u in left:
        for v in right:
            if rng.random() < p:
                G.add_edge(u, v)
    return G


def perturbed_path_or_cycle(i: int, rng: random.Random) -> nx.Graph:
    n = 6 + (i % 9)
    G = nx.path_graph(n) if i % 2 == 0 else nx.cycle_graph(n)
    extra_edges = i % 3
    for _ in range(extra_edges):
        u, v = rng.sample(range(n), 2)
        if abs(u - v) > 1 and not (set((u, v)) == {0, n - 1} and i % 2 == 1):
            G.add_edge(u, v)
    if i % 10 == 0 and n < MAX_EXACT_N:
        G.add_node(n)
        G.add_edge(n, rng.randrange(n))
    return G


def small_structured_graph(i: int) -> nx.Graph:
    kind = i % 5
    n = 6 + ((i // 5) % 9)
    if kind == 0:
        return nx.wheel_graph(max(6, n))
    if kind == 1:
        a = 3 + (i % 4)
        b = 3 + ((i // 4) % 5)
        return nx.complete_bipartite_graph(a, b)
    if kind == 2:
        rungs = 3 + (i % 4)
        return nx.ladder_graph(rungs)
    if kind == 3:
        m1 = 3 + (i % 3)
        m2 = i % 4
        return nx.barbell_graph(m1, m2)

    G = nx.path_graph(n)
    hub = 0
    for v in range(2, n, 3):
        G.add_edge(hub, v)
    if n > 8:
        G.add_edge(2, n - 1)
    return G


def add_exact_case(
    cases: list[BenchmarkCase],
    name: str,
    family: str,
    graph: nx.Graph,
    note: str,
) -> None:
    graph = nx.Graph(graph)
    graph.remove_edges_from(list(nx.selfloop_edges(graph)))
    if graph.number_of_nodes() > MAX_EXACT_N:
        raise ValueError(f"{name} has {graph.number_of_nodes()} nodes")
    exact_solution = frozenset(exact_minimum_dominating_set(graph))
    cases.append(
        BenchmarkCase(
            name=name,
            family=family,
            graph=graph,
            reference_size=len(exact_solution),
            reference_type="exact_exhaustive",
            reference_note=note,
            exact_solution=exact_solution,
        )
    )


def benchmark_cases() -> list[BenchmarkCase]:
    rng = random.Random(BENCHMARK_SEED)
    cases: list[BenchmarkCase] = []

    for i in range(200):
        n = 6 + (i % 9)
        p = 0.08 + 0.02 * (i % 5)
        seed = rng.randrange(10**9)
        add_exact_case(
            cases,
            f"er_sparse_{i:03d}",
            "erdos_renyi_sparse",
            nx.gnp_random_graph(n, p, seed=seed),
            f"G(n,p) with n={n}, p={p:.2f}, seed={seed}.",
        )

    for i in range(200):
        n = 6 + (i % 9)
        p = 0.24 + 0.03 * (i % 5)
        seed = rng.randrange(10**9)
        add_exact_case(
            cases,
            f"er_medium_{i:03d}",
            "erdos_renyi_medium",
            nx.gnp_random_graph(n, p, seed=seed),
            f"G(n,p) with n={n}, p={p:.2f}, seed={seed}.",
        )

    for i in range(150):
        n = 6 + (i % 9)
        p = 0.48 + 0.04 * (i % 4)
        seed = rng.randrange(10**9)
        add_exact_case(
            cases,
            f"er_dense_{i:03d}",
            "erdos_renyi_dense",
            nx.gnp_random_graph(n, p, seed=seed),
            f"G(n,p) with n={n}, p={p:.2f}, seed={seed}.",
        )

    for i in range(150):
        a = 3 + (i % 5)
        b = 3 + ((i // 5) % 5)
        p = 0.22 + 0.05 * (i % 6)
        add_exact_case(
            cases,
            f"bipartite_{i:03d}",
            "random_bipartite",
            random_bipartite_graph(a, b, p, rng),
            f"Random bipartite graph with parts {a},{b} and p={p:.2f}.",
        )

    for i in range(100):
        n = 6 + (i % 9)
        add_exact_case(
            cases,
            f"tree_{i:03d}",
            "random_tree",
            random_tree_graph(n, rng),
            f"Random Prufer tree with n={n}.",
        )

    for i in range(100):
        add_exact_case(
            cases,
            f"path_cycle_{i:03d}",
            "perturbed_path_cycle",
            perturbed_path_or_cycle(i, rng),
            "Path/cycle family with deterministic small perturbations.",
        )

    for i in range(100):
        add_exact_case(
            cases,
            f"structured_{i:03d}",
            "small_structured",
            small_structured_graph(i),
            "Small structured graph: wheel, complete bipartite, ladder, barbell, or chorded path.",
        )

    if len(cases) != BENCHMARK_INSTANCE_COUNT:
        raise AssertionError(f"expected {BENCHMARK_INSTANCE_COUNT} cases, got {len(cases)}")
    return cases


def cleaned_graph(G: nx.Graph) -> tuple[nx.Graph, set[Any]]:
    H = G.copy()
    H.remove_edges_from(list(nx.selfloop_edges(H)))
    isolates = set(nx.isolates(H))
    H.remove_nodes_from(isolates)
    return H, isolates


def lifted_tscc_baker_candidate(H: nx.Graph, eps: float = 1.0) -> set[Any]:
    if H.number_of_nodes() == 0:
        return set()
    reduced, forced, lift = tscc_ds_reduction.reduce_to_tscc_for_ds(H)
    if reduced.number_of_nodes() == 0:
        return prune_redundant_vertices_dominating(H, set(forced))
    mapping = {u: i for i, u in enumerate(reduced.nodes())}
    unmapping = {i: u for u, i in mapping.items()}
    reduced_for_baker = baker_algo.Graph(reduced.number_of_nodes())
    for u, v in reduced.edges():
        reduced_for_baker.add_edge(mapping[u], mapping[v])
    reduced_solution = baker_algo.baker_ptas(reduced_for_baker, eps, verbose=False)
    lifted = lift({unmapping[u] for u in reduced_solution})
    return prune_redundant_vertices_dominating(H, set(lifted))


CandidateFn = Callable[[nx.Graph], set[Any]]


STRATEGIES: list[tuple[str, CandidateFn]] = [
    ("closed_degree", greedy_closed_degree_dominating_set),
    ("low_degree_witness", low_degree_witness_dominating_set),
    ("medium_degree_witness", medium_degree_witness_dominating_set),
    ("order_ownership_late", lambda G: order_ownership_witness_dominating_set(G, "late")),
    ("order_ownership_early", lambda G: order_ownership_witness_dominating_set(G, "early")),
    ("seed_and_complete", seed_and_complete_dominating_set),
    ("salvador_auxiliary", salvador_planar_bipartite_baker_candidate),
    ("reverse_delete_input", lambda G: reverse_delete_dominating_set(G, "input")),
    ("reverse_delete_reverse", lambda G: reverse_delete_dominating_set(G, "reverse_input")),
    ("reverse_delete_high_degree", lambda G: reverse_delete_dominating_set(G, "high_degree")),
    ("reverse_delete_low_degree", lambda G: reverse_delete_dominating_set(G, "low_degree")),
    ("lifted_tscc_baker", lifted_tscc_baker_candidate),
]


def has_universal_vertex(G: nx.Graph) -> bool:
    n = G.number_of_nodes()
    return any(G.degree(v) == n - 1 for v in G.nodes())


def finalize_candidate(
    original: nx.Graph, working: nx.Graph, isolates: set[Any], candidate: set[Any]
) -> tuple[set[Any], bool]:
    if working.number_of_nodes() == 0:
        final = set(isolates)
    else:
        candidate = {v for v in set(candidate) if v in working}
        candidate = prune_redundant_vertices_dominating(working, candidate)
        final = candidate | set(isolates)
    return final, bool(nx.is_dominating_set(original, final))


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[int(pos)]
    return ordered[lower] * (upper - pos) + ordered[upper] * (pos - lower)


def run_measurements() -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    cases = benchmark_cases()
    instance_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []

    for case in cases:
        G = case.graph
        H, isolates = cleaned_graph(G)
        case_strategy_rows: list[dict[str, Any]] = []

        start = time.perf_counter()
        final_solution = set(find_dominating_set(G))
        final_seconds = time.perf_counter() - start
        final_valid = bool(nx.is_dominating_set(G, final_solution))
        final_ratio = len(final_solution) / case.reference_size if final_valid else None
        final_optimal = bool(final_valid and len(final_solution) == case.reference_size)

        for strategy_name, strategy_fn in STRATEGIES:
            start = time.perf_counter()
            error = None
            try:
                raw_candidate = set(strategy_fn(H))
                candidate, valid = finalize_candidate(G, H, isolates, raw_candidate)
            except Exception as exc:  # pragma: no cover - report path
                candidate = set()
                valid = False
                error = f"{type(exc).__name__}: {exc}"
            seconds = time.perf_counter() - start
            ratio = len(candidate) / case.reference_size if valid else None
            row = {
                "case": case.name,
                "family": case.family,
                "strategy": strategy_name,
                "n": G.number_of_nodes(),
                "m": G.number_of_edges(),
                "optimum_size": case.reference_size,
                "reference_type": case.reference_type,
                "candidate_size": len(candidate),
                "valid": valid,
                "ratio_to_optimum": ratio,
                "seconds": seconds,
                "error": error,
            }
            case_strategy_rows.append(row)

        valid_candidate_sizes = [
            row["candidate_size"] for row in case_strategy_rows if row["valid"]
        ]
        min_valid_size = min(valid_candidate_sizes) if valid_candidate_sizes else None
        for row in case_strategy_rows:
            row["min_valid_candidate_size"] = min_valid_size
            row["attains_min_valid_candidate"] = bool(
                row["valid"] and min_valid_size is not None and row["candidate_size"] == min_valid_size
            )
            row["attains_optimum"] = bool(row["valid"] and row["candidate_size"] == case.reference_size)
            instance_rows.append(row)

        case_rows.append(
            {
                "case": case.name,
                "family": case.family,
                "n": G.number_of_nodes(),
                "m": G.number_of_edges(),
                "optimum_size": case.reference_size,
                "reference_type": case.reference_type,
                "reference_note": case.reference_note,
                "exact_solution": stable_node_list(case.exact_solution),
                "has_universal_vertex": has_universal_vertex(G),
                "final_size": len(final_solution),
                "final_valid": final_valid,
                "final_ratio_to_optimum": final_ratio,
                "final_optimal": final_optimal,
                "final_seconds": final_seconds,
                "min_valid_candidate_size": min_valid_size,
                "min_valid_candidate_ratio_to_optimum": (
                    min_valid_size / case.reference_size if min_valid_size is not None else None
                ),
                "min_attaining_strategies": [
                    row["strategy"] for row in case_strategy_rows if row["attains_min_valid_candidate"]
                ],
            }
        )

    summary_rows: list[dict[str, Any]] = []
    total_cases = len(cases)
    for strategy_name, _ in STRATEGIES:
        rows = [row for row in instance_rows if row["strategy"] == strategy_name]
        valid_rows = [row for row in rows if row["valid"]]
        ratios = [
            row["ratio_to_optimum"]
            for row in valid_rows
            if row["ratio_to_optimum"] is not None and math.isfinite(row["ratio_to_optimum"])
        ]
        min_attaining = sum(1 for row in rows if row["attains_min_valid_candidate"])
        optimal = sum(1 for row in rows if row["attains_optimum"])
        summary_rows.append(
            {
                "strategy": strategy_name,
                "instances": total_cases,
                "valid_instances": len(valid_rows),
                "valid_percent": 100.0 * len(valid_rows) / total_cases,
                "optimal_instances": optimal,
                "optimal_percent": 100.0 * optimal / total_cases,
                "min_attaining_instances": min_attaining,
                "min_attaining_percent": 100.0 * min_attaining / total_cases,
                "mean_size_valid": (
                    sum(row["candidate_size"] for row in valid_rows) / len(valid_rows)
                    if valid_rows
                    else None
                ),
                "mean_ratio_to_optimum_valid": mean(ratios),
                "median_ratio_to_optimum_valid": statistics.median(ratios) if ratios else None,
                "p95_ratio_to_optimum_valid": percentile(ratios, 0.95),
                "max_ratio_to_optimum_valid": max(ratios) if ratios else None,
                "total_seconds": sum(row["seconds"] for row in rows),
            }
        )

    family_rows: list[dict[str, Any]] = []
    for family in sorted({row["family"] for row in case_rows}):
        rows = [row for row in case_rows if row["family"] == family]
        ratios = [row["final_ratio_to_optimum"] for row in rows if row["final_ratio_to_optimum"] is not None]
        optimal_count = sum(1 for row in rows if row["final_optimal"])
        family_rows.append(
            {
                "family": family,
                "instances": len(rows),
                "n_min": min(row["n"] for row in rows),
                "n_max": max(row["n"] for row in rows),
                "m_min": min(row["m"] for row in rows),
                "m_max": max(row["m"] for row in rows),
                "optimum_min": min(row["optimum_size"] for row in rows),
                "optimum_max": max(row["optimum_size"] for row in rows),
                "final_valid_percent": 100.0 * sum(1 for row in rows if row["final_valid"]) / len(rows),
                "final_optimal_percent": 100.0 * optimal_count / len(rows),
                "mean_final_ratio_to_optimum": mean(ratios),
                "median_final_ratio_to_optimum": statistics.median(ratios) if ratios else None,
                "max_final_ratio_to_optimum": max(ratios) if ratios else None,
            }
        )

    final_ratios = [
        row["final_ratio_to_optimum"]
        for row in case_rows
        if row["final_valid"] and row["final_ratio_to_optimum"] is not None
    ]
    min_candidate_ratios = [
        row["min_valid_candidate_ratio_to_optimum"]
        for row in case_rows
        if row["min_valid_candidate_ratio_to_optimum"] is not None
    ]
    final_optimal_count = sum(1 for row in case_rows if row["final_optimal"])
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "tooling": "ChatGPT-assisted Codex run; Python script generated by ChatGPT and executed on the local current Furones source tree.",
        "furones_root": str(FURONES_ROOT),
        "furones_version_imported": FURONES_VERSION,
        "benchmark_name": "furones_v0_3_4_car_1000_exact_small_graphs",
        "benchmark_seed": BENCHMARK_SEED,
        "benchmark_scope": "Deterministic 1000-instance benchmark of small graphs with exact optima found by exhaustive search; finite evidence only, not a universal approximation proof.",
        "instances": total_cases,
        "max_exact_nodes": MAX_EXACT_N,
        "strategies": [name for name, _ in STRATEGIES],
        "observed_final_ratio_constant": max(final_ratios) if final_ratios else None,
        "observed_best_candidate_ratio_constant": max(min_candidate_ratios)
        if min_candidate_ratios
        else None,
        "mean_final_ratio_to_optimum": mean(final_ratios),
        "median_final_ratio_to_optimum": statistics.median(final_ratios) if final_ratios else None,
        "p95_final_ratio_to_optimum": percentile(final_ratios, 0.95),
        "final_optimal_instances": final_optimal_count,
        "final_optimal_percent": 100.0 * final_optimal_count / total_cases,
        "all_final_outputs_valid": all(row["final_valid"] for row in case_rows),
    }
    return metadata, case_rows, instance_rows, summary_rows, family_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col)) for col in columns) + " |")
    return "\n".join(lines)


def write_report(
    metadata: dict[str, Any],
    case_rows: list[dict[str, Any]],
    strategy_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
) -> None:
    worst_rows = [
        {
            "case": row["case"],
            "family": row["family"],
            "n": row["n"],
            "m": row["m"],
            "optimum_size": row["optimum_size"],
            "final_size": row["final_size"],
            "final_ratio_to_optimum": row["final_ratio_to_optimum"],
            "min_valid_candidate_size": row["min_valid_candidate_size"],
            "min_attaining_strategies": ", ".join(row["min_attaining_strategies"]),
        }
        for row in sorted(
            case_rows,
            key=lambda r: (
                -(r["final_ratio_to_optimum"] or 0),
                r["case"],
            ),
        )[:20]
    ]
    summary_display = [
        {
            "strategy": row["strategy"],
            "valid_percent": row["valid_percent"],
            "optimal_percent": row["optimal_percent"],
            "min_attaining_percent": row["min_attaining_percent"],
            "mean_ratio_to_optimum_valid": row["mean_ratio_to_optimum_valid"],
            "max_ratio_to_optimum_valid": row["max_ratio_to_optimum_valid"],
        }
        for row in sorted(
            summary_rows,
            key=lambda r: (-r["min_attaining_percent"], -r["optimal_percent"], r["strategy"]),
        )
    ]
    family_display = [
        {
            "family": row["family"],
            "instances": row["instances"],
            "n_min": row["n_min"],
            "n_max": row["n_max"],
            "optimum_min": row["optimum_min"],
            "optimum_max": row["optimum_max"],
            "final_optimal_percent": row["final_optimal_percent"],
            "mean_final_ratio_to_optimum": row["mean_final_ratio_to_optimum"],
            "max_final_ratio_to_optimum": row["max_final_ratio_to_optimum"],
        }
        for row in family_rows
    ]
    report = f"""# Furones v0.3.4 Integrity Report

Generated: {metadata["generated_at_utc"]}

This is a human-readable Integrity Report prepared for Gauge Freedom Journal's
AI-assistance/CAR expectation. It documents a ChatGPT-assisted measurement run
on the current local Furones source tree:

`{metadata["furones_root"]}`

The report intentionally checks only two requested questions:

1. Measure the finite-benchmark constant suggested by an exact-optimum
   benchmark.
2. Run each exposed strategy and measure candidate sizes plus min-attainment
   percentages on the selected CAR benchmark.

The benchmark contains {metadata["instances"]} small instances with at most
{metadata["max_exact_nodes"]} vertices. The exact domination number of every
instance was found by exhaustive search. This is useful as a reproducible
regression and integrity check, but it is not an exhaustive benchmark and does
not prove a universal constant approximation ratio for Minimum Dominating Set.

## CAR-001: Observed Ratio Constant

For this finite exact-optimum benchmark, the smallest constant fitting the final
Furones outputs is:

`rho_hat_final = {fmt(metadata["observed_final_ratio_constant"])}`

The smallest constant fitting the best valid candidate produced by the measured
strategy pool is:

`rho_hat_best_candidate = {fmt(metadata["observed_best_candidate_ratio_constant"])}`

All final Furones outputs valid: `{metadata["all_final_outputs_valid"]}`.

Final output optimal instances: `{metadata["final_optimal_instances"]}/{metadata["instances"]}`
(`{fmt(metadata["final_optimal_percent"])}%`).

Mean final ratio to optimum: `{fmt(metadata["mean_final_ratio_to_optimum"])}`.
Median final ratio to optimum: `{fmt(metadata["median_final_ratio_to_optimum"])}`.
95th percentile final ratio to optimum: `{fmt(metadata["p95_final_ratio_to_optimum"])}`.

### Family Summary

{markdown_table(family_display, [
    "family",
    "instances",
    "n_min",
    "n_max",
    "optimum_min",
    "optimum_max",
    "final_optimal_percent",
    "mean_final_ratio_to_optimum",
    "max_final_ratio_to_optimum",
])}

### Worst Final-Ratio Cases

{markdown_table(worst_rows, [
    "case",
    "family",
    "n",
    "m",
    "optimum_size",
    "final_size",
    "final_ratio_to_optimum",
    "min_valid_candidate_size",
    "min_attaining_strategies",
])}

## CAR-002: Per-Strategy Ablation

The percentage column is analogous to the Hvala per-candidate ablation:
it records how often a strategy attained the smallest valid candidate size
among the measured strategy pool. Ties count for every tied strategy, so the
percentages need not sum to 100.

{markdown_table(summary_display, [
    "strategy",
    "valid_percent",
    "optimal_percent",
    "min_attaining_percent",
    "mean_ratio_to_optimum_valid",
    "max_ratio_to_optimum_valid",
])}

## AI Assistance Scope

ChatGPT was used to design this 1000-instance exact benchmark, generate this
script, and format the report. The Python run itself executed the current local
Furones code and wrote the JSON/CSV outputs in this folder. The author remains
responsible for interpreting the results and for any manuscript claims.

## Files

- `CAR.furones-v0.3.4.json`
- `CAR-001-ratio-constant.json`
- `CAR-002-strategy-ablation.json`
- `car_benchmark_cases.csv`
- `family_summary.csv`
- `strategy_ablation_by_instance.csv`
- `strategy_ablation_summary.csv`
- `run_integrity_measurements.py`
"""
    (OUT_DIR / "INTEGRITY_REPORT.md").write_text(report, encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(metadata: dict[str, Any]) -> None:
    artifact_names = [
        "CAR-001-ratio-constant.json",
        "CAR-002-strategy-ablation.json",
        "INTEGRITY_REPORT.md",
        "car_benchmark_cases.csv",
        "family_summary.csv",
        "strategy_ablation_by_instance.csv",
        "strategy_ablation_summary.csv",
        "run_integrity_measurements.py",
    ]
    provenance = []
    for name in artifact_names:
        claim_type = Path(name).stem.replace("_", "-")
        provenance.append(
            {
                "claim_type": claim_type,
                "path": f"car/{name}",
                "sha256": f"sha256:{sha256_file(OUT_DIR / name)}",
            }
        )
    manifest = {
        "id": f"car:{sha256_file(OUT_DIR / 'CAR-001-ratio-constant.json')}",
        "run_id": "furones-v0.3.4-car-2026-06-22-1000-exact",
        "created_at": metadata["generated_at_utc"],
        "run": {
            "kind": "measurement",
            "name": "Furones v0.3.4 1000-instance exact small-graph CAR benchmark",
            "model": "ChatGPT-assisted Codex session",
            "version": f"furones-{metadata['furones_version_imported']}",
            "seed": metadata["benchmark_seed"],
            "steps": [
                {
                    "id": "step-1",
                    "run_id": "furones-v0.3.4-car-2026-06-22-1000-exact",
                    "order_index": 1,
                    "checkpoint_type": "scripted-exact-measurement",
                    "model": "ChatGPT-assisted Codex session",
                    "prompt": "Measure Furones on 1000 small instances with exact domination numbers, compare ratios, and measure each strategy's candidate size and percent wins.",
                    "token_budget": None,
                    "proof_mode": "replayable-local-run-with-exhaustive-optima",
                }
            ],
            "sampler": {"temp": None, "top_p": None, "rng": "python-random"},
        },
        "proof": {
            "match_kind": "replayable-script-plus-hashes",
            "ratio_claim_measured": f"rho_hat_final = {metadata['observed_final_ratio_constant']} on {metadata['instances']} exact small instances",
            "scope_warning": "Finite exact benchmark evidence only; not a proof of a universal constant approximation ratio.",
        },
        "policy_ref": {
            "hash": provenance[2]["sha256"],
            "egress": False,
            "estimator": "local-scripted-exhaustive-measurement",
        },
        "budgets": {"usd": None, "tokens": None, "nature_cost": None},
        "provenance": provenance,
        "checkpoints": ["ckpt:furones-v0.3.4-car-2026-06-22-1000-exact:step-1"],
        "sgrade": {
            "score": None,
            "components": {
                "provenance": 1.0,
                "energy": None,
                "replay": 1.0,
                "consent": None,
                "incidents": 0.0,
            },
        },
        "signer_public_key": None,
        "signatures": [],
    }
    (OUT_DIR / "CAR.furones-v0.3.4.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    metadata, case_rows, strategy_rows, summary_rows, family_rows = run_measurements()
    (OUT_DIR / "CAR-001-ratio-constant.json").write_text(
        json.dumps(
            {
                "car_id": "CAR-001",
                "title": "Observed finite-benchmark ratio constant on 1000 exact instances",
                "metadata": metadata,
                "family_summary": family_rows,
                "cases": case_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "CAR-002-strategy-ablation.json").write_text(
        json.dumps(
            {
                "car_id": "CAR-002",
                "title": "Per-strategy Furones candidate ablation on 1000 exact instances",
                "metadata": metadata,
                "summary": summary_rows,
                "by_instance": strategy_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_csv(OUT_DIR / "car_benchmark_cases.csv", case_rows)
    write_csv(OUT_DIR / "family_summary.csv", family_rows)
    write_csv(OUT_DIR / "strategy_ablation_by_instance.csv", strategy_rows)
    write_csv(OUT_DIR / "strategy_ablation_summary.csv", summary_rows)
    write_report(metadata, case_rows, strategy_rows, summary_rows, family_rows)
    write_manifest(metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
