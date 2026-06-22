"""Generate Furones v0.3.4 CAR-style integrity measurements.

This script is intentionally self-contained.  It imports the current Furones
source tree, builds a compact benchmark covering the targeted regression
families, measures the finite-benchmark ratio constant for the final solver,
and runs each exposed strategy as a candidate ablation.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
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


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    family: str
    graph: nx.Graph
    reference_size: int | None
    reference_type: str
    reference_note: str


def deterministic_planted_pair_decoy_graph(q: int, g: int = 5) -> nx.Graph:
    boosters = 5 * q - 9
    G = nx.Graph()
    planted = ["p0", "p1"]
    decoys = [f"d{i}" for i in range(q)]
    boosters_nodes = [f"b{i}" for i in range(boosters)]
    selector = planted + decoys
    G.add_nodes_from(selector + boosters_nodes)
    for i, u in enumerate(selector):
        for v in selector[i + 1 :]:
            G.add_edge(u, v)
    for b in boosters_nodes:
        G.add_edge("p0", b)
        for d in decoys:
            G.add_edge(d, b)
    for i, d in enumerate(decoys):
        e0 = [f"e0_{i}_{j}" for j in range(g)]
        e1 = [f"e1_{i}_{j}" for j in range(g)]
        G.add_nodes_from(e0 + e1)
        for group, planted_vertex in ((e0, "p0"), (e1, "p1")):
            for x in group:
                G.add_edge(planted_vertex, x)
                G.add_edge(d, x)
            for j, x in enumerate(group):
                G.add_edge(x, group[(j + 1) % g])
    return G


def random_set_cover_graph(
    q: int, b: int, p: float, seed: int = 0, beta: int = 0
) -> nx.Graph:
    rng = random.Random(seed)
    G = nx.Graph()
    decoys = [f"d{i}" for i in range(q)]
    planted = ["p0", "p1"]
    e0 = [f"e0_{j}" for j in range(b)]
    e1 = [f"e1_{j}" for j in range(b)]
    boosters = [f"b{j}" for j in range(2 * beta)]
    G.add_nodes_from(decoys + planted + e0 + e1 + boosters)
    selector = planted + decoys
    for i, u in enumerate(selector):
        for v in selector[i + 1 :]:
            G.add_edge(u, v)
    for e in e0:
        G.add_edge("p0", e)
    for e in e1:
        G.add_edge("p1", e)
    for d in decoys:
        for e in e0 + e1:
            if rng.random() < p:
                G.add_edge(d, e)
        for booster in boosters:
            G.add_edge(d, booster)
    for booster in boosters:
        G.add_edge("p0", booster)
    return G


def planted_dominator_graph(
    k: int = 2, r: int = 160, p: float = 0.5, seed: int = 331507
) -> nx.Graph:
    rng = random.Random(seed)
    G = nx.Graph()
    planted = [f"u{i}" for i in range(k)]
    witnesses = [f"x{j}" for j in range(r)]
    G.add_nodes_from(planted + witnesses)
    for j, x in enumerate(witnesses):
        G.add_edge(x, planted[j % k])
    for i, xi in enumerate(witnesses):
        for xj in witnesses[i + 1 :]:
            if rng.random() < p:
                G.add_edge(xi, xj)
    return G


def decoy_clique_private_witness_graph(k: int, t: int) -> nx.Graph:
    G = nx.Graph()
    planted = [f"u{i}" for i in range(k)]
    decoys = [f"d{j}" for j in range(t)]
    witnesses = [f"e{j}" for j in range(t)]
    G.add_nodes_from(planted + decoys + witnesses)
    for i, u in enumerate(decoys):
        for v in decoys[i + 1 :]:
            G.add_edge(u, v)
    for j in range(t):
        u = planted[j % k]
        d = decoys[j]
        e = witnesses[j]
        G.add_edge(d, e)
        G.add_edge(u, e)
        G.add_edge(u, d)
    return G


def multiblock_adversary(
    k: int, q: int, b: int, p: float, beta: int, seed: int = 0
) -> nx.Graph:
    rng = random.Random(seed)
    G = nx.Graph()
    planted = [f"p{i}" for i in range(k)]
    decoys = [f"d{j}" for j in range(q)]
    elements = [[f"e{i}_{a}" for a in range(b)] for i in range(k)]
    boosters = [[f"b{i}_{a}" for a in range(beta)] for i in range(k)]
    G.add_nodes_from(decoys + planted)
    for block in elements:
        G.add_nodes_from(block)
    for block in boosters:
        G.add_nodes_from(block)
    selector = planted + decoys
    for i, u in enumerate(selector):
        for v in selector[i + 1 :]:
            G.add_edge(u, v)
    for i, p_vertex in enumerate(planted):
        for e in elements[i]:
            G.add_edge(p_vertex, e)
        for b_vertex in boosters[i]:
            G.add_edge(p_vertex, b_vertex)
    all_elements = [e for block in elements for e in block]
    all_boosters = [b_vertex for block in boosters for b_vertex in block]
    for d in decoys:
        for e in all_elements:
            if rng.random() < p:
                G.add_edge(d, e)
        for b_vertex in all_boosters:
            G.add_edge(d, b_vertex)
    return G


def benchmark_cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase(
            "path_12",
            "smoke",
            nx.path_graph(12),
            4,
            "exact_formula",
            "gamma(P_n)=ceil(n/3).",
        ),
        BenchmarkCase(
            "cycle_12",
            "smoke",
            nx.cycle_graph(12),
            4,
            "exact_formula",
            "gamma(C_n)=ceil(n/3).",
        ),
        BenchmarkCase(
            "complete_bipartite_4_5",
            "salvador_sanity",
            nx.complete_bipartite_graph(4, 5),
            2,
            "exact_argument",
            "One vertex from each side dominates K_{4,5}; no single vertex is universal.",
        ),
        BenchmarkCase(
            "planted_pair_decoy_q8",
            "planted_pair_decoy_booster",
            deterministic_planted_pair_decoy_graph(8),
            2,
            "exact_argument",
            "{p0,p1} dominates and no single vertex is universal.",
        ),
        BenchmarkCase(
            "planted_pair_decoy_q12",
            "planted_pair_decoy_booster",
            deterministic_planted_pair_decoy_graph(12),
            2,
            "exact_argument",
            "{p0,p1} dominates and no single vertex is universal.",
        ),
        BenchmarkCase(
            "random_set_cover_q20_b120",
            "random_set_cover",
            random_set_cover_graph(20, 120, 0.50, seed=0, beta=0),
            2,
            "exact_argument_checked",
            "{p0,p1} dominates; the script checks absence of a universal vertex.",
        ),
        BenchmarkCase(
            "boosted_set_cover_q20_b120_beta40",
            "boosted_set_cover",
            random_set_cover_graph(20, 120, 0.35, seed=0, beta=40),
            2,
            "exact_argument_checked",
            "{p0,p1} dominates; the script checks absence of a universal vertex.",
        ),
        BenchmarkCase(
            "planted_dominator_k2_r160",
            "planted_dominator",
            planted_dominator_graph(2, 160, 0.5, seed=331507),
            2,
            "exact_argument_checked",
            "The planted pair dominates; the script checks absence of a universal vertex.",
        ),
        BenchmarkCase(
            "decoy_clique_k2_t10",
            "decoy_private_witness",
            decoy_clique_private_witness_graph(2, 10),
            2,
            "exact_argument",
            "Two planted vertices dominate; no one vertex dominates both private-witness sides.",
        ),
        BenchmarkCase(
            "decoy_clique_k4_t24",
            "decoy_private_witness",
            decoy_clique_private_witness_graph(4, 24),
            4,
            "planted_upper_bound",
            "The planted set of size k is a valid dominating set; ratio is to this bound.",
        ),
        BenchmarkCase(
            "multiblock_k4_q50_b40",
            "multiblock",
            multiblock_adversary(4, 50, 40, 0.08, 8, seed=0),
            4,
            "planted_upper_bound",
            "The planted set of size k is a valid dominating set; ratio is to this bound.",
        ),
        BenchmarkCase(
            "multiblock_k8_q70_b40",
            "near_threshold_multiblock",
            multiblock_adversary(8, 70, 40, 0.055, 8, seed=2),
            8,
            "planted_upper_bound",
            "The planted set of size k is a valid dominating set; ratio is to this bound.",
        ),
    ]


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


def planted_set_valid(case: BenchmarkCase) -> bool | None:
    name = case.name
    G = case.graph
    if "planted_pair" in name or "set_cover" in name:
        return nx.is_dominating_set(G, {"p0", "p1"})
    if name.startswith("planted_dominator"):
        return nx.is_dominating_set(G, {"u0", "u1"})
    if "decoy_clique_k4" in name:
        return nx.is_dominating_set(G, {f"u{i}" for i in range(4)})
    if "multiblock_k4" in name:
        return nx.is_dominating_set(G, {f"p{i}" for i in range(4)})
    if "multiblock_k8" in name:
        return nx.is_dominating_set(G, {f"p{i}" for i in range(8)})
    return None


def finalize_candidate(
    original: nx.Graph, working: nx.Graph, isolates: set[Any], candidate: set[Any]
) -> tuple[set[Any], bool]:
    if working.number_of_nodes() == 0:
        final = set(isolates)
    else:
        candidate = set(candidate)
        candidate = {v for v in candidate if v in working}
        candidate = prune_redundant_vertices_dominating(working, candidate)
        final = candidate | set(isolates)
    return final, bool(nx.is_dominating_set(original, final))


def run_measurements() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
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
        final_ratio = (
            len(final_solution) / case.reference_size
            if final_valid and case.reference_size
            else None
        )

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
            ratio = len(candidate) / case.reference_size if valid and case.reference_size else None
            row = {
                "case": case.name,
                "family": case.family,
                "strategy": strategy_name,
                "n": G.number_of_nodes(),
                "m": G.number_of_edges(),
                "reference_size": case.reference_size,
                "reference_type": case.reference_type,
                "candidate_size": len(candidate),
                "valid": valid,
                "ratio_to_reference": ratio,
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
            instance_rows.append(row)

        case_rows.append(
            {
                "case": case.name,
                "family": case.family,
                "n": G.number_of_nodes(),
                "m": G.number_of_edges(),
                "reference_size": case.reference_size,
                "reference_type": case.reference_type,
                "reference_note": case.reference_note,
                "planted_set_valid": planted_set_valid(case),
                "has_universal_vertex": has_universal_vertex(G),
                "final_size": len(final_solution),
                "final_valid": final_valid,
                "final_ratio_to_reference": final_ratio,
                "final_seconds": final_seconds,
                "min_valid_candidate_size": min_valid_size,
                "min_valid_candidate_ratio_to_reference": (
                    min_valid_size / case.reference_size
                    if min_valid_size is not None and case.reference_size
                    else None
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
            row["ratio_to_reference"]
            for row in valid_rows
            if row["ratio_to_reference"] is not None and math.isfinite(row["ratio_to_reference"])
        ]
        summary_rows.append(
            {
                "strategy": strategy_name,
                "instances": total_cases,
                "valid_instances": len(valid_rows),
                "valid_percent": 100.0 * len(valid_rows) / total_cases,
                "min_attaining_instances": sum(
                    1 for row in rows if row["attains_min_valid_candidate"]
                ),
                "min_attaining_percent": 100.0
                * sum(1 for row in rows if row["attains_min_valid_candidate"])
                / total_cases,
                "mean_size_valid": (
                    sum(row["candidate_size"] for row in valid_rows) / len(valid_rows)
                    if valid_rows
                    else None
                ),
                "mean_ratio_to_reference_valid": (
                    sum(ratios) / len(ratios) if ratios else None
                ),
                "max_ratio_to_reference_valid": max(ratios) if ratios else None,
                "total_seconds": sum(row["seconds"] for row in rows),
            }
        )

    final_ratios = [
        row["final_ratio_to_reference"]
        for row in case_rows
        if row["final_valid"] and row["final_ratio_to_reference"] is not None
    ]
    min_candidate_ratios = [
        row["min_valid_candidate_ratio_to_reference"]
        for row in case_rows
        if row["min_valid_candidate_ratio_to_reference"] is not None
    ]
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "tooling": "ChatGPT-assisted Codex run; Python script generated by ChatGPT and executed on the local current Furones source tree.",
        "furones_root": str(FURONES_ROOT),
        "furones_version_imported": FURONES_VERSION,
        "benchmark_name": "furones_v0_3_4_car_compact_targeted_benchmark",
        "benchmark_scope": "Compact targeted benchmark selected for CAR/integrity reporting; not an exhaustive benchmark and not a universal approximation proof.",
        "instances": total_cases,
        "strategies": [name for name, _ in STRATEGIES],
        "observed_final_ratio_constant": max(final_ratios) if final_ratios else None,
        "observed_best_candidate_ratio_constant": max(min_candidate_ratios)
        if min_candidate_ratios
        else None,
        "all_final_outputs_valid": all(row["final_valid"] for row in case_rows),
    }
    return metadata, case_rows, instance_rows, summary_rows


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
) -> None:
    ratio_rows = [
        {
            "case": row["case"],
            "family": row["family"],
            "reference_size": row["reference_size"],
            "reference_type": row["reference_type"],
            "final_size": row["final_size"],
            "final_ratio_to_reference": row["final_ratio_to_reference"],
            "min_valid_candidate_size": row["min_valid_candidate_size"],
            "min_attaining_strategies": ", ".join(row["min_attaining_strategies"]),
        }
        for row in case_rows
    ]
    summary_display = [
        {
            "strategy": row["strategy"],
            "valid_percent": row["valid_percent"],
            "min_attaining_percent": row["min_attaining_percent"],
            "mean_size_valid": row["mean_size_valid"],
            "mean_ratio_to_reference_valid": row["mean_ratio_to_reference_valid"],
            "max_ratio_to_reference_valid": row["max_ratio_to_reference_valid"],
        }
        for row in sorted(
            summary_rows,
            key=lambda r: (-r["min_attaining_percent"], r["strategy"]),
        )
    ]
    report = f"""# Furones v0.3.4 Integrity Report

Generated: {metadata["generated_at_utc"]}

This is a human-readable Integrity Report prepared for Gauge Freedom Journal's
AI-assistance/CAR expectation.  It documents a ChatGPT-assisted measurement run
on the current local Furones source tree:

`{metadata["furones_root"]}`

The report intentionally checks only two requested questions:

1. Measure the finite-benchmark constant suggested by the experiment.
2. Run each exposed strategy and measure candidate sizes plus min-attainment
   percentages on the selected CAR benchmark.

The benchmark is compact and targeted.  It is useful as a reproducible
regression and integrity check, but it is not an exhaustive benchmark and does
not prove a universal constant approximation ratio for Minimum Dominating Set.

## CAR-001: Observed Ratio Constant

For this finite benchmark, the smallest constant fitting the final Furones
outputs against the listed exact or planted reference sizes is:

`rho_hat_final = {fmt(metadata["observed_final_ratio_constant"])}`

The smallest constant fitting the best valid candidate produced by the measured
strategy pool is:

`rho_hat_best_candidate = {fmt(metadata["observed_best_candidate_ratio_constant"])}`

All final Furones outputs valid: `{metadata["all_final_outputs_valid"]}`.

{markdown_table(ratio_rows, [
    "case",
    "family",
    "reference_size",
    "reference_type",
    "final_size",
    "final_ratio_to_reference",
    "min_valid_candidate_size",
    "min_attaining_strategies",
])}

## CAR-002: Per-Strategy Ablation

The percentage column is analogous to the Hvala per-candidate ablation:
it records how often a strategy attained the smallest valid candidate size
among the measured strategy pool.  Ties count for every tied strategy, so the
percentages need not sum to 100.

{markdown_table(summary_display, [
    "strategy",
    "valid_percent",
    "min_attaining_percent",
    "mean_size_valid",
    "mean_ratio_to_reference_valid",
    "max_ratio_to_reference_valid",
])}

## AI Assistance Scope

ChatGPT was used to design this compact CAR benchmark, generate this script,
and format the report.  The Python run itself executed the current local
Furones code and wrote the JSON/CSV outputs in this folder.  The author remains
responsible for interpreting the results and for any manuscript claims.

## Files

- `CAR-001-ratio-constant.json`
- `CAR-002-strategy-ablation.json`
- `car_benchmark_cases.csv`
- `strategy_ablation_by_instance.csv`
- `strategy_ablation_summary.csv`
- `run_integrity_measurements.py`
"""
    (OUT_DIR / "INTEGRITY_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    metadata, case_rows, strategy_rows, summary_rows = run_measurements()
    (OUT_DIR / "CAR-001-ratio-constant.json").write_text(
        json.dumps(
            {
                "car_id": "CAR-001",
                "title": "Observed finite-benchmark ratio constant",
                "metadata": metadata,
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
                "title": "Per-strategy Furones candidate ablation",
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
    write_csv(OUT_DIR / "strategy_ablation_by_instance.csv", strategy_rows)
    write_csv(OUT_DIR / "strategy_ablation_summary.csv", summary_rows)
    write_report(metadata, case_rows, strategy_rows, summary_rows)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
