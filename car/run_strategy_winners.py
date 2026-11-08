#!/usr/bin/env python3
# Created on 07/10/2026
# Author: Frank Vega
"""
CAR strategy-winner report for Furones (Minimum Dominating Set).

Runs every candidate strategy of ``furones.algorithm.find_dominating_set``
separately on each instance, prunes each candidate exactly as the portfolio
does (``prune_redundant_vertices_dominating``), checks that it dominates the
graph, and records which strategy (or strategies, on ties) produced the
smallest set.  The four improvement strategies of ``furones.improve`` (union
re-prune, swap search, consensus forcing, final refinement) are computed from
the pruned base candidates of the instance and compete in the same table.  The full Furones answer and, when requested, the exact optimum
are reported alongside.

Instances come from built-in generator families and/or a directory of DIMACS
files (the same format ``iris``/``furones`` read).

Outputs (in --out, default: this folder):
    strategy_winners_by_instance.csv   one row per instance
    strategy_winners_summary.csv       one row per strategy
and a summary table on the console.

Examples:
    python car/run_strategy_winners.py
    python car/run_strategy_winners.py --sizes 30 60 120 --seeds 10 --exact milp
    python car/run_strategy_winners.py --input-dir benchmarks --families none
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from furones import algorithm as A  # noqa: E402
from furones import baker_algo, tscc_ds_reduction  # noqa: E402
from furones.improve import improvement_steps  # noqa: E402


# ---------------------------------------------------------------------------
# Strategies (the same candidates the Furones portfolio compares)
# ---------------------------------------------------------------------------


def _lifted_tscc_baker(G: nx.Graph, eps: float = 1.0) -> Set[Any]:
    """C_L: TSCC pendant cascade + Baker on the reduced graph, lifted back."""
    G_reduced, forced_ds, lift = tscc_ds_reduction.reduce_to_tscc_for_ds(G)
    if not G_reduced:
        return set(forced_ds)
    mapping = {u: k for k, u in enumerate(G_reduced.nodes())}
    unmapping = {k: u for u, k in mapping.items()}
    B = baker_algo.Graph(G_reduced.number_of_nodes())
    for u, v in G_reduced.edges():
        B.add_edge(mapping[u], mapping[v])
    sol = baker_algo.baker_ptas(B, eps, verbose=False)
    return lift({unmapping[u] for u in sol})


STRATEGIES: List[Tuple[str, Callable[[nx.Graph], Set[Any]]]] = [
    ("lifted_tscc_baker", _lifted_tscc_baker),
    ("closed_degree_sweep", A.greedy_closed_degree_dominating_set),
    ("greedy_max_coverage", A.greedy_max_coverage_dominating_set),
    ("low_degree_witness", A.low_degree_witness_dominating_set),
    ("medium_degree_witness", A.medium_degree_witness_dominating_set),
    ("ownership_late", lambda G: A.order_ownership_witness_dominating_set(G, "late")),
    ("ownership_early", lambda G: A.order_ownership_witness_dominating_set(G, "early")),
    ("seed_and_complete", A.seed_and_complete_dominating_set),
    ("salvador_baker", lambda G: A.salvador_planar_bipartite_baker_candidate(G, eps=1)),
    ("max_cut_double_cover", A.max_cut_bipartite),
    ("spanning_tree", A.spanning_tree_dominating_set),
    ("reverse_delete_input", lambda G: A.reverse_delete_dominating_set(G, "input")),
    ("reverse_delete_reverse", lambda G: A.reverse_delete_dominating_set(G, "reverse_input")),
    ("reverse_delete_high", lambda G: A.reverse_delete_dominating_set(G, "high_degree")),
    ("reverse_delete_low", lambda G: A.reverse_delete_dominating_set(G, "low_degree")),
]
# Improvement strategies (furones.improve): computed from the pruned base candidates of the same instance.
IMPROVEMENT_NAMES = ["union_reprune", "swap_search", "consensus_forcing", "final_refinement"]
STRATEGY_NAMES = [name for name, _ in STRATEGIES] + IMPROVEMENT_NAMES


def _working_graph(graph: nx.Graph) -> Tuple[nx.Graph, Set[Any]]:
    """Same preprocessing as find_dominating_set: drop self-loops and isolates."""
    W = nx.Graph(graph)
    W.remove_edges_from(list(nx.selfloop_edges(W)))
    isolates = set(nx.isolates(W))
    W.remove_nodes_from(isolates)
    return W, isolates


def run_strategies(graph: nx.Graph) -> Dict[str, Dict[str, Any]]:
    """Return {strategy: {size, valid, seconds, error}} for one instance."""
    W, isolates = _working_graph(graph)
    out: Dict[str, Dict[str, Any]] = {}
    base_sets: List[Set[Any]] = []

    def record(name: str, produce: Callable[[], Set[Any]], t0: float) -> None:
        rec: Dict[str, Any] = {"size": None, "valid": False, "seconds": 0.0, "error": ""}
        try:
            if W.number_of_nodes() == 0:
                D: Set[Any] = set()
            else:
                D = A.prune_redundant_vertices_dominating(W, set(produce()))
            full = D | isolates
            rec["valid"] = nx.is_dominating_set(graph, full)
            rec["size"] = len(full)
            if rec["valid"] and name not in IMPROVEMENT_NAMES:
                base_sets.append(D)
        except Exception as exc:  # a failing strategy must not stop the run
            rec["error"] = f"{type(exc).__name__}: {exc}"[:200]
        rec["seconds"] = time.perf_counter() - t0
        out[name] = rec

    for name, fn in STRATEGIES:
        record(name, lambda fn=fn: fn(W), time.perf_counter())

    t0 = time.perf_counter()
    try:
        produced = dict(improvement_steps(W, base_sets)) if W.number_of_nodes() else {}
    except Exception:
        produced = {}
    phase = time.perf_counter() - t0
    for name in IMPROVEMENT_NAMES:
        record(name, lambda name=name: produced.get(name, set()), time.perf_counter() - phase / len(IMPROVEMENT_NAMES))
    return out


# ---------------------------------------------------------------------------
# Exact optimum (optional)
# ---------------------------------------------------------------------------


def exact_optimum(graph: nx.Graph, mode: str, brute_limit: int) -> Optional[int]:
    if mode == "none":
        return None
    n = graph.number_of_nodes()
    if mode == "brute" or (mode == "auto" and n <= brute_limit):
        if n > brute_limit:
            return None
        D = A.find_dominating_set_brute_force(graph)
        return None if D is None else len(D)
    # MILP via SciPy (a Furones dependency)
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except Exception:
        return None
    if n == 0:
        return 0
    idx = {v: i for i, v in enumerate(graph.nodes())}
    M = lil_matrix((n, n))
    for v in graph.nodes():
        i = idx[v]
        M[i, i] = 1
        for u in graph.neighbors(v):
            M[i, idx[u]] = 1
    res = milp(
        np.ones(n),
        constraints=LinearConstraint(M.tocsr(), 1, np.inf),
        integrality=np.ones(n),
        bounds=Bounds(0, 1),
    )
    return int(round(res.fun)) if res.success else None


# ---------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------


def _tree_plus(n: int, extra: int, rng: random.Random) -> nx.Graph:
    G = nx.random_labeled_tree(n, seed=rng.randrange(2**31))
    while G.number_of_edges() < n - 1 + extra:
        u, v = rng.sample(range(n), 2)
        G.add_edge(u, v)
    return G


def _caterpillar(n: int, rng: random.Random) -> nx.Graph:
    spine = max(2, n // 3)
    G = nx.path_graph(spine)
    for v in range(spine, n):
        G.add_edge(v, rng.randrange(spine))
    return G


def _planted(n: int, rng: random.Random) -> nx.Graph:
    """k hidden dominators plus noise edges."""
    k = max(1, n // 10)
    G = nx.empty_graph(n)
    for v in range(k, n):
        G.add_edge(v, rng.randrange(k))
    for _ in range(n):
        u, v = rng.sample(range(k, n), 2)
        G.add_edge(u, v)
    return G


FAMILIES: Dict[str, Callable[[int, random.Random], nx.Graph]] = {
    "tree": lambda n, r: nx.random_labeled_tree(n, seed=r.randrange(2**31)),
    "caterpillar": _caterpillar,
    "tree_plus_n8": lambda n, r: _tree_plus(n, max(1, n // 8), r),
    "gnp_avgdeg3": lambda n, r: nx.gnp_random_graph(n, 3 / max(1, n - 1), seed=r.randrange(2**31)),
    "gnp_avgdeg8": lambda n, r: nx.gnp_random_graph(n, 8 / max(1, n - 1), seed=r.randrange(2**31)),
    "gnp_dense": lambda n, r: nx.gnp_random_graph(n, 0.3, seed=r.randrange(2**31)),
    "grid": lambda n, r: nx.convert_node_labels_to_integers(
        nx.grid_2d_graph(max(2, int(n ** 0.5)), max(2, n // max(2, int(n ** 0.5))))
    ),
    "geometric": lambda n, r: nx.random_geometric_graph(n, (6 / n) ** 0.5, seed=r.randrange(2**31)),
    "barabasi_albert_2": lambda n, r: nx.barabasi_albert_graph(n, 2, seed=r.randrange(2**31)),
    "watts_strogatz_4": lambda n, r: nx.connected_watts_strogatz_graph(n, 4, 0.2, seed=r.randrange(2**31)),
    "random_regular_3": lambda n, r: nx.random_regular_graph(3, n + (n % 2), seed=r.randrange(2**31)),
    "planted_dominators": _planted,
}


def generated_instances(
    families: List[str], sizes: List[int], seeds: int, base_seed: int
) -> Iterable[Tuple[str, str, int, int, nx.Graph]]:
    for fam in families:
        for n in sizes:
            for s in range(seeds):
                rng = random.Random(f"{base_seed}-{fam}-{n}-{s}")
                G = FAMILIES[fam](n, rng)
                yield f"{fam}_n{n}_s{s}", fam, n, s, G


def file_instances(directory: str) -> Iterable[Tuple[str, str, int, int, nx.Graph]]:
    from furones import parser, utils

    for path in sorted(utils.get_file_names(directory)):
        full = path if os.path.isabs(path) else os.path.join(directory, path)
        try:
            G = parser.read(full)
        except Exception as exc:
            print(f"  skip {full}: {exc}", file=sys.stderr)
            continue
        yield utils.get_file_name(full), "file", G.number_of_nodes(), -1, G


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Which Furones strategy wins on each instance.")
    ap.add_argument("--families", nargs="+", default=list(FAMILIES),
                    help=f"generator families, or 'none' (choices: {', '.join(FAMILIES)})")
    ap.add_argument("--sizes", nargs="+", type=int, default=[20, 60, 150])
    ap.add_argument("--seeds", type=int, default=5, help="instances per family and size")
    ap.add_argument("--base-seed", type=int, default=2026)
    ap.add_argument("--input-dir", default=None, help="also run every DIMACS file in this directory")
    ap.add_argument("--exact", choices=["auto", "brute", "milp", "none"], default="auto",
                    help="exact optimum: auto = brute force up to --brute-limit nodes, MILP above")
    ap.add_argument("--brute-limit", type=int, default=20)
    ap.add_argument("--no-furones", action="store_true", help="skip the full find_dominating_set run")
    ap.add_argument("--out", default=HERE)
    args = ap.parse_args()

    families = [] if args.families == ["none"] else args.families
    unknown = [f for f in families if f not in FAMILIES]
    if unknown:
        ap.error(f"unknown families: {unknown}")

    instances: List[Iterable] = [generated_instances(families, args.sizes, args.seeds, args.base_seed)]
    if args.input_dir:
        instances.append(file_instances(args.input_dir))

    os.makedirs(args.out, exist_ok=True)
    by_inst_path = os.path.join(args.out, "strategy_winners_by_instance.csv")
    summary_path = os.path.join(args.out, "strategy_winners_summary.csv")

    wins = Counter()          # strategy reached the best size (ties count for all)
    unique_wins = Counter()   # strategy was the only one to reach the best size
    optimal_hits = Counter()  # strategy reached the exact optimum
    invalid = Counter()
    ratio_sum = defaultdict(float)
    ratio_cnt = Counter()
    time_sum = defaultdict(float)
    family_winners: Dict[str, Counter] = defaultdict(Counter)
    total = 0

    fields = (["instance", "family", "n", "m", "seed", "optimum", "furones_size",
               "best_strategy_size", "winners", "unique_winner"]
              + [f"{s}_size" for s in STRATEGY_NAMES])

    with open(by_inst_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for source in instances:
            for inst, fam, n, seed, G in source:
                total += 1
                res = run_strategies(G)
                sizes = {s: r["size"] for s, r in res.items() if r["valid"]}
                for s, r in res.items():
                    time_sum[s] += r["seconds"]
                    if not r["valid"]:
                        invalid[s] += 1
                best = min(sizes.values()) if sizes else None
                winners = [s for s in STRATEGY_NAMES if sizes.get(s) == best]
                for s in winners:
                    wins[s] += 1
                    family_winners[fam][s] += 1
                if len(winners) == 1:
                    unique_wins[winners[0]] += 1

                opt = exact_optimum(G, args.exact, args.brute_limit)
                ref = opt if opt is not None else best
                for s, k in sizes.items():
                    if opt is not None and k == opt:
                        optimal_hits[s] += 1
                    if ref:
                        ratio_sum[s] += k / ref
                        ratio_cnt[s] += 1

                fur = None
                if not args.no_furones:
                    try:
                        fur = len(A.find_dominating_set(G))
                    except Exception as exc:
                        fur = f"error: {type(exc).__name__}"

                row = {
                    "instance": inst, "family": fam, "n": G.number_of_nodes(),
                    "m": G.number_of_edges(), "seed": seed,
                    "optimum": "" if opt is None else opt,
                    "furones_size": "" if fur is None else fur,
                    "best_strategy_size": "" if best is None else best,
                    "winners": ";".join(winners),
                    "unique_winner": winners[0] if len(winners) == 1 else "",
                }
                for s in STRATEGY_NAMES:
                    row[f"{s}_size"] = sizes.get(s, "invalid" if not res[s]["error"] else "error")
                writer.writerow(row)
                fh.flush()

                tag = f"opt={opt} " if opt is not None else ""
                print(f"{inst:32s} n={G.number_of_nodes():<5d} {tag}best={best} "
                      f"furones={fur}  winners: {', '.join(winners)}")

    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["strategy", "instances", "wins", "win_rate", "unique_wins",
                         "optimal_hits", "mean_ratio", "invalid", "total_seconds"])
        for s in STRATEGY_NAMES:
            writer.writerow([
                s, total, wins[s], f"{wins[s] / total:.4f}" if total else "",
                unique_wins[s], optimal_hits[s],
                f"{ratio_sum[s] / ratio_cnt[s]:.4f}" if ratio_cnt[s] else "",
                invalid[s], f"{time_sum[s]:.3f}",
            ])

    ratio_label = "ratio vs optimum" if args.exact != "none" else "ratio vs best"
    print(f"\n{total} instances.  wins = reached the smallest size (ties count for every tied strategy)")
    print(f"{'strategy':26s} {'wins':>6s} {'unique':>7s} {'optimal':>8s} {'mean ' + ratio_label:>22s} {'invalid':>8s} {'sec':>8s}")
    for s in sorted(STRATEGY_NAMES, key=lambda x: (-wins[x], -unique_wins[x])):
        mr = f"{ratio_sum[s] / ratio_cnt[s]:.4f}" if ratio_cnt[s] else "-"
        print(f"{s:26s} {wins[s]:6d} {unique_wins[s]:7d} {optimal_hits[s]:8d} {mr:>22s} {invalid[s]:8d} {time_sum[s]:8.2f}")

    print("\nTop strategy per family (by wins):")
    for fam, c in family_winners.items():
        top = c.most_common(3)
        print(f"  {fam:20s} " + ", ".join(f"{s} ({k})" for s, k in top))

    print(f"\nWrote {by_inst_path}\nWrote {summary_path}")


if __name__ == "__main__":
    main()
