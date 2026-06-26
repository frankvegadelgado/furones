#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAR-003: High-degree / dense regime experiment for the Furones near-threshold
ratio hypothesis.

This script probes the regime left open by the low-degree proposition: graphs
with maximum degree Delta > sqrt(n)/e. It generates 1000 dense instances drawn
from high-degree families (complete, cocktail-party, dense circulant, random
regular, dense Erdos-Renyi, perturbed hypercube, perturbed Paley), and for each
instance it computes

  * the exact domination number  gamma(G)  by increasing-size exhaustive search
    from the packing lower bound (cheap because dense graphs have small gamma),
  * the size returned by the INSTALLED Furones solver
    (furones.algorithm.find_dominating_set), i.e. the full portfolio,
  * the realised ratio  |D| / gamma(G),
  * the domination tightness  tau(G) = gamma(G) * (Delta + 1) / n,
  * the target  rho(n) = max(4, 0.5 * ln n),

and records whether the hypothesis  |D| <= rho(n) * gamma(G)  holds.

Every instance is kept small (n <= 16) and dense, so the optimum is found in a
feasible amount of time even across 1000 instances. The pool deliberately
includes perfect-code stress cases (complete graphs, dense circulants,
hypercubes, Paley graphs) whose tightness tau = 1.

Requirements:
    pip install furones        # v0.3.6 or later
    (networkx is a Furones dependency)

Reproducibility:
    Instance i is built from its own RNG seeded with (BASE_SEED + i), and the
    family is chosen as i % (#families).

Usage:
    python run_high_degree_experiment.py [--instances 1000] [--seed 12345] [--max-n 16]

Outputs:
    CAR-003-high-degree.json        per-family + overall summary
    high_degree_by_instance.csv     one row per instance
"""

import argparse
import csv
import itertools
import json
import math
import random
from datetime import datetime, timezone

import networkx as nx

from furones import __version__ as FURONES_VERSION
from furones.algorithm import find_dominating_set


# --------------------------------------------------------------------------- #
def run_furones(G):
    """Call the installed Furones solver (full portfolio) and return a set."""
    try:
        return set(find_dominating_set(G))
    except TypeError:
        return set(find_dominating_set(G, 1.0))


# --------------------------------------------------------------------------- #
# Exact domination number via increasing-size search from the packing bound.
# --------------------------------------------------------------------------- #
def exact_gamma(G):
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return 0
    idx = {v: i for i, v in enumerate(nodes)}
    full = (1 << n) - 1
    mask = []
    for v in nodes:
        m = 1 << idx[v]
        for w in G.neighbors(v):
            m |= 1 << idx[w]
        mask.append(m)
    Delta = max(d for _, d in G.degree())
    lower = max(1, -(-n // (Delta + 1)))  # ceil(n / (Delta + 1))
    for k in range(lower, n + 1):
        for combo in itertools.combinations(range(n), k):
            covered = 0
            for j in combo:
                covered |= mask[j]
                if covered == full:
                    break
            if covered == full:
                return k
    return n


def dominates(G, D):
    covered = set()
    for d in D:
        covered.add(d)
        covered.update(G.neighbors(d))
    return covered >= set(G.nodes())


def finalize(G):
    G = nx.convert_node_labels_to_integers(G)
    iso = [v for v in G.nodes() if G.degree(v) == 0]
    for v in iso:
        u = (v + 1) % G.number_of_nodes()
        if u != v:
            G.add_edge(v, u)
    return G


def _perturb(G, r, t):
    nodes = list(G.nodes())
    for _ in range(t):
        u, v = r.sample(nodes, 2)
        G.add_edge(u, v)
    return G


# --------------------------------------------------------------------------- #
# Dense, high-degree graph families (all satisfy Delta > sqrt(n)/e). Each takes
# an RNG and the size cap, and returns an nx.Graph.
# --------------------------------------------------------------------------- #
def gen_complete(r, max_n):
    return nx.complete_graph(r.randint(6, max_n))


def gen_cocktail(r, max_n):
    n = r.choice([k for k in range(6, max_n + 1) if k % 2 == 0])
    G = nx.complete_graph(n)
    for i in range(0, n - 1, 2):
        if G.has_edge(i, i + 1):
            G.remove_edge(i, i + 1)
    return G


def gen_circulant(r, max_n):
    n = r.randint(8, max_n)
    d = r.randint(2, max(2, n // 2))
    return nx.circulant_graph(n, list(range(1, d + 1)))


def gen_random_regular(r, max_n):
    n = r.randint(8, max_n)
    dmin = math.ceil(math.sqrt(n))
    d = r.randint(dmin, max(dmin, n // 2))
    d = max(2, min(d, n - 1))
    if (n * d) % 2 == 1:
        d -= 1
    d = max(2, d)
    try:
        return nx.random_regular_graph(d, n, seed=r.randint(0, 2**31 - 1))
    except Exception:
        return nx.circulant_graph(n, list(range(1, max(2, d // 2) + 1)))


def gen_erdos(r, max_n):
    n = r.randint(8, max_n)
    p = r.uniform(0.45, 0.8)
    return nx.gnp_random_graph(n, p, seed=r.randint(0, 2**31 - 1))


def gen_hypercube(r, max_n):
    k = r.choice([3, 4])
    G = nx.convert_node_labels_to_integers(nx.hypercube_graph(k))
    return _perturb(G, r, r.randint(0, 3))


def gen_paley(r, max_n):
    p = r.choice([5, 13])
    squares = {(x * x) % p for x in range(1, p)}
    G = nx.Graph()
    G.add_nodes_from(range(p))
    for u, v in itertools.combinations(range(p), 2):
        if (u - v) % p in squares:
            G.add_edge(u, v)
    return _perturb(G, r, r.randint(0, 2))


FAMILIES = [
    ("complete", gen_complete),
    ("cocktail_party", gen_cocktail),
    ("circulant", gen_circulant),
    ("random_regular", gen_random_regular),
    ("erdos_renyi", gen_erdos),
    ("hypercube", gen_hypercube),
    ("paley", gen_paley),
]


def build_one(i, r, max_n):
    name, gen = FAMILIES[i % len(FAMILIES)]
    G = finalize(gen(r, max_n))
    return name, G


# --------------------------------------------------------------------------- #
def rho(n):
    return max(4.0, 0.5 * math.log(n))


def evaluate(family, G):
    n = G.number_of_nodes()
    deg = [d for _, d in G.degree()]
    Delta, delta = max(deg), min(deg)
    gamma = exact_gamma(G)
    D = run_furones(G)
    valid = dominates(G, D)
    size = len(D)
    ratio = size / gamma if gamma else 1.0
    r = rho(n)
    return {
        "family": family,
        "n": n,
        "m": G.number_of_edges(),
        "delta": delta,
        "Delta": Delta,
        "high_degree": Delta > math.sqrt(n) / math.e,
        "gamma": gamma,
        "furones_size": size,
        "furones_valid": bool(valid),
        "ratio": round(ratio, 4),
        "tau": round(gamma * (Delta + 1) / n, 4),
        "rho_n": round(r, 4),
        "within_rho": bool(valid and ratio <= r + 1e-9),
        "optimal": bool(valid and size == gamma),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--max-n", type=int, default=16)
    args = ap.parse_args()

    rows = []
    for i in range(args.instances):
        seed_i = args.seed + i
        r = random.Random(seed_i)
        family, G = build_one(i, r, args.max_n)
        rec = evaluate(family, G)
        rec["index"] = i
        rec["seed"] = seed_i
        rows.append(rec)

    fams = {}
    for rec in rows:
        fams.setdefault(rec["family"], []).append(rec)

    families = []
    for key, rs in sorted(fams.items()):
        ratios = [x["ratio"] for x in rs]
        families.append({
            "family": key,
            "instances": len(rs),
            "n_range": [min(x["n"] for x in rs), max(x["n"] for x in rs)],
            "Delta_range": [min(x["Delta"] for x in rs), max(x["Delta"] for x in rs)],
            "tau_range": [min(x["tau"] for x in rs), max(x["tau"] for x in rs)],
            "all_valid": all(x["furones_valid"] for x in rs),
            "mean_ratio": round(sum(ratios) / len(ratios), 4),
            "max_ratio": round(max(ratios), 4),
            "pct_optimal": round(100.0 * sum(x["optimal"] for x in rs) / len(rs), 2),
            "pct_within_rho": round(100.0 * sum(x["within_rho"] for x in rs) / len(rs), 2),
        })

    overall = {
        "instances": len(rows),
        "all_high_degree": all(r["high_degree"] for r in rows),
        "all_valid": all(r["furones_valid"] for r in rows),
        "pct_within_rho": round(100.0 * sum(r["within_rho"] for r in rows) / len(rows), 3),
        "pct_optimal": round(100.0 * sum(r["optimal"] for r in rows) / len(rows), 3),
        "mean_ratio": round(sum(r["ratio"] for r in rows) / len(rows), 4),
        "max_ratio": round(max(r["ratio"] for r in rows), 4),
        "num_violations": sum(1 for r in rows if not r["within_rho"]),
    }

    manifest = {
        "car_id": "CAR-003-high-degree",
        "title": "High-degree / dense regime probe of the near-threshold ratio hypothesis",
        "generated": datetime.now(timezone.utc).isoformat(),
        "base_seed": args.seed,
        "instances_requested": args.instances,
        "solver": f"furones.algorithm.find_dominating_set (v{FURONES_VERSION}, full portfolio)",
        "rho_definition": "rho(n) = max(4, 0.5 * ln n)",
        "regime": "Delta > sqrt(n)/e (the regime left open by the low-degree proposition)",
        "note": ("Each instance is solved by the installed Furones solver (the full "
                 "portfolio) and compared against the exact domination number found by "
                 "increasing-size exhaustive search."),
        "families": families,
        "overall": overall,
    }

    with open("CAR-003-high-degree.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    cols = ["index", "seed", "family", "n", "m", "delta", "Delta", "high_degree",
            "gamma", "furones_size", "furones_valid", "ratio", "tau", "rho_n",
            "within_rho", "optimal"]
    with open("high_degree_by_instance.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for rec in rows:
            w.writerow({c: rec[c] for c in cols})

    print(f"furones version: {FURONES_VERSION}")
    print(f"instances: {overall['instances']}  (all high-degree: {overall['all_high_degree']}, all valid: {overall['all_valid']})")
    print(f"within rho(n): {overall['pct_within_rho']}%   optimal: {overall['pct_optimal']}%")
    print(f"mean ratio: {overall['mean_ratio']}   max ratio: {overall['max_ratio']}")
    print(f"violations (ratio > rho(n)): {overall['num_violations']}")
    print("wrote CAR-003-high-degree.json and high_degree_by_instance.csv")


if __name__ == "__main__":
    main()
