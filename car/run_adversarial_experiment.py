#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAR-004: Adversarial stress test of the Furones near-threshold ratio hypothesis.

Goal
----
Search for the WORST adversarial instances for the hypothesis

        |D_furones|  <=  rho(n) * gamma(G),     rho(n) = max(4, 0.5 * ln n),

by generating 10 000 small graphs that are deliberately designed to push the
ratio  |D_furones| / gamma(G)  as high as possible, while staying small enough
that the exact domination number gamma(G) is cheap to compute. The actual
Furones solver (v0.3.6) is invoked on every instance and compared against the
exact optimum.

Why these instances are "adversarial but feasible"
--------------------------------------------------
The families below are the ones theory flags as hardest (Theorem on the
high-degree regime + the (near-)perfect-code remark) plus the classic
greedy-defeating constructions:

  * greedy_trap          - k private blocks behind k centres, with decoy edges
                           that inflate non-centre degrees to mislead any
                           degree-driven heuristic; optimum stays tiny.
  * geometric_setcover   - domination version of the tight set-cover instance
                           where naive greedy is lured into a logarithmic chain
                           while two/three centres already dominate.
  * perfect_code_perturbed - hypercubes / circulants that admit an efficient
                           dominating set (tau = 1), lightly perturbed.
  * circulant_efficient  - circulants engineered to (nearly) admit a perfect
                           code, the residual hard case for the hypothesis.
  * cocktail_multipartite- complete multipartite / cocktail-party graphs whose
                           optimum is 2, stressing the constant floor of 4.
  * random_regular       - dense random regular graphs.
  * random_dense         - dense Erdos-Renyi graphs.

By construction every optimum is small (usually 1-5), so increasing-size
exhaustive search from the packing lower bound finds gamma(G) almost instantly,
keeping the 10 000-instance sweep feasible.

Note on rho(n): exact gamma is only feasible for small n, where 0.5*ln(n) < 4,
so the BINDING bound on every tested instance is the constant floor rho = 4.
This run therefore stress-tests precisely the constant-floor branch of the
hypothesis: can any adversary make Furones exceed 4x the optimum?

Reproducibility
---------------
Instance i is built from its own RNG seeded with (BASE_SEED + i), and the family
is chosen as i % (#families). The worst instances reported in the manifest can be
regenerated exactly from (family, index, seed, n).

Usage
-----
    pip install furones            # v0.3.6 already installed
    python run_adversarial_experiment.py [--instances 10000] [--seed 1000] [--max-n 16]

Outputs (written, not committed here):
    CAR-004-adversarial.json       per-family + overall summary, worst cases
    adversarial_by_instance.csv    one row per instance
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
# Furones invocation (robust to the optional epsilon/consistency arguments).
# --------------------------------------------------------------------------- #
def run_furones(G):
    """Call the installed Furones solver and return its result as a set."""
    try:
        D = find_dominating_set(G)
    except TypeError:
        # fall back to the (graph, epsilon) signature if required by the build
        D = find_dominating_set(G, 1.0)
    return set(D)


# --------------------------------------------------------------------------- #
# Exact domination number via increasing-size search from the packing bound.
# Fast here because every adversarial family keeps the optimum small.
# --------------------------------------------------------------------------- #
def exact_gamma(G):
    nodes = list(G.nodes())
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    full = (1 << n) - 1
    mask = []
    for v in nodes:
        m = 1 << idx[v]
        for w in G.neighbors(v):
            m |= 1 << idx[w]
        mask.append(m)
    Delta = max((G.degree(v) for v in nodes), default=0)
    lower = max(1, -(-n // (Delta + 1)))  # ceil(n / (Delta+1))
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


# --------------------------------------------------------------------------- #
# Helpers to keep instances clean (relabelled 0..n-1, no isolated vertices).
# --------------------------------------------------------------------------- #
def finalize(G):
    G = nx.convert_node_labels_to_integers(G)
    iso = [v for v in G.nodes() if G.degree(v) == 0]
    for v in iso:
        u = (v + 1) % G.number_of_nodes()
        if u != v:
            G.add_edge(v, u)
    return G


# --------------------------------------------------------------------------- #
# Adversarial graph families. Each takes an RNG and returns an nx.Graph.
# The exact optimum is recomputed afterwards, so perturbations are free.
# --------------------------------------------------------------------------- #
def greedy_trap(r):
    """k private blocks behind k centres, plus decoy edges to fool degree greedy."""
    k = r.randint(2, 4)
    G = nx.Graph()
    centres = []
    blocks = []
    nxt = 0
    for _ in range(k):
        c = nxt; nxt += 1
        centres.append(c)
        b = r.randint(2, 4)
        blk = list(range(nxt, nxt + b)); nxt += b
        blocks.append(blk)
        for v in blk:
            G.add_edge(c, v)
    # decoy edges: inflate non-centre degrees inside and across blocks
    allblock = [v for blk in blocks for v in blk]
    extra = r.randint(len(allblock) // 2, len(allblock) + 2)
    for _ in range(extra):
        u, v = r.sample(allblock, 2)
        G.add_edge(u, v)
    return finalize(G)


def geometric_setcover(r):
    """Domination version of the tight set-cover instance (geometric decoys)."""
    m = r.choice([6, 8, 10, 12])
    G = nx.Graph()
    elements = list(range(m))
    G.add_nodes_from(elements)
    # two optimal centres covering the two halves
    c1, c2 = m, m + 1
    for e in elements[: m // 2]:
        G.add_edge(c1, e)
    for e in elements[m // 2:]:
        G.add_edge(c2, e)
    # geometric decoy centres covering 1/2, 1/4, ... of the elements
    nxt = m + 2
    size = m // 2
    while size >= 1:
        d = nxt; nxt += 1
        targets = r.sample(elements, size)
        for e in targets:
            G.add_edge(d, e)
        size //= 2
    return finalize(G)


def perfect_code_perturbed(r):
    """Hypercube or circulant with an efficient dominating set, lightly perturbed."""
    if r.random() < 0.5:
        k = r.choice([3, 4])
        G = nx.hypercube_graph(k)
        G = nx.convert_node_labels_to_integers(G)
    else:
        n = r.choice([7, 9, 11, 13, 15])
        offs = [1, 2]
        G = nx.circulant_graph(n, offs)
    t = r.randint(0, 3)
    nodes = list(G.nodes())
    for _ in range(t):
        u, v = r.sample(nodes, 2)
        G.add_edge(u, v)
    return finalize(G)


def circulant_efficient(r):
    """Dense circulant tuned toward a (near-)perfect code."""
    n = r.choice([10, 12, 14, 16])
    d = r.randint(2, max(2, n // 3))
    offs = list(range(1, d + 1))
    G = nx.circulant_graph(n, offs)
    return finalize(G)


def cocktail_multipartite(r):
    """Cocktail-party or complete multipartite graph (optimum usually 2)."""
    if r.random() < 0.5:
        m = r.choice([3, 4, 5, 6, 7])  # K_{m x 2}
        G = nx.complete_multipartite_graph(*([2] * m))
    else:
        parts = [r.randint(2, 4) for _ in range(r.randint(2, 4))]
        G = nx.complete_multipartite_graph(*parts)
    return finalize(G)


def random_regular(r):
    n = r.choice([8, 10, 12, 14, 16])
    d = r.choice([3, 4, 5, max(3, n // 2)])
    if d >= n:
        d = n - 1
    if (n * d) % 2 == 1:
        d -= 1
    seed = r.randint(0, 2**31 - 1)
    G = nx.random_regular_graph(d, n, seed=seed)
    return finalize(G)


def random_dense(r):
    n = r.choice([8, 10, 12, 14, 16])
    p = r.uniform(0.3, 0.7)
    seed = r.randint(0, 2**31 - 1)
    G = nx.gnp_random_graph(n, p, seed=seed)
    return finalize(G)


FAMILIES = [
    ("greedy_trap", greedy_trap),
    ("geometric_setcover", geometric_setcover),
    ("perfect_code_perturbed", perfect_code_perturbed),
    ("circulant_efficient", circulant_efficient),
    ("cocktail_multipartite", cocktail_multipartite),
    ("random_regular", random_regular),
    ("random_dense", random_dense),
]


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
    ratio = size / gamma if gamma else float("inf")
    r = rho(n)
    return {
        "family": family,
        "n": n,
        "m": G.number_of_edges(),
        "delta": delta,
        "Delta": Delta,
        "gamma": gamma,
        "furones_size": size,
        "furones_valid": bool(valid),
        "ratio": round(ratio, 4),
        "tau": round(gamma * (Delta + 1) / n, 4),
        "rho_n": round(r, 4),
        "within_rho": valid and ratio <= r + 1e-9,
        "optimal": valid and size == gamma,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--max-n", type=int, default=16)
    args = ap.parse_args()

    rows = []
    for i in range(args.instances):
        seed_i = args.seed + i
        r = random.Random(seed_i)
        fam_name, gen = FAMILIES[i % len(FAMILIES)]
        G = gen(r)
        if G.number_of_nodes() > args.max_n:
            # keep exact optimum feasible
            G = nx.convert_node_labels_to_integers(G.subgraph(list(G.nodes())[: args.max_n]).copy())
            G = finalize(G)
        rec = evaluate(fam_name, G)
        rec["index"] = i
        rec["seed"] = seed_i
        rows.append(rec)

    # per-family aggregation
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
            "gamma_range": [min(x["gamma"] for x in rs), max(x["gamma"] for x in rs)],
            "all_valid": all(x["furones_valid"] for x in rs),
            "mean_ratio": round(sum(ratios) / len(ratios), 4),
            "max_ratio": round(max(ratios), 4),
            "pct_optimal": round(100.0 * sum(x["optimal"] for x in rs) / len(rs), 2),
            "pct_within_rho": round(100.0 * sum(x["within_rho"] for x in rs) / len(rs), 2),
        })

    worst = sorted(rows, key=lambda x: x["ratio"], reverse=True)[:15]
    violations = [x for x in rows if not x["within_rho"]]

    overall = {
        "instances": len(rows),
        "all_valid": all(x["furones_valid"] for x in rows),
        "pct_within_rho": round(100.0 * sum(x["within_rho"] for x in rows) / len(rows), 3),
        "pct_optimal": round(100.0 * sum(x["optimal"] for x in rows) / len(rows), 3),
        "mean_ratio": round(sum(x["ratio"] for x in rows) / len(rows), 4),
        "max_ratio": round(max(x["ratio"] for x in rows), 4),
        "num_violations": len(violations),
    }

    manifest = {
        "car_id": "CAR-004-adversarial",
        "title": "Adversarial worst-case stress test of the near-threshold ratio hypothesis",
        "generated": datetime.now(timezone.utc).isoformat(),
        "base_seed": args.seed,
        "instances_requested": args.instances,
        "solver": f"furones.algorithm.find_dominating_set (v{FURONES_VERSION}, full portfolio)",
        "rho_definition": "rho(n) = max(4, 0.5 * ln n)",
        "binding_bound_note": ("All feasible instance sizes have 0.5*ln(n) < 4, so the binding "
                               "bound is the constant floor rho = 4; this run stress-tests that floor."),
        "objective": "maximize the observed ratio |D_furones| / gamma(G) over adversarial families",
        "families": families,
        "overall": overall,
        "worst_instances": worst,
        "violations": violations,
    }

    with open("CAR-004-adversarial.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    cols = ["index", "seed", "family", "n", "m", "delta", "Delta", "gamma",
            "furones_size", "furones_valid", "ratio", "tau", "rho_n",
            "within_rho", "optimal"]
    with open("adversarial_by_instance.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for rec in rows:
            w.writerow({c: rec[c] for c in cols})

    print(f"instances: {overall['instances']}  all valid: {overall['all_valid']}")
    print(f"within rho(n): {overall['pct_within_rho']}%   optimal: {overall['pct_optimal']}%")
    print(f"mean ratio: {overall['mean_ratio']}   max ratio: {overall['max_ratio']}")
    print(f"violations (ratio > rho(n)): {overall['num_violations']}")
    print("wrote CAR-004-adversarial.json and adversarial_by_instance.csv")


if __name__ == "__main__":
    main()
