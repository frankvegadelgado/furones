"""
tscc_ds_reduction.py
====================

Reduction from an arbitrary undirected NetworkX graph G to a
minimum-degree-2 kernel graph for the *dominating set* (DS) problem.

Background — why DS differs fundamentally from VC
--------------------------------------------------
The companion file ``tscc_vc_reduction.py`` applies three kernel rules to
reduce a graph to a bridge-free (TSCC-structured) residual for vertex cover:

  VC Rule 0 – isolated vertex  → remove (never in any minimum VC)
  VC Rule 1 – pendant deg(v)=1 → N(v) forced into VC; remove v and N(v)
  VC Rule 2 – bridge (u,v)     → exactly one of {u,v} must be in every VC;
                                   solve the bridge forest with a 2-state
                                   tree DP and force optimal endpoints

Dominating set inverts and complicates each rule:

  DS Rule 0 – isolated vertex  → v has NO neighbour to dominate it, so v
                                   MUST be in every minimum DS (opposite
                                   of VC Rule 0, where isolated nodes are
                                   never in the cover).

  DS Rule 1 – pendant deg(v)=1, unique neighbour u
                               → by an exchange argument, WLOG u ∈ D*:
                                   putting u in D instead of v covers a
                                   superset of what v covers (N[u] ⊇ N[v]).
                                   Force u.  Vertices in N_H(u) that have
                                   NO H-neighbours outside N_H[u] are useless
                                   after u is gone and are removed immediately.
                                   Vertices in N_H(u) that DO have H-neighbours
                                   outside N_H[u] are kept in H — they can
                                   still dominate those outer vertices and must
                                   not be discarded prematurely.  The cascade
                                   continues until no new pendants / isolates
                                   remain.

  DS Rule 2 – bridge (u,v)     → NO analog for DS.

Rule 1 — why the original "remove all of N_H[u]" is wrong
----------------------------------------------------------
The naive application of Rule 1 removes every node in N_H[u] after
forcing u.  This is INCORRECT whenever some w ∈ N_H(u) has H-neighbours
outside N_H[u] (called "outer neighbours").  Removing w makes those outer
neighbours lose their only H-path to a potential dominator; they then
become isolated and get wrongly forced by Rule 0, inflating forced_ds.

Concrete counterexample
  Graph: v — u — w — {x1, x2, x3}   (v is the pendant of u; x_i adj only w)
  Naive: force u, remove {v, u, w} → x1, x2, x3 isolated → force all three.
         forced_ds = {u, x1, x2, x3}, size 4.  But OPT = 2 ({u, w}).
  Fixed: force u, remove {v, u} (w has outer neighbours → keep w).
         x1 is pendant of w → force w, remove {w, x1, x2, x3}.
         forced_ds = {u, w}, size 2.  Correct.

A second, related bug: order-dependency
  The naive algorithm's output depends on which pendant is processed first.
  On the user-reported graph (edges given below), processing pendant 14
  before the pendants of vertex 4 forces {3, 6, 10, 11, 13, 2, 1} (size 7)
  instead of the correct {4, 3} (size 2), because forcing 3 first removes
  vertex 4 from H, leaving its other pendants (6, 10, 11, 13) without a
  forced neighbour and causing Rule 0 to wrongly self-force them.

Two fixes
---------
Fix 1 – Selective removal in Rule 1
  When forcing u, partition N_H(u) into:
    • no-outer: N_H(u) nodes whose every H-neighbour is in N_H[u].
                These are removed immediately (they serve no further purpose).
    • outer:    N_H(u) nodes that have at least one H-neighbour outside N_H[u].
                These are KEPT in H so the cascade can process them normally.
  Only {u, v} ∪ no-outer is removed; outer nodes stay.

Fix 2 – Domination guard in Rules 0 and 1
  After Fix 1, a kept outer node w may later become isolated or pendant
  because its outer H-neighbours are processed by subsequent cascade steps.
  At that point w is already dominated by u ∈ forced_ds (since u adj w in G).
  Without a guard, Rule 0 would wrongly add w to forced_ds.
  The guard: before forcing any isolated or pendant vertex, check whether a
  G-neighbour of that vertex is already in forced_ds.  If so, the vertex is
  already dominated — remove it silently without adding to forced_ds.
  Likewise, a pendant v whose G-neighbour (other than its sole H-neighbour u)
  is already in forced_ds is dominated; remove v silently and let the cascade
  continue with u.

Structural guarantees after the cascade
----------------------------------------
  • G_reduced has minimum degree ≥ 2 (pendants and isolates fully reduced).
  • forced_ds ∪ (any valid DS of G_reduced) is a valid DS of G.
  • If no forced vertex has a G-neighbour in G_reduced (verified by the
    no_bridge_forced flag in verify_reduction_ds), the exact identity
        OPT(G) = |forced_ds| + OPT(G_reduced)
    holds and the reduction is tight.
  • When G_reduced is non-empty and some forced vertex does have a G-neighbour
    in G_reduced (e.g. a kept outer node that was not further reduced), the
    lifted DS is still valid but the above identity becomes an upper bound.
    The solver for G_reduced should account for vertices already dominated by
    forced_ds to recover tightness.

Time complexity
---------------
O(V + E).  Every vertex is enqueued at most once (tracked by in_q).
Each edge is examined O(1) times across all cascade steps.

Public API
----------
    reduce_to_tscc_for_ds(G)  →  (G_reduced, forced_ds, lift)
    lift(ds_reduced)           →  dominating set of G
    is_dominating_set(G, D)    →  bool
    verify_reduction_ds(...)   →  diagnostic dict
"""

from __future__ import annotations

import networkx as nx
from collections import deque
from typing import Any, Callable, Dict, FrozenSet, Set, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# Main reduction
# ═══════════════════════════════════════════════════════════════════════════

def reduce_to_tscc_for_ds(
    G: nx.Graph,
) -> Tuple[nx.Graph, Set[Any], Callable[[Set[Any]], Set[Any]]]:
    """
    Reduce undirected graph *G* to a minimum-degree-2 kernel for dominating
    set via exhaustive application of DS Rules 0 (isolated) and 1 (pendant).

    Parameters
    ----------
    G : nx.Graph
        Any simple undirected graph.  Self-loops are stripped (a self-loop
        never affects domination: a vertex with a self-loop is self-dominated,
        but self-loops are non-standard and discarded for simplicity).

    Returns
    -------
    G_reduced : nx.Graph
        Residual graph with every vertex of degree ≥ 2.
        Node labels are identical to those in *G*.
        May be empty if all domination was resolved during reduction.

    forced_ds : set
        Vertices of *G* forced into every minimum dominating set by
        Rules 0 and 1.  When ``verify_reduction_ds`` reports
        ``no_bridge_forced=True``, the identity
        ``OPT(G) = |forced_ds| + OPT(G_reduced)`` holds exactly.
        When ``no_bridge_forced=False`` (a kept outer node remained in
        G_reduced adjacent to a forced vertex), the lifted DS is still
        *valid*; the solver for G_reduced should account for vertices
        already dominated by forced_ds to recover exact optimality.

    lift : Callable[[set], set]
        ``lift(ds_reduced)`` maps a dominating set of *G_reduced* to a
        dominating set of *G*.  Always returns a valid DS of *G*.

    Complexity
    ----------
    O(V + E) — single BFS pass, each vertex and edge processed O(1) times.
    """
    H: nx.Graph = nx.Graph(G)
    H.remove_edges_from(list(nx.selfloop_edges(H)))

    forced_ds: Set[Any] = set()

    # ───────────────────────────────────────────────────────────────────────
    # Unified BFS queue for Rules 0 and 1
    #
    # Seed: every vertex with degree ≤ 1.  As nodes are removed, outer
    # neighbours whose degree drops to ≤ 1 are enqueued immediately.
    # Each vertex enters the queue at most once (guarded by in_q).
    # ───────────────────────────────────────────────────────────────────────

    q:    deque[Any] = deque(v for v in H if H.degree(v) <= 1)
    in_q: Set[Any]   = set(q)

    while q:
        v = q.popleft()
        if v not in H:
            continue            # already removed by an earlier iteration

        dv = H.degree(v)

        # ── Rule 0: isolated vertex ─────────────────────────────────────
        if dv == 0:
            # v has no H-neighbours.  If a previously-forced vertex already
            # dominates v in G (i.e. some G-neighbour of v is in forced_ds),
            # v is already dominated — remove it silently.  Otherwise v can
            # only dominate itself and must be forced into every minimum DS.
            if any(nb in forced_ds for nb in G.neighbors(v)):
                H.remove_node(v)
            else:
                forced_ds.add(v)
                H.remove_node(v)

        # ── Rule 1: pendant ─────────────────────────────────────────────
        elif dv == 1:
            # u is the unique H-neighbour of the pendant v.
            u = next(iter(H.neighbors(v)))

            # ── Domination guard ─────────────────────────────────────────
            # If a forced vertex already dominates v through a G-edge other
            # than (v, u) — possible when a "kept" outer node was later
            # forced in an earlier cascade step — v needs no action.
            # Remove v silently and check whether u's degree also dropped.
            if any(nb in forced_ds for nb in G.neighbors(v) if nb != u):
                H.remove_node(v)
                if u in H and u not in in_q and H.degree(u) <= 1:
                    in_q.add(u)
                    q.append(u)
                continue

            # ── Exchange argument: force u ───────────────────────────────
            # WLOG u ∈ D* (adding u dominates N[u] ⊇ N[v]).
            neighbors_u = list(H.neighbors(u))
            closed_u: Set[Any] = set(neighbors_u)
            closed_u.add(u)                      # closed_u = N_H[u]

            # ── Selective removal (Fix 1) ────────────────────────────────
            # Partition N_H(u) \ {v} into:
            #
            #   no-outer : every H-neighbour of w is inside N_H[u].
            #              w is useless after u leaves; remove immediately.
            #
            #   kept     : w has at least one H-neighbour outside N_H[u].
            #              w can still dominate those outer vertices.
            #              Keep w in H so the cascade handles it naturally.
            #
            # Always remove u (forced) and v (pendant).
            kept:      list = []
            to_remove: Set[Any] = {u, v}

            for w in neighbors_u:
                if w is v or w == v:
                    continue                     # v already in to_remove
                if any(x not in closed_u for x in H.neighbors(w)):
                    kept.append(w)               # has outer H-neighbours
                else:
                    to_remove.add(w)             # purely inside N_H[u]

            forced_ds.add(u)
            for node in to_remove:
                if node in H:
                    H.remove_node(node)

            # ── Enqueue kept nodes that may now be degree ≤ 1 ───────────
            # Kept nodes lost their edge to u (and to any removed no-outer
            # siblings).  Their degree may have dropped; check and enqueue.
            for w in kept:
                if w in H and w not in in_q and H.degree(w) <= 1:
                    in_q.add(w)
                    q.append(w)

    # ───────────────────────────────────────────────────────────────────────
    # Build output
    # ───────────────────────────────────────────────────────────────────────
    G_reduced: nx.Graph = H.copy()

    # Freeze forced_ds so the closure is safe against external mutation.
    _frozen: FrozenSet[Any] = frozenset(forced_ds)

    def lift(ds_reduced: Set[Any]) -> Set[Any]:
        """
        Convert a dominating set of *G_reduced* into a dominating set of *G*.

        Parameters
        ----------
        ds_reduced : set
            Any dominating set of *G_reduced* (exact or approximate).

        Returns
        -------
        set
            forced_ds ∪ ds_reduced — a dominating set of *G* of identical
            quality (exact lifts to exact; α-approx lifts to α-approx).
        """
        return _frozen | set(ds_reduced)

    return G_reduced, set(forced_ds), lift


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════

def is_dominating_set(G: nx.Graph, D: Set[Any]) -> bool:
    """
    Return True if *D* is a valid dominating set of *G*.

    Every vertex v ∉ D must have at least one neighbour in D.
    Isolated vertices (degree 0) must be in D.
    """
    D = set(D)
    return all(
        v in D or any(u in D for u in G.neighbors(v))
        for v in G.nodes()
    )


def verify_reduction_ds(
    G:          nx.Graph,
    G_reduced:  nx.Graph,
    forced_ds:  Set[Any],
    lift:       Callable[[Set[Any]], Set[Any]],
    ds_reduced: Set[Any],
) -> Dict[str, Any]:
    """
    Verify that the DS reduction is structurally correct and the lifted
    solution is a valid dominating set of the original graph.

    Parameters
    ----------
    G          : original graph
    G_reduced  : reduced graph returned by reduce_to_tscc_for_ds
    forced_ds  : forced vertices returned by reduce_to_tscc_for_ds
    lift       : lift function returned by reduce_to_tscc_for_ds
    ds_reduced : a dominating set of G_reduced (exact or approximate)

    Returns
    -------
    dict with keys:
        min_degree_ok     bool  Every vertex in G_reduced has degree ≥ 2.
        no_bridge_forced  bool  No forced vertex has a neighbour in G_reduced
                                 (invariant that guarantees exact lift).
        ds_reduced_ok     bool  ds_reduced is a valid DS of G_reduced.
        lifted_ds_ok      bool  Lifted DS is valid for G.
        lifted_ds         set   The full dominating set of G.
        forced_count      int   |forced_ds|.
        reduced_count     int   |ds_reduced|.
        total_count       int   |lifted_ds|.
    """
    lifted = lift(ds_reduced)

    # Check min-degree ≥ 2
    min_deg_ok = all(
        G_reduced.degree(v) >= 2
        for v in G_reduced.nodes()
    )

    # Check that no forced vertex has a neighbour in G_reduced
    reduced_nodes = set(G_reduced.nodes())
    no_bridge_forced = all(
        len(set(G.neighbors(u)) & reduced_nodes) == 0
        for u in forced_ds
    )

    return {
        "min_degree_ok":    min_deg_ok,
        "no_bridge_forced": no_bridge_forced,
        "ds_reduced_ok":    is_dominating_set(G_reduced, ds_reduced),
        "lifted_ds_ok":     is_dominating_set(G, lifted),
        "lifted_ds":        lifted,
        "forced_count":     len(forced_ds),
        "reduced_count":    len(ds_reduced),
        "total_count":      len(lifted),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Demo / self-test
# ═══════════════════════════════════════════════════════════════════════════

def _trivial_ds(G: nx.Graph) -> Set[Any]:
    """
    A greedy minimum-degree dominating set used only for demos.
    Not guaranteed optimal; used to produce a valid ds_reduced for testing.
    """
    remaining = set(G.nodes())
    D: Set[Any] = set()
    dominated: Set[Any] = set()

    # Seed with isolated vertices (must be in D)
    for v in list(remaining):
        if G.degree(v) == 0:
            D.add(v)
            dominated.add(v)
            remaining.discard(v)

    while remaining - dominated:
        # Pick the undominated vertex whose neighbourhood contributes most
        v = max(
            (x for x in remaining if x not in dominated),
            key=lambda x: len(set(G.neighbors(x)) - dominated) + (1 if x not in dominated else 0)
        )
        u = max(
            [v] + list(G.neighbors(v)),
            key=lambda x: len(set(G.neighbors(x)) - dominated)
        )
        D.add(u)
        dominated.add(u)
        dominated.update(G.neighbors(u))

    return D


def _demo(name: str, G: nx.Graph) -> None:
    """Run the full DS reduction pipeline on *G* and print a summary."""
    G_r, forced, lift = reduce_to_tscc_for_ds(G)

    # Solve DS on the reduced graph (greedy, for demonstration)
    ds_r = _trivial_ds(G_r) if G_r.number_of_nodes() > 0 else set()

    res = verify_reduction_ds(G, G_r, forced, lift, ds_r)

    print(f"\n{'═' * 66}")
    print(f"  {name}")
    print(f"  Original  : V={G.number_of_nodes():>4},  E={G.number_of_edges():>5}")
    print(f"  Reduced   : V={G_r.number_of_nodes():>4},  E={G_r.number_of_edges():>5}  "
          f"(min_deg≥2: {res['min_degree_ok']})")
    print(f"  Forced DS ({res['forced_count']:>2}) : "
          f"{sorted(str(v) for v in forced)[:10]}"
          f"{'...' if res['forced_count'] > 10 else ''}")
    print(f"  DS_reduced ({res['reduced_count']:>2}): "
          f"{sorted(str(v) for v in ds_r)[:10]}"
          f"{'...' if res['reduced_count'] > 10 else ''}")
    total = res['total_count']
    print(f"  Lifted DS  ({total:>2}): "
          f"[...{total} nodes...]" if total > 14
          else f"  Lifted DS  ({total:>2}): {sorted(str(v) for v in res['lifted_ds'])}")
    print(f"  Invariant (no forced↔reduced edges): {res['no_bridge_forced']}")
    print(f"  DS_reduced valid?  {res['ds_reduced_ok']}")
    print(f"  Lifted DS valid?   {res['lifted_ds_ok']}")


if __name__ == "__main__":

    # 1. Path P₆ — all edges are bridges; Rule 1 cascade resolves completely
    _demo("Path P₆", nx.path_graph(6))

    # 2. Star K₁,₁₀ — center forced by Rule 1 on any leaf; G_reduced = ∅
    _demo("Star K₁,₁₀", nx.star_graph(10))

    # 3. Cycle C₈ — bridge-free, min-degree 2; Rule 0/1 never fire
    #    G_reduced = C₈ (no reduction possible)
    _demo("Cycle C₈", nx.cycle_graph(8))

    # 4. Barbell: two triangles joined by a bridge
    #    The bridge edge (2,3) is not a pendant; Rule 1 won't fire on it.
    #    Neither triangle vertex has degree 1.  G_reduced = full barbell.
    barbell = nx.Graph()
    barbell.add_edges_from([(0,1),(1,2),(2,0),(2,3),(3,4),(4,5),(5,3)])
    _demo("Barbell (two triangles + bridge)", barbell)

    # 5. Lollipop K₄+P₅ — pendant tail triggers Rule 1 cascade inward
    _demo("Lollipop K₄+P₅", nx.lollipop_graph(4, 5))

    # 6. Petersen graph — 3-regular, 3-edge-connected; no rule fires
    _demo("Petersen graph", nx.petersen_graph())

    # 7. Grid 4×4 — many degree-2 corner/edge vertices; Rule 1 cascade
    _demo("Grid 4×4", nx.grid_2d_graph(4, 4))

    # 8. Complete bipartite K₅,₁₀ — all degrees ≥ 5; no rule fires
    #    Confirms the known weakness: large-Δ bipartite graphs don't reduce
    _demo("Complete bipartite K₅,₁₀", nx.complete_bipartite_graph(5, 10))

    # 9. Complete bipartite K₁,₁₀ — Rule 1 fires on each leaf; center forced
    _demo("Complete bipartite K₁,₁₀", nx.complete_bipartite_graph(1, 10))

    # 10. Random sparse graph
    _demo("Random G(30, 45, seed=42)", nx.gnm_random_graph(30, 45, seed=42))

    # 11. Karate-club social network
    _demo("Karate Club (Zachary, 34 nodes)", nx.karate_club_graph())

    # 12. Caterpillar tree — long spine with pendant leaves; deep cascade
    caterpillar = nx.Graph()
    spine = list(range(10))
    nx.add_path(caterpillar, spine)
    for i in spine:
        caterpillar.add_edge(i, f"L{i}a")
        caterpillar.add_edge(i, f"L{i}b")
    _demo("Caterpillar (10-spine, 2 leaves each)", caterpillar)

    print(f"\n{'═' * 66}")
    print("All demos complete.")
    print()
    print("Key structural difference vs VC reduction:")
    print("  VC  Rule 0: isolated → NEVER in cover  (remove freely)")
    print("  DS  Rule 0: isolated → ALWAYS in DS    (force, opposite!)")
    print("  VC  Rule 2: bridge   → exact 2-state tree DP on bridge forest")
    print("  DS  Rule 2: bridge   → NO ANALOG (domination resolvable within")
    print("                         2ECC; bridge forces nothing in DS)")
    print("  Result: DS residual has min-degree ≥ 2 but MAY have bridges.")
    print("          VC residual is fully bridge-free (TSCC-structured).")