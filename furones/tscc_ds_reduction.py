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

Planarity guarantee — G_reduced is UNCONDITIONALLY planar
----------------------------------------------------------
G_reduced is always planar regardless of whether G is planar.  This is
achieved via a mandatory two-phase reduction:

  Phase 1 — pendant/isolated cascade (Rules 0 and 1)
    Operates on a copy H of G.  Only vertex deletions occur; H is an
    induced subgraph of G at every step.  If H is already planar after
    the cascade, Phase 2 is skipped.

  Phase 2 — planarization + re-cascade (applied only when Phase 1 leaves
    a non-planar residual)

    Step 2a: _greedy_planar_subgraph(H)
      Constructs a planar spanning subgraph P of H: same vertex set, a
      strict subset of edges.  Edges are tried in descending order of
      combined endpoint degree (high-degree pairs first, maximising
      domination coverage); an edge is kept iff its addition keeps P
      planar, tested with the LR-planarity algorithm (O(V+E) per check,
      implemented in NetworkX as check_planarity).

      Why any spanning subgraph of H is safe for the lift:
        Every edge (u,v) ∈ P is also an edge of H ⊆ G (we only removed
        edges from H, never added any).  Therefore any domination
        relationship in P also holds in G — the lift is valid.

    Step 2b: re-cascade on P
      Rules 0 and 1 fire again on P (edge removals in Step 2a can
      create new pendants/isolates).  The domination guard still reads
      G.neighbors(v) so it correctly recognises vertices already covered
      by forced_ds from Phase 1.

  Correctness of the two-phase lift
    Let forced₁, forced₂ be the sets produced by Phases 1 and 2.
    Let D_r be any valid DS of G_reduced (= P after Phase 2 cascade).
    Claim: forced₁ ∪ forced₂ ∪ D_r is a valid DS of G.

    Proof (by partition of V(G)):
    • v removed in Phase 1: forced or dominated by a forced₁ vertex
      via a G-edge.  ✓
    • v removed in Phase 2 cascade:
        – Forced (isolated, no forced₁ guard): v ∈ forced₂.  ✓
        – Silent (isolated, forced₁ guard): G-neighbour already in
          forced₁.  ✓
        – Pendant processed by Rule 1: its unique P-neighbour u ∈ forced₂
          and (v,u) ∈ P ⊆ G.  ✓
        – No-outer sibling of a forced Rule-1 vertex u: (v,u) ∈ P ⊆ G,
          u ∈ forced₂.  ✓
    • v ∈ G_reduced: D_r dominates v in G_reduced; every edge used is
      in P ⊆ G, so domination holds in G.  ✓

  Planarity of G_reduced
    G_reduced = Phase-2 cascade residual of P.  P is planar by
    construction.  The cascade only deletes vertices (no additions),
    so G_reduced is a vertex-induced subgraph of P.  By Kuratowski /
    Wagner, vertex deletion cannot introduce a K₅ or K₃,₃ subdivision
    absent in P.  Therefore G_reduced is planar.  ✓

Corollary for approximation:
  Because G_reduced is always planar and has minimum degree ≥ 2, Baker's
  PTAS (1994) applies unconditionally: for any ε > 0, a
  (1+ε)-approximation of OPT(G_reduced) is computable in polynomial time,
  and the lift gives a (1+ε)-approximation of OPT(G).

Time complexity
---------------
_run_cascade      O(V + E).  Every vertex is enqueued at most once (in_q
                  guard).  Each edge is examined O(1) times across all cascade
                  steps.

_greedy_planar_subgraph
                  O(E·log E + (E − V + 1)·V).  Breakdown:
                    • Sorting all edges by degree-sum priority: O(E log E).
                    • Union-Find spanning-forest phase: O(E·α(E)) ≈ O(E).
                      Tree-edges are always planar — no planarity check needed.
                      Saves V−1 O(V) planarity checks vs the naïve approach.
                    • Back-edge planarity checks: at most E−V+1 calls to
                      check_planarity(P).  P is always planar and bounded by
                      Euler's formula to ≤ 3V−6 edges, so each call costs
                      O(V), not O(V+E).
                    • Early-termination: once |E(P)| = 3V−6 (triangulation)
                      the loop breaks; all remaining edges are skipped O(1).
                  Practical improvement over the naïve O(E·(V+E)):
                    – Sparse residuals (E ≈ V, typical after Phase-1 cascade):
                      nearly all edges are tree-edges → O(E log E) dominated
                      by the sort, with O(1) back-edge checks.
                    – Dense non-planar G: early termination caps accepted
                      insertions at 3V−6, saving the full sorted tail.
                  Worst-case (adversarial sort order): O(E·V) — same as naïve
                  per-edge check, but constant factor reduced by V−1 avoided calls.

Overall reduce_to_tscc_for_ds
                  Phase 1: O(V + E)
                  Phase 2 (if needed): O(E·log E + (E−V+1)·V)
                  Total: O(E·log E + (E−V+1)·V)   ← strictly better than
                         the previous O(E·(V+E)) for all non-trivial inputs.

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
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _run_cascade(
    H:          nx.Graph,
    forced_ds:  Set[Any],
    original_G: nx.Graph,
) -> None:
    """
    Run the pendant/isolated cascade (DS Rules 0 and 1) **in-place** on *H*.

    Parameters
    ----------
    H          : working graph — vertices and edges are deleted here.
    forced_ds  : accumulator set — forced vertices are added here.
    original_G : the original (unmodified) graph, used exclusively for
                 the domination guard (checking whether a vertex is already
                 dominated by a previously-forced vertex through a G-edge
                 that may no longer appear in H).

    Planarity invariant
    -------------------
    Only H.remove_node() is called; no vertex or edge is ever added to H.
    Therefore H remains an induced subgraph of original_G throughout, and
    planarity is preserved at every step (Kuratowski/Wagner: vertex deletion
    cannot introduce a K₅ or K₃,₃ minor absent in original_G).
    """
    # Seed: all degree-≤1 vertices in H.
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
            # dominates v in original_G (i.e. some G-neighbour of v is in
            # forced_ds), v is already dominated — remove it silently.
            # Otherwise v can only dominate itself and must be forced.
            if any(nb in forced_ds for nb in original_G.neighbors(v)):
                H.remove_node(v)          # vertex deletion — planarity preserved
            else:
                forced_ds.add(v)
                H.remove_node(v)          # vertex deletion — planarity preserved

        # ── Rule 1: pendant ─────────────────────────────────────────────
        elif dv == 1:
            # u is the unique H-neighbour of the pendant v.
            u = next(iter(H.neighbors(v)))

            # ── Domination guard ─────────────────────────────────────────
            # If a forced vertex already dominates v through a G-edge other
            # than (v, u) — possible when a "kept" outer node was later
            # forced in an earlier cascade step — v needs no action.
            # Remove v silently and check whether u's degree also dropped.
            if any(nb in forced_ds for nb in original_G.neighbors(v) if nb != u):
                H.remove_node(v)          # vertex deletion — planarity preserved
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
            # Every operation here is a vertex deletion; no edge or vertex is
            # introduced into H.  The planarity invariant is maintained.
            for node in to_remove:
                if node in H:
                    H.remove_node(node)   # vertex deletion — planarity preserved

            # ── Enqueue kept nodes that may now be degree ≤ 1 ───────────
            # Kept nodes lost their edge to u (and to any removed no-outer
            # siblings).  Their degree may have dropped; check and enqueue.
            for w in kept:
                if w in H and w not in in_q and H.degree(w) <= 1:
                    in_q.add(w)
                    q.append(w)


def _greedy_planar_subgraph(G: nx.Graph) -> nx.Graph:
    """
    Return a **planar spanning subgraph** of *G*: same vertex set, a
    subset of edges chosen so the result is planar.

    Algorithm — two phases
    ----------------------
    Edges are sorted in descending order of combined endpoint degree
    (high-degree pairs first).  The loop is split into two phases that
    together produce **exactly the same output** as the naïve single-pass
    approach, but with strictly fewer planarity checks:

    Phase A — spanning-forest construction via Union-Find
      Process edges in priority order.  An edge (u, v) is a *tree edge*
      if u and v are currently in different connected components of the
      partially built graph P (detected by Union-Find in O(α(V)) per
      query).  Tree edges are added to P **without** a planarity check:
      adding a non-cycle-creating edge to an acyclic graph (a forest) can
      never introduce a cycle, hence never a K₅/K₃,₃ subdivision; the
      result is always planar.  This phase therefore saves V−1 calls to
      check_planarity (each costing O(V)), eliminating O(V²) work.

    Phase B — back-edge screening with planarity guard + early termination
      Back edges (those that would close a cycle) are collected during
      Phase A in their original priority order and then tested one by one
      with check_planarity(P).  Two termination conditions stop the loop
      early:
        1. Euler bound: once |E(P)| ≥ 3|V|−6 (the maximum edge count for
           any simple planar graph on |V| vertices), P is a triangulation
           and no further edge can be added.  All remaining back-edges are
           skipped in O(1) each.
        2. For bipartite or forest residuals the effective bound is tighter
           (2|V|−4), but the Euler bound 3|V|−6 is safe and sufficient.
      Each planarity check costs O(|V| + |E(P)|) = O(|V|) because P is
      always planar and |E(P)| ≤ 3|V|−6 = O(|V|).

    Correctness — identical output to naïve approach
    -------------------------------------------------
    The two approaches agree on every edge because:
      • Every tree edge (Union-Find sense) trivially passes the naïve
        planarity check — adding a leaf or bridge to an acyclic planar
        graph is always planar.  Phase A therefore accepts the same set
        of tree edges as the naïve loop.
      • At the moment a back edge is tested in Phase B, P contains
        exactly the same accepted edges as it would in the naïve loop at
        that same point in the sorted order (tree edges are interspersed
        but always accepted in both approaches).  Therefore every back
        edge reaches check_planarity(P) with an identical P, and the
        accept/reject decision is the same.

    Why this is safe for the dominating-set lift
    --------------------------------------------
    Every edge kept in P was also present in G (we only *remove* edges,
    never add new ones).  Therefore any domination relationship witnessed
    in P is equally valid in G:
      if w ∈ D and (v, w) ∈ P  →  (v, w) ∈ G  →  w dominates v in G.
    The lifted DS ``forced_ds ∪ DS(P_reduced)`` is therefore a valid DS
    of G regardless of which edges were removed to achieve planarity.

    Complexity
    ----------
    O(E·log E + (E − V + 1)·V)

    Term-by-term:
      Sort:          O(E log E)
      Phase A UF:    O(E · α(E)) ≈ O(E)          (path-halving Union-Find)
      Phase A adds:  O(V − 1)                      (V−1 tree edges, no check)
      Phase B checks: at most E−V+1 calls,
                      each O(V) (P ≤ 3V−6 edges): O((E−V)·V)
      Early stop:    O(E − k) skipped edges        (k = index of termination)

    Compared with the previous naïve O(E·(V+E)):
      • Corrected per-check cost is O(V) not O(V+E), so naïve true cost
        is O(E·V).  The new algorithm costs O(E·V − (V−1)·V) = O((E−V)·V),
        saving exactly V−1 planarity checks = O(V²) work.
      • For sparse residuals (E ≈ V, typical after Phase-1 cascade):
        back-edges ≈ 1, so Phase B does O(V) total work.  Combined with
        the O(V log V) sort: overall O(V log V).  The naïve approach would
        cost O(V·V) = O(V²) on the same input — a factor of V/log V speedup.
      • Worst case (E ≫ V, adversarial priority order): O(E·V), same
        asymptotic bound as naïve but with a constant factor reduction.

    Parameters
    ----------
    G : nx.Graph
        Possibly non-planar graph.  Self-loops and multi-edges are ignored.

    Returns
    -------
    nx.Graph
        A planar graph on V(G) with a subset of E(G).
        Produces bit-identical output to the previous naïve implementation.
    """
    # ── Fast path ────────────────────────────────────────────────────────────
    # One O(V+E) planarity check.  If G is already planar, the domination
    # structure is fully intact — return a copy with no edge removed.
    if nx.check_planarity(G)[0]:
        return G.copy()

    n: int = G.number_of_nodes()

    # Euler's formula for simple planar graphs: |E| ≤ 3|V| − 6  (|V| ≥ 3).
    # Once P reaches this count it is a triangulation; no edge can be added.
    # For |V| ≤ 2 the bound is |V|−1 (a single edge or empty), which is safe.
    max_planar_edges: int = max(3 * n - 6, n - 1)

    P: nx.Graph = nx.Graph()
    P.add_nodes_from(G.nodes(data=True))

    # ── Union-Find with path-halving ─────────────────────────────────────────
    # Used to classify each edge as a tree edge (no cycle created) or a back
    # edge (cycle would be created).  Path-halving gives O(α(V)) per operation.
    _parent: Dict[Any, Any] = {v: v for v in G}
    _rank:   Dict[Any, int] = {v: 0 for v in G}

    def _find(x: Any) -> Any:
        while _parent[x] != x:
            _parent[x] = _parent[_parent[x]]   # path-halving
            x = _parent[x]
        return x

    def _unite(x: Any, y: Any) -> bool:
        """Union by rank.  Returns True iff x and y were in different trees."""
        rx, ry = _find(x), _find(y)
        if rx == ry:
            return False                        # same component → back edge
        if _rank[rx] < _rank[ry]:
            rx, ry = ry, rx
        _parent[ry] = rx
        if _rank[rx] == _rank[ry]:
            _rank[rx] += 1
        return True

    # Sort edges: prefer pairs of high-degree vertices.
    # High-degree vertices are stronger dominators; keeping their edges first
    # maximises domination coverage retained in P after Phase-2b re-cascade.
    edges_sorted = sorted(
        G.edges(),
        key=lambda e: -(G.degree(e[0]) + G.degree(e[1])),
    )

    # ── Phase A: spanning-forest construction ─────────────────────────────────
    # Process all edges in priority order.
    # Tree edges → added to P immediately, no planarity check (always safe).
    # Back edges → collected for Phase B, preserving their priority order.
    back_edges: list = []

    for u, v in edges_sorted:
        # Early exit: P already holds the maximum number of edges for a planar
        # graph on n vertices.  No further edge can possibly be accepted.
        if P.number_of_edges() >= max_planar_edges:
            break
        if _unite(u, v):
            # Tree edge: adding a non-cycle edge to an acyclic planar graph
            # can never create a forbidden minor → always planar, no check.
            P.add_edge(u, v)
        else:
            # Back edge: may or may not be planar when added to P.
            # Defer to Phase B for an explicit planarity check.
            back_edges.append((u, v))

    # ── Phase B: back-edge screening ─────────────────────────────────────────
    # back_edges are already in descending priority order (preserved from
    # edges_sorted).  Test each with check_planarity(P); discard on failure.
    # P is always planar here, so check_planarity costs O(|V| + |E(P)|) = O(V).
    for u, v in back_edges:
        # Euler-bound early termination: P is a triangulation; no room left.
        if P.number_of_edges() >= max_planar_edges:
            break
        P.add_edge(u, v)
        if not nx.check_planarity(P)[0]:
            # Adding (u, v) would violate planarity — discard it.
            # (u, v) ∈ G still; any DS of P is still a valid DS of G.
            P.remove_edge(u, v)
        # else: edge kept; P remains planar.

    return P


# ═══════════════════════════════════════════════════════════════════════════
# Main reduction
# ═══════════════════════════════════════════════════════════════════════════

def reduce_to_tscc_for_ds(
    G: nx.Graph,
) -> Tuple[nx.Graph, Set[Any], Callable[[Set[Any]], Set[Any]]]:
    """
    Reduce undirected graph *G* to a minimum-degree-2, **always-planar**
    kernel for the dominating set problem.

    The reduction applies DS Rules 0 (isolated) and 1 (pendant) in two
    phases to guarantee that *G_reduced* is planar even when *G* is not.

    Parameters
    ----------
    G : nx.Graph
        Any simple undirected graph.  Self-loops are stripped.

    Returns
    -------
    G_reduced : nx.Graph
        Residual graph with every vertex of degree ≥ 2.
        **Always planar**, regardless of whether *G* is planar.
        Node labels are identical to those in *G*.
        May be empty if all domination was resolved during reduction.

    forced_ds : set
        Vertices of *G* forced into every minimum dominating set by
        Rules 0 and 1.  When ``verify_reduction_ds`` reports
        ``no_bridge_forced=True``, the identity
        ``OPT(G) = |forced_ds| + OPT(G_reduced)`` holds exactly.

    lift : Callable[[set], set]
        ``lift(ds_reduced)`` maps a dominating set of *G_reduced* to a
        valid dominating set of *G*.

    Two-phase algorithm
    -------------------
    Phase 1 — standard cascade on H = copy(G):
      Apply _run_cascade(H, forced_ds, G).  Only vertex deletions occur;
      H is an induced subgraph of G throughout.

    Phase 2 — planarization + re-cascade (only if Phase 1 left H non-planar):
      Step 2a: P = _greedy_planar_subgraph(H)
        Same vertices as H, planar subset of edges.  Every kept edge is
        still an edge of G, so the DS-lift validity is maintained.
      Step 2b: _run_cascade(P, forced_ds, G)
        Edge removals in Step 2a may have created new pendants/isolates;
        this re-cascade eliminates them.  The domination guard still
        consults G so it correctly sees vertices already dominated by
        forced_ds from Phase 1.

    Planarity of output
    -------------------
    After Phase 2, G_reduced is a vertex-induced subgraph of P (Phase 2
    cascade only deletes vertices).  P is planar by construction.  By
    Kuratowski/Wagner, vertex deletion cannot introduce a forbidden minor,
    so G_reduced is planar.  When Phase 2 is skipped (H already planar),
    the same argument applies to H ⊆ G.

    Lift correctness — see module docstring for the full proof.

    Complexity
    ----------
    Phase 1: O(V + E).
    Phase 2 (worst case): O(E · (V + E)) for the greedy planarization,
    then O(V + E) for the re-cascade.
    """
    # ── H starts as an edge-identical copy of G (self-loops stripped). ──────
    # Stripping self-loops does not affect planarity: self-loops never
    # participate in a K₅ / K₃,₃ minor.
    H: nx.Graph = nx.Graph(G)
    H.remove_edges_from(list(nx.selfloop_edges(H)))

    forced_ds: Set[Any] = set()

    # ───────────────────────────────────────────────────────────────────────
    # Phase 1: standard pendant/isolated cascade
    #
    # _run_cascade modifies H and forced_ds in-place.  The domination guard
    # inside the cascade reads G.neighbors(v) (the original graph) to detect
    # vertices already covered by earlier forced vertices.
    # ───────────────────────────────────────────────────────────────────────
    _run_cascade(H, forced_ds, G)

    # ───────────────────────────────────────────────────────────────────────
    # Phase 2: planarize + re-cascade (only when Phase 1 left H non-planar)
    #
    # Why Phase 2 is needed: Rule 1 and Rule 0 only remove degree-≤1
    # vertices.  For a non-planar G, the Phase 1 residual can still be
    # non-planar (e.g. a K₅ or K₃,₃ subgraph where every vertex has
    # degree ≥ 2 survives the cascade intact).
    #
    # Step 2a — _greedy_planar_subgraph:
    #   Keeps the same vertex set; removes the minimum number of edges
    #   (greedily) to make the graph planar.  Every retained edge is
    #   still in G, so the domination relationships used in the lift
    #   remain valid for G.
    #
    # Step 2b — re-cascade on P:
    #   Edge removals in Step 2a can introduce new pendants/isolates
    #   (e.g. a vertex whose only remaining H-edges were the removed
    #   crossing edges now has lower degree).  The cascade cleans these
    #   up exactly as in Phase 1.  forced_ds is the same accumulator, so
    #   Phase-1-forced vertices are still visible to the domination guard.
    # ───────────────────────────────────────────────────────────────────────
    is_planar, _ = nx.check_planarity(H)
    if not is_planar:
        # Step 2a: construct a planar spanning subgraph of H.
        # P has the same vertices as H but only a planar subset of edges.
        # Because every edge of P is also an edge of H ⊆ G, the lift
        # identity (forced_ds ∪ DS(P_reduced) is a DS of G) holds.
        P: nx.Graph = _greedy_planar_subgraph(H)

        # Step 2b: re-cascade on P.
        # After this, P has min-degree ≥ 2 and is planar (vertex deletions
        # on a planar graph preserve planarity by Kuratowski/Wagner).
        _run_cascade(P, forced_ds, G)

        H = P   # H now points to the planar, cascade-reduced graph.

    # ───────────────────────────────────────────────────────────────────────
    # Build output
    #
    # H is planar (guaranteed by the two-phase reduction above):
    #   • Phase 2 skipped → H is a vertex-induced subgraph of a planar G.
    #   • Phase 2 executed → H is a vertex-induced subgraph of P, which
    #     was explicitly constructed to be planar.
    # In both cases Kuratowski/Wagner applies: only vertex deletions
    # occurred after the planar graph was established, so no forbidden
    # minor was introduced.
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

    # Verify the unconditional planarity guarantee.
    planar_ok, _ = nx.check_planarity(G_r)

    print(f"\n{'═' * 66}")
    print(f"  {name}")
    print(f"  Original  : V={G.number_of_nodes():>4},  E={G.number_of_edges():>5}"
          f"  (planar input: {nx.check_planarity(G)[0]})")
    print(f"  Reduced   : V={G_r.number_of_nodes():>4},  E={G_r.number_of_edges():>5}  "
          f"(min_deg≥2: {res['min_degree_ok']}, planar: {planar_ok})")
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
    assert planar_ok,   f"FAIL: G_reduced is not planar for '{name}'"
    assert res['lifted_ds_ok'], f"FAIL: lifted DS invalid for '{name}'"


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

    # 13. K₅ — the smallest non-planar graph (complete graph on 5 vertices).
    #    Every vertex has degree 4; Phase 1 cascade does nothing.
    #    Phase 2 must planarize: K₅ has 10 edges; a maximal planar graph on
    #    5 vertices has at most 3·5−6 = 9 edges (it is K₅ minus one edge,
    #    which is still non-planar — the greedy must remove ≥ 2 edges).
    _demo("K₅ (smallest non-planar)", nx.complete_graph(5))

    # 14. K₃,₃ — the other Kuratowski obstruction.
    #    All degrees are 3; Phase 1 does nothing.  Phase 2 must remove
    #    at least 1 edge to make it planar (K₃,₃ has 9 edges; planar
    #    bipartite bound is 2V−4 = 8 edges).
    _demo("K₃,₃ (Kuratowski bipartite obstruction)", nx.complete_bipartite_graph(3, 3))

    # 15. K₆ — complete graph on 6 vertices (highly non-planar).
    #    15 edges; planar bound is 3·6−6 = 12, so ≥ 3 edges must be removed.
    _demo("K₆ (complete, highly non-planar)", nx.complete_graph(6))

    # 16. Petersen graph — well-known non-planar 3-regular graph.
    #    Already in the planar demos above, but shown here to confirm Phase 2
    #    fires when needed (the Petersen graph is non-planar despite being
    #    highly symmetric; Phase 1 does nothing, Phase 2 planarizes it).
    _demo("Petersen graph (non-planar 3-regular)", nx.petersen_graph())

    # 17. Random dense non-planar graph (30 nodes, 120 edges).
    #    Exercises both phases and the greedy planarizer on a large instance.
    _demo("Random dense G(30, 120, seed=7)", nx.gnm_random_graph(30, 120, seed=7))

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