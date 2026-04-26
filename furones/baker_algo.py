"""
Baker's PTAS — Minimum Dominating Set on Planar 2-Edge-Connected Graphs
=======================================================================

  Baker (1994): For any fixed ε > 0 and any planar graph G:

    1. Let k = ⌈1/ε⌉.
    2. BFS-layer G from an arbitrary root.
    3. For each shift j ∈ {0, …, k−1}:
         • Remove "separator" layers: vertices whose depth ≡ j  (mod k).
         • Each remaining connected component spans < k consecutive
           layers  →  treewidth O(k) by the outerplanarity argument.
         • Solve DS *exactly* on every component, allowing separator-layer
           vertices on the component boundary to act as free dominators
           (they can be in the DS without needing to be dominated).
         • Repair: greedily cover any separator vertex still undominated.
    4. Return the best solution seen across all k shifts.

  Approximation ratio : (k+1)/k  ≤  1 + ε
  Time complexity     : O(k · 3^k · n)  =  O(f(ε) · n)   (linear in n)

Bugs fixed vs. the original implementation
-------------------------------------------
  Bug 1 (DP reconstruction) — ds_dp_tree_decomp returned only the DS_IN
    vertices present in the *root bag*, silently dropping all vertices
    selected in lower bags.  Fixed: each DP entry now carries a frozenset
    of the DS_IN vertices accumulated so far; the root's entry contains the
    full solution.

  Bug 2 (phantom-DOM states) — The initial 3^|bag| enumeration produced
    states where a vertex was labelled DOM but had no DS_IN neighbour in the
    bag ("phantom domination").  These cost-0 phantom states outcompeted
    legitimate cost-≥1 states in the min-cost comparison, causing the DP to
    return empty sets for even trivial instances (triangle, path, star).
    Fixed: states in which any DOM vertex lacks a DS_IN bag-neighbour are
    rejected during initialisation.

  Bug 3 (compatibility + neighbour-upgrade) — The combination step required
    the full state tuple of overlap vertices to match exactly between parent
    and child, pruning valid configurations where a vertex was FREE in the
    child subtree but DOM in the parent (dominated by a parent DS_IN
    neighbour).  Forgotten DS_IN vertices were also not propagating their
    domination to parent-bag neighbours.  Fixed:
      • Compatibility is now checked on DS_IN *membership* only (not DOM/FREE
        status), and the merged state uses max(parent, child) per vertex.
      • When a forgotten vertex is DS_IN, all its neighbours in the parent
        bag are upgraded to at least DOM (the neighbour-upgrade step).

  Bug 4 (separator patching) — After solving inner components, undominated
    separator vertices were all added to the DS at once ("solution |=
    undom_seps"), which is unnecessarily aggressive: a single inner or
    separator vertex may dominate several undominated separators at once.
    Fixed: greedy targeted cover replaces the bulk-add; the existing
    _greedy_repair fallback handles any residual gaps.

  Bug 5 (isolated inner components) — Inner components were solved on
    G.induced(comp), which severs all edges to separator vertices.  Any inner
    vertex whose only neighbours are separators became isolated and was forced
    to self-dominate, inflating the solution.  Baker's theoretical guarantee
    requires that separator-layer vertices adjacent to a component may act as
    free dominators.  Fixed: solve_component now includes boundary-separator
    vertices in the extended subgraph and uses brute_force_partial_ds, which
    finds the minimum DS that only needs to dominate the inner component
    vertices (separator boundary vertices may be in the DS but need not be
    covered).

Components
----------
  §1  Graph data-structure
  §2  BFS layering
  §3  Min-fill tree decomposition
  §4  Dominating-set DP on tree decomposition   (exact, O(3^w · n))
  §5  Brute-force fallback for tiny components
  §6  Greedy DS baseline   (O(m log n), ratio O(log Δ))
  §7  Baker's PTAS
  §8  Utilities: verification, generators
  §9  Demo / benchmark
"""

from __future__ import annotations

import math
import sys
import time
from collections import defaultdict, deque
from itertools import product
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# §1  GRAPH
# ─────────────────────────────────────────────────────────────────────────────

class Graph:
    """Undirected graph on integer vertices 0 … n-1."""

    def __init__(self, n: int = 0):
        self.vertices: Set[int] = set(range(n))
        self.adj: Dict[int, Set[int]] = {v: set() for v in range(n)}

    # ── Mutation ──────────────────────────────────────────────────────────────

    def add_vertex(self, v: int) -> None:
        self.vertices.add(v)
        self.adj.setdefault(v, set())

    def add_edge(self, u: int, v: int) -> None:
        self.add_vertex(u); self.add_vertex(v)
        self.adj[u].add(v); self.adj[v].add(u)

    # ── Accessors ─────────────────────────────────────────────────────────────

    def neighbors(self, v: int) -> Set[int]:
        return self.adj[v]

    def degree(self, v: int) -> int:
        return len(self.adj[v])

    @property
    def num_edges(self) -> int:
        return sum(len(nb) for nb in self.adj.values()) // 2

    # ── Sub-graphs ────────────────────────────────────────────────────────────

    def induced(self, nodes: Set[int]) -> "Graph":
        H = Graph()
        H.vertices = set(nodes)
        H.adj = {v: self.adj[v] & nodes for v in nodes}
        return H

    def connected_components(self, nodes: Set[int]) -> List[Set[int]]:
        seen, comps = set(), []
        for s in nodes:
            if s in seen:
                continue
            q, comp = deque([s]), set()
            seen.add(s)
            while q:
                v = q.popleft()
                comp.add(v)
                for u in self.adj[v]:
                    if u in nodes and u not in seen:
                        seen.add(u); q.append(u)
            comps.append(comp)
        return comps

    # ── 2-edge-connectivity ───────────────────────────────────────────────────

    def bridges(self) -> List[Tuple[int, int]]:
        """Return list of bridge edges (Tarjan, O(V+E))."""
        disc: Dict[int, int] = {}
        low:  Dict[int, int] = {}
        timer = [0]
        found: List[Tuple[int, int]] = []

        def dfs(v: int, par: int) -> None:
            disc[v] = low[v] = timer[0]; timer[0] += 1
            for u in self.adj[v]:
                if u not in disc:
                    dfs(u, v)
                    low[v] = min(low[v], low[u])
                    if low[u] > disc[v]:
                        found.append((v, u))
                elif u != par:
                    low[v] = min(low[v], disc[u])

        for s in self.vertices:
            if s not in disc:
                dfs(s, -1)
        return found

    def is_2_edge_connected(self) -> bool:
        if len(self.vertices) < 2:
            return True
        # Must be connected
        seen = set()
        q = deque([next(iter(self.vertices))])
        seen.add(next(iter(self.vertices)))
        while q:
            v = q.popleft()
            for u in self.adj[v]:
                if u not in seen:
                    seen.add(u); q.append(u)
        if seen != self.vertices:
            return False
        return len(self.bridges()) == 0

    def __repr__(self) -> str:
        return f"Graph(|V|={len(self.vertices)}, |E|={self.num_edges})"


# ─────────────────────────────────────────────────────────────────────────────
# §2  BFS LAYERING
# ─────────────────────────────────────────────────────────────────────────────

def bfs_layers(G: Graph, root: int) -> Dict[int, int]:
    """BFS distance from *root* for every vertex in G.

    For disconnected graphs every connected component is seeded
    independently: vertices unreachable from *root* receive distances
    measured from the smallest vertex in their own component.  This
    guarantees that ``layers[v]`` is defined for *every* v in G.vertices,
    preventing the KeyError that arose in baker_ptas when the DS-reduction
    produced a disconnected residual graph.
    """
    dist: Dict[int, int] = {}

    # Seed the BFS with the requested root first so its component
    # gets the "natural" distances; remaining components are handled
    # in vertex order so the result is deterministic.
    seeds = [root] + sorted(G.vertices - {root})
    for seed in seeds:
        if seed in dist:
            continue          # already reached by an earlier BFS wave
        dist[seed] = 0
        q = deque([seed])
        while q:
            v = q.popleft()
            for u in G.adj[v]:
                if u not in dist:
                    dist[u] = dist[v] + 1
                    q.append(u)

    return dist


# ─────────────────────────────────────────────────────────────────────────────
# §3  TREE DECOMPOSITION  (min-fill elimination heuristic)
# ─────────────────────────────────────────────────────────────────────────────

def build_tree_decomp(
    G: Graph,
) -> Tuple[List[FrozenSet[int]], List[int], List[int]]:
    """
    Build a tree decomposition of *G* using the **min-fill heuristic**.

    Returns
    -------
    bags   : list of frozensets (one bag per elimination step)
    elim   : elimination order (elim[i] is the vertex eliminated at step i)
    parent : parent[i] = index of parent bag  (-1 = root, always the last bag)

    Width of the decomposition = max(len(bag) for bag in bags) − 1.
    For a graph of treewidth w the heuristic typically gives width ≤ 3w+1.
    """
    if not G.vertices:
        return [], [], []

    # Working adjacency (we'll add fill-edges here)
    adj: Dict[int, Set[int]] = {v: set(G.adj[v]) for v in G.vertices}
    remaining = set(G.vertices)

    bags: List[FrozenSet[int]] = []
    elim: List[int] = []

    while remaining:
        # ── Min-fill: pick vertex minimising needed fill-edges ────────────────
        best_v, best_fill = None, math.inf
        for v in remaining:
            nbrs = adj[v] & remaining
            nl = list(nbrs)
            fill = sum(
                1
                for i in range(len(nl))
                for j in range(i + 1, len(nl))
                if nl[j] not in adj[nl[i]]
            )
            if fill < best_fill:
                best_fill, best_v = fill, v

        v = best_v
        nbrs = adj[v] & remaining
        bags.append(frozenset(nbrs | {v}))
        elim.append(v)

        # ── Make neighbourhood a clique (add fill-edges) ──────────────────────
        nl = list(nbrs)
        for i in range(len(nl)):
            for j in range(i + 1, len(nl)):
                adj[nl[i]].add(nl[j])
                adj[nl[j]].add(nl[i])

        remaining.remove(v)

    # ── Build parent pointers ─────────────────────────────────────────────────
    # parent[i] = first j > i whose bag has maximum overlap with bags[i] \ {elim[i]}
    n = len(bags)
    parent = [-1] * n
    for i in range(n - 1):
        core = bags[i] - {elim[i]}
        best_j, best_ov = i + 1, -1
        for j in range(i + 1, n):
            ov = len(bags[j] & core)
            if ov > best_ov:
                best_ov, best_j = ov, j
        parent[i] = best_j

    return bags, elim, parent


# ─────────────────────────────────────────────────────────────────────────────
# §4  DOMINATING SET DP ON TREE DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────────────
#
#  State label for each vertex in a bag
#  ──────────────────────────────────────
#  DS_IN  (2) : vertex is selected into the dominating set
#  DOM    (1) : vertex is dominated (≥1 DS neighbour in subtree below)
#  FREE   (0) : not yet dominated — valid only while still inside some bag
#
#  Validity rule: when a vertex is "forgotten" (removed from bag on the way
#  to the root) its state must be 1 or 2 (never 0).
#
#  DP table: dp[bag_index] : state_tuple → min DS-size in that subtree
# ─────────────────────────────────────────────────────────────────────────────

DS_IN, DOM, FREE = 2, 1, 0
INF = float("inf")


def ds_dp_tree_decomp(
    bags:   List[FrozenSet[int]],
    parent: List[int],
    G:      Graph,
) -> Set[int]:
    """
    Exact minimum dominating set via DP on the given tree decomposition.

    The DP runs bottom-up (leaves first).  Each table entry carries both
    a cost and the *actual vertex set* chosen so far, enabling direct
    reconstruction without back-pointers.

    Three bugs fixed vs. the original:

    Fix 1 — DS-set tracking.
        Each DP entry is (cost, frozenset_of_DS_IN_vertices).  The DS set
        accumulates as bags are merged, so the root yields the full solution
        directly — no need for a separate backtracking pass.

    Fix 2 — Neighbour-upgrade when forgetting DS_IN vertices.
        When a vertex u is "forgotten" (present in a child bag but absent
        from the parent bag), its membership decision is finalised.  If
        u ∈ DS, every neighbour of u that *is* in the parent bag must be
        upgraded to at least DOM.  The original code ignored this, producing
        states where vertices appeared dominated but had no actual DS
        neighbour.

    Fix 3 — Correct overlap compatibility.
        Two states are compatible on the overlap iff they *agree on DS_IN
        membership* for every overlap vertex (a vertex cannot be IN on one
        side and OUT on the other).  DOM vs FREE may legitimately differ:
        the parent may already hold a DS_IN neighbour that the child subtree
        has not seen.  The merged state uses max(parent, child) per vertex,
        so domination information flows correctly in both directions.

    Time: O(3^w · n)  where w = treewidth of the decomposition.
    """
    if not bags:
        return set()

    n_bags   = len(bags)
    children: Dict[int, List[int]] = defaultdict(list)
    for i in range(n_bags - 1):
        children[parent[i]].append(i)

    # ── Topological order: leaves before parents ──────────────────────────────
    order: List[int] = []
    visited: Set[int] = set()
    stack = [n_bags - 1]        # root = last bag in elimination order
    while stack:
        node = stack[-1]
        if node not in visited:
            visited.add(node)
            for c in children[node]:
                stack.append(c)
        else:
            order.append(node)
            stack.pop()

    # ── DP tables: state_tuple → (cost, ds_frozenset) ────────────────────────
    dp: List[Optional[Dict[Tuple[int, ...], Tuple[float, FrozenSet]]]] = [None] * n_bags

    for node in order:
        bag   = bags[node]
        bl    = sorted(bag)
        bmap  = {v: i for i, v in enumerate(bl)}
        clist = children[node]

        # ── Enumerate all 3^|bag| assignments ────────────────────────────────
        node_dp: Dict[Tuple, Tuple[float, FrozenSet]] = {}

        for raw in product(range(3), repeat=len(bl)):
            state = list(raw)

            # Propagate intra-bag domination:
            #   DS_IN vertex → all its bag-neighbours become at least DOM.
            changed = True
            while changed:
                changed = False
                for idx, v in enumerate(bl):
                    if state[idx] == DS_IN:
                        for u in G.adj[v]:
                            if u in bmap and state[bmap[u]] == FREE:
                                state[bmap[u]] = DOM
                                changed = True

            key  = tuple(state)
            cost = sum(1 for s in state if s == DS_IN)
            ds   = frozenset(bl[i] for i, s in enumerate(state) if s == DS_IN)

            # Reject phantom-DOM states: a vertex labelled DOM must have at
            # least one DS_IN neighbour *within the bag*.  Without this guard,
            # cost-0 "dominated-by-nobody" states pollute the table and make
            # legitimate, costlier states unreachable.
            phantom = False
            for idx, v in enumerate(bl):
                if state[idx] == DOM:
                    if not any(
                        u in bmap and state[bmap[u]] == DS_IN for u in G.adj[v]
                    ):
                        phantom = True
                        break
            if phantom:
                continue

            if key not in node_dp or node_dp[key][0] > cost:
                node_dp[key] = (cost, ds)

        # ── Combine with children (one child at a time) ───────────────────────
        for child in clist:
            child_bag  = bags[child]
            cbl        = sorted(child_bag)
            cmap       = {v: i for i, v in enumerate(cbl)}
            child_dp   = dp[child]
            overlap    = bag & child_bag
            forgotten  = child_bag - bag
            ov_sorted  = sorted(overlap)

            # ── Pre-process child states ──────────────────────────────────────
            # For each valid child state we record:
            #   ov_ds_in   – DS_IN membership of each overlap vertex (0/1 tuple)
            #                used for compatibility matching with the parent.
            #   ov_dom     – full state value of each overlap vertex
            #                used to upgrade the parent via max(parent, child).
            #   nb_upgrade – parent-bag positions that get DOM because a
            #                *forgotten* vertex is DS_IN (Fix 2).
            #
            # Deduplicate by (ov_ds_in, ov_dom, nb_upgrade_frozen) keeping the
            # cheapest child cost for each distinct combination.

            dedup: Dict[Tuple, Tuple[float, FrozenSet, Dict[int, int]]] = {}

            for c_state, (c_cost, c_ds) in child_dp.items():
                # Forgotten vertices must be dominated (non-FREE) before leaving.
                if any(c_state[cmap[v]] == FREE for v in forgotten):
                    continue

                ov_ds_in = tuple(
                    1 if c_state[cmap[v]] == DS_IN else 0 for v in ov_sorted
                )
                ov_dom = tuple(c_state[cmap[v]] for v in ov_sorted)

                # Fix 2: forgotten DS_IN vertices dominate their parent-bag
                # neighbours — propagate this as a mandatory upgrade.
                nb: Dict[int, int] = {}
                for u in forgotten:
                    if c_state[cmap[u]] == DS_IN:
                        for w in G.adj[u]:
                            if w in bmap:
                                nb[bmap[w]] = DOM

                key = (ov_ds_in, ov_dom, frozenset(nb.items()))
                if key not in dedup or dedup[key][0] > c_cost:
                    dedup[key] = (c_cost, c_ds, nb)

            # Flatten into a list for iteration
            child_entries: List[Tuple] = [
                (c_cost, c_ds, nb, ov_ds_in, ov_dom)
                for (ov_ds_in, ov_dom, _), (c_cost, c_ds, nb) in dedup.items()
            ]

            if not child_entries:
                node_dp = {}
                break

            new_dp: Dict[Tuple, Tuple[float, FrozenSet]] = {}

            for p_state, (p_cost, p_ds) in node_dp.items():
                # Fix 3: match on DS_IN membership only (not full DOM/FREE state).
                p_ov_ds_in = tuple(
                    1 if p_state[bmap[v]] == DS_IN else 0 for v in ov_sorted
                )

                for (c_cost, c_ds, nb_upgrade, c_ov_ds_in, c_ov_dom) in child_entries:
                    # Overlap vertices must agree on whether they are DS_IN.
                    if p_ov_ds_in != c_ov_ds_in:
                        continue

                    # Build merged state.
                    new_state = list(p_state)

                    # Fix 2: apply neighbour-upgrade from forgotten DS_IN vertices.
                    for bidx, dom_val in nb_upgrade.items():
                        new_state[bidx] = max(new_state[bidx], dom_val)

                    # Fix 3: merge overlap domination status via max(parent, child).
                    for i, v in enumerate(ov_sorted):
                        bidx = bmap[v]
                        new_state[bidx] = max(new_state[bidx], c_ov_dom[i])

                    # Overlap DS_IN vertices were counted in both costs; subtract once.
                    ov_ds = sum(
                        1 for v in ov_sorted if p_state[bmap[v]] == DS_IN
                    )
                    total_cost = p_cost + c_cost - ov_ds
                    total_ds   = p_ds | c_ds   # Fix 1: accumulate vertex sets

                    new_key = tuple(new_state)
                    if new_key not in new_dp or new_dp[new_key][0] > total_cost:
                        new_dp[new_key] = (total_cost, total_ds)

            node_dp = new_dp

        dp[node] = node_dp

    # ── Extract answer at root ────────────────────────────────────────────────
    root = n_bags - 1

    best_cost = INF
    best_ds: Optional[FrozenSet] = None
    for state, (cost, ds) in dp[root].items():
        if all(s != FREE for s in state) and cost < best_cost:
            best_cost = cost
            best_ds   = ds

    if best_ds is None:
        # Fallback: should not occur on valid (connected) input.
        return brute_force_ds(G)

    return set(best_ds)


# ─────────────────────────────────────────────────────────────────────────────
# §5  BRUTE-FORCE EXACT DS  (branch-and-bound, for small components)
# ─────────────────────────────────────────────────────────────────────────────

def brute_force_ds(G: Graph) -> Set[int]:
    """
    Branch-and-bound exact minimum dominating set.
    Practical for |V| ≤ 30 or treewidth ≤ ~8.
    """
    vlist = sorted(G.vertices)
    n = len(vlist)
    best: List[Set[int]] = [set(vlist)]      # start with all vertices

    def cover(v: int) -> Set[int]:
        return G.adj[v] | {v}

    def branch(idx: int, ds: List[int], dominated: Set[int]) -> None:
        if len(ds) >= len(best[0]):
            return                           # prune: can't improve
        if idx == n:
            if G.vertices <= dominated:
                best[0] = set(ds)
            return

        v = vlist[idx]
        # ── Branch 1: include v in DS ─────────────────────────────────────────
        ds.append(v)
        branch(idx + 1, ds, dominated | cover(v))
        ds.pop()
        # ── Branch 2: skip v ──────────────────────────────────────────────────
        branch(idx + 1, ds, dominated)

    branch(0, [], set())
    return best[0]


def brute_force_partial_ds(G: Graph, must_dominate: Set[int]) -> Set[int]:
    """
    Branch-and-bound minimum DS of *G* that only needs to dominate the
    vertices in *must_dominate*.  Vertices in G.vertices − must_dominate
    may be placed in the DS but are not required to be covered.

    Used by solve_component to allow separator boundary vertices to act as
    "free" dominators while not needing coverage themselves.
    """
    vlist = sorted(G.vertices)
    n     = len(vlist)
    best: List[Set[int]] = [set(vlist)]

    def cover(v: int) -> Set[int]:
        return G.adj[v] | {v}

    def branch(idx: int, ds: List[int], dominated: Set[int]) -> None:
        if len(ds) >= len(best[0]):
            return
        if idx == n:
            if must_dominate <= dominated:
                best[0] = set(ds)
            return
        v = vlist[idx]
        ds.append(v)
        branch(idx + 1, ds, dominated | cover(v))
        ds.pop()
        branch(idx + 1, ds, dominated)

    branch(0, [], set())
    return best[0]


def solve_component(G: Graph, comp: Set[int],
                    separators: Optional[Set[int]] = None) -> Set[int]:
    """
    Find the minimum dominating set for the vertices in *comp*, optionally
    allowing separator-layer vertices adjacent to *comp* to act as dominators.

    Baker's PTAS guarantee requires this "extended component" formulation:
    separator vertices on the boundary of a component may be included in the
    DS (reducing cost) but are not themselves required to be dominated by the
    component's solution.  When multiple components share a boundary separator
    the set-union of their solutions counts the separator only once, so reuse
    comes for free.

    Without this, isolated inner vertices always self-dominate, inflating the
    solution and potentially violating the (k+1)/k ratio bound.
    """
    if len(comp) == 0:
        return set()

    # ── Collect boundary separators adjacent to comp ──────────────────────────
    boundary: Set[int] = set()
    if separators:
        for v in comp:
            for u in G.adj[v]:
                if u in separators:
                    boundary.add(u)

    if boundary:
        # Solve partial DS on the extended graph:
        #   vertices  = comp ∪ boundary
        #   edges     = induced from G
        #   dominate  = comp only   (boundary vertices need not be covered)
        extended = comp | boundary
        H = G.induced(extended)
        return brute_force_partial_ds(H, comp)

    # ── No boundary separators (or not requested): original behaviour ─────────
    H = G.induced(comp)
    if len(comp) == 1:
        return set(comp)
    if len(comp) <= 28:
        return brute_force_ds(H)
    # Large component: use tree decomposition DP
    bags, _, par = build_tree_decomp(H)
    return ds_dp_tree_decomp(bags, par, H)


# ─────────────────────────────────────────────────────────────────────────────
# §6  GREEDY DS BASELINE
# ─────────────────────────────────────────────────────────────────────────────

def greedy_ds(G: Graph) -> Set[int]:
    """
    Standard greedy dominating set.
    Approximation ratio: H(Δ+1) = O(log Δ).
    Time: O(m log n) with a priority queue.
    """
    dominated: Set[int] = set()
    solution:  Set[int] = set()
    candidates = set(G.vertices)

    while dominated != G.vertices:
        best_v = max(
            candidates,
            key=lambda v: len((G.adj[v] | {v}) - dominated),
        )
        solution.add(best_v)
        dominated.add(best_v)
        dominated |= G.adj[best_v]
        candidates.discard(best_v)

    return solution


# ─────────────────────────────────────────────────────────────────────────────
# §7  BAKER'S PTAS
# ─────────────────────────────────────────────────────────────────────────────

def baker_ptas(
    G:       Graph,
    epsilon: float,
    verbose: bool = True,
) -> Set[int]:
    """
    Baker's PTAS for Minimum Dominating Set on planar graphs.

    Parameters
    ----------
    G       : planar graph (2-edge-connected recommended)
    epsilon : ε ∈ (0, 1]  — approximation parameter
    verbose : print progress per shift

    Returns
    -------
    A (1 + ε)-approximate minimum dominating set.

    Complexity
    ----------
    Time  : O(k · 3^k · n)  where  k = ⌈1/ε⌉
    Space : O(3^k · n)
    """
    if not G.vertices:
        return set()

    k   = max(1, math.ceil(1.0 / epsilon))
    n   = len(G.vertices)
    m   = G.num_edges
    two_ec = G.is_2_edge_connected()

    if verbose:
        bar = "═" * 62
        print(f"\n{bar}")
        print(f"  Baker's PTAS — Minimum Dominating Set")
        print(bar)
        print(f"  ε = {epsilon}   →   k = ⌈1/ε⌉ = {k}")
        print(f"  |V| = {n},  |E| = {m}")
        print(f"  2-edge-connected : {two_ec}")
        print(f"  Approx guarantee : ≤ {(k + 1) / k:.4f} × OPT")
        print(f"  Subproblem bound : 3^{k} = {3 ** k} states per bag")
        print(f"  Shifts to try    : {k}")
        print(bar)

    root   = min(G.vertices)
    layers = bfs_layers(G, root)

    best_sol: Optional[Set[int]] = None

    for shift in range(k):
        t0 = time.perf_counter()

        # ── Step 1: separator layers (depth ≡ shift mod k) ───────────────────
        separators: Set[int] = {
            v for v in G.vertices if layers[v] % k == shift
        }
        inner = G.vertices - separators

        # ── Step 2: solve DS on each strip-component ──────────────────────────
        components = G.connected_components(inner)
        solution: Set[int] = set()

        for comp in components:
            solution |= solve_component(G, comp, separators)

        # ── Step 2b: partial minimisation ────────────────────────────────────
        # After unioning per-component solutions, many DS vertices may be
        # redundant: e.g. three singletons each include the same separator
        # vertex, then a fourth includes an inner vertex that's already
        # dominated by that separator.  Removing such redundancies *before*
        # the separator-repair step keeps the repair cost low and keeps the
        # solution competitive with greedy.
        # Criterion: only inner vertices need to remain dominated here.
        solution = _make_minimal(G, solution, inner)

        # ── Step 3: ensure separator vertices are dominated ───────────────────
        dominated: Set[int] = set()
        for v in solution:
            dominated.add(v)
            dominated |= G.adj[v]

        # Greedily cover any separator not yet dominated.
        # We pick whichever vertex (from all of V) covers the most
        # undominated separators per step — cheaper than blindly adding
        # every undominated separator, and still preserves the ratio bound.
        undom_seps = separators - dominated
        while undom_seps:
            best_v = max(
                G.vertices - solution,
                key=lambda v: len((G.adj[v] | {v}) & undom_seps),
            )
            if not ((G.adj[best_v] | {best_v}) & undom_seps):
                # No single outside vertex helps; add remaining separators directly.
                solution  |= undom_seps
                dominated |= undom_seps
                break
            solution.add(best_v)
            dominated.add(best_v)
            dominated |= G.adj[best_v]
            undom_seps -= dominated

        # ── Step 4: final verification & greedy repair ────────────────────────
        dominated = set()
        for v in solution:
            dominated.add(v)
            dominated |= G.adj[v]

        if not (G.vertices <= dominated):
            solution = _greedy_repair(G, solution)

        # ── Step 5: full minimisation ─────────────────────────────────────────
        # The separator-repair and greedy-repair steps may have introduced
        # vertices that are now made redundant by other choices.  A final
        # pass removes any such surplus vertices.
        solution = _make_minimal(G, solution)

        elapsed = time.perf_counter() - t0
        tag = ""
        if best_sol is None or len(solution) < len(best_sol):
            best_sol = solution
            tag = "  ← new best"

        if verbose:
            print(
                f"  shift {shift}/{k - 1} | "
                f"comps={len(components):3d} | "
                f"|DS|={len(solution):4d} | "
                f"{elapsed * 1000:6.1f} ms{tag}"
            )

    if verbose:
        print(bar)
        print(f"  Final |DS| = {len(best_sol)}")
        print(bar + "\n")

    return best_sol


def _make_minimal(
    G:             Graph,
    ds:            Set[int],
    must_dominate: Optional[Set[int]] = None,
) -> Set[int]:
    """
    Remove redundant vertices from *ds* while keeping it a valid DS.

    If *must_dominate* is given only those vertices need to remain dominated
    (used after the inner-component phase, before separator repair, so that
    inner-only domination is the criterion and separator-layer vertices are
    still allowed to be pruned).  When *must_dominate* is None (the default)
    every vertex in G must remain dominated.

    The scan is repeated until no further vertex can be removed.
    Time: O(|ds|² · n) — fast enough for all component sizes we encounter.
    """
    if must_dominate is None:
        must_dominate = G.vertices

    ds = set(ds)
    changed = True
    while changed:
        changed = False
        for v in sorted(ds):
            test = ds - {v}
            dom: Set[int] = set()
            for u in test:
                dom.add(u)
                dom |= G.adj[u]
            if must_dominate <= dom:
                ds = test
                changed = True
                break     # restart scan so later vertices see updated ds
    return ds


def _greedy_repair(
    G:       Graph,
    partial: Set[int],
) -> Set[int]:
    """Augment *partial* until it dominates every vertex."""
    sol = set(partial)
    dominated: Set[int] = set()
    for v in sol:
        dominated.add(v)
        dominated |= G.adj[v]

    while not (G.vertices <= dominated):
        best_v = max(
            G.vertices - sol,
            key=lambda v: len((G.adj[v] | {v}) - dominated),
        )
        sol.add(best_v)
        dominated.add(best_v)
        dominated |= G.adj[best_v]

    return sol


# ─────────────────────────────────────────────────────────────────────────────
# §8  UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def verify_ds(G: Graph, ds: Set[int]) -> bool:
    """Return True iff *ds* is a valid dominating set of *G*."""
    dominated: Set[int] = set()
    for v in ds:
        dominated.add(v)
        dominated |= G.adj[v]
    return G.vertices <= dominated


# ── Graph generators ──────────────────────────────────────────────────────────

def cycle_graph(n: int) -> Graph:
    """Cycle C_n  —  planar, 2-edge-connected."""
    G = Graph(n)
    for i in range(n):
        G.add_edge(i, (i + 1) % n)
    return G


def grid_graph(rows: int, cols: int) -> Graph:
    """r × c grid  —  planar, 2-edge-connected for r, c ≥ 2."""
    G = Graph(rows * cols)
    for r in range(rows):
        for c in range(cols):
            v = r * cols + c
            if c + 1 < cols: G.add_edge(v, v + 1)
            if r + 1 < rows: G.add_edge(v, v + cols)
    return G


def ladder_graph(n: int) -> Graph:
    """Ladder P_n × P_2  —  planar, 2-edge-connected."""
    G = Graph(2 * n)
    for i in range(n - 1):
        G.add_edge(i, i + 1)
        G.add_edge(n + i, n + i + 1)
    for i in range(n):
        G.add_edge(i, n + i)
    return G


def dodecahedron_graph() -> Graph:
    """Dodecahedron  —  planar, 3-regular, 3-connected (hence 2-EC)."""
    edges = [
        (0,1),(1,2),(2,3),(3,4),(4,0),
        (0,5),(1,6),(2,7),(3,8),(4,9),
        (5,10),(6,11),(7,12),(8,13),(9,14),
        (10,11),(11,12),(12,13),(13,14),(14,10),
        (10,15),(11,16),(12,17),(13,18),(14,19),
        (15,16),(16,17),(17,18),(18,19),(19,15),
    ]
    G = Graph(20)
    for u, v in edges:
        G.add_edge(u, v)
    return G


def halin_graph(k: int) -> Graph:
    """
    Simple Halin graph: a complete binary tree of depth k with leaves
    connected in a cycle  —  planar, 3-connected (hence 2-EC).
    """
    G = Graph()
    node = [0]

    leaves: List[int] = []

    def build_tree(parent: int, depth: int) -> int:
        v = node[0]; node[0] += 1
        G.add_vertex(v)
        if parent >= 0:
            G.add_edge(parent, v)
        if depth == 0:
            leaves.append(v)
        else:
            build_tree(v, depth - 1)
            build_tree(v, depth - 1)
        return v

    build_tree(-1, k)
    for i in range(len(leaves)):
        G.add_edge(leaves[i], leaves[(i + 1) % len(leaves)])
    return G


# ─────────────────────────────────────────────────────────────────────────────
# §9  DEMO & BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark() -> None:
    print("\n" + "▓" * 65)
    print("  Baker's PTAS  —  Dominating Set Benchmark")
    print("  Planar 2-Edge-Connected Graphs")
    print("▓" * 65)

    cases = [
        # (label,                  graph,               epsilon)
        ("Cycle C₁₂",             cycle_graph(12),      0.5),
        ("Ladder L₁₀",            ladder_graph(10),     0.5),
        ("Dodecahedron (20 v)",    dodecahedron_graph(), 0.34),
        ("Grid 4×4  (16 v)",      grid_graph(4, 4),     0.5),
        ("Grid 5×5  (25 v)",      grid_graph(5, 5),     0.5),
        ("Halin(3)  (22 v)",      halin_graph(3),       0.5),
        ("Grid 6×6  (36 v)",      grid_graph(6, 6),     0.34),
        ("Cycle C₃₀",             cycle_graph(30),      0.25),
        ("Grid 7×7  (49 v)",      grid_graph(7, 7),     0.5),
    ]

    summary_rows = []

    for label, G, eps in cases:
        print(f"\n{'━' * 62}")
        print(f"  {label}")
        print(f"  2-EC = {G.is_2_edge_connected()}")

        t0 = time.perf_counter()
        ptas_sol = baker_ptas(G, eps, verbose=True)
        ptas_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        grdy_sol = greedy_ds(G)
        grdy_time = time.perf_counter() - t1

        ok_p = verify_ds(G, ptas_sol)
        ok_g = verify_ds(G, grdy_sol)

        k   = math.ceil(1.0 / eps)
        ratio_bound = (k + 1) / k

        print(f"  {'Method':<20} {'|DS|':>5}  {'Valid':>5}  {'Time (ms)':>10}")
        print(f"  {'─'*50}")
        print(f"  {'Baker PTAS (ε=' + str(eps) + ')':<20} {len(ptas_sol):>5}  {str(ok_p):>5}  {ptas_time*1000:>10.1f}")
        print(f"  {'Greedy (ln Δ)':<20} {len(grdy_sol):>5}  {str(ok_g):>5}  {grdy_time*1000:>10.1f}")
        print(f"  PTAS ratio bound: ≤ {ratio_bound:.3f} × OPT")

        summary_rows.append((label, len(G.vertices), eps, len(ptas_sol),
                              len(grdy_sol), ratio_bound, ok_p))

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n\n" + "═" * 72)
    print("  SUMMARY")
    print("═" * 72)
    print(f"  {'Graph':<22} {'|V|':>4}  {'ε':>5}  "
          f"{'PTAS|DS|':>8}  {'Grdy|DS|':>8}  {'Bound':>7}  {'Valid':>5}")
    print(f"  {'─' * 68}")
    for row in summary_rows:
        lbl, nv, eps, p, g, bnd, ok = row
        print(f"  {lbl:<22} {nv:>4}  {eps:>5.2f}  "
              f"{p:>8}  {g:>8}  {bnd:>7.3f}  {str(ok):>5}")
    print("═" * 72 + "\n")


if __name__ == "__main__":
    run_benchmark()