# Furones: Approximate Dominating Set Solver

![In Loving Memory of Asia Furones (The Grandmother I Never Knew)](docs/furones.jpg)

This work builds upon [The Furones Algorithm](https://www.preprints.org/manuscript/202504.0522).

---

# Overview of the Minimum Dominating Set (MDS)

## Definition

A **dominating set** in a graph $G = (V, E)$ is a subset $D \subseteq V$ such that every vertex not in $D$ is adjacent to at least one vertex in $D$. The **minimum dominating set (MDS)** is the smallest possible dominating set in terms of the number of vertices.

## Computational Complexity

- **NP-Hard**: Finding the minimum dominating set is NP-hard; no polynomial-time exact algorithm is known for general graphs.
- **Approximation**: The greedy Set Cover heuristic gives a ratio $H(\Delta+1) \le 1 + \ln(\Delta+1) = \mathcal{O}(\log \Delta)$, where $\Delta$ is the maximum degree. This is tight and matches the $o(\ln n)$-in-$n$ inapproximability threshold (Dinur–Steurer).

## Applications

- Network monitoring and wireless sensor coverage
- Influence maximisation in social networks
- Facility placement and logistics
- Protein–protein interaction modelling

---

# The Furones Algorithm

Furones v0.4.0 is a **linear-time candidate-comparison solver** for MDS. It builds a portfolio of dominating-set candidates on the input graph and returns the smallest validated one.

> **New in v0.4.0**: an **improvement phase** (`furones.improve`) applied to the pruned candidates — *union re-prune*, *weighted swap search*, *consensus forcing* and *final refinement* (see steps 7–8 below). A `candidates` option returns the ensemble to the caller.

The pipeline consists of:

1. **Preprocessing**: self-loop removal, isolated-vertex separation.
2. **TSCC-style pendant cascade** (`ReduceToTSCCForDS`): iteratively commits pendant supports and isolated vertices to a forced set $F$, producing a reduced residual.
3. **Baker-style reduced solve** on the residual (treated as planar by construction): used only as a *validated candidate generator* — its output enters the pool and is admitted only after passing the domination check; none of the unconditional guarantees depend on it.
4. **Lifted candidate** $C_L = F \cup \ell(D_R)$: lifts the reduced solution back to original labels.
5. **Original-graph linear candidates** (all computed in $\mathcal{O}(n+m)$):
   - $C_G$ — **greedy maximum-coverage** (certifies the $H(\Delta+1)$ ratio)
   - $C_D$ — closed-degree coverage sweep
   - $C_W$, $C_M$ — low- and medium-degree witness sweeps
   - $C_O$ — order-ownership candidates (early and late)
   - $C_S$ — seed-and-complete
   - $C_B$ — Salvador-style bipartite auxiliary
   - $C_X$ — **max-cut double-cover** (min-side maximum cut of the bipartite double cover $B(H)$, decoded by first coordinate; validated)
   - $C_T$ — **spanning-tree candidate**: exact tree DP for MDS on a maximum degree-sum spanning tree (bucket Kruskal, $\mathcal{O}(n + m\,\alpha(n))$), then pruned in the original graph; exact on forests
   - $C_R$ — reverse-delete (multiple deterministic orders)
6. **Pruning** (`PruneRedundantDominating`): each candidate is made inclusion-minimal.
7. **Portfolio minimum**: the smallest valid (dominates the original graph) pruned candidate is the baseline answer.
8. **Improvement phase** (`improve=True`, the default; `furones.improve`), run on the working graph with a work budget linear in $n + m$:
   - **Union re-prune** — the vertices chosen by at least half of the candidates, completed greedily by maximum coverage into a dominating set and pruned.
   - **Weighted swap search** (`furones.weighted_search`) — from the smallest set, a fixed-size set may leave vertices undominated; the weights of the undominated vertices grow at every step, a vertex is swapped out by best-of-sample score and one from an undominated vertex's closed neighbourhood is swapped in under configuration checking; every dominating set found is recorded and one vertex is removed to try the next smaller size.
   - **Consensus forcing** — the vertices chosen by at least two thirds of the candidates are forced, the rest is completed greedily, pruned and improved by the swap search.
   - **Final refinement** — the swap search again, with another seed, from the smallest set found.

   The result replaces the portfolio minimum only when it is a smaller dominating set, so steps 1–7 and their guarantees are unchanged.

### Unconditional guarantees

- **Feasibility** (Theorem): every normal return is a dominating set of the input graph.
- **Approximation ratio** (Theorem): because the portfolio contains $C_G$, every returned set $D$ satisfies $|D| \le H(\Delta+1)\,\gamma(G) \le (1+\ln(\Delta+1))\,\gamma(G) = \mathcal{O}(\log \Delta)$.
- **Improvement phase** (Theorem): it never returns a set larger than the portfolio minimum and every set it returns is validated, so the bounds above hold with it enabled.
- **Constant factor on bounded-degree graphs** (Corollary): for any fixed $\Delta$, the ratio is at most $1 + \ln(\Delta+1)$.

### Near-threshold ratio hypothesis

Furones is conjectured to satisfy $|D| \le \max\\{4,\,\frac{1}{2}\ln n\\}\cdot\gamma(G)$ on every graph. Proving this for all graphs would imply **P = NP** (via Dinur–Steurer). The hypothesis is already proved unconditionally for all graphs of maximum degree $\Delta \le \sqrt{n}/e$.

### Runtime

$\mathcal{O}(n+m)$ for fixed parameters (large hidden constant). The Baker reduced-solve uses the default $\varepsilon = 1$ (shift $k = 1$), a constant factor; the improvement phase uses a work budget of $\max(2(n+m),\,50\,000)$ per swap search, so it is $\mathcal{O}(n+m)$ as well; the max-cut double-cover candidate $C_X$ is also $\mathcal{O}(n+m)$ (build the double cover with $2n$ nodes and $2m$ edges, then one BFS min-side cut).

---

# CAR Benchmarks

The `car/` folder contains three reproducible exact-optimum studies. Every study imports the **installed** Furones package and compares it against the exact domination number $\gamma(G)$ computed by exhaustive search.

## CAR-001: Core Exact Benchmark (`run_integrity_measurements.py`)

- **1 000 instances**, at most 14 vertices, seven graph families (Erdős–Rényi sparse/medium/dense, random bipartite, random trees, perturbed paths/cycles, small structured).
- Exact $\gamma(G)$ by exhaustive search for every instance.
- **Result**: Furones is **optimal on all 1 000 instances** (100.0 %); observed ratio $\hat{\rho} = 1.000$.

| Family | Instances | $n$ | $\gamma$ | Optimal | Mean ratio | Max ratio |
|---|---|---|---|---|---|---|
| Erdős–Rényi sparse | 200 | 6–14 | 2–10 | 100.0 % | 1.000 | 1.000 |
| Erdős–Rényi medium | 200 | 6–14 | 2–5  | 100.0 % | 1.000 | 1.000 |
| Erdős–Rényi dense  | 150 | 6–14 | 1–4  | 100.0 % | 1.000 | 1.000 |
| Random bipartite   | 150 | 6–14 | 2–9  | 100.0 % | 1.000 | 1.000 |
| Random trees       | 100 | 6–14 | 2–6  | 100.0 % | 1.000 | 1.000 |
| Perturbed paths/cycles | 100 | 6–14 | 2–5 | 100.0 % | 1.000 | 1.000 |
| Small structured   | 100 | 6–14 | 1–4  | 100.0 % | 1.000 | 1.000 |
| **Overall** | **1000** | **6–14** | **1–10** | **100.0 %** | **1.000** | **1.000** |

## CAR-002: Per-Strategy Ablation (`run_integrity_measurements.py`)

Each exposed strategy is measured independently on the same 1 000-instance benchmark. The script also measures the four improvement strategies (`union_reprune`, `swap_search`, `consensus_forcing`, `final_refinement`), each computed from the pruned base candidates of the instance; `run_strategy_winners.py` reports the winners over any graph set.

| Strategy | Valid | Optimal | Mean ratio | Max ratio |
|---|---|---|---|---|
| Seed-and-complete | 100.0 % | 99.8 % | 1.000 | 1.250 |
| Greedy max-coverage ($C_G$) | 100.0 % | 92.7 % | 1.023 | 1.500 |
| Closed-degree coverage | 100.0 % | 89.0 % | 1.048 | 2.500 |
| Reverse delete, low-degree | 100.0 % | 88.7 % | 1.049 | 2.500 |
| Medium-degree witnesses | 100.0 % | 88.3 % | 1.054 | 2.500 |
| TSCC/Baker/lift | 100.0 % | 87.8 % | 1.052 | 2.000 |
| Salvador auxiliary | 100.0 % | 84.9 % | 1.107 | 5.000 |
| Low-degree witnesses | 100.0 % | 84.8 % | 1.080 | 3.000 |
| Order ownership, late | 79.2 % | 65.8 % | 1.060 | 2.000 |
| Order ownership, early | 79.2 % | 64.5 % | 1.065 | 2.000 |
| Reverse delete, reverse | 100.0 % | 61.2 % | 1.176 | 3.500 |
| **Max-cut double cover ($C_X$)** | 100.0 % | 58.4 % | 1.217 | 5.000 |
| Reverse delete, input order | 100.0 % | 57.9 % | 1.241 | 5.000 |
| Reverse delete, high-degree | 100.0 % | 27.9 % | 1.428 | 5.000 |

No single strategy is universally optimal; the portfolio minimum combines them to achieve 100 % optimal on every instance.

## CAR-003: High-Degree / Dense Regime (`run_high_degree_experiment.py`)

Probes the regime $\Delta > \sqrt{n}/e$ left open by the low-degree proposition. Base seed 12 345, **1 000 instances** from seven dense families (complete, cocktail-party, dense circulant, random regular, dense Erdős–Rényi, perturbed hypercube, perturbed Paley), including perfect-code stress cases ($\tau(G) = 1$).

| Family | Inst. | $n$ | $\Delta$ | $\tau$ | Mean ratio | Max ratio | ≤ ρ(n) |
|---|---|---|---|---|---|---|---|
| Complete | 143 | 6–16 | 5–15 | 1.00 | 1.000 | 1.000 | 100 % |
| Cocktail party | 143 | 6–16 | 4–14 | 1.67–1.88 | 1.000 | 1.000 | 100 % |
| Dense circulant | 143 | 8–16 | 4–15 | 1.00–1.88 | 1.000 | 1.000 | 100 % |
| Random regular | 143 | 8–16 | 2–8 | 1.00–1.69 | 1.000 | 1.000 | 100 % |
| Dense Erdős–Rényi | 143 | 8–16 | 4–15 | 1.00–2.44 | 1.000 | 1.000 | 100 % |
| Hypercube (perturbed) | 143 | 8–16 | 3–6 | 1.00–1.75 | 1.000 | 1.000 | 100 % |
| Paley (perturbed) | 142 | 5–13 | 2–7 | 1.00–1.85 | 1.000 | 1.000 | 100 % |
| **Overall** | **1000** | **5–16** | **2–15** | **1.00–2.44** | **1.000** | **1.000** | **100 %** |

Furones is **optimal on all 1 000 high-degree instances**, including every $\tau(G) = 1$ perfect-code case. Zero violations of $\rho(n) = \max\\{4, \frac{1}{2}\ln n\\}$.

## CAR-004: Adversarial Stress Test (`run_adversarial_experiment.py`)

Actively searches for worst-case instances. Base seed 1 000, **10 000 instances** from seven adversarial families: greedy traps, geometric set-cover gadgets, perturbed perfect-code graphs, near-efficient circulants, cocktail-party/multipartite graphs, dense random-regular, and dense Erdős–Rényi. Exact $\gamma(G)$ by exhaustive search on every instance.

| Adversarial family | Inst. | $n$ | $\gamma$ | Mean ratio | Max ratio | Optimal | ≤ ρ(n) |
|---|---|---|---|---|---|---|---|
| Greedy trap | 1 429 | 6–16 | 2–4 | 1.000 | 1.000 | 100.0 % | 100 % |
| Geometric set-cover | 1 429 | 10–16 | 3–5 | 1.000 | 1.000 | 100.0 % | 100 % |
| Perfect-code perturbed | 1 429 | 7–16 | 1–4 | 1.000 | 1.000 | 100.0 % | 100 % |
| Efficient circulant | 1 429 | 10–16 | 2–4 | 1.000 | 1.000 | 100.0 % | 100 % |
| Cocktail / multipartite | 1 428 | 4–16 | 2 | 1.000 | 1.000 | 100.0 % | 100 % |
| Random dense | 1 428 | 8–16 | 1–5 | 1.000 | 1.000 | 100.0 % | 100 % |
| Random regular | 1 428 | 8–16 | 2–5 | 1.001 | 1.333 | 99.7 % | 100 % |
| **Overall** | **10 000** | **4–16** | **1–5** | **1.000** | **1.333** | **99.95 %** | **100 %** |

**Zero violations** of $\rho(n)$ across all 10 000 adversarial instances. The worst observed ratio was 1.333 (a dense random-regular graph, $n = 16$, $\gamma = 3$, returned size 4), well below $\rho = 4$.

---

# Problem Statement

Input: A Boolean Adjacency Matrix $M$.

Answer: Find a Minimum Dominating Set.

### Example Instance: 5 × 5 matrix

|        | c1  | c2  | c3  | c4  | c5  |
| ------ | --- | --- | --- | --- | --- |
| **r1** | 0   | 0   | 1   | 0   | 1   |
| **r2** | 0   | 0   | 0   | 1   | 0   |
| **r3** | 1   | 0   | 0   | 0   | 1   |
| **r4** | 0   | 1   | 0   | 0   | 0   |
| **r5** | 1   | 0   | 1   | 0   | 0   |

The input for undirected graphs is provided in [DIMACS](http://dimacs.rutgers.edu/Challenges) format:

```
p edge 5 4
e 1 3
e 1 5
e 2 4
e 3 5
```

_Example Solution:_

Dominating Set Found `1, 2`: nodes `1` and `2` constitute an optimal solution.

---

# Compile and Environment

## Prerequisites

- Python ≥ 3.12

## Installation

```bash
pip install furones
```

## Execution

1. Clone the repository:

   ```bash
   git clone https://github.com/frankvegadelgado/furones.git
   cd furones
   ```

2. Use it from Python:

   ```python
   import networkx as nx
   from furones.algorithm import find_dominating_set

   G = nx.gnp_random_graph(200, 0.03, seed=1)
   D = find_dominating_set(G)                    # with the improvement phase
   D0 = find_dominating_set(G, improve=False)    # portfolio minimum only
   pool = []
   find_dominating_set(G, candidates=pool)       # pool receives the ensemble's candidate sets
   ```

3. Run the script:

   ```bash
   asia -i ./benchmarks/testMatrix1
   ```

   **Example Output:**

   ```
   testMatrix1: Dominating Set Found 1, 2
   ```

---

## Dominating Set Size

Use the `-c` flag to count the nodes in the Dominating Set:

```bash
asia -i ./benchmarks/testMatrix2 -c
```

**Output:**

```
testMatrix2: Dominating Set Size 4
```

---

# Command Options

```bash
asia -h
```

**Output:**

```
usage: asia [-h] -i INPUTFILE [-a] [-b] [-c] [-v] [-l] [--consistency] [--version]

Solve the Approximate Minimum Dominating Set for undirected graph encoded in DIMACS format.

options:
  -h, --help            show this help message and exit
  -i INPUTFILE, --inputFile INPUTFILE
                        input file path
  -a, --approximation   enable comparison with a polynomial-time approximation approach within a logarithmic factor
  -b, --bruteForce      enable comparison with the exponential-time brute-force approach
  -c, --count           calculate the size of the Dominating Set
  -v, --verbose         enable verbose output
  -l, --log             enable file logging
  --consistency         require a linear-time certificate for the Furones approximation bound
  --version             show program's version number and exit
```

---

# Batch Execution

```bash
batch_asia -h
```

**Output:**

```
usage: batch_asia [-h] -i INPUTDIRECTORY [-a] [-b] [-c] [-v] [-l] [--consistency] [--version]

Solve the Approximate Minimum Dominating Set for all undirected graphs encoded in DIMACS format and stored in a directory.

options:
  -h, --help            show this help message and exit
  -i INPUTDIRECTORY, --inputDirectory INPUTDIRECTORY
                        Input directory path
  -a, --approximation   enable comparison with a polynomial-time approximation approach within a logarithmic factor
  -b, --bruteForce      enable comparison with the exponential-time brute-force approach
  -c, --count           calculate the size of the Dominating Set
  -v, --verbose         enable verbose output
  -l, --log             enable file logging
  --consistency         require a linear-time certificate for the Furones approximation bound
  --version             show program's version number and exit
```

---

# Testing Application

```bash
usage: test_asia [-h] -d DIMENSION [-n NUM_TESTS] [-s SPARSITY] [-a] [-b] [-c] [-w] [-v] [-l] [--consistency] [--version]

The Furones Testing Application using randomly generated, large sparse matrices.

options:
  -h, --help            show this help message and exit
  -d DIMENSION, --dimension DIMENSION
                        an integer specifying the dimensions of the square matrices
  -n NUM_TESTS, --num_tests NUM_TESTS
                        an integer specifying the number of tests to run
  -s SPARSITY, --sparsity SPARSITY
                        sparsity of the matrices (0.0 for dense, close to 1.0 for very sparse)
  -a, --approximation   enable comparison with a polynomial-time approximation approach within a logarithmic factor
  -b, --bruteForce      enable comparison with the exponential-time brute-force approach
  -c, --count           calculate the size of the Dominating Set
  -w, --write           write the generated random matrix to a file in the current directory
  -v, --verbose         enable verbose output
  -l, --log             enable file logging
  --consistency         require a linear-time certificate for the Furones approximation bound
  --version             show program's version number and exit
```

---

# Code

- Python implementation by **Frank Vega**.

---

# Complexity

```diff
+ We present a linear-time portfolio algorithm for MDS with an unconditional H(Δ+1) approximation guarantee and exact performance across the CAR benchmarks: optimal on all 1 000 core and 1 000 high-degree instances, and zero factor-4 violations across 10 000 adversarial instances.
```

---

# License

- MIT License.
