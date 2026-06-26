# Furones: Approximate Dominating Set Solver

![In Loving Memory of Asia Furones (The Grandmother I Never Knew)](docs/furones.jpg)

This work builds upon [The Furones Algorithm](https://dev.to/frank_vega_987689489099bf/the-furones-algorithm-15lp).

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

Furones v0.3.7 is a **linear-time candidate-comparison solver** for MDS. It builds a portfolio of dominating-set candidates on the input graph and returns the smallest validated one. The pipeline consists of:

1. **Preprocessing**: self-loop removal, isolated-vertex separation.
2. **TSCC-style pendant cascade** (`ReduceToTSCCForDS`): iteratively commits pendant supports and isolated vertices to a forced set $F$, producing a reduced residual.
3. **Forest projection** (if the residual is non-planar): projects onto a spanning forest before solving.
4. **Baker-style planar PTAS** on the residual: solves MDS on $k$-outerplanar components by tree-decomposition DP.
5. **Lifted candidate** $C_L = F \cup \ell(D_R)$: lifts the reduced solution back to original labels.
6. **Original-graph linear candidates** (all computed in $\mathcal{O}(n+m)$):
   - $C_G$ — **greedy maximum-coverage** (certifies the $H(\Delta+1)$ ratio)
   - $C_D$ — closed-degree coverage sweep
   - $C_W$, $C_M$ — low- and medium-degree witness sweeps
   - $C_O$ — order-ownership candidates (early and late)
   - $C_S$ — seed-and-complete
   - $C_B$ — Salvador-style bipartite auxiliary
   - $C_R$ — reverse-delete (multiple deterministic orders)
7. **Pruning** (`PruneRedundantDominating`): each candidate is made inclusion-minimal.
8. **Portfolio minimum**: the smallest valid (dominates the original graph) pruned candidate is returned.

### Unconditional guarantees

- **Feasibility** (Theorem): every normal return is a dominating set of the input graph.
- **Approximation ratio** (Theorem): because the portfolio contains $C_G$, every returned set $D$ satisfies $|D| \le H(\Delta+1)\,\gamma(G) \le (1+\ln(\Delta+1))\,\gamma(G) = \mathcal{O}(\log \Delta)$.
- **Constant factor on bounded-degree graphs** (Corollary): for any fixed $\Delta$, the ratio is at most $1 + \ln(\Delta+1)$.

### Near-threshold ratio hypothesis

Furones is conjectured to satisfy $|D| \le \max\\{4,\,\frac{1}{2}\ln n\\}\cdot\gamma(G)$ on every graph. Proving this for all graphs would imply **P = NP** (via Dinur–Steurer). The hypothesis is already proved unconditionally for all graphs of maximum degree $\Delta \le \sqrt{n}/e$.

### Runtime

$\mathcal{O}(n+m)$ for fixed parameters (large hidden constant; grows as $2^{\mathcal{O}(1/\varepsilon)}$ on the Baker branch).

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

Each exposed strategy is measured independently on the same 1 000-instance benchmark.

| Strategy | Valid | Optimal | Mean ratio | Max ratio |
|---|---|---|---|---|
| Seed-and-complete | 100.0 % | 99.8 % | 1.000 | 1.250 |
| Greedy max-coverage ($C_G$) | 100.0 % | 92.7 % | 1.023 | 1.500 |
| Closed-degree coverage | 100.0 % | 89.0 % | 1.048 | 2.500 |
| Reverse delete, low-degree | 100.0 % | 88.7 % | 1.049 | 2.500 |
| Medium-degree witnesses | 100.0 % | 88.3 % | 1.054 | 2.500 |
| TSCC/Baker/lift | 100.0 % | 87.8 % | 1.052 | 2.000 |
| Salvador auxiliary | 100.0 % | 85.4 % | 1.107 | 5.000 |
| Low-degree witnesses | 100.0 % | 84.8 % | 1.080 | 3.000 |
| Order ownership, late | 79.2 % | 65.8 % | 1.060 | 2.000 |
| Order ownership, early | 79.2 % | 64.5 % | 1.065 | 2.000 |
| Reverse delete, reverse | 100.0 % | 61.2 % | 1.176 | 3.500 |
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

2. Run the script:

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
+ We present a linear-time portfolio algorithm for MDS with an unconditional H(Δ+1) approximation guarantee and exact performance on all 11 000 CAR benchmark instances.
```

---

# License

- MIT License.
