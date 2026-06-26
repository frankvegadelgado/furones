# Furones v0.3.6 Integrity Report

Generated: 2026-06-26T06:37:09.305264+00:00

This is a human-readable Integrity Report prepared for Gauge Freedom Journal's
AI-assistance/CAR expectation. It documents the exact-optimum measurement run
on the Furones source tree:

`D:\Work\NP\furones`

CAR-001 and CAR-002 below are produced by this script; CAR-003 and CAR-004 are
companion studies produced by `run_high_degree_experiment.py` and
`run_adversarial_experiment.py`.

This script checks two questions:

1. Measure the finite-benchmark constant suggested by an exact-optimum
   benchmark.
2. Run each exposed strategy and measure candidate sizes plus min-attainment
   percentages on the selected CAR benchmark. The strategy pool now includes
   the dynamic greedy maximum-coverage candidate (`greedy_max_coverage`), which
   certifies the H(Δ+1) approximation guarantee.

The benchmark contains 1000 small instances with at most
14 vertices. The exact domination number of every
instance was found by exhaustive search. This is useful as a reproducible
regression and integrity check, but it is not an exhaustive benchmark and does
not prove a universal constant approximation ratio for Minimum Dominating Set.

## CAR-001: Observed Ratio Constant

For this finite exact-optimum benchmark, the smallest constant fitting the final
Furones outputs is:

`rho_hat_final = 1.000`

The smallest constant fitting the best valid candidate produced by the measured
strategy pool is:

`rho_hat_best_candidate = 1.000`

All final Furones outputs valid: `True`.

Final output optimal instances: `1000/1000`
(`100.000%`).

Mean final ratio to optimum: `1.000`.
Median final ratio to optimum: `1.000`.
95th percentile final ratio to optimum: `1.000`.

### Family Summary

| family | instances | n_min | n_max | optimum_min | optimum_max | final_optimal_percent | mean_final_ratio_to_optimum | max_final_ratio_to_optimum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| erdos_renyi_dense | 150 | 6 | 14 | 1 | 4 | 100.000 | 1.000 | 1.000 |
| erdos_renyi_medium | 200 | 6 | 14 | 2 | 5 | 100.000 | 1.000 | 1.000 |
| erdos_renyi_sparse | 200 | 6 | 14 | 2 | 10 | 100.000 | 1.000 | 1.000 |
| perturbed_path_cycle | 100 | 6 | 14 | 2 | 5 | 100.000 | 1.000 | 1.000 |
| random_bipartite | 150 | 6 | 14 | 2 | 9 | 100.000 | 1.000 | 1.000 |
| random_tree | 100 | 6 | 14 | 2 | 6 | 100.000 | 1.000 | 1.000 |
| small_structured | 100 | 6 | 14 | 1 | 4 | 100.000 | 1.000 | 1.000 |

### Worst Final-Ratio Cases

| case | family | n | m | optimum_size | final_size | final_ratio_to_optimum | min_valid_candidate_size | min_attaining_strategies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bipartite_000 | random_bipartite | 6 | 2 | 4 | 4 | 1.000 | 4 | greedy_max_coverage, closed_degree, low_degree_witness, medium_degree_witness, seed_and_complete, salvador_auxiliary, reverse_delete_input, reverse_delete_low_degree, lifted_tscc_baker |
| bipartite_001 | random_bipartite | 7 | 3 | 4 | 4 | 1.000 | 4 | greedy_max_coverage, closed_degree, low_degree_witness, medium_degree_witness, seed_and_complete, salvador_auxiliary, reverse_delete_reverse, reverse_delete_low_degree, lifted_tscc_baker |
| bipartite_002 | random_bipartite | 8 | 5 | 4 | 4 | 1.000 | 4 | closed_degree, low_degree_witness, medium_degree_witness, seed_and_complete, salvador_auxiliary, reverse_delete_low_degree, lifted_tscc_baker |
| bipartite_003 | random_bipartite | 9 | 4 | 6 | 6 | 1.000 | 6 | greedy_max_coverage, closed_degree, low_degree_witness, medium_degree_witness, seed_and_complete, salvador_auxiliary, reverse_delete_input, reverse_delete_reverse, reverse_delete_high_degree, reverse_delete_low_degree, lifted_tscc_baker |
| bipartite_004 | random_bipartite | 10 | 7 | 4 | 4 | 1.000 | 4 | greedy_max_coverage, closed_degree, low_degree_witness, medium_degree_witness, order_ownership_late, order_ownership_early, seed_and_complete, salvador_auxiliary, reverse_delete_input, reverse_delete_low_degree, lifted_tscc_baker |
| bipartite_005 | random_bipartite | 7 | 9 | 2 | 2 | 1.000 | 2 | greedy_max_coverage, closed_degree, order_ownership_late, order_ownership_early, seed_and_complete, reverse_delete_low_degree |
| bipartite_006 | random_bipartite | 8 | 3 | 5 | 5 | 1.000 | 5 | greedy_max_coverage, closed_degree, low_degree_witness, medium_degree_witness, seed_and_complete, salvador_auxiliary, reverse_delete_input, reverse_delete_low_degree, lifted_tscc_baker |
| bipartite_007 | random_bipartite | 9 | 6 | 4 | 4 | 1.000 | 4 | greedy_max_coverage, closed_degree, low_degree_witness, medium_degree_witness, seed_and_complete, salvador_auxiliary, reverse_delete_input, reverse_delete_low_degree, lifted_tscc_baker |
| bipartite_008 | random_bipartite | 10 | 6 | 6 | 6 | 1.000 | 6 | greedy_max_coverage, closed_degree, low_degree_witness, medium_degree_witness, order_ownership_late, order_ownership_early, seed_and_complete, salvador_auxiliary, reverse_delete_reverse, reverse_delete_low_degree, lifted_tscc_baker |
| bipartite_009 | random_bipartite | 11 | 10 | 6 | 6 | 1.000 | 6 | greedy_max_coverage, closed_degree, order_ownership_late, order_ownership_early, seed_and_complete, salvador_auxiliary, reverse_delete_low_degree, lifted_tscc_baker |
| bipartite_010 | random_bipartite | 8 | 9 | 3 | 3 | 1.000 | 3 | greedy_max_coverage, closed_degree, low_degree_witness, medium_degree_witness, order_ownership_late, order_ownership_early, seed_and_complete, salvador_auxiliary, reverse_delete_reverse, reverse_delete_low_degree, lifted_tscc_baker |
| bipartite_011 | random_bipartite | 9 | 13 | 3 | 3 | 1.000 | 3 | greedy_max_coverage, closed_degree, order_ownership_late, order_ownership_early, seed_and_complete, salvador_auxiliary, reverse_delete_low_degree, lifted_tscc_baker |
| bipartite_012 | random_bipartite | 10 | 6 | 5 | 5 | 1.000 | 5 | greedy_max_coverage, closed_degree, low_degree_witness, medium_degree_witness, seed_and_complete, salvador_auxiliary, reverse_delete_low_degree, lifted_tscc_baker |
| bipartite_013 | random_bipartite | 11 | 8 | 5 | 5 | 1.000 | 5 | greedy_max_coverage, closed_degree, low_degree_witness, medium_degree_witness, order_ownership_late, order_ownership_early, seed_and_complete, salvador_auxiliary, reverse_delete_input, reverse_delete_high_degree, reverse_delete_low_degree, lifted_tscc_baker |
| bipartite_014 | random_bipartite | 12 | 13 | 5 | 5 | 1.000 | 5 | closed_degree, low_degree_witness, medium_degree_witness, order_ownership_late, order_ownership_early, seed_and_complete, salvador_auxiliary, reverse_delete_low_degree, lifted_tscc_baker |
| bipartite_015 | random_bipartite | 9 | 10 | 3 | 3 | 1.000 | 3 | greedy_max_coverage, closed_degree, low_degree_witness, medium_degree_witness, order_ownership_late, order_ownership_early, seed_and_complete, salvador_auxiliary, reverse_delete_reverse, reverse_delete_low_degree, lifted_tscc_baker |
| bipartite_016 | random_bipartite | 10 | 13 | 3 | 3 | 1.000 | 3 | greedy_max_coverage, closed_degree, order_ownership_late, order_ownership_early, seed_and_complete, salvador_auxiliary, reverse_delete_low_degree, lifted_tscc_baker |
| bipartite_017 | random_bipartite | 11 | 13 | 4 | 4 | 1.000 | 4 | greedy_max_coverage, closed_degree, order_ownership_late, order_ownership_early, seed_and_complete, lifted_tscc_baker |
| bipartite_018 | random_bipartite | 12 | 5 | 8 | 8 | 1.000 | 8 | greedy_max_coverage, closed_degree, low_degree_witness, medium_degree_witness, seed_and_complete, salvador_auxiliary, reverse_delete_input, reverse_delete_reverse, reverse_delete_high_degree, reverse_delete_low_degree, lifted_tscc_baker |
| bipartite_019 | random_bipartite | 13 | 8 | 7 | 7 | 1.000 | 7 | greedy_max_coverage, closed_degree, low_degree_witness, medium_degree_witness, seed_and_complete, salvador_auxiliary, reverse_delete_input, reverse_delete_high_degree, reverse_delete_low_degree, lifted_tscc_baker |

## CAR-002: Per-Strategy Ablation

The percentage column is analogous to the Hvala per-candidate ablation:
it records how often a strategy attained the smallest valid candidate size
among the measured strategy pool. Ties count for every tied strategy, so the
percentages need not sum to 100.

| strategy | valid_percent | optimal_percent | min_attaining_percent | mean_ratio_to_optimum_valid | max_ratio_to_optimum_valid |
| --- | --- | --- | --- | --- | --- |
| seed_and_complete | 100.000 | 99.800 | 99.800 | 1.000 | 1.250 |
| greedy_max_coverage | 100.000 | 92.700 | 92.700 | 1.023 | 1.500 |
| closed_degree | 100.000 | 89.000 | 89.000 | 1.048 | 2.500 |
| reverse_delete_low_degree | 100.000 | 88.700 | 88.700 | 1.049 | 2.500 |
| medium_degree_witness | 100.000 | 88.300 | 88.300 | 1.054 | 2.500 |
| lifted_tscc_baker | 100.000 | 87.800 | 87.800 | 1.052 | 2.000 |
| salvador_auxiliary | 100.000 | 85.400 | 85.400 | 1.107 | 5.000 |
| low_degree_witness | 100.000 | 84.800 | 84.800 | 1.080 | 3.000 |
| order_ownership_late | 79.200 | 65.800 | 65.800 | 1.060 | 2.000 |
| order_ownership_early | 79.200 | 64.500 | 64.500 | 1.065 | 2.000 |
| reverse_delete_reverse | 100.000 | 61.200 | 61.200 | 1.176 | 3.500 |
| reverse_delete_input | 100.000 | 57.900 | 57.900 | 1.241 | 5.000 |
| reverse_delete_high_degree | 100.000 | 27.900 | 27.900 | 1.428 | 5.000 |

## Companion CAR Studies

- **CAR-003** (`run_high_degree_experiment.py`, `CAR-003-high-degree.json`):
  high-degree / dense regime probe (Delta > sqrt(n)/e), run against the
  installed Furones solver.
- **CAR-004** (`run_adversarial_experiment.py`, `CAR-004-adversarial.json`):
  10,000-instance adversarial worst-case stress test against the installed
  Furones solver.

## AI Assistance Scope

CAR-001 and CAR-002 were designed with ChatGPT assistance; CAR-003 and CAR-004
were designed with Claude. The v0.3.6 update of this script, including the
`greedy_max_coverage` ablation entry, was made with Claude. Every run executed
the Furones code directly and wrote the JSON/CSV outputs in this folder.
CAR-004 was run against the public release installed via `pip install furones`
(v0.3.6) and invoked as `from furones.algorithm import find_dominating_set`.
The author reviewed all results and remains responsible for any manuscript
claims.

## Files

- `CAR.furones-v0.3.6.json`
- `CAR-001-ratio-constant.json`
- `CAR-002-strategy-ablation.json`
- `CAR-003-high-degree.json`
- `CAR-004-adversarial.json`
- `car_benchmark_cases.csv`
- `family_summary.csv`
- `strategy_ablation_by_instance.csv`
- `strategy_ablation_summary.csv`
- `high_degree_by_instance.csv`
- `adversarial_by_instance.csv`
- `run_integrity_measurements.py`
- `run_high_degree_experiment.py`
- `run_adversarial_experiment.py`
