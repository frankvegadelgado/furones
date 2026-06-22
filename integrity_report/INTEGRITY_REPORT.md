# Furones v0.3.4 Integrity Report

Generated: 2026-06-22T04:15:32.364595+00:00

This is a human-readable Integrity Report prepared for Gauge Freedom Journal's
AI-assistance/CAR expectation.  It documents a ChatGPT-assisted measurement run
on the current local Furones source tree:

`D:\Work\NP\furones`

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

`rho_hat_final = 1.000`

The smallest constant fitting the best valid candidate produced by the measured
strategy pool is:

`rho_hat_best_candidate = 1.000`

All final Furones outputs valid: `True`.

| case | family | reference_size | reference_type | final_size | final_ratio_to_reference | min_valid_candidate_size | min_attaining_strategies |
| --- | --- | --- | --- | --- | --- | --- | --- |
| path_12 | smoke | 4 | exact_formula | 4 | 1.000 | 4 | closed_degree, low_degree_witness, medium_degree_witness, seed_and_complete, salvador_auxiliary, reverse_delete_input, reverse_delete_reverse, reverse_delete_low_degree, lifted_tscc_baker |
| cycle_12 | smoke | 4 | exact_formula | 4 | 1.000 | 4 | closed_degree, low_degree_witness, medium_degree_witness, seed_and_complete, salvador_auxiliary, reverse_delete_input, reverse_delete_reverse, reverse_delete_high_degree, reverse_delete_low_degree, lifted_tscc_baker |
| complete_bipartite_4_5 | salvador_sanity | 2 | exact_argument | 2 | 1.000 | 2 | seed_and_complete, lifted_tscc_baker |
| planted_pair_decoy_q8 | planted_pair_decoy_booster | 2 | exact_argument | 2 | 1.000 | 2 | low_degree_witness, order_ownership_early, seed_and_complete, reverse_delete_reverse |
| planted_pair_decoy_q12 | planted_pair_decoy_booster | 2 | exact_argument | 2 | 1.000 | 2 | low_degree_witness, medium_degree_witness, order_ownership_early, seed_and_complete, reverse_delete_reverse |
| random_set_cover_q20_b120 | random_set_cover | 2 | exact_argument_checked | 2 | 1.000 | 2 | order_ownership_late, seed_and_complete |
| boosted_set_cover_q20_b120_beta40 | boosted_set_cover | 2 | exact_argument_checked | 2 | 1.000 | 2 | order_ownership_late, seed_and_complete, salvador_auxiliary |
| planted_dominator_k2_r160 | planted_dominator | 2 | exact_argument_checked | 2 | 1.000 | 2 | low_degree_witness, seed_and_complete, reverse_delete_reverse |
| decoy_clique_k2_t10 | decoy_private_witness | 2 | exact_argument | 2 | 1.000 | 2 | low_degree_witness, medium_degree_witness, order_ownership_early, seed_and_complete, reverse_delete_reverse, lifted_tscc_baker |
| decoy_clique_k4_t24 | decoy_private_witness | 4 | planted_upper_bound | 4 | 1.000 | 4 | low_degree_witness, medium_degree_witness, seed_and_complete, reverse_delete_reverse, lifted_tscc_baker |
| multiblock_k4_q50_b40 | multiblock | 4 | planted_upper_bound | 4 | 1.000 | 4 | closed_degree, low_degree_witness, medium_degree_witness, order_ownership_late, seed_and_complete, salvador_auxiliary, reverse_delete_low_degree |
| multiblock_k8_q70_b40 | near_threshold_multiblock | 8 | planted_upper_bound | 8 | 1.000 | 8 | low_degree_witness, medium_degree_witness, order_ownership_late, seed_and_complete |

## CAR-002: Per-Strategy Ablation

The percentage column is analogous to the Hvala per-candidate ablation:
it records how often a strategy attained the smallest valid candidate size
among the measured strategy pool.  Ties count for every tied strategy, so the
percentages need not sum to 100.

| strategy | valid_percent | min_attaining_percent | mean_size_valid | mean_ratio_to_reference_valid | max_ratio_to_reference_valid |
| --- | --- | --- | --- | --- | --- |
| seed_and_complete | 100.000 | 100.000 | 3.167 | 1.000 | 1.000 |
| low_degree_witness | 100.000 | 75.000 | 4.333 | 1.583 | 5.000 |
| medium_degree_witness | 100.000 | 58.333 | 5.083 | 1.958 | 5.000 |
| reverse_delete_reverse | 100.000 | 58.333 | 8.917 | 2.365 | 6.000 |
| lifted_tscc_baker | 100.000 | 41.667 | 10.333 | 3.062 | 6.500 |
| order_ownership_late | 83.333 | 33.333 | 8.000 | 3.000 | 6.000 |
| salvador_auxiliary | 100.000 | 33.333 | 8.500 | 3.031 | 9.000 |
| closed_degree | 100.000 | 25.000 | 10.667 | 3.365 | 6.000 |
| order_ownership_early | 83.333 | 25.000 | 12.400 | 3.350 | 6.000 |
| reverse_delete_low_degree | 100.000 | 25.000 | 10.583 | 3.354 | 6.000 |
| reverse_delete_input | 100.000 | 16.667 | 112.500 | 39.583 | 160.000 |
| reverse_delete_high_degree | 100.000 | 8.333 | 112.667 | 39.646 | 160.000 |

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
