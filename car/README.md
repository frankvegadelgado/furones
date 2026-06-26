# Furones CAR Artifacts

This folder follows the Gauge Freedom Journal template convention: CAR material is
kept in a top-level `car/` directory and referenced from the manuscript
declarations.

The package is standard-first rather than tool-locked. It contains four detailed
machine-readable CAR-style measurements, a human-readable Integrity Report, CSV
tables, and the rerun scripts used to generate the artifacts.

## Reproduction (prerequisite: install Furones)

Every CAR experiment runs against the public release, so install it first:

    pip install furones        # v0.3.6 or later
    python run_integrity_measurements.py     # CAR-001 + CAR-002
    python run_high_degree_experiment.py     # CAR-003
    python run_adversarial_experiment.py     # CAR-004

Each script imports `from furones.algorithm import find_dominating_set`, so the
reported numbers reflect the installed solver. To start from a clean state, run
`clean_generated_bundle.ps1` first to remove previously generated JSON/CSV/report
files.

## Files

- `CAR.furones-v0.3.6.json`: submission-facing CAR manifest in the style of the
  journal template.
- `CAR-001-ratio-constant.json`: 1000-instance exact-benchmark ratio-constant
  measurement.
- `CAR-002-strategy-ablation.json`: per-strategy candidate-size and
  min-attainment measurement (includes the dynamic greedy maximum-coverage
  candidate).
- `run_high_degree_experiment.py` / `CAR-003-high-degree.json`: high-degree /
  dense-regime experiment probing `Delta > sqrt(n)/e`, including perfect-code
  stress cases.
- `run_adversarial_experiment.py` / `CAR-004-adversarial.json`: 10,000-instance
  adversarial worst-case stress test.
- `INTEGRITY_REPORT.md`: human-readable report summarizing all four checks.
- `car_benchmark_cases.csv`: selected benchmark cases, exact optima, and
  Furones output sizes.
- `family_summary.csv`: aggregate family-level exact-ratio summary.
- `strategy_ablation_by_instance.csv`: per-instance strategy outputs.
- `strategy_ablation_summary.csv`: per-strategy aggregate percentages.
- `run_integrity_measurements.py`: rerun script for CAR-001/002.
- `clean_generated_bundle.ps1`: removes generated artifacts for a fresh run.

## Scope

These artifacts collect four exact-optimum CAR studies for Furones v0.3.6. In
every study the exact domination number is computed by exhaustive search before
comparison, so each instance is checked against the true optimum.

- **CAR-001 / CAR-002.** A deterministic 1000-instance benchmark, every graph
  with at most 14 vertices. Supports the finite-benchmark observation
  `rho_hat_final = 1.000` and documents which strategies match or tie the
  optimum (per-strategy ablation); the portfolio returns an exact optimum on all
  1000 instances.
- **CAR-003.** 1000 dense instances over seven families, all with maximum degree
  `Delta > sqrt(n)/e` (the regime left open by the low-degree proposition),
  including perfect-code cases (`tau = 1`). Furones is within
  `rho(n) = max(4, 0.5*ln n)` on 100% of instances and optimal on all 1000.
- **CAR-004.** 10,000 adversarial instances drawn from worst-case families
  (greedy traps, the set-cover-to-domination gadget, perturbed perfect codes,
  near-efficient circulants, cocktail-party/multipartite graphs, dense random
  regular, and dense Erdos-Renyi). Furones is valid on every instance, stays
  within `rho(n)` on 100% with **zero violations**, is optimal on 99.95%, and
  has overall mean ratio 1.0001 (max 1.333). Because exact `gamma` forces small
  `n`, the binding bound throughout is the constant floor `rho = 4`.

None of these studies proves a universal constant approximation ratio for
Minimum Dominating Set; they are reproducible finite evidence consistent with
the manuscript's near-threshold ratio hypothesis.
