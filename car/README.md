# Furones CAR Artifacts

This folder follows the Gauge Freedom Journal template convention: CAR material is
kept in a top-level `car/` directory and referenced from the manuscript
declarations.

The package is standard-first rather than tool-locked.  It contains two detailed
machine-readable CAR-style measurements, a human-readable Integrity Report, CSV
tables, and the rerun script used to generate the artifacts.

## Files

- `CAR.furones-v0.3.4.json`: submission-facing CAR manifest in the style of the
  journal template.
- `CAR-001-ratio-constant.json`: 1000-instance exact-benchmark ratio-constant
  measurement.
- `CAR-002-strategy-ablation.json`: per-strategy candidate-size and
  min-attainment measurement.
- `INTEGRITY_REPORT.md`: human-readable report summarizing both checks.
- `car_benchmark_cases.csv`: selected benchmark cases, exact optima, and
  Furones output sizes.
- `family_summary.csv`: aggregate family-level exact-ratio summary.
- `strategy_ablation_by_instance.csv`: per-instance strategy outputs.
- `strategy_ablation_summary.csv`: per-strategy aggregate percentages.
- `run_integrity_measurements.py`: one-command local rerun script.

## Scope

These artifacts measure a deterministic 1000-instance exact CAR benchmark for
Furones v0.3.4.  Every graph has at most 14 vertices, so the script computes
the exact domination number by exhaustive search before comparing Furones with
the optimum.  The run supports the finite experimental claim
`rho_hat_final = 1.000` on the selected cases, and it documents which
strategies match or tie the optimum on those cases.  It does not prove a
universal constant approximation ratio for Minimum Dominating Set.
