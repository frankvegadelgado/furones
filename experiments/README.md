# Furones v0.3.2 targeted ChatGPT-assisted regressions

This folder contains the targeted reproducibility experiment used in the v0.3.2 manuscript.

The experiment is intentionally small. It is not a full exhaustive benchmark battery. It was created after adversarial examples were found against earlier deterministic versions of the solver.

## Purpose

The script checks that Furones v0.3.2 overcomes the following reported adversarial families without adding special-case detectors:

1. A planted-dominator dense graph with exact optimum 2.
2. A decoy-clique/private-witness family with exact optimum 2 for the small cases and a planted promise set of size k for the larger cases.
3. Small smoke-test graphs.

The important point is that the new solver uses general linear candidates:

- closed-degree coverage sweep;
- low-degree-witness coverage sweep;
- reverse-delete scans in several deterministic orders;
- TSCC/Baker/lift candidate as the reduced-instance path.

The low-degree-witness sweep is not a detector for the decoy construction. It gives higher priority to vertices that dominate many low-degree witnesses. This is a general heuristic motivated by private-neighborhood structure.

## Files

- `chatgpt_ratio4_regression.py`: Python script that constructs the adversarial families and runs the solver.
- `chatgpt_ratio4_regression.json`: JSON output from the targeted run.

## How to run

From the repository root:

```bash
python experiments/chatgpt_ratio4_regression.py