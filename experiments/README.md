# Furones v0.3.3 targeted regressions

This folder contains focused ChatGPT-assisted adversarial regressions. They are not exhaustive benchmarks and they are not proofs of a universal approximation ratio.

## Included regressions

### `chatgpt_ratio8_regression.py`

Checks the deterministic planted-pair decoy-booster family that previously forced the solver to select many decoys. The exact optimum is 2 by construction because `{p0,p1}` dominates the graph and no single vertex is universal.

### `chatgpt_ratio75_regression.py`

Checks the random and boosted set-cover-style family that previously reached ratio 7.5. The generator uses two planted vertices, random decoy-element incidence, and optional booster blocks that raise decoy degrees while preserving `{p0,p1}` as an exact optimum of size 2.

## Expected result

For the listed targeted rows, Furones v0.3.3 should return `{p0,p1}` with ratio 1.0. This only shows that the reported regressions are repaired by the current general heuristics.
