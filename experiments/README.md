# ChatGPT-assisted constant-envelope experiment

This folder contains the reproducible experiment for Furones v0.3.1.
The experiment was created with ChatGPT assistance and uses deterministic graph
generators, seeds, and exact bit-set search for the optimum on the tested
instances.

Run from the repository root:

```bash
python experiments/chatgpt_constant_experiment.py
```

The script writes `chatgpt_constant_experiment.json`, which contains the suite
definitions, environment information, summary statistics, worst-case edge
lists, and row-level size/ratio results.

The v0.3.1 battery includes the universal-vertex adversarial family.  That
family previously exposed the dense-edge loss of the pure forest-projection
path; after adding the general linear closed-degree coverage sweep, the solver
selects a singleton dominating set on every tested member of the family.

Current JSON summary:

- instances: 6,842;
- maximum observed ratio: 2.0;
- rows with ratio greater than 2: 0;
- rows with ratio greater than 3: 0.

This is evidence only.  It does not prove a universal approximation theorem for
general Minimum Dominating Set.
