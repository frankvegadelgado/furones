# ChatGPT-assisted constant-envelope experiment

This folder replaces the previous experiment content for Furones v0.3.0.
The experiment was created with ChatGPT assistance and is designed to be
reproducible from deterministic graph generators and seeds.

Run from the repository root:

```bash
python experiments/chatgpt_constant_experiment.py
```

The script writes `chatgpt_constant_experiment.json`, which contains the
suite definitions, environment information, summary statistics, worst-case
edge lists, and the row-level size/ratio results.

The experiment compares `furones.algorithm.find_dominating_set` with an exact
minimum dominating set found by exhaustive bit-set search.  It tests whether
the observed behavior suggests a bounded constant-type phenomenon.  The data
falsify a strict empirical 2-envelope on the tested battery, because ratio 3
occurs, but no tested exact instance exceeds ratio 3.  This is evidence only,
not a proof of a universal approximation theorem.
