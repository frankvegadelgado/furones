# Furones v0.3.3 targeted ratio-above-4 regression

This folder contains a short ChatGPT-assisted targeted regression for the deterministic planted-pair decoy-booster family that previously drove the v0.3.2 implementation above ratio 4.

The experiment is deliberately small and fast.  It is not an exhaustive benchmark and not a proof of a universal approximation ratio.

Run from the repository root:

```bash
python experiments/chatgpt_ratio8_regression.py
```

Expected result: the adversarial rows q=8,9,10,12,16 all return the planted pair `{p0,p1}`, giving ratio 1 against the exact optimum 2.
