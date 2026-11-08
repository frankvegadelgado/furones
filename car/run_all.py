#!/usr/bin/env python3
# Author: Frank Vega
"""
Runs every Furones CAR experiment in sequence.

    run_integrity_measurements.py   CAR-001 + CAR-002 (exact benchmark and per-strategy ablation,
                                    base and improvement strategies)
    run_high_degree_experiment.py   CAR-003
    run_adversarial_experiment.py   CAR-004
    run_strategy_winners.py         strategy winners

The repository's own ``furones/`` package is put first on ``PYTHONPATH`` for every script, so the
experiments run the same whether or not Furones is installed.

Usage (from anywhere):
    python car/run_all.py
    python car/run_all.py --skip run_strategy_winners.py
    python car/run_all.py --keep-going

Options:
    --skip NAME [NAME ...]   scripts not to run
    --keep-going             continue after a script that fails (default: stop)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

CAR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(CAR)

SCRIPTS = [
    "run_integrity_measurements.py",
    "run_high_degree_experiment.py",
    "run_adversarial_experiment.py",
    "run_strategy_winners.py",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all Furones CAR experiments.")
    parser.add_argument("--skip", nargs="*", default=[], help="scripts not to run")
    parser.add_argument("--keep-going", action="store_true", help="continue after a failing script")
    args = parser.parse_args()

    env = dict(os.environ)
    env["PYTHONPATH"] = REPO + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["FURONES_ROOT"] = REPO

    failed = []
    total = time.perf_counter()
    for name in SCRIPTS:
        if name in args.skip:
            print(f"== skipped {name}", flush=True)
            continue
        print(f"\n== {name}", flush=True)
        start = time.perf_counter()
        code = subprocess.call([sys.executable, os.path.join(CAR, name)], cwd=CAR, env=env)
        print(f"== {name}: exit {code} in {time.perf_counter() - start:.1f} s", flush=True)
        if code != 0:
            failed.append(name)
            if not args.keep_going:
                break

    print(f"\nTotal time: {time.perf_counter() - total:.1f} s")
    if failed:
        print("Failed: " + ", ".join(failed))
        return 1
    print("All experiments finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
