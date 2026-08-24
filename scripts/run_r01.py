#!/usr/bin/env python3
"""Retired unsafe R01 runner.

Use ``scripts/run_experiment.py``. The former runner wrote model and framework
logs into shared locations, so keeping it executable would reintroduce a
concurrent-job overwrite path.
"""

raise SystemExit(
    "scripts/run_r01.py is retired. Use: python scripts/run_experiment.py run "
    "--config experiments/r01_a00_smoke.json --allow-dirty"
)
