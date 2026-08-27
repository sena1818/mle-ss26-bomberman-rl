"""Measure what an M4 route actually costs, before committing to training it.

docs/05 section 5.4 puts this first in the M4 ladder, and is explicit about
why: an extrapolated throughput figure is not evidence.  Three numbers decide
whether the route is viable at all.

**Setup cost.** The official framework gives ``act`` 0.5 s and does not time
``setup``.  Importing PyTorch, building the network and running the first
forward pass costs seconds, so it has to happen in ``setup``; this benchmark
reports both so the split is visible rather than assumed.

**Steady-state inference.** What the 0.5 s per-step budget is actually spent
on.  Reported as p50 and p99 over a real encoded game state, with the
worst-case margin against the official limit.

**Gradient throughput.** With ``train_every = 1`` every environment step also
performs one replay gradient step, so a training job's wall clock is dominated
by this rather than by the game.

Run it single-threaded (the default) to get the number a parallel worker sees;
``BOMBERMAN_TORCH_THREADS`` raises it for a single-job measurement.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_code.research_agent.config import ACTIONS, EXPERIMENTS  # noqa: E402
from agent_code.research_agent.state import encode_state, state_dimension  # noqa: E402
from collect_demonstrations import build_world  # noqa: E402
from experiment_lib import git_provenance, write_json  # noqa: E402


OFFICIAL_STEP_BUDGET_SECONDS = 0.5


def _sample_game_state(scenario: str, seed: int) -> dict:
    """Return one mid-round state from the real framework, not a synthetic one.

    Encoding cost depends on how much is on the board -- crates, coins, bombs --
    so a hand-built empty arena would understate it.
    """
    world = build_world(agent_name="rule_based_agent", scenario=scenario, seed=seed,
                        log_dir=ROOT / "pretraining" / "logs")
    try:
        world.new_round()
        for _ in range(20):
            if not world.running:
                break
            world.do_step()
        state = next((agent.last_game_state for agent in world.agents if agent.last_game_state is not None), None)
        if state is None:
            raise RuntimeError("The framework produced no game state to benchmark against.")
        return state
    finally:
        world.end()


def _time(callable_, repeats: int) -> np.ndarray:
    samples = np.empty(repeats)
    for index in range(repeats):
        started = perf_counter()
        callable_()
        samples[index] = perf_counter() - started
    return samples


def measure_torch_import() -> float:
    """Time the PyTorch import in a cold subprocess.

    Measuring it in-process would only be honest for the first route benchmarked
    and would report ~0 for every route after it, which is exactly the kind of
    number that gets copied into a report and believed.
    """
    import subprocess

    source = "from time import perf_counter; started = perf_counter(); import torch; print(perf_counter() - started)"
    completed = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, check=True)
    return float(completed.stdout.strip())



def _time_learner_step(config, model, state: np.ndarray, repeats: int) -> list[float]:
    """Time one *whole* learner update, not just the weight step.

    ``fit_batch`` is a minority of a Double DQN update.  The rest is drawing the
    batch, decoding it out of uint8, relabelling it under a D4 element, and --
    the expensive part -- two forward passes over the next states, one through
    the online network to pick the argmax and one through the target network to
    value it.  Measuring only ``fit_batch`` understated a gradient step by more
    than half, and every schedule built on that number inherited the error.
    """
    from agent_code.research_agent.learners.base import Transition
    from agent_code.research_agent.learners.replay_q import ReplayQLearner

    if config.replay is None:
        return [0.0]
    learner = ReplayQLearner(config, model, seed=0)
    mask = np.ones(len(ACTIONS), dtype=bool)

    def transition(index: int) -> Transition:
        return Transition(state=state, action_index=index % len(ACTIONS), reward=0.1,
                          next_state=state, next_legal_mask=mask, terminal=False,
                          n_step=config.n_step)

    counter = itertools.count()
    # Fill past min_size first, so the timed observes are the ones that update.
    for _ in range(config.replay.min_size + config.replay.train_every):
        learner.observe(transition(next(counter)))
    if not learner.step_diagnostics().get("gradient_applied", False):
        for _ in range(config.replay.train_every):
            learner.observe(transition(next(counter)))

    def one_update() -> None:
        for _ in range(config.replay.train_every):
            learner.observe(transition(next(counter)))

    return _time(one_update, repeats)

def benchmark_route(route: str, *, game_state: dict, repeats: int, steps_per_round: int, rounds: int,
                    first_in_process: bool) -> dict:
    from agent_code.research_agent.models import build_model

    config = EXPERIMENTS[route]
    dimension = state_dimension(config.state_encoder)

    construction_started = perf_counter()
    model = build_model(config, dimension, seed=0)
    constructed = perf_counter() - construction_started

    first_forward_started = perf_counter()
    model.q_values(np.zeros(dimension, dtype=np.float32))
    first_forward = perf_counter() - first_forward_started

    encode = _time(lambda: encode_state(game_state, config.state_encoder), repeats)
    state = encode_state(game_state, config.state_encoder)
    inference = _time(lambda: model.q_values(state), repeats)

    batch_size = config.replay.batch_size if config.replay is not None else 32
    train_every = config.replay.train_every if config.replay is not None else 1
    states = np.repeat(state[None, :], batch_size, axis=0)
    actions = np.arange(batch_size) % len(ACTIONS)
    targets = np.zeros(batch_size, dtype=np.float32)
    fit_only = _time(lambda: model.fit_batch(states, actions, targets), repeats)
    gradient = _time_learner_step(config, model, state, repeats)

    # A gradient step happens once every ``train_every`` environment steps, so
    # its cost is amortised.  Charging one per step -- which this projection did
    # while every route ran ``train_every: 1`` -- overstates a job by a factor of
    # ``train_every``, and the replay ratio is the main throughput lever on this
    # line, so getting it wrong would misprice the whole schedule.
    per_step = float(
        np.median(encode) + np.median(inference) + np.median(gradient) / train_every
    )
    worst_act = float(np.percentile(encode, 99) + np.percentile(inference, 99))
    return {
        "route": route,
        "model": config.network,
        "state_dimension": dimension,
        "batch_size": batch_size,
        "train_every": train_every,
        "replay_ratio": batch_size / train_every,
        # Excludes the PyTorch import, which is process-wide and reported once.
        # Torch also initialises its kernels lazily, so only the first model
        # built in a process pays for that; a real job always builds exactly one.
        "setup_seconds": {"construction": constructed, "first_forward": first_forward,
                          "total": constructed + first_forward, "first_in_process": first_in_process},
        "encode_seconds": {"p50": float(np.median(encode)), "p99": float(np.percentile(encode, 99))},
        "inference_seconds": {"p50": float(np.median(inference)), "p99": float(np.percentile(inference, 99))},
        "act_seconds": {"p99": worst_act},
        "official_budget_margin": OFFICIAL_STEP_BUDGET_SECONDS / worst_act,
        # ``fit_batch_seconds`` is kept separately because it is what the
        # earlier version of this script reported; the two together say how much
        # of an update is the weight step and how much is everything around it.
        "fit_batch_seconds": {"p50": float(np.median(fit_only))},
        "gradient_step_seconds": {"p50": float(np.median(gradient)), "p99": float(np.percentile(gradient, 99))},
        "gradient_steps_per_second": 1.0 / float(np.median(gradient)),
        "environment_steps_per_second": 1.0 / per_step,
        "projected_training_job": {
            "rounds": rounds,
            "steps_per_round": steps_per_round,
            "environment_steps": steps_per_round * rounds,
            "seconds": per_step * steps_per_round * rounds,
            "minutes": per_step * steps_per_round * rounds / 60.0,
            "hours": per_step * steps_per_round * rounds / 3600.0,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--routes", nargs="+", default=["R07", "R08"], choices=sorted(EXPERIMENTS))
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--scenario", default="classic")
    parser.add_argument("--seed", type=int, default=9001)
    parser.add_argument("--steps-per-round", type=int, default=312,
                        help="measured A03 mean for solo classic; see docs/05 section 0.1")
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    torch_import_seconds = measure_torch_import()
    game_state = _sample_game_state(args.scenario, args.seed)
    results = [benchmark_route(route, game_state=game_state, repeats=args.repeats,
                               steps_per_round=args.steps_per_round, rounds=args.rounds,
                               first_in_process=index == 0)
               for index, route in enumerate(args.routes)]
    report = {
        "measured_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "repeats": args.repeats,
        "scenario": args.scenario,
        "torch_import_seconds": torch_import_seconds,
        "torch_threads": int(os.environ.get("BOMBERMAN_TORCH_THREADS", "1")),
        "routes": results,
        **git_provenance(),
    }
    if args.output:
        write_json(args.output, report)

    print(f"torch import (cold process): {torch_import_seconds * 1e3:.0f} ms, paid once per job in setup()")
    print(f"{'route':>6} {'setup':>9} {'act p99':>10} {'margin':>9} {'grad step':>11} {'ratio':>7} {'env/s':>8} {'10k rounds':>12}")
    for entry in results:
        print(f"{entry['route']:>6} "
              f"{entry['setup_seconds']['total'] * 1e3:>7.0f}ms{'*' if entry['setup_seconds']['first_in_process'] else ' '}"
              f"{entry['act_seconds']['p99'] * 1e3:>8.2f}ms "
              f"{entry['official_budget_margin']:>8.0f}x "
              f"{entry['gradient_step_seconds']['p50'] * 1e3:>9.2f}ms "
              f"{entry['replay_ratio']:>7.0f} "
              f"{entry['environment_steps_per_second']:>8.0f} "
              f"{entry['projected_training_job']['hours']:>10.1f}h")
    print("* first model built in this process; it also pays torch's lazy kernel init. "
          "A real job builds one model, so its setup cost is the import plus this line.")
    if not args.output:
        print()
        print(json.dumps(report["routes"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
