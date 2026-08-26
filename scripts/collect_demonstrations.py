"""Record ``(state, action)`` demonstrations from a scripted agent.

This produces the dataset the M4 behaviour-cloning warm start is fitted on
(docs/05 section 5.4).  It drives the unmodified framework directly rather than
going through ``main.py``: after each ``do_step`` the framework has already
stored, on every agent, the state it was shown and the action it returned, so a
demonstration is read out of the world instead of being reconstructed from a
replay.  Nothing in the framework is patched or subclassed.

States are stored in the same encoding the learning route uses, as ``float16``.
The channels are binary or small fractions, so the representation error is
around 1e-4 -- irrelevant for fitting a policy, and it halves a dataset that is
otherwise dominated by 2316 floats per step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_code.research_agent.config import ACTIONS  # noqa: E402
from agent_code.research_agent.state import encode_state, state_dimension  # noqa: E402
from experiment_lib import SCENARIOS, git_provenance, write_json  # noqa: E402


DEFAULT_OUTPUT = ROOT / "pretraining" / "demonstrations"


def collect_from_world(world, *, encoder: str, rounds: int, max_states: int, agent_name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Play ``rounds`` rounds and return the encoded states and action indices.

    A step is skipped when the demonstrator returned something that is not one
    of the six actions -- ``None`` before its first act, or ``"ERROR"`` for a
    silenced exception.  Cloning those would teach the network to imitate the
    framework's substitutions rather than the demonstrator.
    """
    states: list[np.ndarray] = []
    actions: list[int] = []
    skipped = 0
    played_rounds = 0
    for _ in range(rounds):
        world.new_round()
        played_rounds += 1
        while world.running and len(states) < max_states:
            world.do_step()
            for agent in world.agents:
                if agent.name != agent_name or agent.last_game_state is None:
                    continue
                if agent.last_action not in ACTIONS:
                    skipped += 1
                    continue
                states.append(encode_state(agent.last_game_state, encoder).astype(np.float16))
                actions.append(ACTIONS.index(agent.last_action))
        if world.running:
            world.end_round()
        if len(states) >= max_states:
            break
    summary = {"rounds_played": played_rounds, "skipped_unusable_actions": skipped}
    if not states:
        raise RuntimeError("The demonstrator produced no usable state-action pairs.")
    return np.stack(states), np.asarray(actions, dtype=np.int64), summary


def build_world(*, agent_name: str, scenario: str, seed: int, log_dir: Path):
    from environment import BombeRLeWorld, WorldArgs

    log_dir.mkdir(parents=True, exist_ok=True)
    args = WorldArgs(
        no_gui=True, fps=15, turn_based=False, update_interval=0.1, save_replay=False,
        replay=None, make_video=False, continue_without_training=True, log_dir=str(log_dir),
        save_stats=False, match_name=f"demonstrations_{scenario}_seed{seed}", seed=seed,
        silence_errors=False, scenario=scenario,
    )
    # ``False`` is the per-agent train flag: a demonstrator is never trained.
    return BombeRLeWorld(args, [(agent_name, False)])


def collect(
    *, agent_name: str, scenario: str, seeds: list[int], rounds: int, encoder: str,
    max_states: int, output: Path, log_dir: Path,
) -> Path:
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    summaries = []
    for seed in seeds:
        remaining = max_states - sum(len(chunk) for chunk in all_states)
        if remaining <= 0:
            break
        world = build_world(agent_name=agent_name, scenario=scenario, seed=seed, log_dir=log_dir)
        try:
            states, actions, summary = collect_from_world(
                world, encoder=encoder, rounds=rounds, max_states=remaining, agent_name=agent_name
            )
        finally:
            world.end()
        all_states.append(states)
        all_actions.append(actions)
        summaries.append({"seed": seed, **summary, "states": int(len(states))})
        print(f"seed {seed}: {len(states)} states over {summary['rounds_played']} rounds")

    states = np.concatenate(all_states)
    actions = np.concatenate(all_actions)
    counts = np.bincount(actions, minlength=len(ACTIONS))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, states=states, action_indices=actions)
    metadata = {
        "demonstrator": agent_name,
        "scenario": scenario,
        "seeds": seeds,
        "rounds_per_seed": rounds,
        "state_encoder": encoder,
        "state_dimension": state_dimension(encoder),
        "state_dtype": "float16",
        "states": int(len(states)),
        "action_counts": {action: int(count) for action, count in zip(ACTIONS, counts)},
        "per_seed": summaries,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "collected_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        **git_provenance(),
    }
    write_json(output.with_suffix(".json"), metadata)
    print(json.dumps({key: metadata[key] for key in ("states", "action_counts", "sha256")}, indent=2))
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", default="rule_based_agent", help="the demonstrator to record")
    parser.add_argument("--scenario", default="classic", choices=sorted(SCENARIOS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[9001, 9002, 9003])
    parser.add_argument("--rounds", type=int, default=40, help="rounds per seed")
    parser.add_argument("--encoder", default="board_egocentric_v1")
    parser.add_argument("--max-states", type=int, default=20_000,
                        help="hard cap on dataset size; 20k is ample for cloning six actions")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    output = args.output or DEFAULT_OUTPUT / f"{args.agent}_{args.scenario}_{args.max_states}.npz"
    collect(
        agent_name=args.agent, scenario=args.scenario, seeds=args.seeds, rounds=args.rounds,
        encoder=args.encoder, max_states=args.max_states, output=output,
        log_dir=ROOT / "pretraining" / "logs",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
