#!/usr/bin/env python3
"""Export a self-contained tournament agent from the research checkout.

What the tournament actually runs is one directory copied out of ``agent_code``
into somebody else's checkout, imported as ``agent_code.<name>.callbacks``, with
none of this project's environment variables set and no ``scripts/`` beside it.
Three things in the research agent depend on that context and are pinned here
instead:

* the route, which otherwise falls back to R01 -- the linear baseline, so a
  submitted CNN would quietly play as the first arm this project ever ran;
* the weights, which are addressed relative to the agent directory, never
  absolutely, because the organisers unpack it somewhere nobody can predict;
* the per-action log, which every diagnostic in docs/01 reads and which nothing
  in the tournament ever will.

The export is then *verified by running it*: a fresh interpreter with every
``BOMBERMAN_*`` variable stripped plays real rounds from a temporary copy of the
framework, and the per-step decision time is measured against the official
0.5 s budget rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from experiment_lib import ROOT, git_provenance, write_json

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_code.research_agent.config import EXPERIMENTS, validate_config  # noqa: E402
from agent_code.research_agent.models.ensemble import MANIFEST_SUFFIX  # noqa: E402
from agent_code.research_agent.state import state_dimension  # noqa: E402

SOURCE = ROOT / "agent_code" / "research_agent"
# Everything the runtime imports, and nothing else.  An allowlist rather than a
# deny-list for the reason copy_runtime gives: a deny-list silently ships
# whatever new directory appears, and here that would mean shipping training
# logs or a replay buffer into a submission.
PACKAGE_FILES = ("__init__.py", "callbacks.py", "train.py", "config.py", "state.py",
                 "shaping.py", "symmetry.py", "replay.py", "artifacts.py",
                 "learning.py", "networks.py")
PACKAGE_DIRECTORIES = ("runtime", "learners", "models")
IGNORED = shutil.ignore_patterns("__pycache__", "*.pyc", "tests", "artifacts", "logs")


def _members_of(manifest: Path) -> list[Path]:
    declared = json.loads(manifest.read_text(encoding="utf-8"))
    return [manifest.parent / entry["path"] for entry in declared["members"]]


def export(*, route: str, model: Path, name: str, destination_root: Path,
           action_log: bool = False) -> Path:
    config = validate_config(EXPERIMENTS[route])
    destination = destination_root / name
    if destination.exists():
        raise SystemExit(f"refusing to overwrite {destination}")
    if not model.is_file():
        raise SystemExit(f"model is unavailable: {model}")
    # Read before the export writes anything.  Taken afterwards it describes the
    # tree the export just made untracked, so every package recorded
    # worktree_dirty even when it came from a clean checkout -- a provenance
    # field that is always true says nothing.
    provenance = git_provenance()

    destination.mkdir(parents=True)
    for file_name in PACKAGE_FILES:
        source = SOURCE / file_name
        if source.is_file():
            shutil.copy2(source, destination / file_name)
    for directory in PACKAGE_DIRECTORIES:
        shutil.copytree(SOURCE / directory, destination / directory, ignore=IGNORED)

    if model.name.endswith(MANIFEST_SUFFIX):
        # The manifest resolves its members relative to itself, so the whole
        # member directory travels with it.
        model_name = f"model{MANIFEST_SUFFIX}"
        shutil.copy2(model, destination / model_name)
        members = destination / "members"
        members.mkdir()
        for member in _members_of(model):
            shutil.copy2(member, members / member.name)
        # Rewrite the paths so they point at the copied members.
        declared = json.loads((destination / model_name).read_text(encoding="utf-8"))
        for entry, member in zip(declared["members"], _members_of(model)):
            entry["path"] = f"members/{member.name}"
        (destination / model_name).write_text(
            json.dumps(declared, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        model_name = "model.npz"
        shutil.copy2(model, destination / model_name)

    write_json(destination / "submission.json", {
        "route": route,
        "model": model_name,
        "action_log": bool(action_log),
        "state_representation": config.state_encoder,
        "network": config.network,
        "algorithm": config.algorithm,
        "state_dimension": state_dimension(config.state_encoder),
        "exported_from": str(model),
        **provenance,
    })
    return destination


# The framework logs a decision's duration as ``{:.2f}s``, so anything under
# ten milliseconds reads as 0.00 and the margin against the 0.5 s budget cannot
# be seen at all.  This harness times the packaged ``act`` itself, on states
# taken from a real game rather than synthesised, and reports the maximum --
# the budget is per step, so the worst decision is the one that matters.
_TIMING_HARNESS = """
import json, logging, sys, time, types
from environment import BombeRLeWorld, WorldArgs
import importlib

name = sys.argv[1]
callbacks = importlib.import_module("agent_code.%s.callbacks" % name)
logging.getLogger().addHandler(logging.NullHandler())

args = WorldArgs(no_gui=True, fps=15, turn_based=False, update_interval=0.1, save_replay=False,
                 replay=None, make_video=False, continue_without_training=True, log_dir="logs",
                 save_stats=False, match_name="timing", seed=20260917, silence_errors=False,
                 scenario="classic")
world = BombeRLeWorld(args, [("rule_based_agent", False)] * 4)
states = []
for _ in range(3):
    world.new_round()
    while world.running and len(states) < 400:
        world.do_step()
        for agent in world.agents:
            if agent.last_game_state is not None:
                states.append(agent.last_game_state)
    if world.running:
        world.end_round()
world.end()

fake = types.SimpleNamespace(train=False, logger=logging.getLogger("timing"))
started = time.perf_counter()
callbacks.setup(fake)
setup_seconds = time.perf_counter() - started

times = []
for state in states:
    began = time.perf_counter()
    callbacks.act(fake, state)
    times.append(time.perf_counter() - began)
times.sort()
print(json.dumps({
    "setup_seconds": setup_seconds,
    "decisions_timed": len(times),
    "max_decision_seconds": times[-1],
    "median_decision_seconds": times[len(times) // 2],
    "p99_decision_seconds": times[min(len(times) - 1, int(0.99 * len(times)))],
}))
"""


def _time_decisions(sandbox: Path, name: str, environment: dict) -> dict:
    """Time the packaged ``act`` on real states, inside the sandbox."""
    harness = sandbox / "_timing_harness.py"
    harness.write_text(_TIMING_HARNESS, encoding="utf-8")
    result = subprocess.run([sys.executable, harness.name, name], cwd=sandbox, env=environment,
                            capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(f"timing the packaged agent failed:\n{result.stderr[-4000:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def verify(package: Path, *, rounds: int, opponents: tuple[str, ...], scenario: str) -> dict:
    """Play real rounds from a temporary framework copy, with no project variables.

    The copy is what makes this a test rather than a demonstration: the package
    is exercised through ``agent_code/<name>`` of a tree that holds nothing else
    of this project -- no ``scripts``, no ``experiments``, no ``runs`` -- so an
    import or a path that only resolves in the development checkout fails here
    instead of in the tournament.
    """
    with tempfile.TemporaryDirectory() as directory:
        sandbox = Path(directory) / "framework"
        (sandbox / "agent_code").mkdir(parents=True)
        for module in sorted(ROOT.glob("*.py")):
            shutil.copy2(module, sandbox / module.name)
        shutil.copytree(ROOT / "assets", sandbox / "assets")
        for agent in {package.name, *opponents}:
            source = package if agent == package.name else ROOT / "agent_code" / agent
            shutil.copytree(source, sandbox / "agent_code" / agent, ignore=IGNORED)
        (sandbox / "logs").mkdir()
        stats = sandbox / "stats.json"
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith("BOMBERMAN_")}
        command = [sys.executable, "main.py", "play", "--agents", package.name, *opponents,
                   "--scenario", scenario, "--n-rounds", str(rounds), "--seed", "20260917",
                   "--no-gui", "--save-stats", str(stats), "--match-name", "submission_check"]
        result = subprocess.run(command, cwd=sandbox, env=environment,
                                capture_output=True, text=True)
        if result.returncode:
            raise SystemExit(f"the packaged agent failed to play:\n{result.stderr[-4000:]}")
        played = json.loads(stats.read_text(encoding="utf-8"))
        agent_log = sandbox / "agent_code" / package.name / "logs" / f"{package.name}.log"
        agent_log_written = (sandbox / "agent_code" / package.name / "artifacts").exists()
        game_log = (sandbox / "logs" / "game.log").read_text(encoding="utf-8", errors="replace")
        timing = _time_decisions(sandbox, package.name, environment)
    mine = played["by_agent"][package.name]
    exceeded = [line for line in game_log.splitlines()
                if "exceeded think time" in line and package.name in line]
    report = {
        "rounds": int(mine["rounds"]),
        "score_per_round": mine["score"] / max(1, int(mine["rounds"])),
        "steps": int(mine.get("steps", 0)),
        "invalid_actions": int(mine.get("invalid", 0)),
        "timeouts": len(exceeded),
        "packaged_action_log_written": agent_log_written,
    }
    report.update(timing)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--route", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--model", type=Path, required=True,
                        help="A .npz checkpoint, or a .ensemble.json manifest with its members.")
    parser.add_argument("--name", required=True, help="Directory name; it identifies the agent in the tournament.")
    parser.add_argument("--destination", type=Path, default=ROOT / "agent_code")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--opponents", nargs="*", default=["rule_based_agent"] * 3)
    parser.add_argument("--scenario", default="classic")
    parser.add_argument("--action-log", action="store_true",
                        help="Keep the per-action record in the packaged agent. Off by default.")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    package = export(route=args.route, model=args.model, name=args.name,
                     destination_root=args.destination, action_log=args.action_log)
    print(f"exported {package}")
    for path in sorted(package.rglob("*")):
        if path.is_file() and path.suffix in {".npz", ".json"}:
            print(f"  {path.relative_to(package)}  {path.stat().st_size / 1024:.0f} KiB")
    if args.skip_verify:
        return
    print(f"\nverifying: {args.rounds} rounds of {args.scenario} against "
          f"{', '.join(args.opponents) or 'nobody'}, no BOMBERMAN_* variables, temporary framework copy")
    report = verify(package, rounds=args.rounds, opponents=tuple(args.opponents),
                    scenario=args.scenario)
    for key, value in report.items():
        print(f"  {key:32s} {value}")
    write_json(package / "submission_check.json", report)
    budget = 0.5
    if report["timeouts"]:
        raise SystemExit(f"the packaged agent exceeded the think-time budget {report['timeouts']} times")
    if report["max_decision_seconds"] > budget / 2:
        raise SystemExit(
            f"slowest decision {report['max_decision_seconds']:.3f}s is more than half the "
            f"{budget}s budget; the tournament machine is not this one")
    print(f"\nOK: {report['decisions_timed']} decisions, slowest "
          f"{report['max_decision_seconds'] * 1000:.2f} ms against a {budget * 1000:.0f} ms budget "
          f"({budget / max(report['max_decision_seconds'], 1e-9):.0f}x margin); "
          f"setup {report['setup_seconds']:.2f}s, which the framework does not time")


if __name__ == "__main__":
    main()
