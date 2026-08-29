#!/usr/bin/env python3
"""Count the kill opportunities an arm had, and what it did with them.

docs/01 section 7.26 left the tournament line with one arithmetic gap: an
opponent kill is worth 5 points, the same as five coins, and the agent takes
about 0.05 of them a round.  Counting training events showed why the learner
cannot close that gap on its own -- F2 saw 820 KILLED_OPPONENT events in 25,000
rounds against 156,870 COIN_COLLECTED, so the kill term carries 1.7 percent of
the positive reward mass.  That is a signal-density problem of the kind section
2.1 diagnosed for coins.

Before shaping anything toward opponents, this script asks which half of the
problem is real, because the two answers imply different designs:

  * the agent stands where a bomb would catch an opponent and does not drop one
    -> a decision problem, and shaping the drop has a target;
  * it is almost never in such a position -> an approach problem, and shaping
    the drop would be pointless.

The replay is the one ``diagnose_bomb_escape.py`` uses: every agent, including
the scripted opponents, is driven from the actions recorded in the framework
log, so nothing is re-decided and the opponents' nondeterminism (section 7.15.1)
cannot affect the count.  Acceptance is the same too -- every agent's official
coins, kills, suicides and score must reproduce exactly.

``opponent in the blast`` is measured at the moment of the decision and is an
upper bound on the real opportunity: a bomb dropped now detonates BOMB_TIMER+1
steps later, and the opponent may walk out.  ``opponent adjacent`` is the
condition rule_based_agent actually fires on (callbacks.py line 170), and is
reported next to it so the two are comparable.
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

import settings as s  # noqa: E402
from environment import WorldArgs  # noqa: E402
from experiment_lib import write_json  # noqa: E402
from agent_code.research_agent.config import SHAPING_SPECIFICATIONS  # noqa: E402
from agent_code.research_agent.shaping import PotentialShaping  # noqa: E402
from agent_code.research_agent.state import _blast_coordinates, escape_search  # noqa: E402
from diagnose_bomb_escape import (  # noqa: E402
    ReplayWorld, jsonable, official_totals, read_actions,
)
from framework_log import usable_jobs  # noqa: E402


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def replay_job(job_dir: Path, discount: float) -> dict:
    """Replay one job and count every step where a kill was available."""
    snapshot = json.loads((job_dir / "job.snapshot.json").read_text(encoding="utf-8"))
    actions = read_actions(job_dir)
    command = json.loads((job_dir / "command.json").read_text(encoding="utf-8"))["command"]
    names = command[command.index("--agents") + 1:]
    for flag in ("--scenario", "--seed", "--n-rounds", "--no-gui", "--save-stats", "--match-name"):
        if flag in names:
            names = names[:names.index(flag)]
    roster = [(name, False) for name in names]
    log_dir = Path(tempfile.mkdtemp(prefix="kill_opportunity_"))
    args = WorldArgs(
        no_gui=True, fps=15, turn_based=False, update_interval=0.1, save_replay=False,
        replay=None, make_video=False, continue_without_training=True, log_dir=str(log_dir),
        save_stats=False, match_name=None, seed=snapshot["seed"], silence_errors=False,
        scenario=snapshot["scenario"],
    )
    world = ReplayWorld(args, actions, roster)
    agent = next(a for a in world.agents if a.code_name == "research_agent")

    counts = collections.Counter()
    nearest = collections.Counter()
    # The potential the shaping would see, computed by the shaping itself so the
    # diagnostic and the agent can never drift apart (the discipline section
    # 7.11 applied to escape_search).  A07 minus A06 is exactly the new term.
    v1 = PotentialShaping(SHAPING_SPECIFICATIONS["A06"], discount)
    v2 = PotentialShaping(SHAPING_SPECIFICATIONS["A07"], discount)
    delta = lambda s: v2.potential(s) - v1.potential(s)
    # Drops that created the state A07 rewards, still open for their follow-up.
    open_drops: list[dict] = []
    for _ in range(int(snapshot["budget"]["rounds"])):
        world.new_round()
        world.user_input = None
        # Cleared per round, not once per run.  A bomb dropped in the last few
        # steps of a round has no tick 3 or 4 -- the round ended -- and letting
        # it age into the next round credits its follow-up ticks to a board it
        # never existed on.  That inflates exactly the activation rate this
        # diagnostic is here to measure.  The immediate counts (a drop could
        # hit, could still escape, was or was not taken) are single-step and
        # were never affected.
        open_drops.clear()
        while world.running:
            round_number, step_number = world.round, world.step + 1
            state = world.get_state_for_agent(agent)
            chosen = actions.get((round_number, step_number, agent.name), "WAIT")
            if state is not None:
                counts["steps_alive"] += 1
                # Follow every qualifying drop for the four ticks a bomb lives:
                # an n-step window ending on a zero delta contributes nothing,
                # whatever n is, so this is A07's real activation rate.
                current = delta(state)
                for drop in open_drops:
                    drop["age"] += 1
                    if drop["age"] <= 4:
                        counts["followed_tick_%d" % drop["age"]] += 1
                        if current > 0:
                            counts["still_threatening_tick_%d" % drop["age"]] += 1
                            own = set(_blast_coordinates(drop["origin"], np.asarray(state["field"])))
                            if any(tuple(o[3]) in own for o in state["others"]):
                                counts["still_threatened_by_that_bomb_tick_%d" % drop["age"]] += 1
                open_drops = [d for d in open_drops if d["age"] < 4]
                others = [tuple(o[3]) for o in state["others"]]
                own = tuple(state["self"][3])
                has_bomb = bool(state["self"][2])
                if others:
                    nearest[min(manhattan(own, o) for o in others)] += 1
                if has_bomb:
                    counts["steps_with_a_bomb"] += 1
                    blast = set(_blast_coordinates(own, np.asarray(state["field"])))
                    caught = [o for o in others if o in blast]
                    adjacent = [o for o in others if manhattan(own, o) <= 1]
                    if caught:
                        counts["opportunity_opponent_in_blast"] += 1
                        counts["opportunity_took_it"] += chosen == "BOMB"
                        # Would dropping have left an escape?  A drop the agent
                        # could not survive is not an opportunity it declined.
                        hypothetical = dict(state)
                        hypothetical["bombs"] = list(state["bombs"]) + [(own, s.BOMB_TIMER)]
                        reachable, _ = escape_search(hypothetical, horizon=s.BOMB_TIMER + 1)
                        counts["opportunity_survivable"] += bool(reachable)
                        if reachable:
                            counts["opportunity_survivable_took_it"] += chosen == "BOMB"
                    if adjacent:
                        counts["rule_based_trigger_adjacent"] += 1
                        counts["rule_based_trigger_took_it"] += chosen == "BOMB"
                    if caught and chosen == "BOMB":
                        open_drops.append({"origin": own, "age": 0})
            world.do_step()
        # A round always ends inside do_step: the last agent dying and the step
        # limit both make time_to_stop() true, so there is nothing to close here.

    official = json.loads((job_dir / "official_stats.json").read_text(encoding="utf-8"))
    observed = {
        a.name: {k: int(a.lifetime_statistics.get(k, 0)) for k in ("coins", "kills", "suicides", "bombs")}
                | {"score": int(a.total_score)}
        for a in world.agents
    }
    shutil.rmtree(log_dir, ignore_errors=True)
    return {"job": job_dir.name, "counts": dict(counts),
            "nearest_opponent_distance": dict(nearest),
            "official": official, "observed": observed}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--job-prefix", default="eval_classic_versus_opponents")
    parser.add_argument("--discount", type=float, default=0.95)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--include-degraded", action="store_true",
                        help="Replay jobs whose agents were overridden or skipped for slow "
                             "think time; they cannot reproduce the official stats.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = sorted(p for p in (args.run_dir / "jobs").iterdir() if p.name.startswith(args.job_prefix))
    if not jobs:
        raise SystemExit(f"No jobs matching {args.job_prefix!r} in {args.run_dir}")
    jobs = usable_jobs(jobs, args.include_degraded)

    total = collections.Counter()
    nearest = collections.Counter()
    reproduced = 0
    for job in jobs:
        result = replay_job(job, args.discount)
        total.update(result["counts"])
        nearest.update(result["nearest_opponent_distance"])
        expected = official_totals(result["official"])
        drifted = [name for name, totals in expected.items()
                   if any(result["observed"].get(name, {}).get(k) != totals[k]
                          for k in ("coins", "suicides", "score"))]
        if not drifted:
            reproduced += 1
        else:
            print(f"  MISMATCH in {job.name}: replay diverged from official stats")

    def line(label, value, base):
        share = f"{100 * value / base:5.1f}%" if base else "    --"
        print(f"  {label:<52} {value:8d} {share}")

    print("=" * 70)
    print(f"  {'jobs replayed':<52} {len(jobs):8d}")
    print(f"  {'replay reproduced official statistics':<52} {reproduced:8d}/{len(jobs)}")
    alive, armed = total["steps_alive"], total["steps_with_a_bomb"]
    line("steps alive", alive, alive)
    line("of those, holding a bomb", armed, alive)
    print("-" * 70)
    print("  a bomb dropped now would have caught an opponent")
    caught = total["opportunity_opponent_in_blast"]
    line("such steps", caught, armed)
    line("of those, an escape existed after dropping", total["opportunity_survivable"], caught)
    line("of those, the agent dropped", total["opportunity_took_it"], caught)
    line("survivable AND the agent dropped", total["opportunity_survivable_took_it"],
         total["opportunity_survivable"])
    print("-" * 70)
    print("  rule_based_agent's own trigger: an opponent within one step")
    trigger = total["rule_based_trigger_adjacent"]
    line("such steps", trigger, armed)
    line("of those, the agent dropped", total["rule_based_trigger_took_it"], trigger)
    print("-" * 70)
    print("  after a drop that covered an opponent, is the potential still raised?")
    print("  (an n-step window ending on a zero delta contributes nothing, whatever n)")
    for tick in (1, 2, 3, 4):
        base = total["followed_tick_%d" % tick]
        line(f"tick {tick}: any opponent still in some blast", total[f"still_threatening_tick_{tick}"], base)
        line(f"tick {tick}:   ... in THAT bomb's blast", total[f"still_threatened_by_that_bomb_tick_{tick}"], base)
    print("-" * 70)
    print("  distance to the nearest opponent, over every step alive")
    shown = sum(nearest.values())
    for distance in sorted(nearest)[:8]:
        line(f"manhattan {distance}", nearest[distance], shown)

    if args.out:
        write_json(args.out, jsonable({"run_dir": str(args.run_dir), "job_prefix": args.job_prefix,
                                       "jobs": len(jobs), "reproduced": reproduced,
                                       "totals": dict(total),
                                       "nearest_opponent_distance": dict(nearest)}))
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
