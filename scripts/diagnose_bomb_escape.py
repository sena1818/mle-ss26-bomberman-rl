#!/usr/bin/env python3
"""Ask why a finished run's agent dies to its own bombs.

``approximate_safe_bomb_rate`` is computed per bomb and answers "did this bomb
kill me".  It cannot separate the two failures that need different fixes:

* the agent bombs itself into a position with no way out, or
* the agent bombs a position it could have escaped from and then fails to walk
  out in the ticks it had.

The first is a decision problem in the bombing action; the second is a
short-horizon execution problem.  Only the second is helped by anything that
propagates reward backwards, and only the first is helped by a safer bombing
rule, so a run's remaining suicides have to be split before choosing.

Method.  Each evaluation job is replayed through the unmodified official
``BombeRLeWorld`` with the job's own scenario and seed, driving the agent with
the actions recorded in ``agent/agent.jsonl``.  Nothing is re-inferred, so the
replay is exact rather than approximate; the acceptance test is that every
job's bombs, coins, suicides and score reproduce ``official_stats.json``
exactly, and a job that fails it is reported instead of silently averaged in.

For every bomb the agent places, a time-expanded breadth-first search decides
whether a survivable plan existed at that moment, using the framework's own
blast geometry.  A cell counts as reached safely only if the agent is not
standing in a blast at the tick it arrives, and the goal is a cell outside the
blast of every bomb then on the board -- the strict reading, which also treats
the tile the bomb was just dropped on as no longer enterable.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import shutil
import sys
import tempfile
from collections import deque
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import settings as s  # noqa: E402
from agents import AgentBackend  # noqa: E402
from environment import BombeRLeWorld, WorldArgs  # noqa: E402
from experiment_lib import write_json  # noqa: E402

from agent_code.research_agent.config import shaping_specification  # noqa: E402
from agent_code.research_agent.shaping import PotentialShaping  # noqa: E402
from agent_code.research_agent.state import _blast_coordinates  # noqa: E402

MOVES = {"UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1), "LEFT": (-1, 0)}


class ScriptedBackend(AgentBackend):
    """Replay recorded actions instead of running the agent's callbacks.

    The world calls the backend for ``act`` and for the two training events.
    Only ``act`` has to answer, and it answers from the log, so no model is
    loaded and no inference happens.
    """

    def __init__(self, actions: dict[tuple[int, int], str]):
        self.actions = actions
        self.world = None
        self.pending = None

    def start(self):
        return None

    def send_event(self, event_name, *event_args):
        if event_name == "act":
            state = event_args[0]
            self.pending = self.actions.get((state["round"], state["step"]), "WAIT")

    def get(self, expect_name: str, block=True, timeout=None):
        return None

    def get_with_time(self, expect_name: str, block=True, timeout=None):
        return self.pending, 0.0


class ReplayWorld(BombeRLeWorld):
    """The official world with the agent's decisions supplied from a log."""

    def __init__(self, args: WorldArgs, actions: dict[tuple[int, int], str]):
        self._scripted_actions = actions
        super().__init__(args, [("research_agent", False)])

    def add_agent(self, agent_dir, name, train=False):
        from agents import Agent

        backend = ScriptedBackend(self._scripted_actions)
        color = self.colors.pop()
        self.agents.append(Agent(name, agent_dir, name, train, backend, color, color))


def read_actions(job_dir: Path) -> dict[tuple[int, int], str]:
    """Return the recorded action for every (round, step) the agent acted on."""
    path = job_dir / "agent" / "agent.jsonl.gz"
    opener = gzip.open
    if not path.exists():
        path = job_dir / "agent" / "agent.jsonl"
        opener = open
    actions: dict[tuple[int, int], str] = {}
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("kind") == "action":
                actions[(record["round"], record["step"])] = record["action"]
    return actions


def blast_schedule(field: np.ndarray, bombs: list) -> dict[tuple[int, int], set[int]]:
    """Return, per cell, the set of future ticks at which it is lethal.

    Tick numbering is relative to the decision now being taken: a bomb whose
    observed timer is ``t`` explodes ``t + 1`` ticks from now, matching the
    convention in ``state.future_danger_times``, and the blast lingers for
    ``EXPLOSION_TIMER`` ticks after that.
    """
    lethal: dict[tuple[int, int], set[int]] = collections.defaultdict(set)
    for position, timer in bombs:
        detonation = max(int(timer) + 1, 1)
        for cell in _blast_coordinates(tuple(position), field):
            for offset in range(s.EXPLOSION_TIMER):
                lethal[cell].add(detonation + offset)
    return lethal


def survivable(state: dict, horizon: int) -> tuple[bool, int | None]:
    """Was there a plan that walks out of every current blast in time?

    Returns whether such a plan exists and the length of the shortest one.  The
    search is over ``(cell, tick)`` so that walking through a cell that is about
    to explode is rejected, which a plain distance BFS cannot express.
    """
    field = np.asarray(state["field"])
    bombs = list(state["bombs"])
    if not bombs:
        return True, 0
    start = tuple(state["self"][3])
    blocked = {tuple(position) for position, _ in bombs}
    blocked.update(tuple(other[3]) for other in state["others"])
    lethal = blast_schedule(field, bombs)
    in_any_blast = set()
    for position, _ in bombs:
        in_any_blast.update(_blast_coordinates(tuple(position), field))

    def free(cell: tuple[int, int]) -> bool:
        x, y = cell
        if not (0 <= x < field.shape[0] and 0 <= y < field.shape[1]):
            return False
        return field[x, y] == 0 and cell not in blocked

    if start not in in_any_blast:
        return True, 0
    seen = {(start, 0)}
    queue = deque([(start, 0)])
    while queue:
        cell, tick = queue.popleft()
        if tick >= horizon:
            continue
        candidates = [cell] + [(cell[0] + dx, cell[1] + dy) for dx, dy in MOVES.values()]
        for nxt in candidates:
            if nxt != cell and not free(nxt):
                continue
            step = tick + 1
            if step in lethal.get(nxt, ()):  # standing there when it goes off
                continue
            if nxt not in in_any_blast:
                return True, step
            if (nxt, step) in seen:
                continue
            seen.add((nxt, step))
            queue.append((nxt, step))
    return False, None


def shaping_terms(state: dict, shaping: PotentialShaping) -> dict[str, float]:
    """Return ``gamma * phi(s') - phi(s)`` for WAIT and for each legal move.

    The world's own tick is held fixed and only the agent's cell is varied, so
    the comparison isolates the positional part of the potential -- the part
    under investigation.  Bomb countdowns advance identically whatever the agent
    does, so they cancel from the comparison between actions.
    """
    field = np.asarray(state["field"])
    blocked = {tuple(position) for position, _ in state["bombs"]}
    blocked.update(tuple(other[3]) for other in state["others"])
    origin = tuple(state["self"][3])
    here = shaping.potential(state)
    terms = {"WAIT": shaping.discount * here - here}
    for name, (dx, dy) in MOVES.items():
        cell = (origin[0] + dx, origin[1] + dy)
        x, y = cell
        if not (0 <= x < field.shape[0] and 0 <= y < field.shape[1]):
            continue
        if field[x, y] != 0 or cell in blocked:
            continue
        moved = dict(state)
        name_, score, can_bomb, _ = state["self"]
        moved["self"] = (name_, score, can_bomb, cell)
        terms[name] = shaping.discount * shaping.potential(moved) - here
    return terms


def escape_step_record(state: dict, chosen: str, shaping: PotentialShaping | None) -> dict:
    """Describe one tick of an escape: where safety is, and what shaping pays.

    ``safer_moves`` are the legal moves that strictly reduce the number of ticks
    still needed to stand outside every blast.  Comparing the shaping term of
    those against WAIT is the whole question: a potential that cannot tell the
    agent to hurry will not rank them above standing still.
    """
    here_exists, here_steps = survivable(state, s.BOMB_TIMER + 1)
    record = {
        "position": [int(v) for v in state["self"][3]], "action": chosen,
        "escape_exists": bool(here_exists), "steps_to_safety": here_steps,
        "shaping": None, "safer_moves": [], "wait_beats_every_safer_move": None,
    }
    if here_steps in (None, 0):
        return record

    field = np.asarray(state["field"])
    blocked = {tuple(position) for position, _ in state["bombs"]}
    blocked.update(tuple(other[3]) for other in state["others"])
    origin = tuple(state["self"][3])
    for name, (dx, dy) in MOVES.items():
        cell = (origin[0] + dx, origin[1] + dy)
        x, y = cell
        if not (0 <= x < field.shape[0] and 0 <= y < field.shape[1]):
            continue
        if field[x, y] != 0 or cell in blocked:
            continue
        moved = dict(state)
        name_, score, can_bomb, _ = state["self"]
        moved["self"] = (name_, score, can_bomb, cell)
        _, moved_steps = survivable(moved, s.BOMB_TIMER + 1)
        if moved_steps is not None and moved_steps < here_steps:
            record["safer_moves"].append(name)

    if shaping is not None:
        terms = shaping_terms(state, shaping)
        record["shaping"] = terms
        payoffs = [terms[name] for name in record["safer_moves"] if name in terms]
        if payoffs:
            record["wait_beats_every_safer_move"] = bool(terms["WAIT"] >= max(payoffs))
    return record


def replay_job(job_dir: Path, shaping: PotentialShaping | None) -> dict:
    """Replay one evaluation job and collect every own-bomb episode."""
    snapshot = json.loads((job_dir / "job.snapshot.json").read_text(encoding="utf-8"))
    actions = read_actions(job_dir)
    # The world insists on a log directory; a finished run is read-only as far
    # as this script is concerned, so the framework's chatter goes to a temp dir.
    log_dir = Path(tempfile.mkdtemp(prefix="bomb_escape_"))
    args = WorldArgs(
        no_gui=True, fps=15, turn_based=False, update_interval=0.1, save_replay=False,
        replay=None, make_video=False, continue_without_training=True, log_dir=str(log_dir),
        save_stats=False, match_name=None, seed=snapshot["seed"], silence_errors=False,
        scenario=snapshot["scenario"],
    )
    world = ReplayWorld(args, actions)
    agent = world.agents[0]

    bombs: list[dict] = []
    observed = {"coins": 0, "kills": 0, "suicides": 0, "score": 0, "bombs": 0}
    for _ in range(int(snapshot["budget"]["rounds"])):
        world.new_round()
        # do_step sets this; we read a state before the first step of a round.
        world.user_input = None
        pending: dict | None = None
        while world.running:
            round_number, step_number = world.round, world.step + 1
            chosen = actions.get((round_number, step_number), "WAIT")
            before = world.get_state_for_agent(agent)
            bombs_left_before = agent.bombs_left
            # Ticks after a bomb, while it is still ticking, are the escape the
            # agent has to execute; record what it saw and what it chose.
            if (pending is not None and before is not None
                    and step_number - pending["step"] <= s.BOMB_TIMER):
                pending["escape_window"].append(escape_step_record(before, chosen, shaping))
            world.do_step()
            placed = chosen == "BOMB" and bombs_left_before and not agent.bombs_left
            if placed:
                after = world.get_state_for_agent(agent)
                episode = {
                    "round": round_number, "step": step_number,
                    "position": [int(v) for v in before["self"][3]],
                    "escape_existed": None, "min_escape_steps": None,
                    "died": False, "death_step": None, "last_action": None,
                    "shaping_at_death": None, "escape_window": [],
                }
                if after is not None:
                    exists, length = survivable(after, s.BOMB_TIMER + 1)
                    episode["escape_existed"] = bool(exists)
                    episode["min_escape_steps"] = length
                bombs.append(episode)
                pending = episode
            if agent.dead:
                if pending is not None:
                    pending["died"] = True
                    pending["death_step"] = step_number
                    pending["last_action"] = chosen
                    if shaping is not None and before is not None:
                        pending["shaping_at_death"] = shaping_terms(before, shaping)
                break
        # A round always ends inside do_step: the last agent dying or the step
        # limit both make time_to_stop() true, so there is nothing to close here.
        pending = None

    for key in ("coins", "kills", "suicides", "bombs"):
        observed[key] = int(agent.lifetime_statistics.get(key, 0))
    observed["score"] = int(agent.total_score)
    official = json.loads((job_dir / "official_stats.json").read_text(encoding="utf-8"))
    shutil.rmtree(log_dir, ignore_errors=True)
    return {"job_id": snapshot["job_id"], "bombs": bombs, "observed": observed, "official": official}


def official_totals(official: dict) -> dict[str, int]:
    """Pull the agent's per-run totals out of the framework's stats file."""
    by_agent = official.get("by_agent", {})
    entry = next(iter(by_agent.values()), {})
    return {key: int(entry.get(key, 0)) for key in ("coins", "kills", "suicides", "score", "bombs")}


def jsonable(value):
    """Coerce numpy scalars so the findings survive json.dumps."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--job-prefix", default="eval_round00500",
                        help="Which evaluation jobs to replay (default: the round-500 primary sweep).")
    parser.add_argument("--reward-version", default="A06",
                        help="Reward version whose shaping is evaluated at the fatal step; '' to skip.")
    parser.add_argument("--discount", type=float, default=0.95)
    parser.add_argument("--out", type=Path, help="Optional machine-readable copy of the findings.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shaping = None
    if args.reward_version:
        specification = shaping_specification(args.reward_version)
        if specification is not None:
            shaping = PotentialShaping(specification, args.discount)

    jobs = sorted(p for p in (args.run_dir / "jobs").iterdir() if p.name.startswith(args.job_prefix))
    if not jobs:
        raise SystemExit(f"No jobs matching {args.job_prefix!r} in {args.run_dir}")

    results, mismatched = [], []
    for job in jobs:
        result = replay_job(job, shaping)
        expected = official_totals(result["official"])
        if any(result["observed"][key] != expected[key] for key in ("coins", "suicides", "score")):
            mismatched.append({"job_id": result["job_id"], "replayed": result["observed"], "official": expected})
        results.append(result)
        print(f"  replayed {result['job_id']}: "
              f"{len(result['bombs'])} bombs, {result['observed']['suicides']} suicides", flush=True)

    if mismatched:
        print("\nREPLAY DID NOT REPRODUCE THE OFFICIAL STATISTICS -- findings below are void:")
        for entry in mismatched:
            print(f"  {entry['job_id']}: replayed {entry['replayed']} vs official {entry['official']}")

    every_bomb = [bomb for result in results for bomb in result["bombs"]]
    fatal = [bomb for bomb in every_bomb if bomb["died"]]
    with_escape = [bomb for bomb in every_bomb if bomb["escape_existed"]]
    fatal_with_escape = [bomb for bomb in fatal if bomb["escape_existed"]]
    last_actions = collections.Counter(bomb["last_action"] for bomb in fatal)
    delays = collections.Counter(bomb["death_step"] - bomb["step"] for bomb in fatal)
    lengths = collections.Counter(bomb["min_escape_steps"] for bomb in fatal_with_escape)

    print("\n" + "=" * 66)
    print(f"jobs replayed                                  {len(results)}")
    print(f"replay reproduced official statistics          {len(results) - len(mismatched)}/{len(results)}")
    print(f"bombs placed                                   {len(every_bomb)}")
    print(f"bombs with a survivable plan at placement      {len(with_escape)}/{len(every_bomb)}")
    print(f"deaths attributable to the agent's own bomb    {len(fatal)}")
    print(f"  of those, an escape existed when bombing     {len(fatal_with_escape)}/{len(fatal)}")
    print(f"  last action before dying                     {dict(last_actions.most_common())}")
    print(f"  ticks from bomb to death                     {dict(sorted(delays.items()))}")
    print(f"  shortest escape available (steps)            {dict(sorted(lengths.items(), key=lambda kv: (kv[0] is None, kv[0])))}")

    # Where a fatal escape actually broke: the first tick at which no survivable
    # plan is left any more.  The action taken on the tick *before* that is the
    # one that threw the escape away, and it is usually not the last action.
    point_of_no_return = collections.Counter()
    doomed_before_last = 0
    trapped_at_death = 0
    for bomb in fatal:
        window = bomb["escape_window"]
        lost = next((i for i, tick in enumerate(window) if not tick["escape_exists"]), None)
        if lost is None:
            point_of_no_return["never lost a plan"] += 1
            continue
        if lost == 0:
            point_of_no_return["already doomed when the bomb landed"] += 1
        else:
            culprit = window[lost - 1]
            point_of_no_return[f"tick {lost} after the bomb, by choosing {culprit['action']}"] += 1
            if culprit["safer_moves"]:
                doomed_before_last += 1
        if window and not [name for name in MOVES if name in (window[-1].get("shaping") or {})]:
            trapped_at_death += 1

    ticks = [tick for bomb in every_bomb for tick in bomb["escape_window"]]
    decisive = [tick for tick in ticks if tick["wait_beats_every_safer_move"] is not None]
    wait_wins = [tick for tick in decisive if tick["wait_beats_every_safer_move"]]
    chose_wait = [tick for tick in decisive if tick["action"] == "WAIT"]
    chose_safer = [tick for tick in decisive if tick["action"] in tick["safer_moves"]]
    print("-" * 66)
    print("where each fatal escape was actually lost")
    for label, count in point_of_no_return.most_common():
        print(f"  {label:<44s} {count}")
    print(f"  a safer move still existed at that tick        {doomed_before_last}/{len(fatal)}")
    print(f"  no legal move at all on the final tick         {trapped_at_death}/{len(fatal)}")
    print("-" * 66)
    print("escape ticks (a bomb is ticking and the agent is not yet safe)")
    print(f"  ticks recorded                               {len(ticks)}")
    print(f"  ticks with a strictly safer move available   {len(decisive)}")
    print(f"    agent chose a safer move                   {len(chose_safer)}")
    print(f"    agent chose WAIT                           {len(chose_wait)}")
    if shaping is not None and decisive:
        gaps = [tick["shaping"]["WAIT"] - max(tick["shaping"][name] for name in tick["safer_moves"])
                for tick in decisive]
        print(f"  A06 shaping ranked WAIT >= every safer move  {len(wait_wins)}/{len(decisive)}")
        print(f"  mean WAIT-minus-best-safer-move term         {sum(gaps) / len(gaps):+.4f}")
        print(f"  min / max of that gap                        {min(gaps):+.4f} / {max(gaps):+.4f}")

    if args.out:
        write_json(args.out, jsonable({
            "run_dir": str(args.run_dir), "job_prefix": args.job_prefix,
            "reward_version": args.reward_version, "mismatched_jobs": mismatched,
            "bombs": len(every_bomb), "bombs_with_escape": len(with_escape),
            "fatal": len(fatal), "fatal_with_escape": len(fatal_with_escape),
            "last_actions": dict(last_actions), "bomb_to_death_ticks": {str(k): v for k, v in delays.items()},
            "point_of_no_return": dict(point_of_no_return),
            "fatal_doomed_with_safer_move_available": doomed_before_last,
            "fatal_trapped_on_final_tick": trapped_at_death,
            "escape_ticks": len(ticks), "escape_ticks_with_safer_move": len(decisive),
            "escape_ticks_wait_chosen": len(chose_wait), "escape_ticks_safer_chosen": len(chose_safer),
            "escape_ticks_shaping_prefers_wait": len(wait_wins),
            "shortest_escape_steps": {str(k): v for k, v in lengths.items()},
            "jobs": results,
        }))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
