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
``BombeRLeWorld`` with the job's own scenario and seed, driving every agent with
the actions recorded in the framework's own game log.  Nothing is re-inferred,
so the replay is exact rather than approximate; the acceptance test is that
every job's bombs, coins, suicides and score reproduce ``official_stats.json``
exactly, and a job that fails it is reported instead of silently averaged in.

Reading the log rather than ``agent/agent.jsonl`` is what makes games with
opponents tractable.  ``rule_based_agent.setup`` calls ``np.random.seed()`` with
no argument, so those agents cannot be re-run to the same decisions -- but the
decisions they did make are in the log, one line per agent per step, already in
the order the world executed them.  The board itself is deterministic given the
scenario seed, so replaying recorded decisions reproduces the game exactly even
though re-playing the agents would not.

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
import re
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
from experiment_lib import write_json
from framework_log import usable_jobs  # noqa: E402

from agent_code.research_agent.config import shaping_specification  # noqa: E402
from agent_code.research_agent.shaping import PotentialShaping  # noqa: E402
from agent_code.research_agent.state import (  # noqa: E402
    HANDCRAFTED_V1_LAYOUT, HANDCRAFTED_V3_LAYOUT, _bfs_distances, _crate_adjacent_cells,
    escape_search, handcrafted_v1, handcrafted_v3,
)

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
            # state["self"] is (name, score, bombs_left, position), so one shared
            # action table serves every agent without needing a per-agent copy.
            key = (state["round"], state["step"], state["self"][0])
            self.pending = self.actions.get(key, "WAIT")

    def get(self, expect_name: str, block=True, timeout=None):
        return None

    def get_with_time(self, expect_name: str, block=True, timeout=None):
        return self.pending, 0.0


class ReplayWorld(BombeRLeWorld):
    """The official world with every agent's decisions supplied from a log."""

    def __init__(self, args: WorldArgs, actions: dict[tuple[int, int, str], str],
                 roster: list[tuple[str, bool]]):
        self._scripted_actions = actions
        super().__init__(args, roster)

    def add_agent(self, agent_dir, name, train=False):
        from agents import Agent

        backend = ScriptedBackend(self._scripted_actions)
        color = self.colors.pop()
        self.agents.append(Agent(name, agent_dir, name, train, backend, color, color))


ROUND_START = re.compile(r"STARTING ROUND #(?P<round>\d+)")
STEP_START = re.compile(r"STARTING STEP (?P<step>\d+)")
CHOSE_ACTION = re.compile(r"Agent <(?P<name>[^>]+)> chose action (?P<action>[A-Z_]+) in ")


def read_actions(job_dir: Path) -> dict[tuple[int, int, str], str]:
    """Return every agent's recorded action, keyed by (round, step, agent name).

    The game log is the only place the opponents' decisions survive, and it is
    also the only record of the order the world applied them, which matters
    because moving first decides who gets a contested tile.
    """
    path = job_dir / "framework_game.log.gz"
    opener = gzip.open
    if not path.exists():
        path = job_dir / "framework_game.log"
        opener = open
    actions: dict[tuple[int, int, str], str] = {}
    round_number = step_number = 0
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = ROUND_START.search(line)
            if match:
                round_number = int(match.group("round"))
                step_number = 0
                continue
            match = STEP_START.search(line)
            if match:
                step_number = int(match.group("step"))
                continue
            match = CHOSE_ACTION.search(line)
            if match:
                actions[(round_number, step_number, match.group("name"))] = match.group("action")
    return actions


def survivable(state: dict, horizon: int) -> tuple[bool, int | None]:
    """Was there a plan that walks out of every current blast in time?

    Delegates to the encoder's own ``escape_search`` so that the number this
    diagnostic judges a run by is the very number handcrafted_v2 shows the
    agent.  If the two ever disagreed, the diagnosis would be measuring a
    world the agent does not live in.
    """
    return escape_search(state, horizon=horizon)


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


def bearing_audit(state: dict) -> dict | None:
    """Does the compass bearing to the nearest target name a useful first step?

    ``_target_features`` encodes ``sign(tx - x), sign(ty - y)``: where the
    target lies as the crow flies.  In a maze the first step of the actual route
    often points elsewhere, and nothing else in handcrafted_v1 carries the
    route.  This counts how often the bearing and the route disagree.
    """
    field = np.asarray(state["field"])
    origin = tuple(state["self"][3])
    distances = _bfs_distances(state)
    targets = {tuple(coin) for coin in state["coins"]}
    kind = "coin"
    if not targets:
        targets = _crate_adjacent_cells(state)
        kind = "crate"
    reachable = [t for t in targets if t in distances]
    if not reachable:
        return None
    goal = min(reachable, key=distances.__getitem__)
    if distances[goal] == 0:
        return None

    blocked = {tuple(position) for position, _ in state["bombs"]}
    blocked.update(tuple(other[3]) for other in state["others"])
    # Which legal moves actually shorten the route, by BFS from the goal.
    from collections import deque as _deque
    back = {goal: 0}
    queue = _deque([goal])
    while queue:
        cell = queue.popleft()
        for dx, dy in MOVES.values():
            nxt = (cell[0] + dx, cell[1] + dy)
            x, y = nxt
            if not (0 <= x < field.shape[0] and 0 <= y < field.shape[1]):
                continue
            if nxt in back or field[x, y] != 0 or nxt in blocked:
                continue
            back[nxt] = back[cell] + 1
            queue.append(nxt)
    if origin not in back:
        return None
    routed = set()
    for name, (dx, dy) in MOVES.items():
        cell = (origin[0] + dx, origin[1] + dy)
        x, y = cell
        if not (0 <= x < field.shape[0] and 0 <= y < field.shape[1]):
            continue
        if field[x, y] != 0 or cell in blocked:
            continue
        if back.get(cell, 10 ** 6) < back[origin]:
            routed.add(name)

    # What the bearing suggests, read the way a linear head would read it.
    sx, sy = np.sign(goal[0] - origin[0]), np.sign(goal[1] - origin[1])
    suggested = set()
    if sx > 0:
        suggested.add("RIGHT")
    if sx < 0:
        suggested.add("LEFT")
    if sy > 0:
        suggested.add("DOWN")
    if sy < 0:
        suggested.add("UP")
    return {
        "kind": kind,
        "bearing_names_a_route_step": bool(suggested & routed),
        "bearing_is_entirely_wrong": bool(suggested and not (suggested & routed)),
        "route_step_count": len(routed),
    }


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

    # What the agent's own state encoding says about each neighbour.  If the
    # cell that leads out and the cell that dead-ends carry the same numbers,
    # no learning rule can prefer one, and the failure is in the features.
    features = handcrafted_v1(state)
    block = features[HANDCRAFTED_V1_LAYOUT["danger_current_and_neighbors"]]
    record["danger_by_direction"] = {
        name: [round(float(block[2 + 2 * i]), 6), round(float(block[3 + 2 * i]), 6)]
        for i, name in enumerate(MOVES)
    }
    record["bomb_escape_feature"] = [
        round(float(v), 6) for v in features[HANDCRAFTED_V1_LAYOUT["bomb_escape"]]
    ]
    # The v1 block above is the historical comparison (docs/01 section 7.10).  An
    # arm running handcrafted_v3 does not see it, so the question that decides
    # whether more features are needed is whether *v3's* escape entries separate
    # the fatal turn from a saving one.
    escape_block = handcrafted_v3(state)[HANDCRAFTED_V3_LAYOUT["escape_by_direction"]]
    record["escape_by_direction"] = {
        name: [round(float(escape_block[2 * i]), 6), round(float(escape_block[1 + 2 * i]), 6)]
        for i, name in enumerate(MOVES)
    }
    return record


def replay_job(job_dir: Path, shaping: PotentialShaping | None) -> dict:
    """Replay one evaluation job and collect every own-bomb episode."""
    snapshot = json.loads((job_dir / "job.snapshot.json").read_text(encoding="utf-8"))
    actions = read_actions(job_dir)
    # The roster has to match the recorded game exactly: setup_agents derives the
    # numbered names from this list, and those names key the action table.
    command = json.loads((job_dir / "command.json").read_text(encoding="utf-8"))["command"]
    names = command[command.index("--agents") + 1:]
    for flag in ("--scenario", "--seed", "--n-rounds", "--no-gui", "--save-stats", "--match-name"):
        if flag in names:
            names = names[:names.index(flag)]
    roster = [(name, False) for name in names]
    # The world insists on a log directory; a finished run is read-only as far
    # as this script is concerned, so the framework's chatter goes to a temp dir.
    log_dir = Path(tempfile.mkdtemp(prefix="bomb_escape_"))
    args = WorldArgs(
        no_gui=True, fps=15, turn_based=False, update_interval=0.1, save_replay=False,
        replay=None, make_video=False, continue_without_training=True, log_dir=str(log_dir),
        save_stats=False, match_name=None, seed=snapshot["seed"], silence_errors=False,
        scenario=snapshot["scenario"],
    )
    world = ReplayWorld(args, actions, roster)
    agent = next(a for a in world.agents if a.code_name == "research_agent")

    bombs: list[dict] = []
    deaths: list[dict] = []
    bearings: list[dict] = []
    observed = {"coins": 0, "kills": 0, "suicides": 0, "score": 0, "bombs": 0}
    for _ in range(int(snapshot["budget"]["rounds"])):
        world.new_round()
        # do_step sets this; we read a state before the first step of a round.
        world.user_input = None
        pending: dict | None = None
        died_this_round = False
        while world.running:
            round_number, step_number = world.round, world.step + 1
            chosen = actions.get((round_number, step_number, agent.name), "WAIT")
            before = world.get_state_for_agent(agent)
            bombs_left_before = agent.bombs_left
            # Ticks after a bomb, while it is still ticking, are the escape the
            # agent has to execute; record what it saw and what it chose.
            tick_record = None
            if (pending is not None and before is not None and not died_this_round
                    and step_number - pending["step"] <= s.BOMB_TIMER):
                tick_record = escape_step_record(before, chosen, shaping)
                pending["escape_window"].append(tick_record)
            world.do_step()
            if tick_record is not None:
                # A move the world refused leaves the agent where it was and
                # raises INVALID_ACTION.  During an escape that is the difference
                # between "walked the wrong way" and "was not allowed to walk",
                # and only the second is caused by somebody else standing there.
                blocked = chosen in MOVES and "INVALID_ACTION" in agent.events
                tick_record["move_was_blocked"] = bool(blocked)
                tick_record["blocked_by"] = None
                if blocked:
                    dx, dy = MOVES[chosen]
                    target = (before["self"][3][0] + dx, before["self"][3][1] + dy)
                    field = np.asarray(before["field"])
                    x, y = target
                    in_bounds = 0 <= x < field.shape[0] and 0 <= y < field.shape[1]
                    after = world.get_state_for_agent(agent)
                    # Agents act one after another inside a step, so a cell that
                    # was empty when the state was observed can be taken by an
                    # opponent that moved first.  Checking only the observed
                    # state would miss exactly the case worth finding.
                    was_occupied = any(tuple(o[3]) == target for o in before["others"])
                    now_occupied = bool(after) and any(tuple(o[3]) == target for o in after["others"])
                    if not in_bounds or field[x, y] != 0:
                        tick_record["blocked_by"] = "wall or crate"
                    elif was_occupied:
                        tick_record["blocked_by"] = "opponent standing there"
                    elif now_occupied:
                        tick_record["blocked_by"] = "opponent moved in first"
                    elif any(tuple(pos) == target for pos, _ in before["bombs"]):
                        tick_record["blocked_by"] = "bomb"
                    else:
                        tick_record["blocked_by"] = "unexplained"
            placed = (not died_this_round and chosen == "BOMB"
                      and bombs_left_before and not agent.bombs_left)
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
            if before is not None:
                audit = bearing_audit(before)
                if audit is not None:
                    bearings.append(audit)
            if agent.dead and not died_this_round:
                died_this_round = True
                # The world has just decided the cause and put it in the agent's
                # events.  With opponents on the board, "the agent died some time
                # after it bombed" is not evidence its own bomb did it: most
                # deaths here are somebody else's blast, and attributing those to
                # the last own bomb produced death delays of 116 and 210 ticks.
                self_inflicted = "KILLED_SELF" in agent.events
                deaths.append({"round": round_number, "step": step_number,
                               "self_inflicted": bool(self_inflicted),
                               "last_action": chosen})
                within_blast_window = (
                    pending is not None
                    and step_number - pending["step"] <= s.BOMB_TIMER + s.EXPLOSION_TIMER
                )
                if pending is not None and self_inflicted and within_blast_window:
                    pending["died"] = True
                    pending["death_step"] = step_number
                    pending["last_action"] = chosen
                    if shaping is not None and before is not None:
                        pending["shaping_at_death"] = shaping_terms(before, shaping)
                # Do not break: with opponents still alive the round continues,
                # and skipping its remaining steps would leave the world's rng
                # out of step so every later round would replay a different board.
        # A round always ends inside do_step: the last agent dying or the step
        # limit both make time_to_stop() true, so there is nothing to close here.
        pending = None

    for key in ("coins", "kills", "suicides", "bombs"):
        observed[key] = int(agent.lifetime_statistics.get(key, 0))
    observed["score"] = int(agent.total_score)
    observed["all_agents"] = {
        a.name: {k: int(a.lifetime_statistics.get(k, 0)) for k in ("coins", "kills", "suicides", "bombs")}
        | {"score": int(a.total_score)}
        for a in world.agents
    }
    official = json.loads((job_dir / "official_stats.json").read_text(encoding="utf-8"))
    shutil.rmtree(log_dir, ignore_errors=True)
    return {"job_id": snapshot["job_id"], "bombs": bombs, "deaths": deaths, "bearings": bearings,
            "observed": observed, "official": official}


def official_totals(official: dict) -> dict[str, dict[str, int]]:
    """Pull every agent's per-run totals out of the framework's stats file.

    Checking all of them, not just ours, is what makes the replay credible with
    opponents in the game: if the scripted opponents had diverged by even one
    action their own coins and suicides would drift, and that would show up here
    long before it quietly distorted our agent's numbers.
    """
    return {
        name: {key: int(entry.get(key, 0)) for key in ("coins", "kills", "suicides", "score", "bombs")}
        for name, entry in official.get("by_agent", {}).items()
    }


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
    parser.add_argument("--include-degraded", action="store_true",
                        help="Replay jobs whose agents were overridden or skipped for slow think "
                             "time.  They cannot reproduce the official stats and the counts "
                             "describe a stalled node, not a policy.")
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
    jobs = usable_jobs(jobs, args.include_degraded)

    results, mismatched = [], []
    for job in jobs:
        result = replay_job(job, shaping)
        expected = official_totals(result["official"])
        replayed = result["observed"]["all_agents"]
        drifted = {
            name: {"replayed": replayed.get(name), "official": totals}
            for name, totals in expected.items()
            if any(replayed.get(name, {}).get(key) != totals[key]
                   for key in ("coins", "suicides", "score"))
        }
        if drifted:
            mismatched.append({"job_id": result["job_id"], "agents": drifted})
        results.append(result)
        print(f"  replayed {result['job_id']}: "
              f"{len(result['bombs'])} bombs, {result['observed']['suicides']} suicides", flush=True)

    if mismatched:
        print("\nREPLAY DID NOT REPRODUCE THE OFFICIAL STATISTICS -- findings below are void:")
        for entry in mismatched:
            print(f"  {entry['job_id']}:")
            for name, sides in entry["agents"].items():
                print(f"    {name}: replayed {sides['replayed']} vs official {sides['official']}")

    every_bomb = [bomb for result in results for bomb in result["bombs"]]
    every_death = [death for result in results for death in result["deaths"]]
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
    self_inflicted = [d for d in every_death if d["self_inflicted"]]
    print(f"deaths                                         {len(every_death)}")
    print(f"  the framework blamed the agent's own bomb    {len(self_inflicted)}"
          f" ({len(self_inflicted) / len(every_death):.1%})" if every_death else "")
    print(f"  tied to a specific own bomb in its window    {len(fatal)}")
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
    indistinguishable = 0
    distinguishable_total = 0
    v3_indistinguishable = 0
    v3_total = 0
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
        culprit = window[lost - 1]
        chosen, safer = culprit["action"], culprit["safer_moves"]
        if chosen in MOVES and safer and culprit.get("danger_by_direction"):
            by_direction = culprit["danger_by_direction"]
            fatal_view = by_direction.get(chosen)
            if fatal_view is not None and any(by_direction.get(name) == fatal_view for name in safer):
                indistinguishable += 1
            distinguishable_total += 1
        if chosen in MOVES and safer and culprit.get("escape_by_direction"):
            escape_view = culprit["escape_by_direction"]
            fatal_escape = escape_view.get(chosen)
            if fatal_escape is not None and any(escape_view.get(name) == fatal_escape for name in safer):
                v3_indistinguishable += 1
            v3_total += 1

    # Deaths where a survivable plan was on offer at every tick: in solo play
    # these were 3.6% of self-inflicted deaths, with opponents they are the
    # largest single bucket, so what actually went wrong in them decides whether
    # more features would help at all.
    never_lost = [bomb for bomb in fatal
                  if bomb["escape_window"] and all(t["escape_exists"] for t in bomb["escape_window"])]
    blocked_runs = [bomb for bomb in never_lost
                    if any(t.get("move_was_blocked") for t in bomb["escape_window"])]
    blocked_by = collections.Counter(
        t.get("blocked_by") for bomb in never_lost for t in bomb["escape_window"]
        if t.get("move_was_blocked"))

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
    if distinguishable_total:
        share = indistinguishable / distinguishable_total
        print(f"  the fatal turn and a saving turn looked the")
        print(f"  same in handcrafted_v1's danger features       "
              f"{indistinguishable}/{distinguishable_total} ({share:.1%})")
    if v3_total:
        share3 = v3_indistinguishable / v3_total
        print(f"  same in handcrafted_v3's escape entries        "
              f"{v3_indistinguishable}/{v3_total} ({share3:.1%})   <-- what the agent actually sees")
    bearings = [entry for result in results for entry in result["bearings"]]
    if bearings:
        useful = sum(entry["bearing_names_a_route_step"] for entry in bearings)
        wrong = sum(entry["bearing_is_entirely_wrong"] for entry in bearings)
        print("-" * 66)
        print("compass bearing to the nearest target vs the actual route")
        print(f"  steps audited                                {len(bearings)}")
        print(f"  bearing names at least one route step        {useful} ({useful / len(bearings):.1%})")
        print(f"  every direction the bearing suggests is wrong{wrong:>6} ({wrong / len(bearings):.1%})")
    if never_lost:
        # If the plan was still there at the last tick, the only remaining
        # question is whether the agent could see it and declined, or could not
        # see it.  Those two call for opposite fixes, so they are counted apart.
        last_ticks = [bomb["escape_window"][-1] for bomb in never_lost]
        with_option = [t for t in last_ticks if t["safer_moves"]]
        took_it = [t for t in with_option if t["action"] in t["safer_moves"]]
        waited = [t for t in with_option if t["action"] == "WAIT"]
        visible = invisible = 0
        for tick in with_option:
            escape_view = tick.get("escape_by_direction") or {}
            chosen_view = escape_view.get(tick["action"])
            saving = [escape_view.get(name) for name in tick["safer_moves"]]
            if chosen_view is not None and saving and all(v == chosen_view for v in saving):
                invisible += 1
            elif saving:
                visible += 1
        print("-" * 66)
        print("deaths where a survivable plan existed at every recorded tick")
        print(f"  such deaths                                  {len(never_lost)}")
        print(f"  at least one move the world refused          {len(blocked_runs)}"
              f" ({len(blocked_runs) / len(never_lost):.1%})")
        print(f"  what refused it                              {dict(blocked_by.most_common())}")
        print(f"  a safer move was on offer at the last tick   {len(with_option)}/{len(never_lost)}")
        print(f"    and the agent took it anyway               {len(took_it)}")
        print(f"    and the agent waited instead               {len(waited)}")
        if visible + invisible:
            print(f"    the saving move was visible in v3         {visible}"
                  f" ({visible / (visible + invisible):.1%})")
            print(f"    the saving move was indistinguishable     {invisible}")
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
            "fatal_turn_indistinguishable_in_features": indistinguishable,
            "fatal_turns_with_a_saving_alternative": distinguishable_total,
            "fatal_turn_indistinguishable_in_v3_escape": v3_indistinguishable,
            "fatal_turns_compared_in_v3_escape": v3_total,
            "deaths_with_a_plan_throughout": len(never_lost),
            "of_those_with_a_refused_move": len(blocked_runs),
            "refused_by": {str(k): v for k, v in blocked_by.items()},
            "escape_ticks": len(ticks), "escape_ticks_with_safer_move": len(decisive),
            "escape_ticks_wait_chosen": len(chose_wait), "escape_ticks_safer_chosen": len(chose_safer),
            "escape_ticks_shaping_prefers_wait": len(wait_wins),
            "shortest_escape_steps": {str(k): v for k, v in lengths.items()},
            "jobs": results,
        }))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
