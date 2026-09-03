"""An opponent that dodges bombs, drops none, and hunts nothing.

Why this exists.  docs/01 section 7.32.6 closed the "weak opponent curriculum"
direction on a specific gap: cornering an opponent is a skill that needs an
opponent which *flees*, and the framework's agents that are weaker than
``rule_based_agent`` do not flee at all -- ``peaceful_agent`` is a pure random
walk and ``random_agent`` blows itself up in 18 steps.  There was no middle
rung, so the conclusion was that a weak-opponent table can only teach "bomb
whatever is standing there".

This is that middle rung, built rather than found.  It keeps ``rule_based``'s
danger model exactly -- the same blast map, the same validity rule (a tile is
rejected when a live explosion covers it or a bomb detonates *this* step, which
is weaker than "no bomb will ever reach it"), and the same directed retreat when
a bomb shares its row or column -- and removes everything else:
it never proposes ``BOMB``, and it never walks towards a coin, a crate or an
agent.  So it is:

* **always available as a target**: it cannot kill itself, having no bombs, and
  it survives an opponent's bomb whenever ``rule_based`` would.  Measured, that
  is *too* good -- 0.075 to 0.25 kills a round against it where three
  ``rule_based`` give 0.200 and ``rule_based`` plus two ``peaceful_agent`` give
  0.625 -- so it is not the kill-dense table it was built to be.  What it is
  instead is the *control* for one: it matches a peaceful table's coin
  competition and bomb density while leaving the kill density an order of
  magnitude lower, which is what isolates kill density as a factor;
* **not a competitor**: it takes only the coins it happens to step on while
  wandering, so it does not dilute the coin signal the way ``coin_collector``
  does;
* **an opponent that flees**, which is the whole point and the one thing
  ``peaceful_agent`` cannot offer.

It is a *training* opponent and a proxy in the evaluation pool.  It is not a
submission candidate: it contains no learning of any kind.
"""

from __future__ import annotations

import os
from random import shuffle

import numpy as np

import settings as s


# How often the agent actually uses its retreat proposals.  1.0 is a perfect
# dodger and, measured, a *harder* target than ``rule_based_agent``: it never
# bombs, so it never walks into its own blast, and it flees ours -- 0.05 kills a
# round against it where rule_based gives 0.95.  That is the opposite of what a
# kill-dense training table needs, so the knob exists to build the rung that is
# actually missing: an opponent that flees *sometimes*, which is killable but
# still has to be predicted rather than merely walked up to.
DEFAULT_ALERTNESS = 1.0


def setup(self):
    np.random.seed()
    self.current_round = 0
    raw = os.environ.get("BOMBERMAN_EVADER_ALERTNESS")
    self.alertness = DEFAULT_ALERTNESS if raw is None else float(raw)
    if not 0.0 <= self.alertness <= 1.0:
        raise ValueError(f"BOMBERMAN_EVADER_ALERTNESS must lie in [0, 1], got {raw!r}")


def _blast_map(game_state: dict) -> np.ndarray:
    """Ticks until each tile becomes lethal; ``rule_based``'s map, unchanged."""
    arena = game_state["field"]
    blast = np.ones(arena.shape) * 5
    for (bomb_x, bomb_y), timer in game_state["bombs"]:
        for (x, y) in ([(bomb_x + offset, bomb_y) for offset in range(-3, 4)]
                       + [(bomb_x, bomb_y + offset) for offset in range(-3, 4)]):
            if 0 < x < blast.shape[0] and 0 < y < blast.shape[1]:
                blast[x, y] = min(blast[x, y], timer)
    return blast


def act(self, game_state: dict) -> str:
    if game_state["round"] != self.current_round:
        self.current_round = game_state["round"]

    arena = game_state["field"]
    _, _, _, (x, y) = game_state["self"]
    bombs = game_state["bombs"]
    bomb_positions = [position for position, _ in bombs]
    others = [other[3] for other in game_state["others"]]
    blast = _blast_map(game_state)

    # Exactly ``rule_based``'s validity test: free floor, no live explosion, no
    # bomb about to reach it, nobody standing there.
    candidates = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles = [tile for tile in candidates
                   if arena[tile] == 0
                   and game_state["explosion_map"][tile] < 1
                   and blast[tile] > 0
                   and tile not in others
                   and tile not in bomb_positions]
    valid_actions = []
    if (x - 1, y) in valid_tiles:
        valid_actions.append("LEFT")
    if (x + 1, y) in valid_tiles:
        valid_actions.append("RIGHT")
    if (x, y - 1) in valid_tiles:
        valid_actions.append("UP")
    if (x, y + 1) in valid_tiles:
        valid_actions.append("DOWN")
    if (x, y) in valid_tiles:
        valid_actions.append("WAIT")

    # Wander by default; the retreat proposals are appended afterwards and the
    # last valid proposal wins, so fleeing always overrides wandering.
    proposals = ["UP", "DOWN", "LEFT", "RIGHT"]
    shuffle(proposals)
    # Below full alertness the retreat proposals are simply not made this step,
    # so the agent keeps wandering while a bomb burns.  The *validity* filter
    # above still stands, which is deliberate: a tile already inside a live
    # blast is never stepped onto, so a distracted evader dies by failing to
    # leave in time rather than by walking into fire, which is how a real agent
    # dies too.
    alert = np.random.random() < self.alertness
    for (bomb_x, bomb_y), _ in (bombs if alert else ()):
        if bomb_x == x and abs(bomb_y - y) < 4:
            proposals.append("UP" if bomb_y > y else "DOWN")
            proposals.extend(("LEFT", "RIGHT"))
        if bomb_y == y and abs(bomb_x - x) < 4:
            proposals.append("LEFT" if bomb_x > x else "RIGHT")
            proposals.extend(("UP", "DOWN"))
    for (bomb_x, bomb_y), _ in (bombs if alert else ()):
        if bomb_x == x and bomb_y == y:
            proposals.extend(proposals[:4])

    while proposals:
        action = proposals.pop()
        if action in valid_actions:
            return action
    # Every direction is blocked or lethal; standing still is all that is left.
    return "WAIT"
