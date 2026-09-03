"""Tests for the dodging, non-bombing training opponent.

The agent exists to be a *killable* opponent that does not compete for coins,
and the measurement that motivated it is in docs/01: at full alertness it turned
out to be harder to kill than ``rule_based_agent``, because not bombing is
itself the largest survival advantage on this board. So what these tests hold is
what the arms depend on: it never bombs, it never steps into a live blast, its
alertness knob really does change how often it retreats, and it always returns a
legal action.
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from agent_code.evader_agent import callbacks as evader


def _state(*, self_pos=(1, 1), bombs=(), others=(), coins=(), explosion_map=None):
    field = np.zeros((17, 17), dtype=int)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    for x in range(2, 16, 2):
        for y in range(2, 16, 2):
            field[x, y] = -1
    return {
        "round": 1, "step": 1, "field": field,
        "self": ("evader_agent", 0, True, self_pos),
        "others": list(others), "bombs": list(bombs), "coins": list(coins),
        "explosion_map": np.zeros_like(field) if explosion_map is None else explosion_map,
        "user_input": None,
    }


def _agent(alertness: str | None = None) -> SimpleNamespace:
    agent = SimpleNamespace()
    environment = {} if alertness is None else {"BOMBERMAN_EVADER_ALERTNESS": alertness}
    with patch.dict(os.environ, environment, clear=False):
        if alertness is None:
            os.environ.pop("BOMBERMAN_EVADER_ALERTNESS", None)
        evader.setup(agent)
    return agent


class ItNeverBombsAndNeverBurnsTest(unittest.TestCase):
    def test_bomb_is_not_in_its_vocabulary(self):
        """The whole point: it can never kill itself, so it stays a target."""
        agent = _agent()
        for step in range(200):
            state = _state(self_pos=(1, 1), bombs=[((1, 3), step % 5)])
            self.assertNotEqual(evader.act(agent, state), "BOMB")

    def test_it_never_steps_onto_a_tile_that_detonates_this_step(self):
        """``rule_based``'s rule, unchanged: reject a tile whose bomb is at zero.

        This is deliberately weaker than "no bomb will ever reach it" -- a tile
        three steps from detonation is still walkable, which is what makes
        escaping through a blast corridor possible at all.
        """
        agent = _agent()
        for _ in range(60):
            state = _state(self_pos=(1, 4), bombs=[((1, 1), 0)])
            action = evader.act(agent, state)
            self.assertIn(action, ("UP", "DOWN", "LEFT", "RIGHT", "WAIT"))
            self.assertNotEqual(action, "UP", "stepped onto a detonating tile")

    def test_it_never_steps_into_a_live_explosion(self):
        agent = _agent()
        state = _state(self_pos=(1, 1))
        state["explosion_map"][1, 2] = 1
        for _ in range(40):
            self.assertNotEqual(evader.act(agent, state), "DOWN")

    def test_it_always_returns_a_legal_action(self):
        agent = _agent()
        generator = np.random.default_rng(0)
        for _ in range(200):
            position = (int(generator.integers(1, 16)), int(generator.integers(1, 16)))
            state = _state(self_pos=position)
            if state["field"][position] != 0:
                continue
            self.assertIn(evader.act(agent, state),
                          ("UP", "DOWN", "LEFT", "RIGHT", "WAIT"))


class TheAlertnessKnobIsRealTest(unittest.TestCase):
    def test_a_distracted_evader_retreats_less_often(self):
        """Measured, not asserted: the retreat proposals are simply not made."""
        # Standing in a bomb's column with a clear sideways escape: an alert
        # evader is pushed away from it, a distracted one wanders.
        def turns_the_corner(alertness: str) -> float:
            """How often it leaves the bomb's column, which is the retreat.

            Standing at (1,3) with a bomb at (1,1): the retreat proposals are
            appended last and popped first, and ``rule_based``'s order prefers
            turning a corner to running further down the column, so an alert
            evader goes RIGHT every time. A distracted one never makes those
            proposals at all and just wanders.
            """
            agent = _agent(alertness)
            corner = 0
            for _ in range(400):
                state = _state(self_pos=(1, 3), bombs=[((1, 1), 3)])
                if evader.act(agent, state) == "RIGHT":
                    corner += 1
            return corner / 400

        alert, distracted = turns_the_corner("1.0"), turns_the_corner("0.0")
        self.assertGreater(alert, 0.95)
        self.assertLess(distracted, 0.6)

    def test_the_default_is_a_full_dodger(self):
        agent = _agent()
        self.assertEqual(agent.alertness, evader.DEFAULT_ALERTNESS)
        self.assertEqual(evader.DEFAULT_ALERTNESS, 1.0)

    def test_an_alertness_outside_the_unit_interval_is_refused(self):
        for value in ("1.5", "-0.1"):
            with self.assertRaises(ValueError):
                _agent(value)
