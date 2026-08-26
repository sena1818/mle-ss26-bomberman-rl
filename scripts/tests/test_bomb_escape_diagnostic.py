"""Tests for the survivability search behind the bomb-escape diagnostic.

The diagnostic's whole value is the claim "a survivable plan existed here".
If that search is wrong in either direction the run's suicides get sorted into
the wrong bucket and the next experiment is chosen on a false premise, so the
cases that decide the answer are pinned here rather than eyeballed.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import diagnose_bomb_escape as diagnostic  # noqa: E402
import settings as s  # noqa: E402


def corridor_state(length: int, agent_x: int, bomb_x: int, timer: int) -> dict:
    """One horizontal corridor of free tiles inside a stone frame."""
    field = np.full((length + 2, 3), -1, dtype=np.int8)
    field[1:length + 1, 1] = 0
    return {
        "field": field,
        "self": ("me", 0, False, (agent_x, 1)),
        "others": [],
        "bombs": [((bomb_x, 1), timer)],
        "coins": [],
        "explosion_map": np.zeros_like(field, dtype=float),
        "round": 1,
        "step": 1,
    }


class SurvivabilityTest(unittest.TestCase):
    def test_a_cell_outside_every_blast_is_already_safe(self):
        # BOMB_POWER is 3, so a corridor of 12 puts the far end out of reach.
        state = corridor_state(12, agent_x=12, bomb_x=1, timer=3)
        self.assertEqual(diagnostic.survivable(state, s.BOMB_TIMER + 1), (True, 0))

    def test_a_reachable_exit_is_found_with_its_distance(self):
        # The blast reaches x = 4, so x = 5 is the first safe tile.
        state = corridor_state(12, agent_x=4, bomb_x=1, timer=3)
        exists, steps = diagnostic.survivable(state, s.BOMB_TIMER + 1)
        self.assertTrue(exists)
        self.assertEqual(steps, 1)

    def test_a_dead_end_shorter_than_the_blast_is_not_survivable(self):
        # The corridor ends inside the blast and the bomb blocks the only way
        # back, so no amount of walking gets the agent out.
        state = corridor_state(3, agent_x=3, bomb_x=1, timer=3)
        self.assertEqual(diagnostic.survivable(state, s.BOMB_TIMER + 1), (False, None))

    def test_an_exit_too_far_for_the_remaining_ticks_is_rejected(self):
        state = corridor_state(12, agent_x=2, bomb_x=1, timer=0)
        # x = 5 is the first safe tile but the bomb goes off after one tick.
        self.assertEqual(diagnostic.survivable(state, s.BOMB_TIMER + 1), (False, None))

    def test_the_bomb_tile_itself_is_not_a_way_through(self):
        # Standing right of the bomb with stone on the right: the only opening
        # is the bomb's own tile, which the strict reading refuses.
        field = np.full((4, 3), -1, dtype=np.int8)
        field[1:3, 1] = 0
        state = {
            "field": field, "self": ("me", 0, False, (2, 1)), "others": [],
            "bombs": [((1, 1), 3)], "coins": [],
            "explosion_map": np.zeros_like(field, dtype=float),
            "round": 1, "step": 1,
        }
        self.assertEqual(diagnostic.survivable(state, s.BOMB_TIMER + 1), (False, None))

    def test_no_bomb_on_the_board_is_trivially_safe(self):
        state = corridor_state(12, agent_x=6, bomb_x=1, timer=3)
        state["bombs"] = []
        self.assertEqual(diagnostic.survivable(state, s.BOMB_TIMER + 1), (True, 0))


class BlastScheduleTest(unittest.TestCase):
    def test_a_tile_is_lethal_for_the_whole_explosion(self):
        field = np.zeros((7, 3), dtype=np.int8)
        schedule = diagnostic.blast_schedule(field, [((1, 1), 3)])
        # Timer 3 detonates on tick 4 and lingers EXPLOSION_TIMER ticks.
        self.assertEqual(schedule[(1, 1)], set(range(4, 4 + s.EXPLOSION_TIMER)))

    def test_an_expired_timer_still_detonates_on_the_next_tick(self):
        field = np.zeros((7, 3), dtype=np.int8)
        schedule = diagnostic.blast_schedule(field, [((1, 1), 0)])
        self.assertEqual(min(schedule[(1, 1)]), 1)

    def test_stone_stops_the_blast(self):
        field = np.zeros((7, 3), dtype=np.int8)
        field[3, 1] = -1
        schedule = diagnostic.blast_schedule(field, [((1, 1), 3)])
        self.assertIn((2, 1), schedule)
        self.assertNotIn((4, 1), schedule)


class ShapingComparisonTest(unittest.TestCase):
    def test_moving_toward_the_only_coin_pays_more_than_waiting(self):
        from agent_code.research_agent.shaping import PotentialShaping
        from agent_code.research_agent.config import shaping_specification

        shaping = PotentialShaping(shaping_specification("A06"), 0.95)
        state = corridor_state(12, agent_x=6, bomb_x=1, timer=3)
        state["coins"] = [(10, 1)]
        terms = diagnostic.shaping_terms(state, shaping)
        self.assertIn("RIGHT", terms)
        self.assertIn("LEFT", terms)
        self.assertGreater(terms["RIGHT"], terms["WAIT"])
        self.assertLess(terms["LEFT"], terms["WAIT"])

    def test_blocked_directions_are_absent_rather_than_zero(self):
        from agent_code.research_agent.shaping import PotentialShaping
        from agent_code.research_agent.config import shaping_specification

        shaping = PotentialShaping(shaping_specification("A06"), 0.95)
        state = corridor_state(12, agent_x=6, bomb_x=1, timer=3)
        terms = diagnostic.shaping_terms(state, shaping)
        self.assertEqual(set(terms) & {"UP", "DOWN"}, set())


if __name__ == "__main__":
    unittest.main()
