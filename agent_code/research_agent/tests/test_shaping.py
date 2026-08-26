"""Tests for A06's potential shaping.

Every test here checks a property the policy-invariance guarantee actually
depends on.  A shaping term that merely "looks like" a potential is worthless:
it silently changes what the optimal policy is, and the change is invisible in
any aggregate metric.
"""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from agent_code.research_agent.config import SHAPING_SPECIFICATIONS, active_config, shaping_specification
from agent_code.research_agent.shaping import PotentialShaping, build_shaping


def board(size: int = 9) -> np.ndarray:
    field = np.zeros((size, size), dtype=int)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    return field


def game_state(position=(3, 3), *, coins=(), crates=(), bombs=(), step=1, field=None) -> dict:
    field = board() if field is None else field.copy()
    for crate in crates:
        field[crate] = 1
    return {
        "round": 1,
        "step": step,
        "field": field,
        "self": ("research_agent", 0, True, position),
        "others": [],
        "bombs": list(bombs),
        "coins": list(coins),
        "explosion_map": np.zeros_like(field),
        "user_input": None,
    }


class PotentialShapingTest(unittest.TestCase):
    def setUp(self):
        self.shaping = PotentialShaping(SHAPING_SPECIFICATIONS["A06"], discount=0.95)

    def test_a_terminal_state_has_potential_zero(self):
        # Without this the episode sum does not telescope to a constant and the
        # optimal policy is no longer preserved.
        self.assertEqual(self.shaping.potential(None), 0.0)

    def test_the_potential_depends_on_the_state_alone(self):
        state = game_state(coins=[(6, 3)])
        # Same state, different histories and different actions leading into it:
        # a potential that is a function of the state cannot tell them apart.
        self.assertEqual(self.shaping.potential(state), self.shaping.potential(game_state(coins=[(6, 3)])))
        self.assertEqual(
            self.shaping.potential(state),
            self.shaping.potential(game_state(coins=[(6, 3)], step=97)),
        )

    def test_moving_towards_a_coin_raises_the_potential(self):
        far = self.shaping.potential(game_state((3, 3), coins=[(6, 3)]))
        near = self.shaping.potential(game_state((5, 3), coins=[(6, 3)]))
        self.assertGreater(near, far)

    def test_standing_in_a_future_blast_lowers_the_potential_by_the_danger_weight(self):
        safe = self.shaping.potential(game_state((3, 3), coins=[(3, 4)]))
        endangered = self.shaping.potential(game_state((3, 3), coins=[(3, 4)], bombs=[((3, 5), 3)]))
        self.assertAlmostEqual(safe - endangered, SHAPING_SPECIFICATIONS["A06"]["danger_weight"], places=6)

    def test_the_potential_is_bounded_by_the_declared_clip(self):
        specification = SHAPING_SPECIFICATIONS["A06"]
        lowest = -specification["coin_weight"] * specification["distance_cap"] - specification["danger_weight"]
        # An empty board with nothing to collect is the worst case by design.
        empty = self.shaping.potential(game_state((1, 1)))
        self.assertGreaterEqual(empty, lowest)
        self.assertLessEqual(empty, 0.0)
        # Unbounded distances would leave a residual drift at gamma < 1.
        wide = np.zeros((41, 41), dtype=int)
        wide[[0, -1], :] = -1
        wide[:, [0, -1]] = -1
        distant = self.shaping.potential(game_state((1, 1), coins=[(39, 39)], field=wide))
        self.assertAlmostEqual(distant, -specification["coin_weight"] * specification["distance_cap"], places=6)

    def test_crate_targets_are_the_reachable_cells_beside_a_crate(self):
        # A crate tile is impassable, so its BFS distance is undefined; using it
        # as the target would make the potential jump around at random.
        beside = self.shaping.potential(game_state((3, 3), crates=[(3, 5)]))
        further = self.shaping.potential(game_state((3, 3), crates=[(3, 7)]))
        self.assertGreater(beside, further)

    def test_the_discounted_shaping_of_a_whole_episode_telescopes_to_a_constant(self):
        # sum_t gamma^t F(s_t, s_t+1) must equal gamma^T phi(s_T) - phi(s_0).
        discount = 0.95
        trajectory = [game_state((x, 3), coins=[(7, 3)]) for x in range(1, 7)] + [None]
        potentials = [self.shaping.potential(state) for state in trajectory]
        total = sum(
            (discount ** index) * self.shaping.shaping_reward(potentials[index], potentials[index + 1])
            for index in range(len(trajectory) - 1)
        )
        expected = (discount ** (len(trajectory) - 1)) * potentials[-1] - potentials[0]
        self.assertAlmostEqual(total, expected, places=6)

    def test_pacing_back_and_forth_cannot_farm_unbounded_shaping(self):
        # A single there-and-back is net positive, which is the honest version
        # of the story; what matters is that the episode total stays bounded no
        # matter how many times it is repeated.
        discount = 0.95
        here, there = game_state((3, 3), coins=[(7, 3)]), game_state((4, 3), coins=[(7, 3)])
        potentials = [self.shaping.potential(here), self.shaping.potential(there)]
        totals = []
        for cycles in (1, 10, 100):
            trajectory = [potentials[index % 2] for index in range(2 * cycles + 1)] + [0.0]
            totals.append(sum(
                (discount ** index) * self.shaping.shaping_reward(trajectory[index], trajectory[index + 1])
                for index in range(len(trajectory) - 1)
            ))
        self.assertLess(max(totals), -potentials[0] + 1e-6)

    def test_shaping_is_derived_from_the_reward_version_and_shares_the_learner_discount(self):
        self.assertIsNone(shaping_specification("A03"))
        self.assertEqual(shaping_specification("A06")["name"], "potential_v1")
        config = replace(active_config(), reward_version="A06")
        shaping = build_shaping(config)
        # A shaping term discounted differently from the learner leaves a
        # residual that is not a potential difference at all.
        self.assertEqual(shaping.discount, config.discount)
        self.assertIsNone(build_shaping(replace(config, reward_version="A03")))

    def test_an_unknown_shaping_function_is_refused(self):
        with self.assertRaises(ValueError):
            PotentialShaping({"name": "made_up", "coin_weight": 1, "distance_cap": 1,
                              "danger_weight": 1, "terminal_potential": 0}, discount=0.95)


if __name__ == "__main__":
    unittest.main()
