"""Tests for A06's potential shaping.

Every test here checks a property the policy-invariance guarantee actually
depends on.  A shaping term that merely "looks like" a potential is worthless:
it silently changes what the optimal policy is, and the change is invisible in
any aggregate metric.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from dataclasses import replace

import numpy as np

from agent_code.research_agent.config import SHAPING_SPECIFICATIONS, active_config, shaping_specification
from agent_code.research_agent.shaping import PotentialShaping, build_shaping


def board(size: int = 9) -> np.ndarray:
    field = np.zeros((size, size), dtype=int)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    return field


def game_state(position=(3, 3), *, coins=(), crates=(), bombs=(), step=1, field=None, others=()) -> dict:
    field = board() if field is None else field.copy()
    for crate in crates:
        field[crate] = 1
    return {
        "round": 1,
        "step": step,
        "field": field,
        "self": ("research_agent", 0, True, position),
        "others": [("rule_based_agent", 0, True, other) for other in others],
        "bombs": list(bombs),
        "coins": list(coins),
        "explosion_map": np.zeros_like(field),
        "user_input": None,
    }


class OpponentBlastPotentialTest(unittest.TestCase):
    """potential_v2's only difference from v1 (docs/01 section 7.27)."""

    def setUp(self):
        self.v1 = PotentialShaping(SHAPING_SPECIFICATIONS["A06"], discount=0.95)
        self.v2 = PotentialShaping(SHAPING_SPECIFICATIONS["A07"], discount=0.95)

    def test_with_no_opponents_the_two_potentials_agree_exactly(self):
        """A07 must reduce to A06 in every solo state, or no solo arm is comparable."""
        for state in (game_state(coins=[(6, 3)]),
                      game_state(coins=[(6, 3)], bombs=[((3, 3), 3)]),
                      game_state(crates=[(4, 3)])):
            self.assertEqual(self.v1.potential(state), self.v2.potential(state))

    def test_an_opponent_standing_in_a_blast_raises_the_potential(self):
        weight = SHAPING_SPECIFICATIONS["A07"]["opponent_blast_weight"]
        without = game_state(coins=[(6, 3)], others=[(5, 3)])
        # Same board, plus a bomb whose blast covers that opponent.
        within = game_state(coins=[(6, 3)], others=[(5, 3)], bombs=[((3, 3), 3)])
        self.assertAlmostEqual(self.v2.potential(within) - self.v2.potential(without),
                               weight + (self.v1.potential(within) - self.v1.potential(without)))

    def test_an_opponent_out_of_the_blast_changes_nothing(self):
        state = game_state(coins=[(6, 3)], others=[(1, 1)], bombs=[((5, 5), 3)])
        self.assertEqual(self.v1.potential(state), self.v2.potential(state))

    def test_the_term_counts_each_opponent_once_however_many_bombs_cover_it(self):
        weight = SHAPING_SPECIFICATIONS["A07"]["opponent_blast_weight"]
        one_bomb = game_state(coins=[(6, 3)], others=[(3, 4)], bombs=[((3, 3), 3)])
        two_bombs = game_state(coins=[(6, 3)], others=[(3, 4)], bombs=[((3, 3), 3), ((3, 5), 3)])
        self.assertAlmostEqual(self.v2.potential(two_bombs) - self.v1.potential(two_bombs), weight)
        self.assertAlmostEqual(self.v2.potential(one_bomb) - self.v1.potential(one_bomb), weight)

    def test_the_term_reads_the_state_alone_and_stays_bounded(self):
        weight = SHAPING_SPECIFICATIONS["A07"]["opponent_blast_weight"]
        crowded = game_state(coins=[(6, 3)], others=[(3, 4), (3, 2), (4, 3)], bombs=[((3, 3), 3)])
        self.assertLessEqual(self.v2.potential(crowded) - self.v1.potential(crowded), 3 * weight + 1e-9)
        self.assertEqual(self.v2.potential(None), 0.0)

    def test_a07_and_a06_declare_identical_event_weights(self):
        from agent_code.research_agent.runtime.experiment import REWARD_TABLES, DEATH_PENALTIES
        self.assertEqual(REWARD_TABLES["A06"], REWARD_TABLES["A07"])
        self.assertEqual(DEATH_PENALTIES["A06"], DEATH_PENALTIES["A07"])


class ShapingSurvivesTheNStepReturnTest(unittest.TestCase):
    """A potential term can telescope to nothing inside the n-step window.

    Shaping contributes ``gamma^n phi(s_t+n) - phi(s_t)`` to an n-step target --
    the telescoping that gives policy invariance in the first place.  A bomb
    lives exactly BOMB_TIMER+1 = 5 transitions, so at n_step=5 the window around
    a drop starts before the bomb exists and ends after it is gone, both
    endpoints carry the same potential, and potential_v2's term contributes
    EXACTLY zero to the target of the decision it was written to shape.

    This is not hypothetical: A07 was first configured at n=5 and would have
    measured nothing.  The guard is that any arm running potential_v2 has to
    keep its n_step below the bomb's life.
    """

    def setUp(self):
        self.v1 = PotentialShaping(SHAPING_SPECIFICATIONS["A06"], discount=0.95)
        self.v2 = PotentialShaping(SHAPING_SPECIFICATIONS["A07"], discount=0.95)
        # Drop a bomb covering an opponent, sit out the fuse, bomb gone.
        self.traj = ([game_state(others=[(5, 3)], coins=[(6, 3)])]
                     + [game_state(others=[(5, 3)], coins=[(6, 3)], bombs=[((3, 3), timer)])
                        for timer in (3, 2, 1, 0)]
                     + [game_state(others=[(5, 3)], coins=[(6, 3)])])

    def _contribution(self, n: int) -> float:
        """What the new term adds to the bomb-drop transition's own n-step target."""
        delta = [self.v2.potential(s) - self.v1.potential(s) for s in self.traj]
        end = min(n, len(self.traj) - 1)
        return 0.95 ** n * delta[end] - delta[0]

    def test_the_term_cancels_exactly_at_the_bomb_lifetime(self):
        self.assertAlmostEqual(self._contribution(5), 0.0, places=12)
        self.assertAlmostEqual(self._contribution(8), 0.0, places=12)

    def test_the_term_survives_below_the_bomb_lifetime(self):
        for n in (1, 2, 3, 4):
            self.assertGreater(self._contribution(n), 0.2, f"n={n} carries no signal")

    def test_every_arm_using_potential_v2_stays_off_the_resonance(self):
        """The configs are the thing that can silently drift back onto it."""
        import json
        experiments = Path(__file__).resolve().parents[3] / "experiments"
        checked = 0
        for path in sorted(experiments.glob("*.json")):
            config = json.loads(path.read_text(encoding="utf-8"))
            if (config.get("shaping") or {}).get("name") != "potential_v2":
                continue
            checked += 1
            n_step = config["agent"]["n_step"]
            self.assertLess(n_step, 5,
                            f"{path.name} runs potential_v2 at n_step={n_step}, where the opponent "
                            f"term contributes exactly zero to the bomb-drop target")
        self.assertGreater(checked, 0, "no potential_v2 config found to check")


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
