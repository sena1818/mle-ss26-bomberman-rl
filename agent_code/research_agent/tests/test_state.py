"""Contract tests for the R01 vector state, legality mask, and danger map."""

from __future__ import annotations

import unittest

import numpy as np

from agent_code.research_agent.config import ACTIONS, FEATURE_DIMENSION
from agent_code.research_agent.learning import choose_action_index
from agent_code.research_agent.networks import LinearQNetwork
from agent_code.research_agent.state import (
    HANDCRAFTED_V1_LAYOUT,
    _blast_coordinates,
    future_danger_times,
    handcrafted_v1,
    legal_action_mask,
)
from items import Bomb


def game_state(*, field=None, self_pos=(3, 3), can_bomb=True, bombs=(), others=(), coins=(), explosion_map=None):
    if field is None:
        field = np.zeros((9, 9), dtype=int)
        field[[0, -1], :] = -1
        field[:, [0, -1]] = -1
    if explosion_map is None:
        explosion_map = np.zeros_like(field)
    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("research_agent", 0, can_bomb, self_pos),
        "others": list(others),
        "bombs": list(bombs),
        "coins": list(coins),
        "explosion_map": explosion_map,
        "user_input": None,
    }


class HandcraftedStateTest(unittest.TestCase):
    def test_handcrafted_v1_has_frozen_44_dimensional_layout(self):
        state = game_state(
            bombs=[((5, 3), 2)],
            others=[("opponent", 1, True, (3, 5))],
            coins=[(4, 4)],
        )
        features = handcrafted_v1(state)

        self.assertEqual(features.shape, (FEATURE_DIMENSION,))
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(HANDCRAFTED_V1_LAYOUT["bomb_escape"].stop, FEATURE_DIMENSION)
        np.testing.assert_array_equal(
            features[HANDCRAFTED_V1_LAYOUT["legal_actions"]],
            legal_action_mask(state).astype(np.float32),
        )

    def test_mask_matches_official_immediate_action_rules(self):
        field = game_state()["field"]
        field[3, 2] = -1  # UP: stone wall
        field[4, 3] = 1   # RIGHT: crate
        state = game_state(
            field=field,
            bombs=[((3, 4), 3)],  # DOWN: occupied by a bomb
            others=[("opponent", 0, True, (2, 3))],  # LEFT: occupied by agent
        )
        expected = np.array([False, False, False, False, True, True])
        np.testing.assert_array_equal(legal_action_mask(state), expected)

        no_bomb = game_state(field=field, can_bomb=False)
        np.testing.assert_array_equal(
            legal_action_mask(no_bomb), np.array([False, False, True, True, True, False])
        )

    def test_greedy_and_exploration_never_select_masked_actions(self):
        state = game_state(can_bomb=False)
        mask = legal_action_mask(state)
        model = LinearQNetwork(FEATURE_DIMENSION)
        model.weights.fill(0.0)
        model.bias[:] = np.arange(len(ACTIONS), dtype=np.float32)  # BOMB would win without a mask.
        features = handcrafted_v1(state)
        generator = np.random.default_rng(17)

        self.assertEqual(choose_action_index(model, features, mask, 0.0, generator), ACTIONS.index("WAIT"))
        sampled = {choose_action_index(model, features, mask, 1.0, generator) for _ in range(100)}
        self.assertTrue(sampled.issubset(set(np.flatnonzero(mask))))

    def test_q_learning_target_excludes_an_illegal_high_value_action(self):
        model = LinearQNetwork(FEATURE_DIMENSION)
        model.weights.fill(0.0)
        model.bias.fill(0.0)
        state = np.zeros(FEATURE_DIMENSION, dtype=np.float32)
        state[0] = 1.0
        next_state = np.zeros(FEATURE_DIMENSION, dtype=np.float32)
        next_state[1] = 1.0
        model.weights[:, 1] = 1.0
        model.weights[ACTIONS.index("WAIT"), 1] = 3.0
        model.weights[ACTIONS.index("BOMB"), 1] = 100.0
        next_mask = np.array([True, True, True, True, True, False])

        td_error = model.q_learning_update(
            state, ACTIONS.index("UP"), reward=0.0, next_state=next_state,
            next_legal_mask=next_mask, learning_rate=1.0, discount=1.0,
        )
        self.assertEqual(td_error, 3.0)


class DangerMapTest(unittest.TestCase):
    def test_blast_coordinates_match_official_bomb_and_crates_do_not_stop_it(self):
        field = game_state()["field"]
        field[4, 3] = 1  # crate is included and blast continues through it
        field[6, 3] = -1  # stone wall blocks before itself

        official = Bomb((3, 3), None, timer=0, power=3, bomb_sprite=None).get_blast_coords(field)
        derived = _blast_coordinates((3, 3), field)
        self.assertEqual(derived, official)
        self.assertIn((4, 3), derived)
        self.assertIn((5, 3), derived)
        self.assertNotIn((6, 3), derived)

    def test_danger_timing_current_explosion_and_no_chain_reaction(self):
        state = game_state(bombs=[((3, 3), 0), ((5, 3), 4)])
        state["explosion_map"][1, 1] = 1
        danger = future_danger_times(state)

        self.assertEqual(danger[3, 3], 1)  # timer zero explodes after this action
        self.assertEqual(danger[1, 1], 0)  # official lingering flame is immediately dangerous
        # The first bomb hits (5, 3), but the framework does not trigger the
        # second bomb.  Its perpendicular blast keeps its own timer (4 + 1).
        self.assertEqual(danger[5, 4], 5)
