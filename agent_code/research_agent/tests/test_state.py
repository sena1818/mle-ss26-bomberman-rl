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
from agent_code.research_agent import state as state_module
from agent_code.research_agent.config import FEATURE_DIMENSION_V2
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


class HandcraftedV2Test(unittest.TestCase):
    """The escape block must answer the question v1 provably could not.

    docs/01 section 7.10: in 94.4% of the turns that threw away a survivable
    escape, the fatal direction and a saving one were bit-for-bit identical in
    ``danger_current_and_neighbors``.  The T-junction below is that situation in
    miniature, and the first test is the one that would have caught it.
    """

    @staticmethod
    def t_junction():
        # One corridor, bomb to the agent's left with one tick left.  Both
        # horizontal neighbours are inside the blast and equally urgent, but the
        # bomb seals the left side, so only the right arm reaches open floor in
        # time.  This is the T-junction of section 7.10 reduced to one line.
        field = np.full((10, 5), -1, dtype=np.int8)
        field[1:9, 3] = 0
        return {
            "field": field,
            "self": ("me", 0, False, (5, 3)),
            "others": [],
            "bombs": [((3, 3), 1)],
            "coins": [],
            "explosion_map": np.zeros_like(field, dtype=float),
            "round": 1,
            "step": 5,
        }

    def test_v1_cannot_tell_the_two_arms_apart(self):
        state = self.t_junction()
        block = state_module.handcrafted_v1(state)[
            state_module.HANDCRAFTED_V1_LAYOUT["danger_current_and_neighbors"]]
        order = list(state_module._DIRECTIONS)
        left = order.index("LEFT")
        right = order.index("RIGHT")
        self.assertTrue(np.array_equal(block[2 + 2 * left: 4 + 2 * left],
                                       block[2 + 2 * right: 4 + 2 * right]))

    def test_v2_tells_the_two_arms_apart(self):
        state = self.t_junction()
        block = state_module.handcrafted_v2(state)[
            state_module.HANDCRAFTED_V2_LAYOUT["escape_by_direction"]]
        order = list(state_module._DIRECTIONS)
        left = order.index("LEFT")
        right = order.index("RIGHT")
        self.assertFalse(np.array_equal(block[2 * left: 2 + 2 * left],
                                        block[2 * right: 2 + 2 * right]))
        # The long arm is escapable, the stub is not.
        self.assertEqual(block[2 * right], 1.0)
        self.assertEqual(block[2 * left], 0.0)

    def test_v2_keeps_every_v1_entry_at_its_original_index(self):
        state = self.t_junction()
        first = state_module.handcrafted_v2(state)[:state_module.FEATURE_DIMENSION]
        self.assertTrue(np.array_equal(first, state_module.handcrafted_v1(state)))

    def test_a_board_without_bombs_reports_safety_everywhere(self):
        state = self.t_junction()
        state["bombs"] = []
        block = state_module.handcrafted_v2(state)[
            state_module.HANDCRAFTED_V2_LAYOUT["escape_here"]]
        self.assertEqual(list(block), [1.0, 0.0])

    def test_a_blocked_direction_reads_as_no_way_out(self):
        state = self.t_junction()
        block = state_module.handcrafted_v2(state)[
            state_module.HANDCRAFTED_V2_LAYOUT["escape_by_direction"]]
        order = list(state_module._DIRECTIONS)
        down = order.index("DOWN")  # stone below the agent
        self.assertEqual(list(block[2 * down: 2 + 2 * down]), [0.0, 1.0])

    def test_the_dimension_matches_the_declared_layout(self):
        state = self.t_junction()
        self.assertEqual(state_module.handcrafted_v2(state).shape,
                         (state_module.state_dimension("handcrafted_v2"),))
        self.assertEqual(max(s.stop for s in state_module.HANDCRAFTED_V2_LAYOUT.values()),
                         state_module.state_dimension("handcrafted_v2"))


class HandcraftedV3Test(unittest.TestCase):
    """The routing block must name the turn the compass bearing gets wrong."""

    @staticmethod
    def dogleg():
        # The coin is up and to the right, but a wall means the only route
        # starts by going right; the bearing suggests UP as well.
        field = np.full((9, 9), -1, dtype=np.int8)
        field[1:8, 5] = 0
        field[7, 1:6] = 0
        return {
            "field": field,
            "self": ("me", 0, True, (2, 5)),
            "others": [],
            "bombs": [],
            "coins": [(7, 1)],
            "explosion_map": np.zeros_like(field, dtype=float),
            "round": 1,
            "step": 5,
        }

    def test_the_bearing_points_up_but_the_route_starts_right(self):
        state = self.dogleg()
        features = state_module.handcrafted_v3(state)
        bearing = features[state_module.HANDCRAFTED_V3_LAYOUT["coin_target"]]
        route = features[state_module.HANDCRAFTED_V3_LAYOUT["coin_route"]]
        order = list(state_module._DIRECTIONS)
        # sign(ty - y) is negative: the coin is "up" as the crow flies.
        self.assertLess(bearing[1], 0.0)
        self.assertEqual(route[order.index("UP")], 0.0)
        self.assertEqual(route[order.index("RIGHT")], 1.0)

    def test_v3_keeps_every_v2_entry_at_its_original_index(self):
        state = self.dogleg()
        first = state_module.handcrafted_v3(state)[:state_module.FEATURE_DIMENSION_V2]
        self.assertTrue(np.array_equal(first, state_module.handcrafted_v2(state)))

    def test_no_reachable_coin_leaves_the_coin_route_zero(self):
        state = self.dogleg()
        state["coins"] = []
        route = state_module.handcrafted_v3(state)[
            state_module.HANDCRAFTED_V3_LAYOUT["coin_route"]]
        self.assertEqual(list(route), [0.0, 0.0, 0.0, 0.0])


class BoardEgocentricV2Test(unittest.TestCase):
    """The M4 representation: what changed from v1, and the grid it lives on."""

    def _state(self):
        field = np.zeros((9, 9), dtype=int)
        field[[0, -1], :] = -1
        field[:, [0, -1]] = -1
        field[2, 2] = 1
        field[4, 4] = 1
        return game_state(
            field=field,
            self_pos=(3, 3),
            bombs=[((5, 3), 2), ((3, 6), 0)],
            others=[("opponent", 1, True, (3, 5))],
            coins=[(4, 4), (6, 6)],
        )

    def test_v1_self_plane_is_the_same_constant_for_every_state(self):
        """The deletion that defines v2 is a proof, not a hypothesis."""
        first = state_module.board_egocentric_v1(self._state())
        second = state_module.board_egocentric_v1(
            game_state(self_pos=(5, 5), coins=[(1, 1)], bombs=[((5, 6), 3)])
        )
        shape = state_module.BOARD_EGOCENTRIC_LAYOUT["board_shape"]
        plane = state_module.BOARD_CHANNELS.index("self")
        size = int(np.prod(shape))
        first_plane = first[:size].reshape(shape)[plane]
        second_plane = second[:size].reshape(shape)[plane]
        np.testing.assert_array_equal(first_plane, second_plane)
        self.assertEqual(first_plane.sum(), 1.0)
        self.assertEqual(first_plane[8, 8], 1.0)
        self.assertNotIn("self", state_module.BOARD_CHANNELS_V2)

    def test_v2_layout_is_seven_planes_and_six_scalars(self):
        vector = state_module.board_egocentric_v2(self._state())
        self.assertEqual(vector.shape, (7 * 17 * 17 + 6,))
        self.assertEqual(vector.dtype, np.float32)
        self.assertEqual(state_module.state_dimension("board_egocentric_v2"), vector.shape[0])
        self.assertIs(
            state_module.layout_for_dimension(vector.shape[0]),
            state_module.BOARD_EGOCENTRIC_V2_LAYOUT,
        )
        board, globals_ = state_module.split_board_and_globals(vector)
        self.assertEqual(board.shape, (7, 17, 17))
        self.assertEqual(globals_.shape, (6,))

    def test_v2_shares_every_surviving_plane_with_v1(self):
        state = self._state()
        v1 = state_module.encode_board_channels_v1(state)
        v2 = state_module.encode_board_channels_v2(state)
        for name in state_module.BOARD_CHANNELS_V2:
            np.testing.assert_array_equal(
                v2[state_module.BOARD_CHANNELS_V2.index(name)],
                v1[state_module.BOARD_CHANNELS.index(name)],
                err_msg=f"channel {name} drifted between v1 and v2",
            )

    def test_v2_scalars_report_board_depletion(self):
        state = self._state()
        globals_ = state_module.global_features_v2(state)
        self.assertEqual(globals_[4], 2 / state_module.MAX_VISIBLE_COINS)
        free_cells = int(np.count_nonzero(state["field"] != -1))
        self.assertAlmostEqual(float(globals_[5]), 2 / free_cells, places=6)
        np.testing.assert_allclose(globals_[:4], state_module.global_features_v1(state), rtol=0, atol=0)

    def test_every_board_value_lies_on_the_declared_quantisation_grid(self):
        """What makes uint8 replay storage lossless rather than approximate."""
        board_size = int(np.prod(state_module.BOARD_EGOCENTRIC_V2_LAYOUT["board_shape"]))
        for timer in range(state_module.BOMB_TIMER + 1):
            state = self._state()
            state["bombs"] = [((5, 3), timer)]
            state["explosion_map"][6, 6] = 1
            board = state_module.board_egocentric_v2(state)[:board_size]
            scaled = board * np.float32(state_module.BOARD_QUANTISATION)
            np.testing.assert_allclose(scaled, np.rint(scaled), rtol=0, atol=1e-4)
            self.assertGreaterEqual(board.min(), 0.0)
            self.assertLessEqual(board.max(), 1.0)
