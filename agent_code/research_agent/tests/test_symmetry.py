"""Tests for the egocentric board state and the D4 symmetry group.

D4 augmentation is only sound if the array transform and the action
permutation are the *same* transform.  If they drift apart, augmentation
teaches the network that pressing LEFT moves right -- and the training curve
will not show it, because the labels are still self-consistent within a batch.
"""

from __future__ import annotations

import unittest

import numpy as np

from agent_code.research_agent.config import ACTIONS
from agent_code.research_agent.state import (
    BOARD_EGOCENTRIC_LAYOUT,
    EGOCENTRIC_RADIUS,
    EGOCENTRIC_WINDOW,
    board_egocentric_v1,
    encode_state,
    split_board_and_globals,
    state_dimension,
)
from agent_code.research_agent.symmetry import (
    ACTION_PERMUTATIONS,
    D4_ORDER,
    transform_action_indices,
    transform_board_states,
    transform_legal_masks,
)


DIRECTIONS = {"UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1), "LEFT": (-1, 0)}
SELF_CHANNEL = BOARD_EGOCENTRIC_LAYOUT["channels"].index("self")
COIN_CHANNEL = BOARD_EGOCENTRIC_LAYOUT["channels"].index("coins")
STONE_CHANNEL = BOARD_EGOCENTRIC_LAYOUT["channels"].index("stone_walls")


def game_state(position=(3, 3), *, coins=(), size: int = 17) -> dict:
    field = np.zeros((size, size), dtype=int)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("research_agent", 0, True, position),
        "others": [],
        "bombs": [],
        "coins": list(coins),
        "explosion_map": np.zeros_like(field),
        "user_input": None,
    }


def board_of(state: np.ndarray) -> np.ndarray:
    board, _ = split_board_and_globals(state)
    return board


class EgocentricStateTest(unittest.TestCase):
    def test_the_agent_is_always_at_the_window_centre(self):
        for position in ((1, 1), (3, 8), (8, 8), (15, 15)):
            with self.subTest(position=position):
                board = board_of(board_egocentric_v1(game_state(position)))
                marked = np.argwhere(board[SELF_CHANNEL] == 1.0)
                np.testing.assert_array_equal(marked, [[EGOCENTRIC_RADIUS, EGOCENTRIC_RADIUS]])

    def test_translating_the_whole_situation_leaves_the_state_unchanged(self):
        # The point of an egocentric frame: the network cannot tell where on the
        # arena it is, only what the local situation looks like.
        first = board_egocentric_v1(game_state((3, 3), coins=[(5, 3)]))
        second = board_egocentric_v1(game_state((8, 8), coins=[(10, 8)]))
        np.testing.assert_allclose(board_of(first)[COIN_CHANNEL], board_of(second)[COIN_CHANNEL])

    def test_outside_the_arena_is_padded_with_stone_rather_than_open_floor(self):
        board = board_of(board_egocentric_v1(game_state((1, 1))))
        # A padded cell must be as impassable as a wall; open floor there would
        # invent escape routes that do not exist.
        self.assertEqual(board[STONE_CHANNEL, 0, 0], 1.0)
        self.assertTrue((board[STONE_CHANNEL, :EGOCENTRIC_RADIUS - 1, :] == 1.0).all())

    def test_the_declared_dimension_matches_what_the_encoder_produces(self):
        state = encode_state(game_state(), "board_egocentric_v1")
        self.assertEqual(state.shape, (state_dimension("board_egocentric_v1"),))
        self.assertEqual(state.dtype, np.float32)
        board, globals_ = split_board_and_globals(state)
        self.assertEqual(board.shape, BOARD_EGOCENTRIC_LAYOUT["board_shape"])
        self.assertEqual(globals_.shape, (BOARD_EGOCENTRIC_LAYOUT["global_dimension"],))


class D4SymmetryTest(unittest.TestCase):
    def _marker_state(self, offset: tuple[int, int]) -> np.ndarray:
        channels, width, height = BOARD_EGOCENTRIC_LAYOUT["board_shape"]
        board = np.zeros((channels, width, height), dtype=np.float32)
        board[COIN_CHANNEL, EGOCENTRIC_RADIUS + offset[0], EGOCENTRIC_RADIUS + offset[1]] = 1.0
        return np.concatenate([board.reshape(-1), np.arange(4, dtype=np.float32)])

    def test_the_board_transform_and_the_action_permutation_are_the_same_map(self):
        for transform in range(D4_ORDER):
            for action, offset in DIRECTIONS.items():
                with self.subTest(transform=transform, action=action):
                    moved = board_of(transform_board_states(self._marker_state(offset), transform))
                    landed = tuple(np.argwhere(moved[COIN_CHANNEL] == 1.0)[0] - EGOCENTRIC_RADIUS)
                    permuted = ACTIONS[ACTION_PERMUTATIONS[transform][ACTIONS.index(action)]]
                    self.assertEqual(landed, DIRECTIONS[permuted])

    def test_the_permutations_form_a_group_of_eight_distinct_elements(self):
        rows = {tuple(row) for row in ACTION_PERMUTATIONS}
        self.assertEqual(len(rows), D4_ORDER)
        np.testing.assert_array_equal(ACTION_PERMUTATIONS[0], np.arange(len(ACTIONS)))

    def test_wait_and_bomb_are_fixed_points(self):
        for action in ("WAIT", "BOMB"):
            index = ACTIONS.index(action)
            np.testing.assert_array_equal(ACTION_PERMUTATIONS[:, index], index)

    def test_the_identity_transform_changes_nothing(self):
        state = self._marker_state((2, -1))
        np.testing.assert_array_equal(transform_board_states(state, 0), state)

    def test_the_global_scalars_are_never_rotated(self):
        state = self._marker_state((1, 0))
        for transform in range(D4_ORDER):
            _, globals_ = split_board_and_globals(transform_board_states(state, transform))
            np.testing.assert_array_equal(globals_, np.arange(4, dtype=np.float32))

    def test_a_legal_mask_follows_its_actions(self):
        mask = np.zeros((1, len(ACTIONS)), dtype=bool)
        mask[0, ACTIONS.index("UP")] = True
        for transform in range(D4_ORDER):
            with self.subTest(transform=transform):
                permuted = transform_legal_masks(mask, transform)
                expected = int(transform_action_indices(np.array([ACTIONS.index("UP")]), transform)[0])
                self.assertEqual(int(np.flatnonzero(permuted[0])[0]), expected)

    def test_applying_a_transform_preserves_the_board_contents(self):
        state = board_egocentric_v1(game_state((4, 6), coins=[(4, 3), (7, 6)]))
        for transform in range(D4_ORDER):
            with self.subTest(transform=transform):
                moved = board_of(transform_board_states(state, transform))
                np.testing.assert_allclose(moved.sum(axis=(1, 2)), board_of(state).sum(axis=(1, 2)))
                # The agent stays at the centre: that is why the group acts here.
                np.testing.assert_array_equal(
                    np.argwhere(moved[SELF_CHANNEL] == 1.0), [[EGOCENTRIC_RADIUS, EGOCENTRIC_RADIUS]]
                )

    def test_a_batch_is_transformed_row_by_row(self):
        states = np.stack([self._marker_state((1, 0)), self._marker_state((0, 1))])
        transformed = transform_board_states(states, 1)
        self.assertEqual(transformed.shape, states.shape)
        for row, offset in enumerate(((1, 0), (0, 1))):
            single = transform_board_states(self._marker_state(offset), 1)
            np.testing.assert_array_equal(transformed[row], single)


if __name__ == "__main__":
    unittest.main()
