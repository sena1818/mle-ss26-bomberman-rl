"""The eight board symmetries and how they act on states, actions and masks.

Bomberman's rules are invariant under the dihedral group D4 -- four rotations
and their mirror images -- provided the state is agent-centred, so that a
rotation of the board is not also a translation of the agent.  Only
``board_egocentric_v1`` satisfies that, which is why ``config.validate_config``
refuses D4 augmentation for the handcrafted representation.

Everything here is derived from one place: ``_DIRECTION_TRANSFORMS``.  The array
operations and the action permutation cannot drift apart, because the
permutation is computed from the same coordinate map the arrays use.
"""

from __future__ import annotations

import numpy as np

from .config import ACTIONS
from .state import BOARD_EGOCENTRIC_LAYOUT, split_board_and_globals


# Direction of each movement action in the framework's (x, y) convention, where
# y grows downwards.  WAIT and BOMB have no direction and are fixed points.
_ACTION_DIRECTIONS = {"UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1), "LEFT": (-1, 0)}


def _rotate(vector: tuple[int, int], quarter_turns: int) -> tuple[int, int]:
    """Apply ``np.rot90(..., axes=(x, y))`` to one offset from the centre.

    ``np.rot90`` with axes ``(0, 1)`` moves the element at ``(x, y)`` to
    ``(-y, x)`` measured from the centre of an odd-sized window; this is the
    same map, written for a single vector.
    """
    x, y = vector
    for _ in range(quarter_turns % 4):
        x, y = -y, x
    return x, y


def _mirror(vector: tuple[int, int]) -> tuple[int, int]:
    """Apply a flip along the first axis, matching ``board[::-1, :]``."""
    x, y = vector
    return -x, y


def _direction_transform(index: int):
    quarter_turns, mirrored = index % 4, index >= 4

    def transform(vector: tuple[int, int]) -> tuple[int, int]:
        rotated = _rotate(vector, quarter_turns)
        return _mirror(rotated) if mirrored else rotated

    return transform


D4_ORDER = 8
_DIRECTION_TRANSFORMS = tuple(_direction_transform(index) for index in range(D4_ORDER))


def _action_permutation(index: int) -> np.ndarray:
    """Return ``permutation[a]``: the action ``a`` becomes under transform ``index``."""
    transform = _DIRECTION_TRANSFORMS[index]
    directions = {_ACTION_DIRECTIONS[action]: position
                  for position, action in enumerate(ACTIONS) if action in _ACTION_DIRECTIONS}
    permutation = np.arange(len(ACTIONS))
    for position, action in enumerate(ACTIONS):
        if action in _ACTION_DIRECTIONS:
            permutation[position] = directions[transform(_ACTION_DIRECTIONS[action])]
    return permutation


ACTION_PERMUTATIONS = np.stack([_action_permutation(index) for index in range(D4_ORDER)])


def transform_board_states(states: np.ndarray, index: int) -> np.ndarray:
    """Return a batch of flat egocentric states under board symmetry ``index``.

    The global scalars are untouched: none of them refer to a direction.
    """
    if index == 0:
        return states
    board, globals_ = split_board_and_globals(np.atleast_2d(states))
    rotated = np.rot90(board, k=index % 4, axes=(2, 3))
    if index >= 4:
        rotated = rotated[:, :, ::-1, :]
    flat = rotated.reshape(board.shape[0], -1)
    result = np.concatenate([flat, globals_], axis=1).astype(states.dtype, copy=False)
    return result.reshape(states.shape)


def transform_action_indices(action_indices: np.ndarray, index: int) -> np.ndarray:
    return ACTION_PERMUTATIONS[index][action_indices]


def transform_legal_masks(masks: np.ndarray, index: int) -> np.ndarray:
    """Permute action masks so that entry ``permutation[a]`` holds mask ``a``."""
    permuted = np.empty_like(masks)
    permuted[..., ACTION_PERMUTATIONS[index]] = masks
    return permuted


def board_size() -> int:
    return int(np.prod(BOARD_EGOCENTRIC_LAYOUT["board_shape"]))
