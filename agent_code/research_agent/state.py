"""State encoders, action legality, and deterministic board-derived features."""

from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np

from .config import (
    ACTIONS, BOMB_POWER, BOMB_TIMER, EXPLOSION_TIMER, FEATURE_DIMENSION,
    FEATURE_DIMENSION_V2, FEATURE_DIMENSION_V3, MAX_STEPS,
)


_DIRECTIONS = {
    "UP": (0, -1),
    "RIGHT": (1, 0),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
}
_BLAST_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_DANGER_HORIZON = BOMB_TIMER

# Stable slices make the 44-dimensional contract inspectable in tests and in
# experiment notebooks without leaking a recommended action into the state.
HANDCRAFTED_V1_LAYOUT = {
    "own_global": slice(0, 5),
    "legal_actions": slice(5, 11),
    "danger_current_and_neighbors": slice(11, 21),
    "coin_target": slice(21, 25),
    "crate_target": slice(25, 29),
    "opponent_target": slice(29, 33),
    "nearest_bomb": slice(33, 38),
    "local_counts": slice(38, 42),
    "bomb_escape": slice(42, 44),
}

# handcrafted_v2 keeps every v1 entry at its original index and appends what the
# escape diagnosis found missing (docs/01 section 7.10): v1 forecasts danger one
# step ahead only, so two neighbouring cells that are both inside a blast look
# identical even when one leads out and the other dead-ends.  In 94.4% of the
# turns that lost a survivable escape, the fatal direction and a saving one were
# bit-for-bit equal in ``danger_current_and_neighbors``.  These entries carry the
# multi-step answer instead.  ``bomb_escape`` is left untouched: it answers "may
# I bomb here", is zero once a bomb is down, and changing it would silently
# redefine a block every earlier arm was trained on.
HANDCRAFTED_V2_LAYOUT = {
    **HANDCRAFTED_V1_LAYOUT,
    "escape_here": slice(44, 46),
    "escape_by_direction": slice(46, 54),
}

# handcrafted_v3 adds the routing answer on top of v2.  ``_target_features``
# encodes ``sign(tx - x), sign(ty - y)`` -- a compass bearing, not a route -- and
# in 16.3% of steps every direction that bearing suggests is wrong, because the
# corridor turns.  These entries say which legal move actually shortens the walk.
HANDCRAFTED_V3_LAYOUT = {
    **HANDCRAFTED_V2_LAYOUT,
    "coin_route": slice(54, 58),
    "crate_route": slice(58, 62),
}


# The spatial contract used by the M4 line.  The window is odd-sized and always
# centred on the agent, so a translation of the whole board is invisible to the
# network and the eight board symmetries act as a group on it.  Radius 8 makes
# the window 17x17, which covers the entire official arena from any cell once
# the outside is padded with stone.
EGOCENTRIC_RADIUS = 8
EGOCENTRIC_WINDOW = 2 * EGOCENTRIC_RADIUS + 1
BOARD_CHANNELS = (
    "stone_walls",
    "crates",
    "self",
    "opponents",
    "coins",
    "bomb_countdown",
    "explosions",
    "future_danger",
)
BOARD_EGOCENTRIC_LAYOUT = {
    "board_shape": (len(BOARD_CHANNELS), EGOCENTRIC_WINDOW, EGOCENTRIC_WINDOW),
    "channels": BOARD_CHANNELS,
    "global_dimension": 4,
    "globals": ("can_bomb", "round_progress", "own_score", "living_opponents"),
    "flat_order": "board channels flattened in C order, then the global scalars",
}


# ``board_egocentric_v2``: the M4 line's actual representation.  Two changes
# from v1, both of them provable rather than hypothesised.
#
# 1. The v1 ``self`` plane is gone.  ``_egocentric_crop`` centres the window on
#    the agent, so that plane is the *same* constant array -- one 1 at the
#    centre -- for every state a v1 agent will ever see.  289 inputs carrying
#    zero information is not a question that needs an ablation.
#
# 2. Two global scalars are added.  "How much of the board is still crates" and
#    "how many coins are lying about" are what tells an agent that farming is
#    finished and hunting is all that is left; neither is recoverable from a
#    17x17 window plus the v1 four.
#
# An ``own_bomb`` channel is the obvious use for the freed plane and is
# deliberately NOT here: the official state reports a bomb as ``((x, y), timer)``
# with no owner (``environment.py:406``), so ownership is not a function of the
# observed state.  Reconstructing it from the agent's own action history would
# make the encoder stateful, and the diagnostics import these functions exactly
# so that what the agent sees and what judges it cannot drift apart.  It would
# also buy nothing in solo training, where every bomb is already the agent's.
BOARD_CHANNELS_V2 = (
    "stone_walls",
    "crates",
    "opponents",
    "coins",
    "bomb_countdown",
    "explosions",
    "future_danger",
)
# Every board channel value is a multiple of 1/20: the binary planes are 0 or 1,
# ``bomb_countdown`` steps by 1/BOMB_TIMER = 1/4 and ``future_danger`` by
# 1/(BOMB_TIMER+1) = 1/5.  That makes the whole board exactly representable as
# uint8 codes, which is what lets a 100k-transition replay of these states fit
# in memory (see ``replay.ReplayBuffer``).  The property is asserted by a test
# over real encoded states rather than trusted.
BOARD_QUANTISATION = 20
# A normaliser, not a claim about the arena: the coin-rich scenarios place 50.
MAX_VISIBLE_COINS = 50
BOARD_EGOCENTRIC_V2_LAYOUT = {
    "board_shape": (len(BOARD_CHANNELS_V2), EGOCENTRIC_WINDOW, EGOCENTRIC_WINDOW),
    "channels": BOARD_CHANNELS_V2,
    "global_dimension": 6,
    "globals": (
        "can_bomb",
        "round_progress",
        "own_score",
        "living_opponents",
        "visible_coins",
        "crate_fraction",
    ),
    "quantisation": BOARD_QUANTISATION,
    "flat_order": "board channels flattened in C order, then the global scalars",
}


def legal_action_mask(game_state: dict) -> np.ndarray:
    """Return which of the six actions are legal in the current observed state.

    This only filters actions that are immediately invalid. It deliberately does
    not make strategic or safety decisions for the learning agent.
    """
    if game_state is None:
        return np.zeros(len(ACTIONS), dtype=bool)

    field = game_state["field"]
    _, _, can_bomb, (x, y) = game_state["self"]
    blocked = {position for position, _ in game_state["bombs"]}
    blocked.update(other[3] for other in game_state["others"])

    mask = np.zeros(len(ACTIONS), dtype=bool)
    for index, action in enumerate(ACTIONS):
        if action in _DIRECTIONS:
            dx, dy = _DIRECTIONS[action]
            nx, ny = x + dx, y + dy
            in_bounds = 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]
            mask[index] = in_bounds and field[nx, ny] == 0 and (nx, ny) not in blocked
        elif action == "WAIT":
            mask[index] = True
        else:  # BOMB
            mask[index] = can_bomb
    return mask


def encode_state(game_state: dict, encoder_name: str) -> np.ndarray | None:
    """Route a configured state representation to its encoder."""
    if game_state is None:
        return None
    if encoder_name == "handcrafted_v1":
        return handcrafted_v1(game_state)
    if encoder_name == "handcrafted_v2":
        return handcrafted_v2(game_state)
    if encoder_name == "handcrafted_v3":
        return handcrafted_v3(game_state)
    if encoder_name == "board_egocentric_v1":
        return board_egocentric_v1(game_state)
    if encoder_name == "board_egocentric_v2":
        return board_egocentric_v2(game_state)
    raise NotImplementedError(f"State encoder {encoder_name!r} has not been implemented yet.")


def state_dimension(encoder_name: str) -> int:
    """Return the flat length one encoder produces, without a game state."""
    if encoder_name == "handcrafted_v1":
        return FEATURE_DIMENSION
    if encoder_name == "handcrafted_v2":
        return FEATURE_DIMENSION_V2
    if encoder_name == "handcrafted_v3":
        return FEATURE_DIMENSION_V3
    if encoder_name in BOARD_LAYOUTS:
        layout = BOARD_LAYOUTS[encoder_name]
        channels, width, height = layout["board_shape"]
        return channels * width * height + layout["global_dimension"]
    raise NotImplementedError(f"State encoder {encoder_name!r} has not been implemented yet.")


def escape_search(
    game_state: dict,
    origin: tuple[int, int] | None = None,
    horizon: int = BOMB_TIMER + 1,
) -> tuple[bool, int | None]:
    """Can the agent stand outside every current blast in time, and how fast?

    The search is over ``(cell, tick)`` rather than plain distance, because a
    cell that is reachable in three steps is useless if it detonates on the
    second.  A tile holding a bomb is not enterable, so the tile a bomb was just
    dropped on does not count as a way through.

    ``scripts/diagnose_bomb_escape.py`` measures the same question about a
    finished run and imports this function, so the feature the agent sees and
    the diagnostic that judges it can never drift apart.
    """
    field = game_state["field"]
    bombs = list(game_state["bombs"])
    if not bombs:
        return True, 0
    start = tuple(origin if origin is not None else game_state["self"][3])
    blocked = {tuple(position) for position, _ in bombs}
    blocked.update(tuple(other[3]) for other in game_state["others"])

    lethal: dict[tuple[int, int], set[int]] = {}
    in_any_blast: set[tuple[int, int]] = set()
    for position, timer in bombs:
        detonation = max(int(timer) + 1, 1)
        for cell in _blast_coordinates(tuple(position), field):
            in_any_blast.add(cell)
            ticks = lethal.setdefault(cell, set())
            for offset in range(EXPLOSION_TIMER):
                ticks.add(detonation + offset)

    if start not in in_any_blast:
        return True, 0
    seen = {(start, 0)}
    queue = deque([(start, 0)])
    while queue:
        cell, tick = queue.popleft()
        if tick >= horizon:
            continue
        for nxt in (cell, *((cell[0] + dx, cell[1] + dy) for dx, dy in _DIRECTIONS.values())):
            if nxt != cell and not _is_free(nxt, field, blocked):
                continue
            step = tick + 1
            if step in lethal.get(nxt, ()):
                continue
            if nxt not in in_any_blast:
                return True, step
            if (nxt, step) in seen:
                continue
            seen.add((nxt, step))
            queue.append((nxt, step))
    return False, None


def _escape_features(game_state: dict) -> np.ndarray:
    """Ten entries saying where safety is, from here and from each neighbour.

    Layout: ``[reachable_from_here, ticks_from_here]`` then, per direction in
    ``_DIRECTIONS`` order, ``[reachable_after_stepping, ticks_after_stepping]``.
    Ticks are scaled by the bomb timer so every entry stays in ``[0, 1]``, and an
    unreachable direction is ``(0, 1)``: not reachable, worst possible cost.
    Blocked directions are ``(0, 1)`` too -- a wall is no way out either.
    """
    field = game_state["field"]
    origin = tuple(game_state["self"][3])
    blocked = {tuple(position) for position, _ in game_state["bombs"]}
    blocked.update(tuple(other[3]) for other in game_state["others"])
    scale = float(BOMB_TIMER + 1)

    def encode(reachable: bool, ticks: int | None) -> list[float]:
        if not reachable or ticks is None:
            return [0.0, 1.0]
        return [1.0, min(ticks, BOMB_TIMER + 1) / scale]

    values = encode(*escape_search(game_state))
    for dx, dy in _DIRECTIONS.values():
        cell = (origin[0] + dx, origin[1] + dy)
        if not _is_free(cell, field, blocked):
            values.extend([0.0, 1.0])
            continue
        values.extend(encode(*escape_search(game_state, origin=cell)))
    return np.asarray(values, dtype=np.float32)


def _route_features(game_state: dict) -> np.ndarray:
    """Eight entries: which legal move actually shortens the walk to a target.

    Four for the nearest reachable coin, four for the nearest reachable cell
    beside a crate, in ``_DIRECTIONS`` order.  The distances come from a BFS run
    backwards from the target, so a direction scores 1.0 exactly when stepping
    there lies on some shortest route.  With no reachable target the block is
    zero, which is also what ``_target_features`` reports in that case.
    """
    field = game_state["field"]
    origin = tuple(game_state["self"][3])
    blocked = {tuple(position) for position, _ in game_state["bombs"]}
    blocked.update(tuple(other[3]) for other in game_state["others"])
    forward = _bfs_distances(game_state)

    def gradient(targets: set[tuple[int, int]]) -> list[float]:
        reachable = [target for target in targets if target in forward]
        if not reachable:
            return [0.0] * len(_DIRECTIONS)
        goal = min(reachable, key=forward.__getitem__)
        backward = {goal: 0}
        queue = deque([goal])
        while queue:
            cell = queue.popleft()
            for dx, dy in _DIRECTIONS.values():
                nxt = (cell[0] + dx, cell[1] + dy)
                if nxt in backward or not _is_free(nxt, field, blocked):
                    continue
                backward[nxt] = backward[cell] + 1
                queue.append(nxt)
        here = backward.get(origin)
        if here is None:
            return [0.0] * len(_DIRECTIONS)
        values = []
        for dx, dy in _DIRECTIONS.values():
            cell = (origin[0] + dx, origin[1] + dy)
            values.append(1.0 if _is_free(cell, field, blocked) and backward.get(cell, here) < here else 0.0)
        return values

    coins = {tuple(coin) for coin in game_state["coins"]}
    return np.asarray(gradient(coins) + gradient(_crate_adjacent_cells(game_state)), dtype=np.float32)


def handcrafted_v2(game_state: dict) -> np.ndarray:
    """v1 plus the multi-step escape answer; see ``HANDCRAFTED_V2_LAYOUT``."""
    features = np.concatenate([handcrafted_v1(game_state), _escape_features(game_state)])
    assert features.shape == (FEATURE_DIMENSION_V2,)
    return features.astype(np.float32)


def handcrafted_v3(game_state: dict) -> np.ndarray:
    """v2 plus the routing answer; see ``HANDCRAFTED_V3_LAYOUT``."""
    features = np.concatenate([handcrafted_v2(game_state), _route_features(game_state)])
    assert features.shape == (FEATURE_DIMENSION_V3,)
    return features.astype(np.float32)


def handcrafted_v1(game_state: dict) -> np.ndarray:
    """The agreed 44-dimensional feature vector for R01--R04.

    Feature groups are: own/global state (5), immediate legality (6), forecast
    danger for the current and four adjacent cells (10), reachable coin/crate/
    opponent targets (12), closest bomb (5), local entity counts (4), and
    bomb-escape capability (2).  No entry encodes a recommended action.
    """
    _, _, can_bomb, (x, y) = game_state["self"]
    field = game_state["field"]
    danger = future_danger_times(game_state)
    own = np.array(
        [
            x / (field.shape[0] - 1),
            y / (field.shape[1] - 1),
            float(can_bomb),
            min(game_state["step"], MAX_STEPS) / MAX_STEPS,
            len(game_state["others"]) / 3.0,
        ],
        dtype=np.float32,
    )
    legal = legal_action_mask(game_state).astype(np.float32)

    danger_positions = [(x, y)] + [(x + dx, y + dy) for dx, dy in _DIRECTIONS.values()]
    danger_features = np.concatenate([_danger_features(pos, danger) for pos in danger_positions])

    distances = _bfs_distances(game_state)
    coin = _target_features((x, y), set(game_state["coins"]), distances, field.shape)
    crate = _target_features((x, y), _crate_adjacent_cells(game_state), distances, field.shape)
    opponent = _target_features((x, y), _opponent_adjacent_cells(game_state), distances, field.shape)
    bomb = _nearest_target_features((x, y), [pos for pos, _ in game_state["bombs"]], include_timer=True,
                                    timed_items=game_state["bombs"])
    local = _local_counts(game_state, radius=2)
    escape = _bomb_escape_features(game_state, danger, distances)

    features = np.concatenate(
        [own, legal, danger_features, coin, crate, opponent, bomb, local, escape]
    ).astype(np.float32)
    assert features.shape == (FEATURE_DIMENSION,)
    return features


def encode_board_channels_v1(game_state: dict) -> np.ndarray:
    """Return the already-agreed spatial representation; R01 never calls it.

    This is a state-definition boundary only, not a CNN implementation.  It is
    retained so later spatial experiments use the frozen channel contract.
    """
    field = game_state["field"]
    channels = np.zeros((8, *field.shape), dtype=np.float32)
    channels[0] = field == -1  # stone walls
    channels[1] = field == 1   # crates
    _, _, _, own_pos = game_state["self"]
    channels[2][own_pos] = 1.0
    for other in game_state["others"]:
        channels[3][other[3]] = 1.0
    for coin in game_state["coins"]:
        channels[4][coin] = 1.0
    for position, timer in game_state["bombs"]:
        channels[5][position] = 1.0 - min(max(timer, 0), BOMB_TIMER) / BOMB_TIMER
    channels[6] = np.clip(game_state["explosion_map"], 0.0, 1.0)
    danger = future_danger_times(game_state)
    channels[7] = np.where(
        danger <= _DANGER_HORIZON,
        (_DANGER_HORIZON + 1 - danger) / (_DANGER_HORIZON + 1),
        0.0,
    )
    return channels


def board_egocentric_v1(game_state: dict) -> np.ndarray:
    """Return the M4 state: an agent-centred board tensor plus global scalars.

    The eight board channels are the frozen ``encode_board_channels_v1``
    contract; the four scalars are the frozen ``global_features_v1`` contract.
    The only new thing here is the framing: the board is padded with stone and
    cropped around the agent, so the agent always sits at the window centre.

    The result is flat because ``Transition.state`` is one vector for every
    route.  The shape is public through ``BOARD_EGOCENTRIC_LAYOUT`` and is
    restored by ``split_board_and_globals``; only the CNN model needs it.
    """
    channels = encode_board_channels_v1(game_state)
    window = _egocentric_crop(channels, game_state["self"][3])
    return np.concatenate([window.reshape(-1), global_features_v1(game_state)]).astype(np.float32)


BOARD_LAYOUTS = {
    "board_egocentric_v1": BOARD_EGOCENTRIC_LAYOUT,
    "board_egocentric_v2": BOARD_EGOCENTRIC_V2_LAYOUT,
}


def layout_for_dimension(dimension: int) -> dict:
    """Return the board layout that produces flat states of this length.

    The two spatial representations have different lengths (2316 and 2029), so
    one number identifies one layout.  Inferring it here rather than passing it
    down means the CNN adapter, the replay buffer and the D4 transforms all
    stay layout-agnostic instead of each growing a version switch.
    """
    for layout in BOARD_LAYOUTS.values():
        if int(np.prod(layout["board_shape"])) + layout["global_dimension"] == dimension:
            return layout
    known = ", ".join(str(int(np.prod(one["board_shape"])) + one["global_dimension"])
                      for one in BOARD_LAYOUTS.values())
    raise ValueError(f"No board layout produces a state of length {dimension}; known lengths are {known}.")


def encode_board_channels_v2(game_state: dict) -> np.ndarray:
    """Return the seven v2 board channels: the v1 set without the dead plane.

    Every channel except ``opponents`` is bit-identical to its v1 counterpart;
    only the ``self`` plane is dropped.  See ``BOARD_CHANNELS_V2`` for why that
    is a deletion rather than a replacement.
    """
    field = game_state["field"]
    channels = np.zeros((len(BOARD_CHANNELS_V2), *field.shape), dtype=np.float32)
    channels[0] = field == -1  # stone walls
    channels[1] = field == 1   # crates
    for other in game_state["others"]:
        channels[2][other[3]] = 1.0
    for coin in game_state["coins"]:
        channels[3][coin] = 1.0
    for position, timer in game_state["bombs"]:
        channels[4][position] = 1.0 - min(max(timer, 0), BOMB_TIMER) / BOMB_TIMER
    channels[5] = np.clip(game_state["explosion_map"], 0.0, 1.0)
    danger = future_danger_times(game_state)
    channels[6] = np.where(
        danger <= _DANGER_HORIZON,
        (_DANGER_HORIZON + 1 - danger) / (_DANGER_HORIZON + 1),
        0.0,
    )
    return channels


def global_features_v2(game_state: dict) -> np.ndarray:
    """Return the six v2 scalars: the v1 four plus two board-depletion terms.

    ``visible_coins`` counts what is on the floor now, which is all the state
    reports -- an agent cannot know how many coins a scenario started with.
    ``crate_fraction`` is measured against the arena's free cells, so it is a
    ratio the agent can actually compute rather than one that needs the
    scenario's parameters.
    """
    field = game_state["field"]
    free_cells = int(np.count_nonzero(field != -1))
    _, score, can_bomb, _ = game_state["self"]
    return np.array(
        [
            float(can_bomb),
            min(game_state["step"], MAX_STEPS) / MAX_STEPS,
            np.clip(score / 10.0, 0.0, 1.0),
            len(game_state["others"]) / 3.0,
            min(len(game_state["coins"]), MAX_VISIBLE_COINS) / MAX_VISIBLE_COINS,
            int(np.count_nonzero(field == 1)) / max(free_cells, 1),
        ],
        dtype=np.float32,
    )


def board_egocentric_v2(game_state: dict) -> np.ndarray:
    """Return the M4 state: seven agent-centred planes plus six global scalars.

    The framing is exactly ``board_egocentric_v1``'s -- stone-padded crop
    centred on the agent -- so the D4 group acts on it the same way.
    """
    channels = encode_board_channels_v2(game_state)
    window = _egocentric_crop(channels, game_state["self"][3])
    return np.concatenate([window.reshape(-1), global_features_v2(game_state)]).astype(np.float32)


def quantised_board_spec(encoder_name: str) -> tuple[int, int]:
    """Return ``(board prefix length, steps per unit)`` for uint8 replay storage.

    ``(0, 0)`` means the representation has no such grid, which is the case for
    every handcrafted vector: its entries are BFS distances and ratios, not
    channel levels.  Only a layout that declares ``quantisation`` opts in.
    """
    layout = BOARD_LAYOUTS.get(encoder_name)
    if layout is None or "quantisation" not in layout:
        return 0, 0
    return int(np.prod(layout["board_shape"])), int(layout["quantisation"])


def split_board_and_globals(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Restore the (channels, width, height) board and the global scalars.

    Accepts a single flat state or a batch of them; the board keeps a leading
    batch axis exactly when the input had one.  Which spatial layout the state
    belongs to follows from its length -- see ``layout_for_dimension``.
    """
    layout = layout_for_dimension(int(state.shape[-1]))
    shape = layout["board_shape"]
    board_size = int(np.prod(shape))
    board = state[..., :board_size].reshape(*state.shape[:-1], *shape)
    globals_ = state[..., board_size:]
    return board, globals_


def _egocentric_crop(channels: np.ndarray, origin: tuple[int, int]) -> np.ndarray:
    """Crop a fixed window around ``origin``, padding outside with stone wall.

    Padding with the stone-wall channel set (and everything else zero) is the
    honest completion: a cell beyond the arena is exactly as impassable as a
    wall, so the network never sees a padded cell as open floor.
    """
    padded = np.pad(channels, ((0, 0), (EGOCENTRIC_RADIUS,) * 2, (EGOCENTRIC_RADIUS,) * 2))
    padded[0, :EGOCENTRIC_RADIUS, :] = 1.0
    padded[0, -EGOCENTRIC_RADIUS:, :] = 1.0
    padded[0, :, :EGOCENTRIC_RADIUS] = 1.0
    padded[0, :, -EGOCENTRIC_RADIUS:] = 1.0
    x, y = origin
    return padded[:, x:x + EGOCENTRIC_WINDOW, y:y + EGOCENTRIC_WINDOW]


def global_features_v1(game_state: dict) -> np.ndarray:
    """Return the agreed four non-spatial values; R01 never calls them."""
    _, score, can_bomb, _ = game_state["self"]
    return np.array(
        [
            float(can_bomb),
            min(game_state["step"], MAX_STEPS) / MAX_STEPS,
            np.clip(score / 10.0, 0.0, 1.0),
            len(game_state["others"]) / 3.0,
        ],
        dtype=np.float32,
    )


def future_danger_times(game_state: dict) -> np.ndarray:
    """Earliest known dangerous time per tile, capped outside the short horizon.

    This mirrors the SS26 framework's current `Bomb.get_blast_coords`: stone
    walls stop a blast, whereas crates are destroyed but do not stop it. The
    public framework has no bomb-chain reaction in this code path.  A timer of
    zero explodes after the action currently being chosen, hence danger time 1.
    """
    field = game_state["field"]
    danger = np.full(field.shape, _DANGER_HORIZON + 1, dtype=np.int8)
    danger[np.asarray(game_state["explosion_map"]) > 0] = 0
    for position, timer in game_state["bombs"]:
        # Agents choose an action before this observed timer is processed, so a
        # timer of zero becomes dangerous on the next transition.
        time_to_blast = min(max(timer + 1, 1), _DANGER_HORIZON + 1)
        for blast_pos in _blast_coordinates(position, field):
            danger[blast_pos] = min(danger[blast_pos], time_to_blast)
    return danger


def _danger_features(position: tuple[int, int], danger: np.ndarray) -> np.ndarray:
    x, y = position
    if not (0 <= x < danger.shape[0] and 0 <= y < danger.shape[1]):
        return np.array([1.0, 0.0], dtype=np.float32)
    time_to_danger = int(danger[x, y])
    is_safe = float(time_to_danger > _DANGER_HORIZON)
    urgency = 0.0 if is_safe else (_DANGER_HORIZON + 1 - time_to_danger) / (_DANGER_HORIZON + 1)
    return np.array([urgency, is_safe], dtype=np.float32)


def _bfs_distances(game_state: dict) -> dict[tuple[int, int], int]:
    field = game_state["field"]
    origin = game_state["self"][3]
    blocked = {position for position, _ in game_state["bombs"]}
    blocked.update(other[3] for other in game_state["others"])
    distances = {origin: 0}
    queue = deque([origin])
    while queue:
        x, y = queue.popleft()
        for dx, dy in _DIRECTIONS.values():
            nxt = (x + dx, y + dy)
            if nxt in distances or not _is_free(nxt, field, blocked):
                continue
            distances[nxt] = distances[(x, y)] + 1
            queue.append(nxt)
    return distances


def _target_features(
    origin: tuple[int, int],
    targets: set[tuple[int, int]],
    distances: dict[tuple[int, int], int],
    shape: tuple[int, int],
) -> np.ndarray:
    reachable = [target for target in targets if target in distances]
    if not reachable:
        return np.zeros(4, dtype=np.float32)
    tx, ty = min(reachable, key=distances.__getitem__)
    x, y = origin
    return np.array(
        [
            np.sign(tx - x),
            np.sign(ty - y),
            distances[(tx, ty)] / sum(shape),
            1.0,
        ],
        dtype=np.float32,
    )


def _crate_adjacent_cells(game_state: dict) -> set[tuple[int, int]]:
    field = game_state["field"]
    blocked = {position for position, _ in game_state["bombs"]}
    blocked.update(other[3] for other in game_state["others"])
    targets = set()
    for x, y in zip(*np.where(field == 1)):
        for dx, dy in _DIRECTIONS.values():
            candidate = (int(x + dx), int(y + dy))
            if _is_free(candidate, field, blocked):
                targets.add(candidate)
    return targets


def _opponent_adjacent_cells(game_state: dict) -> set[tuple[int, int]]:
    field = game_state["field"]
    blocked = {position for position, _ in game_state["bombs"]}
    blocked.update(other[3] for other in game_state["others"])
    targets = set()
    for other in game_state["others"]:
        x, y = other[3]
        for dx, dy in _DIRECTIONS.values():
            candidate = (x + dx, y + dy)
            if _is_free(candidate, field, blocked):
                targets.add(candidate)
    return targets


def _local_counts(game_state: dict, radius: int) -> np.ndarray:
    x, y = game_state["self"][3]
    field = game_state["field"]
    in_radius = lambda pos: abs(pos[0] - x) + abs(pos[1] - y) <= radius
    area = 1 + 2 * radius * (radius + 1)
    return np.array(
        [
            np.count_nonzero([(field[i, j] == 1) for i in range(field.shape[0]) for j in range(field.shape[1])
                              if abs(i - x) + abs(j - y) <= radius]) / area,
            sum(in_radius(coin) for coin in game_state["coins"]) / area,
            sum(in_radius(position) for position, _ in game_state["bombs"]) / area,
            sum(in_radius(other[3]) for other in game_state["others"]) / area,
        ],
        dtype=np.float32,
    )


def _bomb_escape_features(
    game_state: dict,
    danger: np.ndarray,
    distances: dict[tuple[int, int], int],
) -> np.ndarray:
    """Describe escape capacity after a bomb; it does not prescribe bombing."""
    _, _, can_bomb, origin = game_state["self"]
    if not can_bomb:
        return np.zeros(2, dtype=np.float32)
    blast = set(_blast_coordinates(origin, game_state["field"]))
    safe_cells = [
        position for position, distance in distances.items()
        if 1 <= distance <= BOMB_TIMER and position not in blast and danger[position] > distance
    ]
    return np.array([float(bool(safe_cells)), min(len(safe_cells), 4) / 4.0], dtype=np.float32)


def _nearest_target_features(
    origin: tuple[int, int],
    targets: Iterable[tuple[int, int]],
    *,
    include_timer: bool,
    timed_items: Iterable[tuple[tuple[int, int], int]] = (),
) -> np.ndarray:
    targets = list(targets)
    if not targets:
        return np.zeros(5 if include_timer else 4, dtype=np.float32)

    x, y = origin
    tx, ty = min(targets, key=lambda pos: abs(pos[0] - x) + abs(pos[1] - y))
    distance = abs(tx - x) + abs(ty - y)
    values = [np.sign(tx - x), np.sign(ty - y), distance / 32.0, 1.0]
    if include_timer:
        timers = dict(timed_items)
        values.append(timers[(tx, ty)] / BOMB_TIMER)
    return np.asarray(values, dtype=np.float32)


def _blast_coordinates(origin: tuple[int, int], field: np.ndarray) -> list[tuple[int, int]]:
    """Match the official fixed-power blast geometry for the current framework."""
    x, y = origin
    coordinates = [origin]
    # Keep the framework's order too.  It does not affect gameplay, but makes
    # this helper exactly comparable with Bomb.get_blast_coords in unit tests.
    for dx, dy in _BLAST_DIRECTIONS:
        for distance in range(1, BOMB_POWER + 1):
            nx, ny = x + dx * distance, y + dy * distance
            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]) or field[nx, ny] == -1:
                break
            coordinates.append((nx, ny))
    return coordinates


def _is_free(position: tuple[int, int], field: np.ndarray, blocked: set[tuple[int, int]]) -> bool:
    x, y = position
    return 0 <= x < field.shape[0] and 0 <= y < field.shape[1] and field[x, y] == 0 and position not in blocked
