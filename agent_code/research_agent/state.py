"""State encoders, action legality, and deterministic board-derived features."""

from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np

from .config import ACTIONS, BOMB_POWER, BOMB_TIMER, FEATURE_DIMENSION, MAX_STEPS


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
    raise NotImplementedError(f"State encoder {encoder_name!r} has not been implemented yet.")


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
