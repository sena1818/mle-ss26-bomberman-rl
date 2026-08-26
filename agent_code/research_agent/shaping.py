"""Potential-based reward shaping (docs/05 section 4).

The shaped reward is ``r + gamma * phi(s') - phi(s)``.  Because the extra term
is the discounted difference of a function of the *state alone*, the discounted
sum over a whole episode telescopes to ``gamma^T phi(s_T) - phi(s_0)``; with
``phi(terminal) = 0`` that is a constant which depends only on the start state.
The optimal policy is therefore unchanged (Ng, Harada & Russell 1999), and the
shaping cannot be farmed for unbounded reward by cycling.

Three constraints make that guarantee real rather than decorative, and each one
is covered by a test:

1. ``phi`` reads the game state only -- never the action, the events, or any
   history.  A term that depends on the transition is not a potential.
2. ``phi(terminal) = 0``.
3. ``phi`` uses the learner's own ``gamma`` and is bounded, so no residual
   drift survives at ``gamma < 1``.
"""

from __future__ import annotations

from .config import ExperimentConfig, shaping_specification
from .state import _bfs_distances, _crate_adjacent_cells, _DANGER_HORIZON, future_danger_times


class PotentialShaping:
    """The ``potential_v1`` potential and its transition term."""

    def __init__(self, specification: dict, discount: float):
        if specification.get("name") != "potential_v1":
            raise ValueError(f"Unknown shaping function {specification.get('name')!r}.")
        self.specification = dict(specification)
        self.discount = float(discount)
        self.coin_weight = float(specification["coin_weight"])
        self.distance_cap = int(specification["distance_cap"])
        self.danger_weight = float(specification["danger_weight"])
        self.terminal_potential = float(specification["terminal_potential"])
        if self.distance_cap < 1:
            raise ValueError("shaping.distance_cap must be positive so that phi stays bounded.")

    def potential(self, game_state: dict | None) -> float:
        """Return ``phi(s)``; a terminal state has the declared constant potential."""
        if game_state is None:
            return self.terminal_potential
        distances = _bfs_distances(game_state)
        targets = self._collection_targets(game_state)
        reachable = [distances[target] for target in targets if target in distances]
        # No reachable target is treated as the worst finite distance rather than
        # as zero, so phi never rewards running out of things to collect.
        distance = min(reachable) if reachable else self.distance_cap
        potential = -self.coin_weight * min(distance, self.distance_cap)
        if self._in_future_blast(game_state):
            potential -= self.danger_weight
        return potential

    def shaping_reward(self, potential: float, next_potential: float) -> float:
        """Return ``gamma * phi(s') - phi(s)`` for one transition."""
        return self.discount * next_potential - potential

    def _collection_targets(self, game_state: dict) -> set[tuple[int, int]]:
        """Coins if any are collectable, otherwise legal cells next to a crate.

        A crate tile itself is impassable, so its BFS distance is undefined; the
        reachable cell beside it is what the agent actually has to walk to.
        """
        coins = {tuple(coin) for coin in game_state["coins"]}
        return coins if coins else _crate_adjacent_cells(game_state)

    def _in_future_blast(self, game_state: dict) -> bool:
        danger = future_danger_times(game_state)
        return bool(danger[game_state["self"][3]] <= _DANGER_HORIZON)


def build_shaping(config: ExperimentConfig) -> PotentialShaping | None:
    """Return the shaping a reward version switches on, sharing the learner's gamma."""
    specification = shaping_specification(config.reward_version)
    if specification is None:
        return None
    return PotentialShaping(specification, config.discount)
