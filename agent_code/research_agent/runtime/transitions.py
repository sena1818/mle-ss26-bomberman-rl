"""The runtime's transition pipeline: one held-back step, then an n-step window.

Two separate problems are solved here, in this order.

**Terminality.** The official framework decides whether a round ends *after* it
has already delivered the step through ``game_events_occurred``, so terminality
is not knowable at delivery time.  ``PendingTransition`` is one transition held
back until the next callback resolves it.

**Credit assignment.** ``NStepAssembler`` turns the resolved one-step
transitions into n-step ones.  A bomb dropped at ``t`` explodes at ``t+4``, so
the events that justify the bomb land four or five transitions later; ``n = 1``
has to propagate that through four bootstraps, while ``n = 5`` carries the real
reward back in one update.  See docs/05 section 5.3.

Every pushed one-step transition becomes the head of exactly one emitted
n-step transition, so the number of learner updates in a round still equals the
number of steps the agent acted on -- which is what the terminal-transition
regression check counts.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..learners.base import Transition


@dataclass
class PendingTransition:
    """One encoded transition held back until its terminality is known."""

    key: tuple[int, int]
    state: np.ndarray
    action_index: int
    next_state: np.ndarray | None
    next_legal_mask: np.ndarray | None
    events: list[str] = field(default_factory=list)
    # phi(s) and phi(s'), captured while the game states were still available.
    potential: float = 0.0
    next_potential: float = 0.0


class NStepAssembler:
    """Accumulate one-step transitions into n-step returns."""

    def __init__(self, n_step: int, discount: float):
        if n_step < 1:
            raise ValueError("n_step must be at least 1.")
        self.n_step = int(n_step)
        self.discount = float(discount)
        self._window: deque[Transition] = deque()

    def push(self, transition: Transition) -> list[Transition]:
        """Add one resolved one-step transition and return what is now complete."""
        if transition.n_step != 1:
            raise ValueError("NStepAssembler consumes one-step transitions only.")
        self._window.append(transition)
        if transition.terminal:
            # Nothing further will arrive, so every suffix of the window is a
            # complete return that needs no bootstrap.
            return self._drain()
        if len(self._window) < self.n_step:
            return []
        emitted = self._aggregate(0, self.n_step)
        self._window.popleft()
        return [emitted]

    def flush(self) -> list[Transition]:
        """Emit the shorter returns left over when a round ends without a terminal.

        A truncated round is still bootstrapped, so each remaining window keeps
        the last observed next state and its own, shorter, discount exponent.
        """
        return self._drain()

    def reset(self) -> None:
        self._window.clear()

    def pending_count(self) -> int:
        """How many one-step transitions are still waiting to be emitted."""
        return len(self._window)

    def _drain(self) -> list[Transition]:
        emitted = [self._aggregate(start, len(self._window) - start) for start in range(len(self._window))]
        self._window.clear()
        return emitted

    def _aggregate(self, start: int, length: int) -> Transition:
        head = self._window[start]
        tail = self._window[start + length - 1]
        discounted_return = 0.0
        for offset in range(length):
            discounted_return += (self.discount ** offset) * self._window[start + offset].reward
        return Transition(
            state=head.state,
            action_index=head.action_index,
            reward=discounted_return,
            next_state=None if tail.terminal else tail.next_state,
            next_legal_mask=None if tail.terminal else tail.next_legal_mask,
            terminal=tail.terminal,
            n_step=length,
        )
