"""Reserved replay-buffer boundary for R04+; R01 does not need a buffer."""

from collections import deque


class ReplayBuffer:
    def __init__(self, capacity: int):
        self._items = deque(maxlen=capacity)

    def append(self, transition: tuple) -> None:
        self._items.append(transition)

    def __len__(self) -> int:
        return len(self._items)
