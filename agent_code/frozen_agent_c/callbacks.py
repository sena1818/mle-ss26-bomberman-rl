"""A second frozen seat: ``frozen_agent`` reading the ``BOMBERMAN_FROZEN_C_*`` variables.

A mixed table seats several frozen checkpoints at once, and the framework
picks an opponent's code by directory name while the checkpoint comes from
process-global variables.  So each seat is its own directory and its own
prefix; the policy is ``frozen_agent``'s, unchanged.
"""

from __future__ import annotations

from ..frozen_agent.callbacks import ENVIRONMENT_PREFIXES, act, setup_from


def setup(self):
    setup_from(self, ENVIRONMENT_PREFIXES["frozen_agent_c"])


__all__ = ("setup", "act")
