"""A frozen research_agent checkpoint, playable as an opponent.

Self-play needs the trained agent on the other side of the board, and the
official framework selects opponents by directory name.  Three copies of
``research_agent`` cannot do it: every ``BOMBERMAN_*`` variable is
process-global, so the opponents would read the trainer's route, and -- worse
-- ``ExperimentRuntime.select_action`` appends one record per step to
``artifact_root()/agent.jsonl``, which resolves the artifact directory at call
time.  Three opponents in the training process would interleave their steps
into the trainer's own action log, and that log is what sections 7.32, 7.34 and
7.38 are computed from.

So this is a separate directory with its own variables and no side effects at
all.  It reads:

  ``BOMBERMAN_FROZEN_EXPERIMENT``   route whose encoder and network to build
                                    (default ``R02_9``)
  ``BOMBERMAN_FROZEN_MODEL_PATH``   absolute path to the ``.npz`` to load

It writes nothing, learns nothing, and holds no per-round state.

The policy is greedy over the legal actions, which is exactly what an
evaluation job does: ``_epsilon_for_game_state`` returns 0 when ``train`` is
false and noisy layers have ``noise_enabled`` set to false there too.  Building
the same behaviour from three lines here rather than reusing the runtime is the
point -- the runtime's ``select_action`` is the thing with the side effect.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..research_agent.config import ACTIONS, EXPERIMENTS, validate_config
from ..research_agent.models import load_model
from ..research_agent.state import encode_state, legal_action_mask


# One directory per frozen seat, because the framework selects an opponent's
# code by directory name and these variables are process-global: three seats
# reading the same prefix would all play the same checkpoint.  Mirrored in
# scripts/experiment_lib.FROZEN_OPPONENT_AGENTS, with a test holding the two
# together.  frozen_agent_b and frozen_agent_c are two-line wrappers over
# ``setup_from`` and ``act``.
ENVIRONMENT_PREFIXES = {
    "frozen_agent": "BOMBERMAN_FROZEN",
    "frozen_agent_b": "BOMBERMAN_FROZEN_B",
    "frozen_agent_c": "BOMBERMAN_FROZEN_C",
}


def setup(self):
    setup_from(self, ENVIRONMENT_PREFIXES["frozen_agent"])


def setup_from(self, prefix: str):
    route = os.environ.get(f"{prefix}_EXPERIMENT", "R02_9")
    try:
        config = EXPERIMENTS[route]
    except KeyError as exc:
        raise ValueError(
            f"Unknown frozen route {route!r}; declared routes: {sorted(EXPERIMENTS)}"
        ) from exc
    selected = os.environ.get(f"{prefix}_MODEL_PATH")
    if not selected:
        raise ValueError(
            f"a frozen seat requires {prefix}_MODEL_PATH. It plays a fixed "
            "checkpoint and has no fallback: an opponent that silently played "
            "random weights would look like a weak opponent, not like a "
            "misconfiguration."
        )
    path = Path(selected).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Frozen opponent checkpoint is unavailable: {path}")
    self.config = validate_config(config)
    self.model = load_model(self.config, path)
    if getattr(self.model, "noisy", False):
        # Same definition of "greedy" the reported numbers use.
        self.model.noise_enabled = False
    features = encode_state(_EMPTY_WARM_UP_STATE, self.config.state_encoder)
    if features is not None:
        # setup is outside the 0.5 s per-step budget; pay for the first forward
        # pass and any lazy allocation here rather than on the opening move.
        self.model.q_values(features)


def act(self, game_state: dict) -> str:
    features = encode_state(game_state, self.config.state_encoder)
    if features is None:
        return "WAIT"
    values = np.asarray(self.model.q_values(features), dtype=np.float64)
    mask = np.asarray(legal_action_mask(game_state), dtype=bool)
    if not mask.any():
        return "WAIT"
    values = np.where(mask, values, -np.inf)
    return ACTIONS[int(np.argmax(values))]


# A board with nothing on it: enough for the encoder to produce a vector of the
# right width, which is all warm-up needs.
_EMPTY_WARM_UP_STATE = {
    "round": 1,
    "step": 1,
    "field": np.zeros((17, 17), dtype=np.int8),
    "bombs": [],
    "explosion_map": np.zeros((17, 17), dtype=np.int8),
    "coins": [],
    "self": ("frozen_agent", 0, True, (1, 1)),
    "others": [],
    "user_input": None,
}
