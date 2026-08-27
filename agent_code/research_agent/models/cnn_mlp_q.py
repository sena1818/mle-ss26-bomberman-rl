"""The M4 line's convolutional QModel: a CNN board branch plus a scalar branch.

This is the only adapter that depends on PyTorch.  The dependency is contained
here and imported lazily by ``models.build_model``, so the M1--M3 lines keep
running in a NumPy-only environment.  The course image already installs
PyTorch (see the repository ``Dockerfile``).

Two variants share this file because they differ in exactly one place -- the
head.  ``dueling=False`` is the anchor required by docs/05 section 5.4;
``dueling=True`` adds the value/advantage split as a single declared increment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from ..config import ACTIONS
from ..state import layout_for_dimension, split_board_and_globals


_TORCH_CONFIGURED = False
_DEVICE = None


def _torch():
    """Import PyTorch on demand with an actionable message when it is absent.

    The experiment runner parallelises across *processes*, so a torch instance
    that helpfully spreads one small batch over every core turns 8 concurrent
    jobs into heavy contention.  One thread per process is the right default
    here; ``BOMBERMAN_TORCH_THREADS`` overrides it for a single-job benchmark.
    """
    global _TORCH_CONFIGURED
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "The M4 routes (cnn_mlp_q, dueling_cnn_mlp_q) need PyTorch. "
            "Install it in the training environment; the project Dockerfile already does."
        ) from exc
    if not _TORCH_CONFIGURED:
        torch.set_num_threads(max(1, int(os.environ.get("BOMBERMAN_TORCH_THREADS", "1"))))
        _TORCH_CONFIGURED = True
    return torch


def device():
    """Return the torch device this process trains on.

    ``cpu`` is the default because the submitted agent is scored on a single
    CPU thread and must never depend on an accelerator being present.
    ``BOMBERMAN_TORCH_DEVICE=cuda`` moves training to a GPU; a request for a
    device torch cannot see is an error rather than a silent fallback, because
    a cluster job that quietly ran 20x slower on CPU is worse than one that
    failed in its first second.
    """
    global _DEVICE
    torch = _torch()
    if _DEVICE is None:
        name = os.environ.get("BOMBERMAN_TORCH_DEVICE", "cpu").strip().lower() or "cpu"
        if name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"BOMBERMAN_TORCH_DEVICE={name!r} was requested but torch reports no CUDA device."
            )
        if name == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("BOMBERMAN_TORCH_DEVICE='mps' was requested but torch reports no MPS device.")
        _DEVICE = torch.device(name)
    return _DEVICE


class CnnMlpQModel:
    """Egocentric board CNN fused with a global-scalar MLP, six Q heads.

    Inference takes the same flat state vector as every other adapter and
    reshapes it internally, so the runtime, the replay buffer and the frozen
    ``Transition`` never learn that this route is spatial.

    Training uses Adam on the Huber loss of the selected heads.  Huber rather
    than squared error because a deep net trained off a bootstrapped target
    needs the gradient bounded; the reported TD error is still the plain
    ``target - prediction`` so it stays comparable with the other routes.
    Optimizer state is deliberately not part of a checkpoint: a warm start
    restores parameters and begins a fresh Adam, which is the usual convention
    and keeps a checkpoint file a pure description of the policy.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_layers: tuple[int, ...] = (256,),
        dueling: bool = False,
        seed: int = 0,
        learning_rate: float = 2.5e-4,
    ):
        torch = _torch()
        layout = layout_for_dimension(int(input_dim))
        self.layout = layout
        if len(hidden_layers) != 1 or hidden_layers[0] < 1:
            raise ValueError("cnn_mlp_q takes exactly one positive hidden width for its fused head.")
        self.input_dim = int(input_dim)
        self.hidden_layers = tuple(int(width) for width in hidden_layers)
        self.dueling = bool(dueling)
        self.seed = int(seed)
        self.learning_rate = float(learning_rate)
        torch.manual_seed(self.seed)
        self.device = device()
        self.network = _QNetwork(
            board_shape=layout["board_shape"],
            global_dim=layout["global_dimension"],
            hidden_width=self.hidden_layers[0],
            dueling=self.dueling,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.metadata: dict = {}

    @property
    def model_type(self) -> str:
        return "dueling_cnn_mlp_q" if self.dueling else "cnn_mlp_q"

    def q_values(self, state: np.ndarray) -> np.ndarray:
        return self.q_values_batch(np.asarray(state, dtype=np.float32)[None, :])[0]

    def q_values_batch(self, states: np.ndarray) -> np.ndarray:
        torch = _torch()
        board, globals_ = self._split(np.asarray(states, dtype=np.float32))
        self.network.eval()
        with torch.no_grad():
            values = self.network(self._tensor(board), self._tensor(globals_))
        return values.cpu().numpy()

    def fit_batch(self, states: np.ndarray, action_indices: np.ndarray, targets: np.ndarray) -> np.ndarray:
        torch = _torch()
        board, globals_ = self._split(np.asarray(states, dtype=np.float32))
        actions = torch.from_numpy(np.asarray(action_indices, dtype=np.int64)).to(self.device)
        wanted = torch.from_numpy(np.asarray(targets, dtype=np.float32)).to(self.device)
        self.network.train()
        predictions = self.network(self._tensor(board), self._tensor(globals_))
        selected = predictions.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = torch.nn.functional.smooth_l1_loss(selected, wanted)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # A single clipped step keeps one unlucky batch from destroying a run.
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10.0)
        self.optimizer.step()
        return (wanted - selected.detach()).cpu().numpy()

    def fit_policy_batch(self, states: np.ndarray, action_indices: np.ndarray) -> float:
        """One supervised step cloning a demonstrator's action choice.

        The Q head is read as a six-way logit vector and fitted with cross
        entropy.  That deliberately does not fit *values* -- a cloned head is
        only calibrated up to an arbitrary scale and offset -- but it does fit
        the ``argmax``, which is the whole policy.  ``rescale_head`` afterwards
        brings the magnitudes back into the range TD targets live in without
        touching the ordering.  Returns the mean cross-entropy loss.
        """
        torch = _torch()
        board, globals_ = self._split(np.asarray(states, dtype=np.float32))
        labels = torch.from_numpy(np.asarray(action_indices, dtype=np.int64)).to(self.device)
        self.network.train()
        logits = self.network(self._tensor(board), self._tensor(globals_))
        loss = torch.nn.functional.cross_entropy(logits, labels)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10.0)
        self.optimizer.step()
        return float(loss.detach())

    def rescale_head(self, factor: float) -> None:
        """Scale the output head by a positive factor, preserving every argmax.

        Cross-entropy pushes logits apart without bound, so a cloned network
        emits Q-values an order of magnitude larger than any real return.  Left
        alone, the first TD updates would spend themselves shrinking that scale.
        Multiplying the final layer by ``factor > 0`` is an exact rescaling of
        the Q function -- ``argmax`` is unchanged, so the cloned policy is
        unchanged -- which also holds for the dueling head, where the value and
        advantage branches are scaled together.
        """
        torch = _torch()
        if factor <= 0.0:
            raise ValueError("rescale_head needs a positive factor; a non-positive one would reorder the actions.")
        layers = [self.network.value, self.network.advantage] if self.dueling else [self.network.head]
        with torch.no_grad():
            for layer in layers:
                layer.weight.mul_(factor)
                layer.bias.mul_(factor)

    def clone(self) -> "CnnMlpQModel":
        copy = CnnMlpQModel(
            self.input_dim,
            hidden_layers=self.hidden_layers,
            dueling=self.dueling,
            seed=self.seed,
            learning_rate=self.learning_rate,
        )
        copy.copy_parameters_from(self)
        return copy

    def copy_parameters_from(self, other: "CnnMlpQModel") -> None:
        if other.model_type != self.model_type or other.hidden_layers != self.hidden_layers:
            raise ValueError(f"Cannot copy parameters from {other.model_type} into {self.model_type}.")
        self.network.load_state_dict(other.network.state_dict())

    def save(self, path: Path, *, metadata: dict | None = None) -> None:
        """Persist parameters as a plain, inspectable ``.npz`` of arrays."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata or {})
        payload: dict[str, np.ndarray] = {
            "model_type": np.asarray(self.model_type),
            "input_dim": np.asarray(self.input_dim, dtype=np.int64),
            "hidden_layers": np.asarray(self.hidden_layers, dtype=np.int64),
            "metadata": np.asarray(json.dumps(self.metadata, sort_keys=True)),
        }
        for name, tensor in self.network.state_dict().items():
            payload[f"parameter::{name}"] = tensor.detach().cpu().numpy()
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: Path, *, learning_rate: float = 2.5e-4) -> "CnnMlpQModel":
        torch = _torch()
        with np.load(path, allow_pickle=False) as data:
            required = {"model_type", "input_dim", "hidden_layers", "metadata"}
            if not required.issubset(data.files):
                raise ValueError(f"Checkpoint {path} is not a CNN Q model.")
            model_type = str(data["model_type"].item())
            if model_type not in {"cnn_mlp_q", "dueling_cnn_mlp_q"}:
                raise ValueError(f"Checkpoint {path} is not a CNN Q model.")
            model = cls(
                int(data["input_dim"]),
                hidden_layers=tuple(int(width) for width in data["hidden_layers"]),
                dueling=model_type == "dueling_cnn_mlp_q",
                learning_rate=learning_rate,
            )
            state_dict = {
                name.removeprefix("parameter::"): torch.from_numpy(data[name])
                for name in data.files if name.startswith("parameter::")
            }
            try:
                metadata = json.loads(str(data["metadata"].item()))
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Checkpoint {path} has invalid metadata.") from exc
        model.network.load_state_dict(state_dict)
        model.network.to(model.device)
        model.metadata = metadata
        return model

    def _split(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        board, globals_ = split_board_and_globals(np.atleast_2d(states))
        return np.ascontiguousarray(board), np.ascontiguousarray(globals_)

    def _tensor(self, array: np.ndarray):
        torch = _torch()
        return torch.from_numpy(array).to(self.device)


def _build_network_module():
    """Define the torch module lazily so importing this file needs no torch."""
    torch = _torch()
    nn = torch.nn

    class QNetwork(nn.Module):
        def __init__(self, board_shape, global_dim: int, hidden_width: int, dueling: bool):
            super().__init__()
            channels, width, height = board_shape
            self.dueling = dueling
            # Two stride-2 layers: 17 -> 9 -> 5, flattening 1600 features.
            #
            # Full resolution throughout was tried first, on the argument that a
            # blast cross is decided one cell at a time and downsampling would
            # blur the cell alignment that decides it.  Measured, that shape
            # costs 104 ms per batch-64 gradient step against 32 ms for this one
            # (scripts/benchmark_cnn.py and the candidate sweep behind docs/06),
            # and a 3x throughput cut is 3x fewer environment steps for a
            # from-scratch deep agent -- the scarcest resource on this line.
            #
            # The argument was also weaker than it sounded.  The window is
            # egocentric and fixed, so the stride grid sits at a *fixed* offset
            # from the agent: a cell one step to the right always lands in the
            # same place in the downsampled map.  The subsampling is consistent
            # across states rather than aliasing, which is the case where a CNN
            # can still separate the two cells in its channels.  If the anchor
            # plateaus in a way that looks like spatial confusion, the
            # full-resolution trunk is the pre-registered single-factor arm to
            # try; it is not worth paying for up front.
            self.board = nn.Sequential(
                nn.Conv2d(channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                board_features = self.board(torch.zeros(1, channels, width, height)).shape[1]
            self.globals = nn.Sequential(nn.Linear(global_dim, 32), nn.ReLU())
            self.fused = nn.Sequential(nn.Linear(board_features + 32, hidden_width), nn.ReLU())
            if dueling:
                self.value = nn.Linear(hidden_width, 1)
                self.advantage = nn.Linear(hidden_width, len(ACTIONS))
            else:
                self.head = nn.Linear(hidden_width, len(ACTIONS))

        def forward(self, board, globals_):
            fused = self.fused(torch.cat([self.board(board), self.globals(globals_)], dim=1))
            if not self.dueling:
                return self.head(fused)
            # Mean-centred advantages: without the shift, V and A are only
            # identified up to a constant and the two heads drift apart.
            advantage = self.advantage(fused)
            return self.value(fused) + advantage - advantage.mean(dim=1, keepdim=True)

    return QNetwork


class _LazyNetworkFactory:
    """Build the torch module class on first use and cache it."""

    def __init__(self):
        self._cls = None

    def __call__(self, **kwargs):
        if self._cls is None:
            self._cls = _build_network_module()
        return self._cls(**kwargs)


_QNetwork = _LazyNetworkFactory()
