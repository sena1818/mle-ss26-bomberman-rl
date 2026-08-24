"""Backward-compatible imports; QModel adapters now live in ``models/``."""

from .models import LinearQModel, build_model

LinearQNetwork = LinearQModel
build_network = build_model

__all__ = ("LinearQNetwork", "build_network")
