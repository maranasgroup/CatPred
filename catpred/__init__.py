from __future__ import annotations

import importlib

__version__ = "0.0.1"

_LAZY_SUBMODULES = {
    "args",
    "constants",
    "data",
    "features",
    "inference",
    "models",
    "nn_utils",
    "rdkit",
    "security",
    "train",
    "uncertainty",
    "utils",
}

__all__ = sorted(_LAZY_SUBMODULES) + ["__version__"]


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f"catpred.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'catpred' has no attribute '{name}'")
