from __future__ import annotations

from .backends import (
    BackendPredictionResult,
    BackendRouterSettings,
    InferenceBackend,
    InferenceBackendError,
    InferenceBackendRouter,
    LocalInferenceBackend,
    ModalHTTPInferenceBackend,
)
from .types import PreparedInputPaths, PredictionRequest

__all__ = [
    "BackendPredictionResult",
    "BackendRouterSettings",
    "InferenceBackend",
    "InferenceBackendError",
    "InferenceBackendRouter",
    "LocalInferenceBackend",
    "ModalHTTPInferenceBackend",
    "PredictionRequest",
    "PreparedInputPaths",
    "prepare_prediction_inputs",
    "run_raw_prediction",
    "postprocess_predictions",
    "run_prediction_pipeline",
]


def __getattr__(name: str):
    if name in {
        "prepare_prediction_inputs",
        "run_raw_prediction",
        "postprocess_predictions",
        "run_prediction_pipeline",
    }:
        from . import service

        value = getattr(service, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'catpred.inference' has no attribute '{name}'")
