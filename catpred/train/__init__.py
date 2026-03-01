from __future__ import annotations

import importlib

_LAZY_ATTRS = {
    "get_metric_func": ("metrics", "get_metric_func"),
    "prc_auc": ("metrics", "prc_auc"),
    "bce": ("metrics", "bce"),
    "rmse": ("metrics", "rmse"),
    "bounded_mse": ("metrics", "bounded_mse"),
    "bounded_mae": ("metrics", "bounded_mae"),
    "bounded_rmse": ("metrics", "bounded_rmse"),
    "accuracy": ("metrics", "accuracy"),
    "f1_metric": ("metrics", "f1_metric"),
    "mcc_metric": ("metrics", "mcc_metric"),
    "sid_metric": ("metrics", "sid_metric"),
    "wasserstein_metric": ("metrics", "wasserstein_metric"),
    "get_loss_func": ("loss_functions", "get_loss_func"),
    "bounded_mse_loss": ("loss_functions", "bounded_mse_loss"),
    "mcc_class_loss": ("loss_functions", "mcc_class_loss"),
    "mcc_multiclass_loss": ("loss_functions", "mcc_multiclass_loss"),
    "sid_loss": ("loss_functions", "sid_loss"),
    "wasserstein_loss": ("loss_functions", "wasserstein_loss"),
    "catpred_train": ("cross_validate", "catpred_train"),
    "cross_validate": ("cross_validate", "cross_validate"),
    "TRAIN_LOGGER_NAME": ("cross_validate", "TRAIN_LOGGER_NAME"),
    "evaluate": ("evaluate", "evaluate"),
    "evaluate_predictions": ("evaluate", "evaluate_predictions"),
    "catpred_predict": ("make_predictions", "catpred_predict"),
    "make_predictions": ("make_predictions", "make_predictions"),
    "load_model": ("make_predictions", "load_model"),
    "set_features": ("make_predictions", "set_features"),
    "load_data": ("make_predictions", "load_data"),
    "predict_and_save": ("make_predictions", "predict_and_save"),
    "catpred_fingerprint": ("molecule_fingerprint", "catpred_fingerprint"),
    "model_fingerprint": ("molecule_fingerprint", "model_fingerprint"),
    "predict": ("predict", "predict"),
    "run_training": ("run_training", "run_training"),
    "train": ("train", "train"),
}

__all__ = sorted(_LAZY_ATTRS.keys())


def __getattr__(name: str):
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module 'catpred.train' has no attribute '{name}'")

    module_name, attr_name = target
    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
