from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from functools import lru_cache
import gzip
import hashlib
import json
import os
import subprocess
import threading
from typing import Tuple

import numpy as np
import pandas as pd
from rdkit import Chem

from .types import PreparedInputPaths, PredictionRequest

_VALID_PARAMETERS = {"kcat", "km", "ki"}
_TARGET_COLUMNS = {
    "kcat": ("log10kcat_max", "s^(-1)"),
    "km": ("log10km_mean", "mM"),
    "ki": ("log10ki_mean", "mM"),
}
_VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")
_MODEL_CACHE_SIZE = max(int(os.environ.get("CATPRED_MODEL_CACHE_SIZE", "6")), 0)
_PREDICTION_CACHE_SIZE = max(int(os.environ.get("CATPRED_PREDICTION_CACHE_SIZE", "128")), 0)
_PREDICTION_CACHE: OrderedDict[tuple, str] = OrderedDict()
_PREDICTION_CACHE_LOCK = threading.Lock()


def _validate_parameter(parameter: str) -> str:
    parameter = parameter.lower()
    if parameter not in _VALID_PARAMETERS:
        raise ValueError(f"Unsupported parameter '{parameter}'. Must be one of: kcat, km, ki.")
    return parameter


def _resolve_repo_root(repo_root: str | None) -> Path:
    root = Path(repo_root) if repo_root else Path.cwd()
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f'Repository root does not exist: "{root}"')
    return root


def _resolve_input_path(input_file: str, repo_root: Path) -> Path:
    path = Path(input_file)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f'Input CSV not found: "{path}"')
    return path


def _resolve_existing_path(path_str: str, repo_root: Path, purpose: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f'{purpose} not found: "{path}"')
    return path


def _validate_and_prepare_dataframe(parameter: str, df: pd.DataFrame, input_csv: Path) -> pd.DataFrame:
    required_columns = {"SMILES", "sequence", "pdbpath"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            f'Missing required column(s) in "{input_csv}": {", ".join(sorted(missing))}.'
        )

    conflicting_pdbpaths = (
        df.groupby("pdbpath")["sequence"]
        .nunique(dropna=False)
        .loc[lambda value: value > 1]
    )
    if len(conflicting_pdbpaths) > 0:
        preview = ", ".join(conflicting_pdbpaths.index.astype(str).tolist()[:5])
        raise ValueError(
            "Found pdbpath values mapped to multiple sequences. "
            f"Each unique sequence must have a unique pdbpath. Examples: {preview}"
        )

    canonical_smiles = []
    for i, raw_smiles in enumerate(df["SMILES"]):
        mol = Chem.MolFromSmiles(raw_smiles)
        if mol is None:
            raise ValueError(f'Invalid SMILES input in row {i + 2}: "{raw_smiles}"')
        smiles = Chem.MolToSmiles(mol)
        if parameter == "kcat" and "." in smiles:
            smiles = ".".join(sorted(smiles.split(".")))
        canonical_smiles.append(smiles)

    for i, sequence in enumerate(df["sequence"]):
        if not isinstance(sequence, str) or not set(sequence).issubset(_VALID_AAS):
            raise ValueError(f'Invalid enzyme sequence in row {i + 2}: "{sequence}"')

    prepared = df.copy()
    prepared["SMILES"] = canonical_smiles
    return prepared


def prepare_prediction_inputs(parameter: str, input_file: str, repo_root: str | None = None) -> PreparedInputPaths:
    parameter = _validate_parameter(parameter)
    root = _resolve_repo_root(repo_root)
    input_csv = _resolve_input_path(input_file, root)

    df = pd.read_csv(input_csv)
    prepared_df = _validate_and_prepare_dataframe(parameter, df, input_csv)

    input_base = input_csv.with_suffix("")
    prepared_input_csv = Path(f"{input_base}_input.csv")
    prepared_df.to_csv(prepared_input_csv, index=False)

    test_prefix = prepared_input_csv.with_suffix("")
    records_file = Path(f"{test_prefix}.json.gz")
    output_csv = Path(f"{test_prefix}_output.csv")

    return PreparedInputPaths(
        input_csv=str(prepared_input_csv),
        records_file=str(records_file),
        output_csv=str(output_csv),
    )


def _build_prediction_commands(
    python_executable: str,
    repo_root: Path,
    paths: PreparedInputPaths,
    checkpoint_dir: str,
) -> Tuple[list[str], list[str]]:
    create_records_cmd = [
        python_executable,
        str(repo_root / "scripts" / "create_pdbrecords.py"),
        "--data_file",
        paths.input_csv,
        "--out_file",
        paths.records_file,
    ]
    predict_cmd = [
        python_executable,
        str(repo_root / "predict.py"),
        "--test_path",
        paths.input_csv,
        "--preds_path",
        paths.output_csv,
        "--checkpoint_dir",
        checkpoint_dir,
        "--uncertainty_method",
        "mve",
        "--smiles_column",
        "SMILES",
        "--individual_ensemble_predictions",
        "--protein_records_path",
        paths.records_file,
    ]
    return create_records_cmd, predict_cmd


def _write_protein_records(input_csv: str, records_file: str) -> None:
    df = pd.read_csv(input_csv)
    required = {"pdbpath", "sequence"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f'Missing required column(s) in "{input_csv}": {", ".join(sorted(missing))}'
        )

    records = {}
    conflicts = []
    for index, row in df.iterrows():
        row_num = index + 2
        pdbpath = row["pdbpath"].strip() if isinstance(row["pdbpath"], str) else row["pdbpath"]
        sequence = row["sequence"].strip() if isinstance(row["sequence"], str) else row["sequence"]

        if not pdbpath:
            raise ValueError(f'Empty "pdbpath" in row {row_num} of "{input_csv}".')
        if not sequence:
            raise ValueError(f'Empty "sequence" in row {row_num} of "{input_csv}".')

        key = os.path.basename(pdbpath)
        existing = records.get(key)
        if existing is not None and existing["seq"] != sequence:
            conflicts.append((row_num, key))
            continue

        records[key] = {"name": key, "seq": sequence}

    if conflicts:
        preview = ", ".join(
            [f'{key} (row {row_num})' for row_num, key in conflicts[:5]]
        )
        raise ValueError(
            "Found pdbpath basenames reused for different sequences. "
            f"Each unique sequence must have a unique pdbpath. Examples: {preview}"
        )

    with gzip.open(records_file, "wt", encoding="utf-8") as handle:
        json.dump(records, handle)


def _build_predict_args(request: PredictionRequest, paths: PreparedInputPaths, repo_root: Path):
    from catpred.args import PredictArgs

    checkpoint_dir = Path(request.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = (repo_root / checkpoint_dir).resolve()

    protein_records_path = paths.records_file
    if request.protein_records_file:
        protein_records_path = str(
            _resolve_existing_path(
                request.protein_records_file,
                repo_root=repo_root,
                purpose="Protein records file",
            )
        )

    args = PredictArgs()
    args.test_path = paths.input_csv
    args.preds_path = paths.output_csv
    args.checkpoint_dir = str(checkpoint_dir)
    args.uncertainty_method = "mve"
    args.smiles_columns = ["SMILES"]
    args.individual_ensemble_predictions = False
    args.save_uncertainty_components = True
    args.protein_records_path = protein_records_path
    args.no_cuda = not request.use_gpu
    args.process_args()
    return args


def _checkpoint_fingerprint(checkpoint_paths: tuple[str, ...]) -> tuple[tuple[str, int, int], ...]:
    fingerprint = []
    for checkpoint_path in checkpoint_paths:
        stat = Path(checkpoint_path).stat()
        fingerprint.append((checkpoint_path, stat.st_mtime_ns, stat.st_size))
    return tuple(fingerprint)


def _deduplicate_prediction_input(paths: PreparedInputPaths) -> tuple[PreparedInputPaths, bool]:
    input_df = pd.read_csv(paths.input_csv)
    key_columns = ["SMILES", "sequence"]
    if any(column not in input_df.columns for column in key_columns):
        return paths, False

    unique_df = input_df.drop_duplicates(subset=key_columns, keep="first")
    if len(unique_df) == len(input_df):
        return paths, False

    input_path = Path(paths.input_csv)
    unique_input_csv = input_path.with_name(f"{input_path.stem}_unique{input_path.suffix}")
    unique_output_csv = unique_input_csv.with_name(f"{unique_input_csv.with_suffix('').name}_output.csv")
    unique_df.to_csv(unique_input_csv, index=False)
    return (
        PreparedInputPaths(
            input_csv=str(unique_input_csv),
            records_file=paths.records_file,
            output_csv=str(unique_output_csv),
        ),
        True,
    )


def _expand_deduplicated_prediction_output(
    original_paths: PreparedInputPaths,
    deduplicated_paths: PreparedInputPaths,
) -> None:
    full_df = pd.read_csv(original_paths.input_csv)
    unique_input_df = pd.read_csv(deduplicated_paths.input_csv)
    unique_output_df = pd.read_csv(deduplicated_paths.output_csv)
    key_columns = ["SMILES", "sequence"]
    prediction_columns = [
        column for column in unique_output_df.columns if column not in unique_input_df.columns
    ]

    if not prediction_columns:
        raise ValueError("Deduplicated prediction output did not contain prediction columns.")

    prediction_lookup = unique_output_df[key_columns + prediction_columns].drop_duplicates(
        subset=key_columns,
        keep="first",
    )
    expanded_df = full_df.merge(prediction_lookup, on=key_columns, how="left", sort=False)

    if expanded_df[prediction_columns].isnull().any().any():
        raise ValueError("Failed to expand deduplicated predictions to all input rows.")

    expanded_df.to_csv(original_paths.output_csv, index=False)


def _file_digest(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prediction_checkpoint_paths(checkpoint_dir: str, repo_root: Path) -> tuple[str, ...]:
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (repo_root / checkpoint_path).resolve()
    if checkpoint_path.is_file():
        return (str(checkpoint_path),)
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f'Checkpoint directory not found: "{checkpoint_path}"')
    model_paths = sorted(str(path.resolve()) for path in checkpoint_path.rglob("model.pt"))
    if not model_paths:
        raise FileNotFoundError(f'No model.pt checkpoints found in "{checkpoint_path}"')
    return tuple(model_paths)


def _prediction_cache_key(
    parameter: str,
    request: PredictionRequest,
    paths: PreparedInputPaths,
    repo_root: Path,
) -> tuple:
    protein_records_digest = None
    if request.protein_records_file:
        protein_records_path = _resolve_existing_path(
            request.protein_records_file,
            repo_root=repo_root,
            purpose="Protein records file",
        )
        protein_records_digest = _file_digest(str(protein_records_path))

    checkpoint_paths = _prediction_checkpoint_paths(request.checkpoint_dir, repo_root)
    return (
        parameter,
        bool(request.use_gpu),
        _file_digest(paths.input_csv),
        protein_records_digest,
        _checkpoint_fingerprint(checkpoint_paths),
    )


def _prediction_cache_get(cache_key: tuple) -> str | None:
    if _PREDICTION_CACHE_SIZE <= 0:
        return None
    with _PREDICTION_CACHE_LOCK:
        cached = _PREDICTION_CACHE.get(cache_key)
        if cached is not None:
            _PREDICTION_CACHE.move_to_end(cache_key)
        return cached


def _prediction_cache_put(cache_key: tuple, csv_text: str) -> None:
    if _PREDICTION_CACHE_SIZE <= 0:
        return
    with _PREDICTION_CACHE_LOCK:
        _PREDICTION_CACHE[cache_key] = csv_text
        _PREDICTION_CACHE.move_to_end(cache_key)
        while len(_PREDICTION_CACHE) > _PREDICTION_CACHE_SIZE:
            _PREDICTION_CACHE.popitem(last=False)


def _write_cached_prediction(
    cached_csv: str,
    paths: PreparedInputPaths,
    repo_root: str | None,
    results_dir: str,
) -> str:
    results_path = Path(results_dir)
    if not results_path.is_absolute():
        results_path = (_resolve_repo_root(repo_root) / results_path).resolve()
    results_path.mkdir(parents=True, exist_ok=True)

    final_output = results_path / Path(paths.output_csv).name
    final_output.write_text(cached_csv, encoding="utf-8")
    return str(final_output)


@lru_cache(maxsize=_MODEL_CACHE_SIZE)
def _load_cached_model_objects(
    checkpoint_paths: tuple[str, ...],
    checkpoint_fingerprint: tuple[tuple[str, int, int], ...],
    use_gpu: bool,
    gpu: int | None,
    pretrained_egnn_feats_path: str,
):
    del checkpoint_fingerprint  # Included in the cache key to invalidate changed checkpoints.

    from catpred.args import PredictArgs
    from catpred.train.make_predictions import load_model

    args = PredictArgs()
    args.checkpoint_paths = list(checkpoint_paths)
    args.no_cuda = not use_gpu
    args.gpu = gpu
    args.pretrained_egnn_feats_path = pretrained_egnn_feats_path
    loaded_args, train_args, models, scalers, num_tasks, task_names = load_model(
        args=args,
        generator=False,
    )
    return train_args, models, scalers, num_tasks, task_names, loaded_args.pretrained_egnn_feats_path


def _load_model_objects_for_prediction(args):
    from catpred.train.make_predictions import load_model
    from catpred.utils import update_prediction_args

    checkpoint_paths = tuple(args.checkpoint_paths)
    if _MODEL_CACHE_SIZE <= 0:
        loaded_args, train_args, models, scalers, num_tasks, task_names = load_model(
            args=args,
            generator=False,
        )
        return loaded_args, train_args, models, scalers, num_tasks, task_names

    train_args, models, scalers, num_tasks, task_names, pretrained_egnn_feats_path = (
        _load_cached_model_objects(
            checkpoint_paths=checkpoint_paths,
            checkpoint_fingerprint=_checkpoint_fingerprint(checkpoint_paths),
            use_gpu=not args.no_cuda,
            gpu=args.gpu,
            pretrained_egnn_feats_path=args.pretrained_egnn_feats_path,
        )
    )
    args.pretrained_egnn_feats_path = pretrained_egnn_feats_path
    update_prediction_args(predict_args=args, train_args=train_args)
    return args, train_args, models, scalers, num_tasks, task_names


def run_inprocess_prediction(request: PredictionRequest, paths: PreparedInputPaths) -> None:
    from catpred.train.make_predictions import make_predictions

    root = _resolve_repo_root(request.repo_root)

    previous_embed_cpu = os.environ.get("PROTEIN_EMBED_USE_CPU")
    os.environ["PROTEIN_EMBED_USE_CPU"] = "0" if request.use_gpu else "1"
    try:
        if not request.protein_records_file:
            _write_protein_records(paths.input_csv, paths.records_file)

        args = _build_predict_args(request, paths, root)
        model_objects = _load_model_objects_for_prediction(args)
        make_predictions(args=args, model_objects=model_objects)
    finally:
        if previous_embed_cpu is None:
            os.environ.pop("PROTEIN_EMBED_USE_CPU", None)
        else:
            os.environ["PROTEIN_EMBED_USE_CPU"] = previous_embed_cpu

    if not os.path.exists(paths.output_csv):
        raise FileNotFoundError(f'Prediction output file was not generated: "{paths.output_csv}"')


def run_raw_prediction(request: PredictionRequest, paths: PreparedInputPaths) -> None:
    root = _resolve_repo_root(request.repo_root)
    create_records_cmd, predict_cmd = _build_prediction_commands(
        python_executable=request.python_executable,
        repo_root=root,
        paths=paths,
        checkpoint_dir=request.checkpoint_dir,
    )

    env = os.environ.copy()
    env["PROTEIN_EMBED_USE_CPU"] = "0" if request.use_gpu else "1"

    protein_records_path = paths.records_file
    if request.protein_records_file:
        protein_records_path = str(
            _resolve_existing_path(
                request.protein_records_file,
                repo_root=root,
                purpose="Protein records file",
            )
        )
    else:
        subprocess.run(create_records_cmd, cwd=str(root), env=env, check=True)

    predict_cmd[-1] = protein_records_path
    subprocess.run(predict_cmd, cwd=str(root), env=env, check=True)

    if not os.path.exists(paths.output_csv):
        raise FileNotFoundError(f'Prediction output file was not generated: "{paths.output_csv}"')


def postprocess_predictions(parameter: str, output_csv: str) -> pd.DataFrame:
    parameter = _validate_parameter(parameter)
    target_col, unit = _TARGET_COLUMNS[parameter]
    unc_col = f"{target_col}_mve_uncal_var"

    df = pd.read_csv(output_csv)
    missing_cols = [col for col in [target_col, unc_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f'Prediction output is missing required column(s): {", ".join(missing_cols)}'
        )

    prediction_log = df[target_col].astype(float).to_numpy()
    unc = df[unc_col].astype(float).to_numpy()
    alea_component_col = f"{target_col}_mve_uncal_aleatoric_var"
    epi_component_col = f"{target_col}_mve_uncal_epistemic_var"
    if alea_component_col in df.columns and epi_component_col in df.columns:
        alea_unc_var = np.maximum(df[alea_component_col].astype(float).to_numpy(), 0.0)
        epi_unc_var = np.maximum(df[epi_component_col].astype(float).to_numpy(), 0.0)
    else:
        model_cols = [col for col in df.columns if col.startswith(target_col) and "model_" in col]
        if not model_cols:
            raise ValueError(
                "Prediction output is missing uncertainty component columns or individual "
                f"ensemble prediction columns for {target_col}."
            )
        epi_unc_var = df[model_cols].astype(float).to_numpy().var(axis=1)
        alea_unc_var = np.maximum(unc - epi_unc_var, 0.0)

    df[f"Prediction_({unit})"] = np.power(10, prediction_log)
    df["Prediction_log10"] = prediction_log
    df["SD_total"] = np.sqrt(np.maximum(unc, 0.0))
    df["SD_aleatoric"] = np.sqrt(alea_unc_var)
    df["SD_epistemic"] = np.sqrt(epi_unc_var)
    return df


def run_prediction_pipeline(request: PredictionRequest, results_dir: str = "../results") -> str:
    parameter = _validate_parameter(request.parameter)
    paths = prepare_prediction_inputs(parameter, request.input_file, request.repo_root)
    run_raw_prediction(request, paths)
    return _write_postprocessed_predictions(parameter, paths, request.repo_root, results_dir)


def run_inprocess_prediction_pipeline(
    request: PredictionRequest,
    results_dir: str = "../results",
) -> str:
    parameter = _validate_parameter(request.parameter)
    paths = prepare_prediction_inputs(parameter, request.input_file, request.repo_root)
    root = _resolve_repo_root(request.repo_root)
    cache_key = _prediction_cache_key(parameter, request, paths, root)
    cached_csv = _prediction_cache_get(cache_key)
    if cached_csv is not None:
        return _write_cached_prediction(cached_csv, paths, request.repo_root, results_dir)

    if request.protein_records_file:
        prediction_paths, was_deduplicated = paths, False
    else:
        prediction_paths, was_deduplicated = _deduplicate_prediction_input(paths)
    run_inprocess_prediction(request, prediction_paths)
    if was_deduplicated:
        _expand_deduplicated_prediction_output(paths, prediction_paths)
    final_output = _write_postprocessed_predictions(parameter, paths, request.repo_root, results_dir)
    _prediction_cache_put(cache_key, Path(final_output).read_text(encoding="utf-8"))
    return final_output


def _write_postprocessed_predictions(
    parameter: str,
    paths: PreparedInputPaths,
    repo_root: str | None,
    results_dir: str,
) -> str:
    output_final = postprocess_predictions(parameter, paths.output_csv)

    results_path = Path(results_dir)
    if not results_path.is_absolute():
        results_path = (_resolve_repo_root(repo_root) / results_path).resolve()
    results_path.mkdir(parents=True, exist_ok=True)

    out_name = Path(paths.output_csv).name
    final_output = results_path / out_name
    output_final.to_csv(final_output, index=False)
    return str(final_output)
