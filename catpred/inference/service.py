from __future__ import annotations

from pathlib import Path
import os
import subprocess
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

    pred_col, pred_logcol, pred_sd_tot, pred_sd_alea, pred_sd_epi = [], [], [], [], []

    for _, row in df.iterrows():
        model_cols = [col for col in row.index if col.startswith(target_col) and "model_" in col]

        unc = row[unc_col]
        prediction_log = row[target_col]
        prediction_linear = np.power(10, prediction_log)

        if model_cols:
            model_outs = np.array([row[col] for col in model_cols])
            epi_unc_var = np.var(model_outs)
        else:
            epi_unc_var = 0.0

        alea_unc_var = max(unc - epi_unc_var, 0.0)
        epi_unc = np.sqrt(epi_unc_var)
        alea_unc = np.sqrt(alea_unc_var)
        total_unc = np.sqrt(max(unc, 0.0))

        pred_col.append(prediction_linear)
        pred_logcol.append(prediction_log)
        pred_sd_tot.append(total_unc)
        pred_sd_alea.append(alea_unc)
        pred_sd_epi.append(epi_unc)

    df[f"Prediction_({unit})"] = pred_col
    df["Prediction_log10"] = pred_logcol
    df["SD_total"] = pred_sd_tot
    df["SD_aleatoric"] = pred_sd_alea
    df["SD_epistemic"] = pred_sd_epi
    return df


def run_prediction_pipeline(request: PredictionRequest, results_dir: str = "../results") -> str:
    parameter = _validate_parameter(request.parameter)
    paths = prepare_prediction_inputs(parameter, request.input_file, request.repo_root)
    run_raw_prediction(request, paths)

    output_final = postprocess_predictions(parameter, paths.output_csv)

    results_path = Path(results_dir)
    if not results_path.is_absolute():
        results_path = (_resolve_repo_root(request.repo_root) / results_path).resolve()
    results_path.mkdir(parents=True, exist_ok=True)

    out_name = Path(paths.output_csv).name
    final_output = results_path / out_name
    output_final.to_csv(final_output, index=False)
    return str(final_output)
