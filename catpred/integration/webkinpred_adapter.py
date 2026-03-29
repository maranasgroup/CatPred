from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import pandas as pd
import torch

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from catpred.inference import PredictionRequest, run_prediction_pipeline
from catpred.security import load_torch_artifact

_VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")
_TARGET_TO_PARAMETER = {
    "kcat": "kcat",
    "Km": "km",
    "km": "km",
    "ki": "ki",
    "Ki": "ki",
}
_PREDICTION_COLUMN = {
    "kcat": "Prediction_(s^(-1))",
    "km": "Prediction_(mM)",
    "ki": "Prediction_(mM)",
}


def _repo_root() -> Path:
    return _REPO_ROOT


def _contains_model_checkpoints(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.rglob("model.pt"))


def _discover_checkpoint_root(repo_root: Path) -> Path:
    production_root = (repo_root / ".e2e-assets" / "pretrained" / "production").resolve()
    checkpoints_root = (repo_root / "checkpoints").resolve()
    if _contains_model_checkpoints(production_root):
        return production_root
    return checkpoints_root


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).resolve() if value else default.resolve()


def _resolve_parameter(payload: dict[str, Any]) -> str:
    target = payload.get("target")
    if isinstance(target, str) and target in _TARGET_TO_PARAMETER:
        return _TARGET_TO_PARAMETER[target]

    params = payload.get("params") or {}
    kinetics_type = str(params.get("kinetics_type", "")).strip().upper()
    if kinetics_type == "KCAT":
        return "kcat"
    if kinetics_type == "KM":
        return "km"
    if kinetics_type == "KI":
        return "ki"
    raise RuntimeError(f"Unsupported target in payload: {target!r}")


def _stable_seq_id(sequence: str) -> str:
    digest = hashlib.sha1(sequence.encode("utf-8")).hexdigest()[:16]
    return f"seq_{digest}"


def _resolve_seq_ids(sequences: list[str], tools_path: Path, media_path: Path) -> list[str]:
    seqmap_cli = tools_path / "seqmap" / "main.py"
    seqmap_db = media_path / "sequence_info" / "seqmap.sqlite3"
    if not seqmap_cli.exists() or not seqmap_db.exists():
        return [_stable_seq_id(sequence) for sequence in sequences]

    seqmap_python = os.environ.get("CATPRED_SEQMAP_PYTHON", sys.executable)
    payload = "\n".join(sequences) + "\n"
    cmd = [
        seqmap_python,
        str(seqmap_cli),
        "--db",
        str(seqmap_db),
        "batch-get-or-create",
        "--stdin",
    ]
    proc = subprocess.run(
        cmd,
        input=payload,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"seqmap failed (rc={proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    seq_ids = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if len(seq_ids) != len(sequences):
        raise RuntimeError(f"seqmap returned {len(seq_ids)} ids for {len(sequences)} sequences")
    return seq_ids


def _load_or_compute_esm2_feature(sequence: str, cache_file: Path) -> Path:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.exists():
        load_torch_artifact(
            cache_file,
            purpose="CatPred ESM2 cache entry",
            map_location="cpu",
            roots=[cache_file.parent],
        )
        return cache_file

    os.environ.setdefault("PROTEIN_EMBED_USE_CPU", "1")
    from catpred.data.esm_utils import get_single_esm_repr

    embedding = get_single_esm_repr(sequence).cpu()
    torch.save(embedding, cache_file)
    return cache_file


def _build_input_dataframe(rows: list[dict[str, Any]], seq_ids: list[str]) -> pd.DataFrame:
    formatted_rows = []
    for row, seq_id in zip(rows, seq_ids):
        substrate = row.get("substrates", row.get("substrate", row.get("Substrate", "")))
        if isinstance(substrate, list):
            if len(substrate) != 1:
                raise RuntimeError("CatPred expects exactly one substrate per row.")
            substrate = substrate[0]
        substrate = str(substrate).strip()
        sequence = str(row.get("sequence", "")).strip()
        formatted_rows.append(
            {
                "SMILES": substrate,
                "sequence": sequence,
                "pdbpath": seq_id,
            }
        )
    return pd.DataFrame(formatted_rows)


def _write_protein_records(
    rows: list[dict[str, Any]],
    seq_ids: list[str],
    parameter: str,
    media_path: Path,
    out_path: Path,
) -> None:
    records: dict[str, dict[str, Any]] = {}
    needs_esm = parameter in {"kcat", "km"}
    esm_cache_dir = media_path / "sequence_info" / "esm2_last" / "per_residue"

    for row, seq_id in zip(rows, seq_ids):
        sequence = str(row.get("sequence", "")).strip()
        record: dict[str, Any] = {"name": seq_id, "seq": sequence}
        if needs_esm:
            cache_file = _load_or_compute_esm2_feature(sequence, esm_cache_dir / f"{seq_id}.pt")
            record["esm2_feats_path"] = str(cache_file.resolve())
        records[seq_id] = record

    with gzip.open(out_path, "wt", encoding="utf-8") as handle:
        json.dump(records, handle)


def run_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("rows") or []
    if not isinstance(rows, list):
        raise RuntimeError("'rows' must be a list in input payload.")

    valid_rows: list[dict[str, Any]] = []
    invalid_indices: list[int] = []
    for idx, row in enumerate(rows):
        sequence = str(row.get("sequence", "")).strip()
        substrate = row.get("substrates", row.get("substrate", row.get("Substrate", "")))
        if isinstance(substrate, list):
            substrate = substrate[0] if len(substrate) == 1 else ""
        substrate = str(substrate).strip()
        if not sequence or not substrate or not set(sequence).issubset(_VALID_AAS):
            invalid_indices.append(idx)
            continue
        valid_rows.append(row)

    predictions: list[float | None] = [None] * len(rows)
    if not valid_rows:
        for idx in range(len(rows)):
            print(f"Progress: {idx + 1}/{len(rows)}", flush=True)
        return {"predictions": predictions, "invalid_indices": invalid_indices}

    repo_root = _env_path("CATPRED_REPO_ROOT", _repo_root())
    media_path = _env_path("CATPRED_MEDIA_PATH", repo_root / "media")
    tools_path = _env_path("CATPRED_TOOLS_PATH", repo_root / "tools")
    checkpoint_root = _env_path("CATPRED_CHECKPOINT_ROOT", _discover_checkpoint_root(repo_root))
    parameter = _resolve_parameter(payload)

    seq_ids = _resolve_seq_ids(
        [str(row.get("sequence", "")).strip() for row in valid_rows],
        tools_path=tools_path,
        media_path=media_path,
    )

    with tempfile.TemporaryDirectory(prefix="catpred_webkinpred_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str).resolve()
        input_csv = tmp_dir / "input.csv"
        protein_records = tmp_dir / "protein_records.json.gz"
        results_dir = tmp_dir / "results"

        _build_input_dataframe(valid_rows, seq_ids).to_csv(input_csv, index=False)
        _write_protein_records(
            rows=valid_rows,
            seq_ids=seq_ids,
            parameter=parameter,
            media_path=media_path,
            out_path=protein_records,
        )

        request = PredictionRequest(
            parameter=parameter,
            input_file=str(input_csv),
            checkpoint_dir=str((checkpoint_root / parameter).resolve()),
            use_gpu=False,
            repo_root=str(repo_root),
            python_executable=sys.executable,
            protein_records_file=str(protein_records),
        )
        output_file = run_prediction_pipeline(request=request, results_dir=str(results_dir))
        output_df = pd.read_csv(output_file)
        value_col = _PREDICTION_COLUMN[parameter]
        if value_col not in output_df.columns:
            raise RuntimeError(f"CatPred output is missing expected column: {value_col}")

        valid_predictions = output_df[value_col].tolist()
        if len(valid_predictions) != len(valid_rows):
            raise RuntimeError(
                f"CatPred produced {len(valid_predictions)} predictions for {len(valid_rows)} rows."
            )

    valid_iter = iter(valid_predictions)
    for idx in range(len(rows)):
        if idx in invalid_indices:
            continue
        predictions[idx] = float(next(valid_iter))

    total = len(rows)
    for idx in range(total):
        print(f"Progress: {idx + 1}/{total}", flush=True)

    return {
        "predictions": predictions,
        "invalid_indices": sorted(set(invalid_indices)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CatPred via the webKinPred subprocess contract.")
    parser.add_argument("--input", required=True, help="Input JSON path.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    result = run_from_payload(payload)

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[CatPred] ERROR: {exc}", file=sys.stderr, flush=True)
        raise
