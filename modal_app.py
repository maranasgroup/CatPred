from __future__ import annotations

from pathlib import Path
import os
import tempfile
from typing import Any, Optional

from fastapi import Header, HTTPException
import modal
from pydantic import BaseModel, Field


image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "libxrender1",
        "libxext6",
        "libsm6",
    )
    .pip_install(
        "fastapi[standard]>=0.115,<1.0",
        "pydantic>=1.10,<2.0",
        "pandas>=1.5,<2.3",
        "numpy>=1.26,<2.3",
        "scikit-learn>=1.3,<1.7",
        "scipy>=1.10,<1.16",
        "torch>=2.1,<2.7",
        "tqdm>=4.66",
        "typed-argument-parser>=1.10",
        "rdkit-pypi>=2022.9.5",
        "descriptastorus>=2.6",
        "transformers>=4.47,<5",
        "sentencepiece>=0.2.0",
        "fair-esm==2.0.0",
        "progres==0.2.7",
        "rotary-embedding-torch==0.6.5",
        "ipdb==0.13.13",
        "pandas-flavor>=0.6.0",
    )
    .add_local_python_source("catpred")
    .add_local_dir("scripts", remote_path="/root/scripts")
    .add_local_file("predict.py", remote_path="/root/predict.py")
)

app = modal.App("catpred-modal-api", image=image)
checkpoints_volume = modal.Volume.from_name("catpred-checkpoints", create_if_missing=True)


class PredictPayload(BaseModel):
    parameter: str = Field(..., description="One of: kcat, km, ki")
    checkpoint_dir: str = Field(..., description="Checkpoint subdirectory inside /checkpoints")
    use_gpu: bool = Field(default=False)
    input_rows: list[dict[str, Any]] = Field(default_factory=list)
    input_filename: Optional[str] = Field(default=None)


def _safe_checkpoint_path(raw_checkpoint_dir: str) -> Path:
    checkpoint_root = Path("/checkpoints").resolve()
    checkpoint_dir = (checkpoint_root / raw_checkpoint_dir).resolve()
    try:
        checkpoint_dir.relative_to(checkpoint_root)
    except ValueError as exc:
        raise ValueError("checkpoint_dir must stay inside /checkpoints.") from exc
    if not checkpoint_dir.is_dir():
        raise ValueError(f'Checkpoint directory not found: "{checkpoint_dir}"')
    return checkpoint_dir


@app.function(
    timeout=60 * 15,
    cpu=4.0,
    memory=16384,
    volumes={"/checkpoints": checkpoints_volume},
)
@modal.fastapi_endpoint(method="POST", docs=True)
def predict(
    payload: PredictPayload,
    authorization: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    import pandas as pd

    from catpred.inference.service import run_prediction_pipeline
    from catpred.inference.types import PredictionRequest

    expected_token = os.environ.get("CATPRED_MODAL_AUTH_TOKEN")
    if expected_token:
        provided = ""
        if authorization:
            lower = authorization.lower()
            if lower.startswith("bearer "):
                provided = authorization[7:].strip()
            else:
                provided = authorization.strip()
        if provided != expected_token:
            raise HTTPException(status_code=401, detail="Unauthorized")

    if not payload.input_rows:
        raise HTTPException(status_code=400, detail="input_rows cannot be empty.")

    parameter = payload.parameter.lower()
    if parameter not in {"kcat", "km", "ki"}:
        raise HTTPException(status_code=400, detail="parameter must be one of: kcat, km, ki.")

    try:
        checkpoint_dir = _safe_checkpoint_path(payload.checkpoint_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    safe_name = Path(payload.input_filename or "api_input.csv").name
    if not safe_name.endswith(".csv"):
        safe_name = f"{safe_name}.csv"

    runtime_dir = Path("/tmp/catpred-modal").resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_input_path = tempfile.mkstemp(prefix="modal_input_", suffix=".csv", dir=str(runtime_dir))
    os.close(fd)

    try:
        pd.DataFrame(payload.input_rows).to_csv(tmp_input_path, index=False)

        request_obj = PredictionRequest(
            parameter=parameter,
            input_file=tmp_input_path,
            checkpoint_dir=str(checkpoint_dir),
            use_gpu=payload.use_gpu,
            repo_root="/root",
            python_executable="python",
        )
        results_dir = str((runtime_dir / "results").resolve())
        output_file = run_prediction_pipeline(request_obj, results_dir=results_dir)
        output_df = pd.read_csv(output_file)

        return {
            "output_rows": output_df.to_dict(orient="records"),
            "output_filename": Path(output_file).name,
            "row_count": int(len(output_df)),
            "backend": "modal",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Modal prediction failed: {exc}") from exc
    finally:
        input_path = Path(tmp_input_path)
        if input_path.exists():
            input_path.unlink()
