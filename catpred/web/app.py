from __future__ import annotations

from pathlib import Path
import os
import subprocess
import tempfile
from typing import Any, Optional

import pandas as pd

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field, root_validator
except ImportError as exc:  # pragma: no cover - import guard for optional dependency
    raise ImportError(
        "catpred.web requires optional dependencies. Install with `pip install .[web]`."
    ) from exc

from catpred.inference import (
    InferenceBackendError,
    InferenceBackendRouter,
    PredictionRequest,
)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


class PredictRequest(BaseModel):
    parameter: str = Field(..., description="One of: kcat, km, ki")
    checkpoint_dir: str = Field(..., description="Path to the checkpoint directory")
    input_file: Optional[str] = Field(default=None, description="Path to input CSV")
    input_rows: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Optional in-request CSV rows. Use this for remote clients.",
    )
    use_gpu: bool = False
    backend: Optional[str] = Field(default=None, description="Override backend (local|modal)")
    fallback_to_local: Optional[bool] = Field(default=None, description="Fallback if modal fails")
    results_dir: str = Field(default="../results", description="Directory for final predictions")
    repo_root: Optional[str] = Field(default=None, description="Repository root path")
    python_executable: str = Field(default="python", description="Python executable to run scripts")

    @root_validator
    def _validate_input_source(cls, values: dict[str, Any]) -> dict[str, Any]:
        has_file = bool(values.get("input_file"))
        has_rows = bool(values.get("input_rows"))
        if has_file == has_rows:
            raise ValueError("Provide exactly one of `input_file` or `input_rows`.")
        return values


class PredictResponse(BaseModel):
    backend: str
    output_file: str
    row_count: int
    preview_rows: list[dict[str, Any]]
    metadata: dict[str, Any] = Field(default_factory=dict)


def _resolve_repo_root(repo_root: Optional[str]) -> Path:
    if repo_root:
        return Path(repo_root).resolve()
    env_repo_root = os.environ.get("CATPRED_REPO_ROOT")
    if env_repo_root:
        return Path(env_repo_root).resolve()
    return Path.cwd().resolve()


def _write_rows_to_temp_csv(rows: list[dict[str, Any]], repo_root: Path) -> tuple[str, str]:
    if len(rows) == 0:
        raise ValueError("`input_rows` cannot be empty.")

    tmp_dir = repo_root / ".e2e-tests" / "api_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="api_input_", suffix=".csv", dir=str(tmp_dir))
    os.close(fd)
    pd.DataFrame(rows).to_csv(tmp_path, index=False)
    return tmp_path, tmp_path


def _resolve_input_file(payload: PredictRequest) -> tuple[str, Optional[str]]:
    if payload.input_file:
        return payload.input_file, None

    repo_root = _resolve_repo_root(payload.repo_root)
    return _write_rows_to_temp_csv(payload.input_rows or [], repo_root)


def _preview_output(output_file: str, preview_limit: int = 5) -> tuple[int, list[dict[str, Any]]]:
    df = pd.read_csv(output_file)
    return len(df), df.head(preview_limit).to_dict(orient="records")


def create_app(router: Optional[InferenceBackendRouter] = None) -> FastAPI:
    app = FastAPI(title="CatPred API", version="0.1.0")
    backend_router = router or InferenceBackendRouter()
    default_fallback = _env_flag("CATPRED_MODAL_FALLBACK_TO_LOCAL", default=False)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict[str, Any]:
        return backend_router.readiness()

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest) -> PredictResponse:
        input_file, temp_file = _resolve_input_file(payload)
        try:
            request_obj = PredictionRequest(
                parameter=payload.parameter.lower(),
                input_file=input_file,
                checkpoint_dir=payload.checkpoint_dir,
                use_gpu=payload.use_gpu,
                repo_root=payload.repo_root,
                python_executable=payload.python_executable,
            )

            fallback = payload.fallback_to_local
            if fallback is None:
                fallback = default_fallback

            result = backend_router.predict(
                request_obj=request_obj,
                results_dir=payload.results_dir,
                backend_name=payload.backend,
                fallback_to_local=fallback,
            )

            row_count, preview_rows = _preview_output(result.output_file)
            return PredictResponse(
                backend=result.backend_name,
                output_file=result.output_file,
                row_count=row_count,
                preview_rows=preview_rows,
                metadata=result.metadata,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (FileNotFoundError, InferenceBackendError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except subprocess.CalledProcessError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction command failed with exit code {exc.returncode}.",
            ) from exc
        finally:
            if temp_file and Path(temp_file).exists():
                Path(temp_file).unlink()

    return app


app = create_app()
