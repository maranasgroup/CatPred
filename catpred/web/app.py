from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import subprocess
import sys
import tempfile
from typing import Any, Optional

import pandas as pd

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
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


_SUPPORTED_PARAMETERS = ("kcat", "km", "ki")


def _contains_model_checkpoints(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any(path.rglob("model.pt"))


def _discover_default_checkpoint_root(repo_root: Path) -> Path:
    default_root = (repo_root / "checkpoints").resolve()
    production_root = (repo_root / ".e2e-assets" / "pretrained" / "production").resolve()
    candidates = [default_root, production_root]

    best_root: Optional[Path] = None
    best_score = -1
    for candidate in candidates:
        if not candidate.exists() or not candidate.is_dir():
            continue
        score = sum(int((candidate / parameter).is_dir()) for parameter in _SUPPORTED_PARAMETERS)
        if score > best_score:
            best_score = score
            best_root = candidate

    if best_root and best_score > 0:
        return best_root

    for candidate in candidates:
        if _contains_model_checkpoints(candidate):
            return candidate

    return default_root


def _discover_available_checkpoints(checkpoint_root: Path) -> dict[str, str]:
    available: dict[str, str] = {}
    for parameter in _SUPPORTED_PARAMETERS:
        param_dir = (checkpoint_root / parameter).resolve()
        if param_dir.is_dir() and _contains_model_checkpoints(param_dir):
            available[parameter] = parameter
    return available


@dataclass(frozen=True)
class APISettings:
    repo_root: str
    python_executable: str
    input_root: str
    results_root: str
    temp_root: str
    checkpoint_root: str
    allow_input_file: bool = False
    allow_unsafe_request_overrides: bool = False
    max_input_rows: int = 1000
    max_input_file_bytes: int = 5_000_000
    preview_rows: int = 5

    @classmethod
    def from_env(cls) -> "APISettings":
        env_repo_root = os.environ.get("CATPRED_REPO_ROOT")
        repo_root = str(Path(env_repo_root).resolve()) if env_repo_root else str(Path.cwd().resolve())
        repo_root_path = Path(repo_root).resolve()
        env_runtime_root = os.environ.get("CATPRED_API_RUNTIME_ROOT")
        if env_runtime_root:
            default_runtime_root = Path(env_runtime_root).resolve()
        elif os.environ.get("VERCEL"):
            default_runtime_root = Path("/tmp/catpred").resolve()
        else:
            default_runtime_root = repo_root_path
        input_root = os.environ.get("CATPRED_API_INPUT_ROOT")
        results_root = os.environ.get("CATPRED_API_RESULTS_ROOT")
        temp_root = os.environ.get("CATPRED_API_TEMP_ROOT")
        checkpoint_root = os.environ.get("CATPRED_API_CHECKPOINT_ROOT")

        return cls(
            repo_root=repo_root,
            python_executable=os.environ.get("CATPRED_PYTHON_EXECUTABLE", sys.executable or "python"),
            input_root=(
                str(Path(input_root).resolve())
                if input_root
                else str((default_runtime_root / "inputs").resolve())
            ),
            results_root=(
                str(Path(results_root).resolve())
                if results_root
                else str((default_runtime_root / "results").resolve())
            ),
            temp_root=(
                str(Path(temp_root).resolve())
                if temp_root
                else str((default_runtime_root / "tmp").resolve())
            ),
            checkpoint_root=(
                str(Path(checkpoint_root).resolve())
                if checkpoint_root
                else str(_discover_default_checkpoint_root(repo_root_path))
            ),
            allow_input_file=_env_flag("CATPRED_API_ALLOW_INPUT_FILE", default=False),
            allow_unsafe_request_overrides=_env_flag("CATPRED_API_ALLOW_UNSAFE_OVERRIDES", default=False),
            max_input_rows=max(int(os.environ.get("CATPRED_API_MAX_INPUT_ROWS", "1000")), 1),
            max_input_file_bytes=max(int(os.environ.get("CATPRED_API_MAX_INPUT_FILE_BYTES", "5000000")), 1024),
            preview_rows=max(int(os.environ.get("CATPRED_API_PREVIEW_ROWS", "5")), 1),
        )


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
    results_dir: str = Field(
        default="results",
        description="Results subdirectory under CATPRED_API_RESULTS_ROOT.",
    )
    repo_root: Optional[str] = Field(
        default=None,
        description="Unsafe override. Disabled by default; only for trusted local workflows.",
    )
    python_executable: Optional[str] = Field(
        default=None,
        description="Unsafe override. Disabled by default; only for trusted local workflows.",
    )

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


def _is_subpath(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_repo_root(repo_root: Optional[str], settings: APISettings) -> Path:
    if repo_root is not None:
        if not settings.allow_unsafe_request_overrides:
            raise ValueError(
                "Request field `repo_root` is disabled. "
                "Set CATPRED_API_ALLOW_UNSAFE_OVERRIDES=1 for trusted local use."
            )
        return Path(repo_root).resolve()
    return Path(settings.repo_root).resolve()


def _resolve_python_executable(python_executable: Optional[str], settings: APISettings) -> str:
    if python_executable is not None:
        if not settings.allow_unsafe_request_overrides:
            raise ValueError(
                "Request field `python_executable` is disabled. "
                "Set CATPRED_API_ALLOW_UNSAFE_OVERRIDES=1 for trusted local use."
            )
        return python_executable
    return settings.python_executable


def _resolve_and_validate_path_under_root(raw_path: str, root: Path, purpose: str) -> Path:
    candidate = Path(raw_path)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not _is_subpath(resolved, root):
        raise ValueError(f"{purpose} must stay under configured root: {root}")
    return resolved


def _resolve_results_dir(raw_results_dir: str, settings: APISettings) -> str:
    results_root = Path(settings.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    resolved = _resolve_and_validate_path_under_root(
        raw_path=raw_results_dir,
        root=results_root,
        purpose="results_dir",
    )
    resolved.mkdir(parents=True, exist_ok=True)
    return str(resolved)


def _resolve_checkpoint_dir(raw_checkpoint_dir: str, settings: APISettings) -> str:
    checkpoint_root = Path(settings.checkpoint_root).resolve()
    resolved = _resolve_and_validate_path_under_root(
        raw_path=raw_checkpoint_dir,
        root=checkpoint_root,
        purpose="checkpoint_dir",
    )
    if not resolved.exists():
        raise FileNotFoundError(f'Checkpoint directory not found: "{resolved}"')
    if not resolved.is_dir():
        raise ValueError(f'checkpoint_dir must be a directory: "{resolved}"')
    return str(resolved)


def _resolve_input_file_path(input_file: str, settings: APISettings) -> Path:
    if not settings.allow_input_file:
        raise ValueError(
            "Request field `input_file` is disabled. Submit `input_rows` instead, "
            "or set CATPRED_API_ALLOW_INPUT_FILE=1 for trusted local use."
        )

    input_root = Path(settings.input_root).resolve()
    input_root.mkdir(parents=True, exist_ok=True)
    resolved = _resolve_and_validate_path_under_root(
        raw_path=input_file,
        root=input_root,
        purpose="input_file",
    )

    if not resolved.exists():
        raise FileNotFoundError(f'Input CSV not found: "{resolved}"')
    if not resolved.is_file():
        raise ValueError(f'Input CSV path is not a file: "{resolved}"')
    if resolved.stat().st_size > settings.max_input_file_bytes:
        raise ValueError(
            f'Input file exceeds CATPRED_API_MAX_INPUT_FILE_BYTES ({settings.max_input_file_bytes}).'
        )

    return resolved


def _write_rows_to_temp_csv(
    rows: list[dict[str, Any]],
    settings: APISettings,
) -> tuple[str, str]:
    if len(rows) == 0:
        raise ValueError("`input_rows` cannot be empty.")
    if len(rows) > settings.max_input_rows:
        raise ValueError(
            f"`input_rows` exceeds CATPRED_API_MAX_INPUT_ROWS ({settings.max_input_rows})."
        )

    tmp_dir = Path(settings.temp_root).resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="api_input_", suffix=".csv", dir=str(tmp_dir))
    os.close(fd)
    pd.DataFrame(rows).to_csv(tmp_path, index=False)
    return tmp_path, tmp_path


def _resolve_input_file(
    payload: PredictRequest,
    settings: APISettings,
) -> tuple[str, Optional[str]]:
    if payload.input_file:
        resolved = _resolve_input_file_path(payload.input_file, settings)
        return str(resolved), None

    return _write_rows_to_temp_csv(payload.input_rows or [], settings=settings)


def _preview_output(output_file: str, preview_limit: int) -> tuple[int, list[dict[str, Any]]]:
    df = pd.read_csv(output_file)
    return len(df), df.head(preview_limit).to_dict(orient="records")


def create_app(
    router: Optional[InferenceBackendRouter] = None,
    settings: Optional[APISettings] = None,
) -> FastAPI:
    api_settings = settings or APISettings.from_env()
    app = FastAPI(title="CatPred API", version="0.2.0")
    backend_router = router or InferenceBackendRouter()
    default_fallback = _env_flag("CATPRED_MODAL_FALLBACK_TO_LOCAL", default=False)
    static_root = (Path(__file__).resolve().parent / "static").resolve()

    if static_root.exists():
        app.mount("/static", StaticFiles(directory=str(static_root)), name="static")

    @app.get("/", include_in_schema=False)
    def root() -> FileResponse:
        dist_index = static_root / "dist" / "index.html"
        if not dist_index.exists():
            raise HTTPException(
                status_code=404,
                detail="Frontend build not found. Run `npm run build` in catpred/web/frontend/.",
            )
        return FileResponse(dist_index)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict[str, Any]:
        readiness = backend_router.readiness()
        checkpoint_root = Path(api_settings.checkpoint_root).resolve()
        available_checkpoints = _discover_available_checkpoints(checkpoint_root)
        readiness["api"] = {
            "allow_input_file": api_settings.allow_input_file,
            "allow_unsafe_request_overrides": api_settings.allow_unsafe_request_overrides,
            "max_input_rows": api_settings.max_input_rows,
            "max_input_file_bytes": api_settings.max_input_file_bytes,
            "input_root": str(Path(api_settings.input_root).resolve()),
            "results_root": str(Path(api_settings.results_root).resolve()),
            "temp_root": str(Path(api_settings.temp_root).resolve()),
            "checkpoint_root": str(checkpoint_root),
            "available_checkpoints": available_checkpoints,
            "missing_checkpoints": [
                parameter for parameter in _SUPPORTED_PARAMETERS if parameter not in available_checkpoints
            ],
        }
        return readiness

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest) -> PredictResponse:
        temp_file: Optional[str] = None
        try:
            repo_root = _resolve_repo_root(payload.repo_root, api_settings)
            python_executable = _resolve_python_executable(payload.python_executable, api_settings)
            input_file, temp_file = _resolve_input_file(payload, settings=api_settings)
            safe_results_dir = _resolve_results_dir(payload.results_dir, api_settings)
            fallback = payload.fallback_to_local
            if fallback is None:
                fallback = default_fallback
            selected_backend = (payload.backend or backend_router.settings.default_backend).lower()
            checkpoint_dir = payload.checkpoint_dir
            if selected_backend == "local" or fallback:
                checkpoint_dir = _resolve_checkpoint_dir(payload.checkpoint_dir, api_settings)
            request_obj = PredictionRequest(
                parameter=payload.parameter.lower(),
                input_file=input_file,
                checkpoint_dir=checkpoint_dir,
                use_gpu=payload.use_gpu,
                repo_root=str(repo_root),
                python_executable=python_executable,
            )

            result = backend_router.predict(
                request_obj=request_obj,
                results_dir=safe_results_dir,
                backend_name=payload.backend,
                fallback_to_local=fallback,
            )

            row_count, preview_rows = _preview_output(
                result.output_file,
                preview_limit=api_settings.preview_rows,
            )
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
