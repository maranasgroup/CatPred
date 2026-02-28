from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import csv
import json
import os
from typing import Any
from urllib import error, request as urllib_request

import pandas as pd

from .types import PredictionRequest


class InferenceBackendError(RuntimeError):
    """Raised when a backend cannot satisfy an inference request."""


@dataclass(frozen=True)
class BackendPredictionResult:
    backend_name: str
    output_file: str
    metadata: dict[str, Any] = field(default_factory=dict)


class InferenceBackend:
    name = "base"

    def readiness(self) -> dict[str, Any]:
        raise NotImplementedError

    def predict(self, request_obj: PredictionRequest, results_dir: str) -> BackendPredictionResult:
        raise NotImplementedError


class LocalInferenceBackend(InferenceBackend):
    name = "local"

    def __init__(self, repo_root: str | None = None) -> None:
        self._repo_root = repo_root

    def readiness(self) -> dict[str, Any]:
        root = Path(self._repo_root) if self._repo_root else Path.cwd()
        root = root.resolve()
        required = [
            root / "predict.py",
            root / "scripts" / "create_pdbrecords.py",
        ]
        missing = [str(path) for path in required if not path.exists()]
        return {
            "configured": True,
            "ready": len(missing) == 0,
            "missing_files": missing,
            "repo_root": str(root),
        }

    def predict(self, request_obj: PredictionRequest, results_dir: str) -> BackendPredictionResult:
        from .service import run_prediction_pipeline

        effective_request = request_obj
        if not request_obj.repo_root and self._repo_root:
            effective_request = replace(request_obj, repo_root=self._repo_root)

        output_file = run_prediction_pipeline(effective_request, results_dir=results_dir)
        return BackendPredictionResult(backend_name=self.name, output_file=output_file)


class ModalHTTPInferenceBackend(InferenceBackend):
    name = "modal"

    def __init__(
        self,
        endpoint: str | None,
        token: str | None = None,
        timeout_seconds: int = 900,
        repo_root: str | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._token = token
        self._timeout_seconds = timeout_seconds
        self._repo_root = repo_root

    def readiness(self) -> dict[str, Any]:
        configured = bool(self._endpoint)
        return {
            "configured": configured,
            "ready": configured,
            "endpoint": self._endpoint,
            "timeout_seconds": self._timeout_seconds,
        }

    def _resolve_input_file(self, request_obj: PredictionRequest) -> Path:
        input_path = Path(request_obj.input_file)
        if not input_path.is_absolute():
            root = Path(request_obj.repo_root or self._repo_root or Path.cwd()).resolve()
            input_path = (root / input_path).resolve()
        if not input_path.exists():
            raise FileNotFoundError(f'Input CSV not found for modal backend: "{input_path}"')
        return input_path

    def _resolve_results_dir(self, results_dir: str, request_obj: PredictionRequest) -> Path:
        out_dir = Path(results_dir)
        if not out_dir.is_absolute():
            root = Path(request_obj.repo_root or self._repo_root or Path.cwd()).resolve()
            out_dir = (root / out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    @staticmethod
    def _load_rows(csv_path: Path) -> list[dict[str, Any]]:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self._endpoint:
            raise InferenceBackendError(
                "Modal backend is not configured. Set CATPRED_MODAL_ENDPOINT."
            )

        encoded_payload = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        req = urllib_request.Request(
            url=self._endpoint,
            method="POST",
            data=encoded_payload,
            headers=headers,
        )
        try:
            with urllib_request.urlopen(req, timeout=self._timeout_seconds) as resp:
                raw = resp.read()
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise InferenceBackendError(
                f"Modal backend request failed with HTTP {exc.code}: {body}"
            ) from exc
        except error.URLError as exc:
            raise InferenceBackendError(
                f"Modal backend request failed: {exc.reason}"
            ) from exc

        try:
            decoded = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise InferenceBackendError("Modal backend returned non-JSON output.") from exc

        if not isinstance(decoded, dict):
            raise InferenceBackendError(
                "Modal backend response must be a JSON object."
            )
        return decoded

    def _materialize_output(
        self,
        response: dict[str, Any],
        input_path: Path,
        results_dir: str,
        request_obj: PredictionRequest,
    ) -> BackendPredictionResult:
        output_file = response.get("output_file")
        if isinstance(output_file, str) and output_file:
            resolved = Path(output_file).resolve()
            if resolved.exists():
                return BackendPredictionResult(
                    backend_name=self.name,
                    output_file=str(resolved),
                    metadata={"endpoint": self._endpoint, "mode": "output_file"},
                )

        output_rows = response.get("output_rows")
        if isinstance(output_rows, list):
            out_dir = self._resolve_results_dir(results_dir, request_obj)
            out_name = response.get("output_filename")
            if not isinstance(out_name, str) or not out_name:
                out_name = f"{input_path.stem}_modal_output.csv"
            if not out_name.endswith(".csv"):
                out_name = f"{out_name}.csv"

            final_output = out_dir / out_name
            pd.DataFrame(output_rows).to_csv(final_output, index=False)
            return BackendPredictionResult(
                backend_name=self.name,
                output_file=str(final_output),
                metadata={"endpoint": self._endpoint, "mode": "output_rows"},
            )

        output_csv_text = response.get("output_csv_text")
        if isinstance(output_csv_text, str) and output_csv_text:
            out_dir = self._resolve_results_dir(results_dir, request_obj)
            out_name = response.get("output_filename")
            if not isinstance(out_name, str) or not out_name:
                out_name = f"{input_path.stem}_modal_output.csv"
            if not out_name.endswith(".csv"):
                out_name = f"{out_name}.csv"
            final_output = out_dir / out_name
            final_output.write_text(output_csv_text, encoding="utf-8")
            return BackendPredictionResult(
                backend_name=self.name,
                output_file=str(final_output),
                metadata={"endpoint": self._endpoint, "mode": "output_csv_text"},
            )

        raise InferenceBackendError(
            "Modal backend response must include one of: output_file, output_rows, output_csv_text."
        )

    def predict(self, request_obj: PredictionRequest, results_dir: str) -> BackendPredictionResult:
        input_path = self._resolve_input_file(request_obj)
        payload = {
            "parameter": request_obj.parameter,
            "checkpoint_dir": request_obj.checkpoint_dir,
            "use_gpu": request_obj.use_gpu,
            "input_rows": self._load_rows(input_path),
            "input_filename": input_path.name,
        }
        response = self._post_json(payload)
        return self._materialize_output(
            response=response,
            input_path=input_path,
            results_dir=results_dir,
            request_obj=request_obj,
        )


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class BackendRouterSettings:
    default_backend: str = "local"
    modal_endpoint: str | None = None
    modal_token: str | None = None
    modal_timeout_seconds: int = 900
    repo_root: str | None = None

    @classmethod
    def from_env(cls) -> "BackendRouterSettings":
        timeout = int(os.environ.get("CATPRED_MODAL_TIMEOUT_SECONDS", "900"))
        return cls(
            default_backend=os.environ.get("CATPRED_DEFAULT_BACKEND", "local").lower(),
            modal_endpoint=os.environ.get("CATPRED_MODAL_ENDPOINT"),
            modal_token=os.environ.get("CATPRED_MODAL_TOKEN"),
            modal_timeout_seconds=timeout,
            repo_root=os.environ.get("CATPRED_REPO_ROOT"),
        )


class InferenceBackendRouter:
    def __init__(self, settings: BackendRouterSettings | None = None) -> None:
        self.settings = settings or BackendRouterSettings.from_env()
        self._backends: dict[str, InferenceBackend] = {
            "local": LocalInferenceBackend(repo_root=self.settings.repo_root),
            "modal": ModalHTTPInferenceBackend(
                endpoint=self.settings.modal_endpoint,
                token=self.settings.modal_token,
                timeout_seconds=self.settings.modal_timeout_seconds,
                repo_root=self.settings.repo_root,
            ),
        }
        if self.settings.default_backend not in self._backends:
            raise ValueError(
                f"Unsupported CATPRED_DEFAULT_BACKEND '{self.settings.default_backend}'. "
                "Use one of: local, modal."
            )

    def available_backends(self) -> list[str]:
        return sorted(self._backends.keys())

    def resolve_backend(self, backend_name: str | None = None) -> InferenceBackend:
        selected = (backend_name or self.settings.default_backend).lower()
        if selected not in self._backends:
            raise ValueError(
                f"Unsupported backend '{selected}'. Use one of: {', '.join(self.available_backends())}."
            )

        backend = self._backends[selected]
        state = backend.readiness()
        if not state.get("configured", False):
            raise InferenceBackendError(
                f"Backend '{selected}' is not configured. Readiness: {state}"
            )
        return backend

    def predict(
        self,
        request_obj: PredictionRequest,
        results_dir: str,
        backend_name: str | None = None,
        fallback_to_local: bool = False,
    ) -> BackendPredictionResult:
        selected_name = (backend_name or self.settings.default_backend).lower()
        backend: InferenceBackend | None = None
        try:
            backend = self.resolve_backend(selected_name)
            return backend.predict(request_obj=request_obj, results_dir=results_dir)
        except Exception as exc:
            if fallback_to_local and selected_name != "local":
                local_backend = self._backends["local"]
                local_result = local_backend.predict(request_obj=request_obj, results_dir=results_dir)
                metadata = dict(local_result.metadata)
                metadata["fallback_from"] = selected_name
                metadata["fallback_reason"] = str(exc)
                return BackendPredictionResult(
                    backend_name=local_result.backend_name,
                    output_file=local_result.output_file,
                    metadata=metadata,
                )
            raise

    def readiness(self) -> dict[str, Any]:
        backends = {name: backend.readiness() for name, backend in self._backends.items()}
        default_state = backends[self.settings.default_backend]
        return {
            "default_backend": self.settings.default_backend,
            "ready": bool(default_state.get("ready", False)),
            "backends": backends,
            "fallback_to_local_enabled": _env_flag("CATPRED_MODAL_FALLBACK_TO_LOCAL", default=False),
        }
