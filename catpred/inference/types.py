from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionRequest:
    parameter: str
    input_file: str
    checkpoint_dir: str
    use_gpu: bool = False
    repo_root: str | None = None
    python_executable: str = "python"
    protein_records_file: str | None = None


@dataclass(frozen=True)
class PreparedInputPaths:
    input_csv: str
    records_file: str
    output_csv: str
