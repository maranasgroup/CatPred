from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _time_runs(
    label: str,
    runner: Callable[[Any, str], str],
    request: Any,
    results_dir: Path,
    warmup: int,
    runs: int,
) -> list[float]:
    for index in range(warmup):
        runner(request, str(results_dir / label / f"warmup_{index}"))

    timings = []
    for index in range(runs):
        started = time.perf_counter()
        runner(request, str(results_dir / label / f"run_{index}"))
        timings.append(time.perf_counter() - started)
    return timings


def _summary(timings: list[float]) -> dict[str, float]:
    return {
        "min_seconds": min(timings),
        "median_seconds": statistics.median(timings),
        "max_seconds": max(timings),
        "mean_seconds": statistics.mean(timings),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark CatPred legacy subprocess inference against the warm "
            "in-process inference path."
        )
    )
    parser.add_argument("--parameter", choices=["kcat", "km", "ki"], required=True)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--results-dir", default="benchmark_results")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument(
        "--skip-subprocess",
        action="store_true",
        help="Only benchmark the in-process path.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write benchmark results as JSON.",
    )
    args = parser.parse_args(argv)

    if args.runs < 1:
        raise ValueError("--runs must be at least 1.")
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative.")

    from catpred.inference.service import (
        run_inprocess_prediction_pipeline,
        run_prediction_pipeline,
    )
    from catpred.inference.types import PredictionRequest

    request = PredictionRequest(
        parameter=args.parameter,
        input_file=args.input_file,
        checkpoint_dir=args.checkpoint_dir,
        use_gpu=args.use_gpu,
        repo_root=args.repo_root,
        python_executable=args.python_executable,
    )
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "parameter": args.parameter,
        "input_file": str(Path(args.input_file).resolve()),
        "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
        "runs": args.runs,
        "warmup": args.warmup,
        "use_gpu": args.use_gpu,
    }

    if not args.skip_subprocess:
        subprocess_timings = _time_runs(
            label="subprocess",
            runner=run_prediction_pipeline,
            request=request,
            results_dir=results_dir,
            warmup=0,
            runs=args.runs,
        )
        report["subprocess"] = _summary(subprocess_timings)

    inprocess_timings = _time_runs(
        label="in_process",
        runner=run_inprocess_prediction_pipeline,
        request=request,
        results_dir=results_dir,
        warmup=args.warmup,
        runs=args.runs,
    )
    report["in_process"] = _summary(inprocess_timings)

    if "subprocess" in report:
        subprocess_median = report["subprocess"]["median_seconds"]  # type: ignore[index]
        inprocess_median = report["in_process"]["median_seconds"]  # type: ignore[index]
        report["median_speedup"] = subprocess_median / inprocess_median
        report["median_latency_reduction_percent"] = (
            (subprocess_median - inprocess_median) / subprocess_median * 100.0
        )

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
