"""
Enzyme kinetics parameter prediction CLI for local/demo usage.

Usage:
    python demo_run.py --parameter <kcat|km|ki> --input_file <path_to_input_csv> --checkpoint_dir <path_to_checkpoint_dir> [--use_gpu]
"""

import argparse
import subprocess

from catpred.inference import PredictionRequest, run_prediction_pipeline


def main(args: argparse.Namespace) -> int:
    request = PredictionRequest(
        parameter=args.parameter.lower(),
        input_file=args.input_file,
        checkpoint_dir=args.checkpoint_dir,
        use_gpu=args.use_gpu,
        repo_root=".",
    )

    print("Predicting.. This will take a while..")
    try:
        final_output = run_prediction_pipeline(request=request, results_dir="../results")
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc))
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"Prediction command failed with exit code {exc.returncode}.")
        return exc.returncode if exc.returncode is not None else 1

    print(f"Output saved to {final_output}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict enzyme kinetics parameters.")
    parser.add_argument(
        "--parameter",
        type=str,
        choices=["kcat", "km", "ki"],
        required=True,
        help="Kinetics parameter to predict (kcat, km, or ki)",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input CSV file",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use GPU for prediction (default is CPU)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the model checkpoint directory",
    )

    raise SystemExit(main(parser.parse_args()))
