import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch


def install_import_stubs() -> None:
    pandas = types.ModuleType("pandas")
    pandas.DataFrame = object
    sys.modules.setdefault("pandas", pandas)

    numpy = types.ModuleType("numpy")
    sys.modules.setdefault("numpy", numpy)

    rdkit = types.ModuleType("rdkit")
    chem = types.ModuleType("rdkit.Chem")
    rdkit.Chem = chem
    sys.modules.setdefault("rdkit", rdkit)
    sys.modules.setdefault("rdkit.Chem", chem)


install_import_stubs()

from catpred.inference.backends import LocalInferenceBackend
from catpred.inference.types import PredictionRequest
from catpred.inference import service


class LocalInferenceBackendTests(unittest.TestCase):
    def _request(self) -> PredictionRequest:
        return PredictionRequest(
            parameter="kcat",
            input_file="input.csv",
            checkpoint_dir="checkpoints/kcat",
            repo_root=".",
        )

    def test_local_backend_uses_inprocess_runner_by_default(self) -> None:
        backend = LocalInferenceBackend(use_subprocess=False)
        with patch(
            "catpred.inference.service.run_inprocess_prediction_pipeline",
            return_value="/tmp/out.csv",
        ) as fast_runner, patch(
            "catpred.inference.service.run_prediction_pipeline",
            return_value="/tmp/slow.csv",
        ) as subprocess_runner:
            result = backend.predict(self._request(), results_dir="/tmp/results")

        self.assertEqual(result.output_file, "/tmp/out.csv")
        self.assertEqual(result.metadata["mode"], "in_process")
        fast_runner.assert_called_once()
        subprocess_runner.assert_not_called()

    def test_local_backend_can_use_subprocess_compatibility_mode(self) -> None:
        backend = LocalInferenceBackend(use_subprocess=True)
        with patch(
            "catpred.inference.service.run_inprocess_prediction_pipeline",
            return_value="/tmp/out.csv",
        ) as fast_runner, patch(
            "catpred.inference.service.run_prediction_pipeline",
            return_value="/tmp/slow.csv",
        ) as subprocess_runner:
            result = backend.predict(self._request(), results_dir="/tmp/results")

        self.assertEqual(result.output_file, "/tmp/slow.csv")
        self.assertEqual(result.metadata["mode"], "subprocess")
        subprocess_runner.assert_called_once()
        fast_runner.assert_not_called()


class ModelCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        service._load_cached_model_objects.cache_clear()

    def test_model_objects_are_reused_for_unchanged_checkpoints(self) -> None:
        load_calls = []

        class FakePredictArgs:
            def __init__(self) -> None:
                self.checkpoint_paths = []
                self.no_cuda = True
                self.gpu = None
                self.pretrained_egnn_feats_path = ""

        fake_args_module = types.ModuleType("catpred.args")
        fake_args_module.PredictArgs = FakePredictArgs

        fake_train_module = types.ModuleType("catpred.train.make_predictions")

        def fake_load_model(args, generator):
            load_calls.append((tuple(args.checkpoint_paths), generator))
            args.pretrained_egnn_feats_path = "resolved-feats.pt"
            return args, "train_args", ["model"], ["scaler"], 1, ["task"]

        fake_train_module.load_model = fake_load_model

        fake_utils_module = types.ModuleType("catpred.utils")

        def fake_update_prediction_args(predict_args, train_args):
            predict_args.updated_from = train_args

        fake_utils_module.update_prediction_args = fake_update_prediction_args

        with TemporaryDirectory() as tmp_dir:
            checkpoint = Path(tmp_dir) / "model.pt"
            checkpoint.write_text("checkpoint", encoding="utf-8")
            first_args = SimpleNamespace(
                checkpoint_paths=[str(checkpoint)],
                no_cuda=True,
                gpu=None,
                pretrained_egnn_feats_path="",
            )
            second_args = SimpleNamespace(
                checkpoint_paths=[str(checkpoint)],
                no_cuda=True,
                gpu=None,
                pretrained_egnn_feats_path="",
            )

            with patch.dict(
                sys.modules,
                {
                    "catpred.args": fake_args_module,
                    "catpred.train.make_predictions": fake_train_module,
                    "catpred.utils": fake_utils_module,
                },
            ):
                first = service._load_model_objects_for_prediction(first_args)
                second = service._load_model_objects_for_prediction(second_args)

        self.assertEqual(len(load_calls), 1)
        self.assertEqual(first[2], ["model"])
        self.assertEqual(second[2], ["model"])
        self.assertEqual(first_args.updated_from, "train_args")
        self.assertEqual(second_args.updated_from, "train_args")


class FastPredictArgsTests(unittest.TestCase):
    def test_fast_predict_args_save_components_without_individual_predictions(self) -> None:
        class FakePredictArgs:
            def process_args(self) -> None:
                self.checkpoint_paths = [str(Path(self.checkpoint_dir) / "model.pt")]

        fake_args_module = types.ModuleType("catpred.args")
        fake_args_module.PredictArgs = FakePredictArgs

        with TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            checkpoint_dir = repo_root / "checkpoints" / "kcat"
            checkpoint_dir.mkdir(parents=True)
            paths = service.PreparedInputPaths(
                input_csv=str(repo_root / "input.csv"),
                records_file=str(repo_root / "input.json.gz"),
                output_csv=str(repo_root / "output.csv"),
            )
            request = PredictionRequest(
                parameter="kcat",
                input_file=str(repo_root / "input.csv"),
                checkpoint_dir="checkpoints/kcat",
                repo_root=str(repo_root),
            )

            with patch.dict(sys.modules, {"catpred.args": fake_args_module}):
                args = service._build_predict_args(request, paths, repo_root)

        self.assertFalse(args.individual_ensemble_predictions)
        self.assertTrue(args.save_uncertainty_components)
        self.assertEqual(args.uncertainty_method, "mve")
        self.assertEqual(args.smiles_columns, ["SMILES"])


if __name__ == "__main__":
    unittest.main()
