import importlib.util
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch


def install_import_stubs() -> None:
    if importlib.util.find_spec("pandas") is None:
        pandas = types.ModuleType("pandas")
        pandas.DataFrame = object
        sys.modules.setdefault("pandas", pandas)

    if importlib.util.find_spec("numpy") is None:
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


class PredictionInputDeduplicationTests(unittest.TestCase):
    def test_duplicate_inputs_are_expanded_back_to_original_rows(self) -> None:
        if not hasattr(service.pd, "read_csv"):
            self.skipTest("pandas is stubbed in this test environment")

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_csv = tmp_path / "input.csv"
            records_file = tmp_path / "input.json.gz"
            output_csv = tmp_path / "input_output.csv"
            input_csv.write_text(
                "\n".join(
                    [
                        "Substrate,SMILES,sequence,pdbpath",
                        "first,C,AAAA,seq1.pdb",
                        "second,C,AAAA,seq1-copy.pdb",
                        "third,O,BBBB,seq2.pdb",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            paths = service.PreparedInputPaths(
                input_csv=str(input_csv),
                records_file=str(records_file),
                output_csv=str(output_csv),
            )

            deduplicated_paths, was_deduplicated = service._deduplicate_prediction_input(paths)
            self.assertTrue(was_deduplicated)

            unique_output = Path(deduplicated_paths.output_csv)
            unique_output.write_text(
                "\n".join(
                    [
                        "Substrate,SMILES,sequence,pdbpath,log10kcat_max,log10kcat_max_mve_uncal_var",
                        "first,C,AAAA,seq1.pdb,1.0,0.1",
                        "third,O,BBBB,seq2.pdb,2.0,0.2",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            service._expand_deduplicated_prediction_output(paths, deduplicated_paths)
            expanded_lines = output_csv.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(expanded_lines), 4)
        self.assertIn("first,C,AAAA,seq1.pdb,1.0,0.1", expanded_lines)
        self.assertIn("second,C,AAAA,seq1-copy.pdb,1.0,0.1", expanded_lines)
        self.assertIn("third,O,BBBB,seq2.pdb,2.0,0.2", expanded_lines)


class PredictionResultCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        service._PREDICTION_CACHE.clear()

    def tearDown(self) -> None:
        service._PREDICTION_CACHE.clear()

    def test_pipeline_reuses_cached_prediction_for_identical_request(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            input_csv = repo_root / "prepared.csv"
            input_csv.write_text("SMILES,sequence\nC,AAAA\n", encoding="utf-8")
            records_file = repo_root / "prepared.json.gz"
            records_file.write_bytes(b"records")
            output_csv = repo_root / "prepared_output.csv"
            checkpoint = repo_root / "checkpoints" / "fold_0" / "model_0" / "model.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")

            paths = service.PreparedInputPaths(
                input_csv=str(input_csv),
                records_file=str(records_file),
                output_csv=str(output_csv),
            )
            request = PredictionRequest(
                parameter="kcat",
                input_file=str(input_csv),
                checkpoint_dir=str(checkpoint.parent.parent.parent),
                repo_root=str(repo_root),
            )

            def write_final(parameter, prepared_paths, repo_root_arg, results_dir):
                final_output = Path(results_dir) / Path(prepared_paths.output_csv).name
                final_output.parent.mkdir(parents=True, exist_ok=True)
                final_output.write_text("SMILES,kcat\nC,1.23\n", encoding="utf-8")
                return str(final_output)

            with patch(
                "catpred.inference.service.prepare_prediction_inputs",
                return_value=paths,
            ), patch(
                "catpred.inference.service.run_inprocess_prediction",
            ) as runner, patch(
                "catpred.inference.service._write_postprocessed_predictions",
                side_effect=write_final,
            ) as postprocess:
                first = service.run_inprocess_prediction_pipeline(
                    request,
                    results_dir=str(repo_root / "results" / "first"),
                )
                second = service.run_inprocess_prediction_pipeline(
                    request,
                    results_dir=str(repo_root / "results" / "second"),
                )
                first_text = Path(first).read_text(encoding="utf-8")
                second_text = Path(second).read_text(encoding="utf-8")

        runner.assert_called_once()
        postprocess.assert_called_once()
        self.assertEqual(first_text, "SMILES,kcat\nC,1.23\n")
        self.assertEqual(second_text, "SMILES,kcat\nC,1.23\n")
        self.assertNotEqual(first, second)


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
