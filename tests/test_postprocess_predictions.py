import importlib.util
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ValueError:
        return False


HAS_PANDAS_NUMPY = _has_module("pandas") and _has_module("numpy")


@unittest.skipUnless(HAS_PANDAS_NUMPY, "pandas and numpy are required")
class PostprocessPredictionTests(unittest.TestCase):
    def setUp(self) -> None:
        if _has_module("rdkit"):
            return
        rdkit = types.ModuleType("rdkit")
        chem = types.ModuleType("rdkit.Chem")
        rdkit.Chem = chem
        sys.modules.setdefault("rdkit", rdkit)
        sys.modules.setdefault("rdkit.Chem", chem)

    def test_prefers_exact_uncertainty_component_columns(self) -> None:
        import pandas as pd

        from catpred.inference.service import postprocess_predictions

        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "preds.csv"
            pd.DataFrame(
                [
                    {
                        "SMILES": "C",
                        "log10kcat_max": 1.0,
                        "log10kcat_max_mve_uncal_var": 0.09,
                        "log10kcat_max_mve_uncal_aleatoric_var": 0.05,
                        "log10kcat_max_mve_uncal_epistemic_var": 0.04,
                    }
                ]
            ).to_csv(output_path, index=False)

            result = postprocess_predictions("kcat", str(output_path))

        self.assertAlmostEqual(result["Prediction_(s^(-1))"].iloc[0], 10.0)
        self.assertAlmostEqual(result["SD_total"].iloc[0], 0.3)
        self.assertAlmostEqual(result["SD_aleatoric"].iloc[0], 0.05 ** 0.5)
        self.assertAlmostEqual(result["SD_epistemic"].iloc[0], 0.2)


if __name__ == "__main__":
    unittest.main()
