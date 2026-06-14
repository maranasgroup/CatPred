import importlib.util
import unittest
from unittest.mock import patch


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ValueError:
        return False


HAS_DATA_DEPS = all(_has_module(name) for name in ("numpy", "pandas", "rdkit", "torch"))
HAS_ESM_DEPS = HAS_DATA_DEPS and _has_module("esm")


@unittest.skipUnless(HAS_DATA_DEPS, "CatPred data dependencies are required")
class PopulateMissingEsmFeaturesTests(unittest.TestCase):
    def test_uses_batch_getter_once_for_unique_missing_sequences(self) -> None:
        from catpred.data import utils

        records = [
            {"name": "protein_a", "seq": "AAA"},
            {"name": "protein_b", "seq": "BBB"},
            {"name": "protein_c", "seq": "AAA"},
        ]
        calls = []

        def batch_getter(sequences, device):
            calls.append((list(sequences), device))
            return {sequence: f"features-{sequence}" for sequence in sequences}

        utils._populate_missing_esm2_features(
            records,
            sequence_feat_getter=None,
            sequence_batch_getter=batch_getter,
        )

        self.assertEqual(calls, [(["AAA", "BBB"], "cpu")])
        self.assertEqual(records[0]["esm2_feats"], "features-AAA")
        self.assertEqual(records[1]["esm2_feats"], "features-BBB")
        self.assertEqual(records[2]["esm2_feats"], "features-AAA")

    def test_fallback_getter_deduplicates_sequences(self) -> None:
        from catpred.data import utils

        records = [
            {"name": "protein_a", "seq": "AAA"},
            {"name": "protein_b", "seq": "BBB"},
            {"name": "protein_c", "seq": "AAA"},
        ]
        calls = []

        def single_getter(sequence, name, device):
            calls.append((sequence, name, device))
            return ([f"features-{sequence}"], None)

        utils._populate_missing_esm2_features(
            records,
            sequence_feat_getter=single_getter,
            sequence_batch_getter=None,
        )

        self.assertEqual(
            calls,
            [
                ("AAA", "protein_a", "cpu"),
                ("BBB", "protein_b", "cpu"),
            ],
        )
        self.assertEqual(records[0]["esm2_feats"], "features-AAA")
        self.assertEqual(records[1]["esm2_feats"], "features-BBB")
        self.assertEqual(records[2]["esm2_feats"], "features-AAA")


@unittest.skipUnless(HAS_ESM_DEPS, "ESM and torch dependencies are required")
class BatchedEsmCacheTests(unittest.TestCase):
    def test_get_many_esm_reprs_skips_cached_and_batches_unique_sequences(self) -> None:
        import torch

        from catpred.data import esm_utils

        cached = {"AAA": torch.tensor([[1.0]])}
        saved = []
        batch_calls = []

        def fake_load(path, cache_key, purpose, map_location):
            return cached.get(cache_key)

        def fake_save(value, path, cache_key):
            saved.append((cache_key, value.detach().cpu().clone()))

        def fake_batch(sequences):
            batch_calls.append(list(sequences))
            return [torch.tensor([[float(index + 2)]]) for index, _ in enumerate(sequences)]

        with patch("catpred.data.esm_utils.load_cache_value", side_effect=fake_load), patch(
            "catpred.data.esm_utils.save_cache_value", side_effect=fake_save
        ), patch(
            "catpred.data.esm_utils._run_esm_batch_with_fallback",
            side_effect=fake_batch,
        ):
            result = esm_utils.get_many_esm_reprs(
                ["AAA", "BBB", "CCC", "BBB"],
                device="cpu",
                batch_size=2,
            )

        self.assertEqual(batch_calls, [["BBB", "CCC"]])
        self.assertEqual([key for key, _ in saved], ["BBB", "CCC"])
        self.assertEqual(set(result), {"AAA", "BBB", "CCC"})
        self.assertTrue(torch.equal(result["AAA"], torch.tensor([[1.0]])))
        self.assertTrue(torch.equal(result["BBB"], torch.tensor([[2.0]])))
        self.assertTrue(torch.equal(result["CCC"], torch.tensor([[3.0]])))


if __name__ == "__main__":
    unittest.main()
