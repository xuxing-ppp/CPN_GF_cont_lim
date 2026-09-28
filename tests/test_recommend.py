import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cpn_gf.config import load_config
from cpn_gf.recommend import _temporary_pilot, recommend_chains


class RecommendChainsTests(unittest.TestCase):
    @staticmethod
    def _config(path, fixed_L=24):
        path.write_text(f'''\
[model]
mul = [0.8, 1.2, 1.0]
[lattice]
L = {fixed_L}
[compute]
device = "cpu"
''', encoding="utf-8")

    def test_fixed_lattice_selects_smallest_candidate_within_95_percent(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.toml"
            self._config(config)
            rates = iter((10.0, 19.0, 20.0, 20.4, 20.5))

            def measure(cfg, model, L, chains):
                return {"chains": chains, "status": "ok", "rate": next(rates),
                        "peak_bytes": 0, "budget_bytes": None,
                        "flow_micro_batch": chains}

            with patch("cpn_gf.recommend._measure_candidate", side_effect=measure):
                result = recommend_chains(config, max_chains=64)
            self.assertEqual(result["L"], 24)
            self.assertEqual(result["mul"], 1.2)
            self.assertEqual(result["recommended_chains"], 8)
            self.assertEqual([row["chains"] for row in result["candidates"]],
                             [2, 4, 8, 16, 32])

    def test_auto_lattice_requires_an_explicit_choice(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.toml"
            self._config(config, fixed_L=0)
            with self.assertRaisesRegex(ValueError, "choose --lattice-size"):
                recommend_chains(config)

    def test_temporary_pilot_uses_only_largest_mul(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.toml"
            self._config(config, fixed_L=0)
            seen = []

            def pilot(cfg, model, seed):
                seen.append(model["mul"])
                return {"recommended_L": 72}

            with patch("cpn_gf.recommend._run_pilot", side_effect=pilot):
                L, mul, chains, result = _temporary_pilot(load_config(config))
            self.assertEqual(seen, [1.2])
            self.assertEqual((L, mul, chains), (72, 1.2, 64))
            self.assertEqual(result["recommended_L"], 72)


if __name__ == "__main__":
    unittest.main()
