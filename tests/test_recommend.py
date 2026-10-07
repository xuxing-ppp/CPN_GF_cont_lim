import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout

import torch

from cpn_gf.config import config_for_mul, load_config
from cpn_gf.recommend import _measure_candidate, recommend_chains
from cpn_gf.runner import pilot_experiment, resume_experiment, run_config
from cpn_gf.config import scaled_model
from cpn_gf.__main__ import main


class RecommendChainsTests(unittest.TestCase):
    def config(self, root, chains="[8, 4]", L=12):
        path = root / "config.toml"
        path.write_text(f'[model]\nmul = [1.2, 0.8]\n[hmc]\nchains = {chains}\n'
                        f'[lattice]\nL = {L}\n[compute]\ndevice = "cpu"\n')
        return path

    @staticmethod
    def measure(cfg, model, L, chains):
        rates = {2: 10, 4: 19, 8: 20, 16: 20.4, 32: 20.5}
        return {"chains": chains, "status": "ok", "rate": rates[chains],
                "peak_bytes": 0, "budget_bytes": None, "flow_micro_batch": chains}

    def test_scalar_array_validation_and_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = load_config(self.config(root))
            self.assertEqual(config_for_mul(cfg, 0.8)["hmc"]["chains"], 4)
            self.assertEqual(cfg["hmc"]["chains"], [8, 4])
            cfg["model"]["mul"] = [0.8, 1.2]
            self.assertEqual(config_for_mul(cfg, 0.8)["hmc"]["chains"], 8)
            cfg = load_config(self.config(root, "6"))
            self.assertEqual(config_for_mul(cfg, 1.2)["hmc"]["chains"], 6)
            for counts in ("[4]", "[4, 1]", "true", "4.0", "[4, false]", "[]"):
                with self.subTest(counts=counts), self.assertRaises((TypeError, ValueError)):
                    load_config(self.config(root, counts))

    def test_fixed_lattice_each_mul_and_persistence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = self.config(root).read_bytes()
            with patch("cpn_gf.recommend._measure_candidate", side_effect=self.measure) as measure, \
                    patch("cpn_gf.runner.pilot_experiment") as pilot:
                result = recommend_chains(root, max_chains=64)
            self.assertEqual(result["recommended_chains"], [8, 8])
            self.assertEqual([row["mul"] for row in result["runs"]], [1.2, 0.8])
            self.assertEqual(measure.call_count, 10)
            pilot.assert_not_called()
            self.assertEqual(original, (root / "config.toml").read_bytes())
            self.assertEqual(json.loads((root / "chains_recommendations.json").read_text()), result)

    def test_run_resolves_and_freezes_each_mul(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config(root)
            with config.open("a") as fh:
                fh.write(f'\n[output]\nroot = "{root.as_posix()}/runs"\n')
            seen = []
            def prepare(cfg, model, child, seed):
                seen.append(cfg["hmc"]["chains"])
                return object(), {}, 0, 0, 0
            with patch("cpn_gf.runner._prepare_new", side_effect=prepare), \
                    patch("cpn_gf.runner._production", return_value={}):
                experiment, _ = run_config(config)
            self.assertEqual(seen, [8, 4])
            self.assertEqual(load_config(experiment / "mul_1.2/config.toml")["hmc"]["chains"], 8)
            self.assertEqual(load_config(experiment / "mul_0.8/config.toml")["hmc"]["chains"], 4)
            self.assertEqual(load_config(experiment / "config.toml")["hmc"]["chains"], [8, 4])

    def test_cli_directory_input_and_copyable_array(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.config(root)
            output = io.StringIO()
            with patch("cpn_gf.recommend._measure_candidate", side_effect=self.measure), redirect_stdout(output):
                main(["recommend-chains", "--run", str(root), "--max-chains", "8"])
            self.assertIn("mul=1.2", output.getvalue())
            self.assertIn("mul=0.8", output.getvalue())
            self.assertIn("hmc.chains = [4, 4]", output.getvalue())

    def _real_candidate(self, device):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config_for_mul(load_config(self.config(Path(directory), L=4)), 1.2)
            cfg["compute"]["device"] = device
            cfg["hmc"]["trajectory_length"] = 0.01
            row = _measure_candidate(cfg, scaled_model(cfg, 1.2), 4, 2, repeats=1, trajectories=1)
            self.assertEqual(row["status"], "ok")
            self.assertGreater(row["rate"], 0)
            self.assertGreater(row["flow_micro_batch"], 0)

    def test_real_cpu_candidate(self):
        self._real_candidate("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_real_cuda_candidate(self):
        self._real_candidate("cuda:0")

    def test_saved_pilots_and_production_chain_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.config(root, L=0)
            def pilot(cfg, model, seed):
                return {"xi": 2.0, "L": 12 if model["mul"] == 1.2 else 18,
                        "recommended_L": 12 if model["mul"] == 1.2 else 18}
            with patch("cpn_gf.runner._run_pilot", side_effect=pilot) as run_pilot, \
                    patch("cpn_gf.recommend._measure_candidate", side_effect=self.measure):
                first = recommend_chains(root, max_chains=32)
                recommend_chains(root, max_chains=32)
                self.assertEqual(run_pilot.call_count, 2)
            self.assertEqual([row["L"] for row in first["runs"]], [12, 18])
            self.config(root, "[16, 2]", L=0)
            seen = []
            def start(cfg, run_dir, manifest):
                seen.append((manifest["chains"], manifest["pilot"]["chains"]))
                return object(), manifest, 0, 0, 0
            with patch("cpn_gf.runner._start_after_pilot", side_effect=start), \
                    patch("cpn_gf.runner._production", side_effect=lambda cfg, rd, eng, man, *args: {"status": "complete"}):
                resume_experiment(root)
            self.assertEqual(seen, [(16, 8), (2, 4)])
            self.assertEqual(load_config(root / "mul_1.2/config.toml")["hmc"]["chains"], 16)
            self.assertEqual(load_config(root / "mul_0.8/config.toml")["hmc"]["chains"], 2)
            self.assertFalse(list(root.glob("mul_*/checkpoint.pt")))

    def test_started_production_keeps_frozen_chains(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.config(root, L=0)
            values = {"xi": 2.0, "L": 12, "recommended_L": 12}
            with patch("cpn_gf.runner._run_pilot", return_value=values.copy()):
                pilot_experiment(root)
            for child in root.glob("mul_*"):
                path = child / "manifest.json"
                manifest = json.loads(path.read_text())
                manifest["phase"] = "production"
                path.write_text(json.dumps(manifest))
            self.config(root, "[16, 2]", L=0)
            seen = []
            def restore(cfg, child):
                seen.append(cfg["hmc"]["chains"])
                return object(), json.loads((child / "manifest.json").read_text()), 0, 0, 0
            with patch("cpn_gf.runner._restore", side_effect=restore), \
                    patch("cpn_gf.runner._production", return_value={"status": "complete"}):
                resume_experiment(root)
            self.assertEqual(seen, [8, 4])

    def test_invalid_saved_pilot_fails_before_benchmark(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.config(root, L=0)
            with patch("cpn_gf.runner._run_pilot", return_value={"xi": 2.0, "L": 12, "recommended_L": 12}):
                pilot_experiment(root)
            path = root / "mul_1.2/manifest.json"
            manifest = json.loads(path.read_text())
            manifest["pilot"]["xi"] = None
            path.write_text(json.dumps(manifest))
            with patch("cpn_gf.recommend._measure_candidate") as measure:
                with self.assertRaisesRegex(ValueError, "invalid saved pilot"):
                    recommend_chains(root)
                measure.assert_not_called()

    def test_pilot_oom_retry_and_failed_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.config(root, L=0)
            values = {"xi": 2.0, "L": 12, "recommended_L": 12}
            with patch("cpn_gf.runner._run_pilot", side_effect=[torch.cuda.OutOfMemoryError(), values.copy(), ValueError("bad pilot")]):
                with self.assertRaisesRegex(ValueError, "bad pilot"):
                    pilot_experiment(root, retry_oom=True)
            manifest = json.loads((root / "mul_1.2/manifest.json").read_text())
            self.assertEqual(manifest["pilot"]["chains"], 4)
            with patch("cpn_gf.runner._run_pilot", return_value=values.copy()) as pilot:
                pilot_experiment(root, retry_oom=True)
                self.assertEqual(pilot.call_count, 1)

    def test_partial_recommendations_survive_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.config(root)
            def measure(cfg, model, L, chains):
                if model["mul"] == 0.8:
                    raise RuntimeError("benchmark failed")
                return self.measure(cfg, model, L, chains)
            with patch("cpn_gf.recommend._measure_candidate", side_effect=measure):
                with self.assertRaisesRegex(RuntimeError, "benchmark failed"):
                    recommend_chains(root, max_chains=32)
            result = json.loads((root / "chains_recommendations.json").read_text())
            self.assertEqual(len(result["runs"]), 1)

    def test_each_mul_can_receive_different_recommendation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.config(root)
            def measure(cfg, model, L, chains):
                row = self.measure(cfg, model, L, chains)
                if model["mul"] == 0.8:
                    row["rate"] = 10.0
                return row
            with patch("cpn_gf.recommend._measure_candidate", side_effect=measure):
                self.assertEqual(recommend_chains(root, max_chains=32)["recommended_chains"], [8, 2])


if __name__ == "__main__":
    unittest.main()
