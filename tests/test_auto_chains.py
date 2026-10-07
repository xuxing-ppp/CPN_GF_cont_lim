import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from cpn_gf.config import (canonical, config_for_mul, fingerprint, load_config,
                           scaled_model, write_config)
from cpn_gf.runner import (_prepare_new, _restore, _run_seed, pilot_experiment,
                           resume_experiment, run_config)
from cpn_gf.recommend import recommend_chains


class AutoChainsTests(unittest.TestCase):
    def config(self, root, auto="true", L=4):
        path = root / "config.toml"
        path.write_text(f'[model]\nmul = [1.2, 0.8]\n[hmc]\nchains = [8, 4]\n'
                        f'auto_chains = {auto}\nwarmup = 0\n[lattice]\nL = {L}\n'
                        '[compute]\ndevice = "cpu"\n')
        return load_config(path)

    @staticmethod
    def measure(cfg, model, L, chains):
        rate = (10 if chains == 2 else 20) if model["mul"] == 1.2 else 10
        return {"chains": chains, "status": "ok", "rate": rate,
                "peak_bytes": 0, "budget_bytes": None, "flow_micro_batch": chains}

    @staticmethod
    def finish(cfg, child, engine, manifest):
        return engine, manifest, 0, 0, 0

    def prepare(self, cfg, root, mul=1.2):
        child = root / f"mul_{mul:g}"
        child.mkdir(exist_ok=True)
        probe = config_for_mul(cfg, mul)
        write_config(child / "config.toml", probe)
        return child, probe

    def test_bool_validation_and_legacy_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = self.config(root, "false")
            legacy = deepcopy(cfg)
            legacy["hmc"].pop("auto_chains")
            self.assertEqual(canonical(cfg), canonical(legacy))
            cfg["hmc"]["auto_chains"] = True
            self.assertEqual(fingerprint(cfg), fingerprint(legacy))
            for value in ("1", '"true"', "[true]"):
                with self.subTest(value=value), self.assertRaisesRegex(TypeError, "boolean"):
                    self.config(root, value)

    def test_fixed_size_selects_per_mul_and_writes_compact_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = self.config(root)
            original = (root / "config.toml").read_bytes()
            output = io.StringIO()
            with patch("cpn_gf.recommend._measure_candidate", side_effect=self.measure), \
                    patch("cpn_gf.runner._run_pilot") as pilot, \
                    patch("cpn_gf.runner._finish_scale", side_effect=self.finish), redirect_stdout(output):
                for mul, count in ((1.2, 4), (0.8, 2)):
                    child, probe = self.prepare(cfg, root, mul)
                    _, manifest, *_ = _prepare_new(probe, scaled_model(probe, mul), child, _run_seed(probe, mul))
                    saved = load_config(child / "config.toml")
                    self.assertEqual(saved["model"]["mul"], [mul])
                    self.assertEqual(saved["hmc"]["chains"], count)
                    self.assertEqual(manifest["chains"], count)
                    self.assertEqual(manifest["config_fingerprint"], fingerprint(saved))
                    self.assertEqual(json.loads((child / "chains_recommendation.json").read_text())["status"], "complete")
                pilot.assert_not_called()
            self.assertIn("mul=1.2 L=4: auto chains = 4", output.getvalue())
            self.assertIn("mul=0.8 L=4: auto chains = 2", output.getvalue())
            self.assertEqual(original, (root / "config.toml").read_bytes())

    def test_auto_pilot_then_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = self.config(root, L=0)
            child, probe = self.prepare(cfg, root)
            with patch("cpn_gf.runner._run_pilot", return_value={"xi": 2.0, "L": 4, "recommended_L": 4}) as pilot, \
                    patch("cpn_gf.recommend._measure_candidate", side_effect=self.measure), \
                    patch("cpn_gf.runner._finish_scale", side_effect=self.finish):
                _, manifest, *_ = _prepare_new(probe, scaled_model(probe, 1.2), child, 7)
            pilot.assert_called_once()
            self.assertEqual(manifest["pilot"]["chains"], 8)
            self.assertEqual(manifest["chains"], 4)
            self.assertTrue((root / "pilot_results.json").is_file())

    def test_pilot_only_then_resume_with_auto_setting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.config(root, "false", L=0)
            values = {"xi": 2.0, "L": 4, "recommended_L": 4}
            with patch("cpn_gf.runner._run_pilot", return_value=values.copy()), \
                    patch("cpn_gf.recommend._measure_candidate") as measure:
                pilot_experiment(root)
                measure.assert_not_called()
            self.config(root, "true", L=0)
            with patch("cpn_gf.runner._run_pilot") as pilot, \
                    patch("cpn_gf.recommend._measure_candidate", side_effect=self.measure), \
                    patch("cpn_gf.runner._finish_scale", side_effect=self.finish), \
                    patch("cpn_gf.runner._production", return_value={"status": "complete"}):
                resume_experiment(root)
                pilot.assert_not_called()
            self.assertEqual(load_config(root / "mul_1.2/config.toml")["hmc"]["chains"], 4)

    def test_cached_selection_survives_preproduction_interruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = self.config(root)
            child, probe = self.prepare(cfg, root)
            with patch("cpn_gf.recommend._measure_candidate", side_effect=self.measure), \
                    patch("cpn_gf.runner._new_engine", side_effect=RuntimeError("interrupted")):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    _prepare_new(probe, scaled_model(probe, 1.2), child, 7)
            with patch("cpn_gf.recommend._measure_candidate") as measure, \
                    patch("cpn_gf.runner._finish_scale", side_effect=self.finish):
                _, manifest, *_ = _restore(load_config(child / "config.toml"), child)
                measure.assert_not_called()
            self.assertEqual(manifest["chains"], 4)

    def test_standalone_recommendation_is_reused_by_auto_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = self.config(root)
            with patch("cpn_gf.recommend._measure_candidate", side_effect=self.measure):
                recommend_chains(root)
            child, probe = self.prepare(cfg, root)
            with patch("cpn_gf.recommend._measure_candidate") as measure, \
                    patch("cpn_gf.runner._finish_scale", side_effect=self.finish):
                _, manifest, *_ = _prepare_new(probe, scaled_model(probe, 1.2), child, 7)
                measure.assert_not_called()
            self.assertEqual(manifest["chains"], 4)
            self.assertTrue((child / "chains_recommendation.json").is_file())

    def test_stale_selection_is_remeasured(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = self.config(root)
            child, probe = self.prepare(cfg, root)
            with patch("cpn_gf.recommend._measure_candidate", side_effect=self.measure), \
                    patch("cpn_gf.runner._new_engine", side_effect=RuntimeError("interrupted")):
                with self.assertRaises(RuntimeError):
                    _prepare_new(probe, scaled_model(probe, 1.2), child, 7)
            path = child / "chains_recommendation.json"
            result = json.loads(path.read_text())
            result["key"]["hardware"]["processor"] = "another machine"
            path.write_text(json.dumps(result))
            with patch("cpn_gf.recommend._measure_candidate", side_effect=self.measure) as measure, \
                    patch("cpn_gf.runner._finish_scale", side_effect=self.finish):
                _restore(load_config(child / "config.toml"), child)
                self.assertGreater(measure.call_count, 0)

    def test_failure_preserves_pilot_and_candidates_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = self.config(root, L=0)
            child, probe = self.prepare(cfg, root)
            with patch("cpn_gf.runner._run_pilot", return_value={"xi": 2.0, "L": 4, "recommended_L": 4}), \
                    patch("cpn_gf.recommend._measure_candidate", side_effect=[self.measure(probe, {"mul": 1.2}, 4, 2), RuntimeError("failed")]):
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    _prepare_new(probe, scaled_model(probe, 1.2), child, 7)
            self.assertEqual(json.loads((child / "manifest.json").read_text())["phase"], "pilot_complete")
            result = json.loads((child / "chains_recommendation.json").read_text())
            self.assertEqual(result["status"], "error")
            self.assertEqual(len(result["candidates"]), 1)
            self.assertFalse((child / "checkpoint.pt").exists())

    def test_existing_checkpoint_never_recommends_again(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = self.config(root)
            child, probe = self.prepare(cfg, root)
            with patch("cpn_gf.recommend._measure_candidate", side_effect=self.measure), \
                    patch("cpn_gf.runner._finish_scale", side_effect=self.finish):
                engine, *_ = _prepare_new(probe, scaled_model(probe, 1.2), child, 7)
            before = engine.state_dict()
            with patch("cpn_gf.runner._recommend_mul") as recommend, \
                    patch("cpn_gf.runner._finish_scale", side_effect=self.finish):
                restored, manifest, *_ = _restore(load_config(child / "config.toml"), child)
                recommend.assert_not_called()
            self.assertEqual(manifest["chains"], 4)
            self.assertTrue((before["z"] == restored.state_dict()["z"]).all())

    def test_complete_cpu_auto_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.toml"
            config.write_text(f'''[model]
mul = [2.0]
[lattice]
L = 4
[hmc]
auto_chains = true
chains = 6
warmup = 50
initial_step_size = 0.005
trajectory_length = 0.01
trajectory_jitter = 0.0
[sampling]
scale_total_samples = 1200
scale_stride = 1
flow_min_total_samples = 12
flow_max_total_samples = 12
convergence_batch_total_samples = 12
relative_error = 0.99
consecutive_checks = 1
minimum_flow_stride = 1
[flow]
rho = [0.01]
epsilon = 0.05
kinds = ["model", "covariant"]
buffer_configurations = 12
[compute]
device = "cpu"
seed = 7
[output]
root = "{root.as_posix()}/runs"
''')
            def measure(cfg, model, L, chains):
                row = self.measure(cfg, {"mul": 1.2}, L, chains)
                return row
            with patch("cpn_gf.recommend._measure_candidate", side_effect=measure):
                experiment, manifests = run_config(config)
            child = experiment / "mul_2"
            self.assertEqual(manifests[0]["chains"], 4)
            self.assertEqual(manifests[0]["completed_total_samples"], 12)
            self.assertTrue((child / "results/model.npz").is_file())
            self.assertTrue((child / "results/covariant.npz").is_file())
            self.assertEqual(load_config(child / "config.toml")["hmc"]["chains"], 4)
            with patch("cpn_gf.runner._recommend_mul") as recommend:
                self.assertEqual(resume_experiment(experiment)["runs"][0]["action"], "skipped")
                recommend.assert_not_called()


if __name__ == "__main__":
    unittest.main()
