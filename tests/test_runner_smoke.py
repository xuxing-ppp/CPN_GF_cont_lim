import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from cpn_gf.analysis import analyze_path
from cpn_gf.config import fingerprint, load_config, run_family_fingerprint
from cpn_gf.runner import (SCHEMA, _flow_buffer_configurations, _restore, _run_pilot,
                           _run_seed, pilot_config, resume_run, run_config)


class RunnerSmokeTests(unittest.TestCase):
    @staticmethod
    def _write_config(path, mul, output, warmup=2000, chains=64):
        path.write_text(f'''\
[model]
mul = {mul}
[hmc]
warmup = {warmup}
chains = {chains}
[compute]
device = "cpu"
seed = 7
[output]
root = "{output.as_posix()}"
''', encoding="utf-8")

    def test_tiny_cpu_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "tiny.toml"
            output = (root / "runs").as_posix()
            config.write_text(f'''\
[model]
mul = [2.0]
[lattice]
L = 4
[hmc]
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
root = "{output}"
''', encoding="utf-8")
            experiment, manifests = run_config(config)
            run_dir = next(experiment.glob("mul_*"))
            self.assertTrue((run_dir / "results" / "model.npz").is_file())
            self.assertTrue((run_dir / "results" / "covariant.npz").is_file())
            self.assertTrue((run_dir / "checkpoint.pt").is_file())
            self.assertFalse(any((run_dir / "observations").glob("*config*")))
            for kind in ("model", "covariant"):
                with np.load(run_dir / "results" / f"{kind}.npz",
                             allow_pickle=False) as data:
                    self.assertEqual(data["n_samples_per_chain"].tolist(), [2] * 6)
                    self.assertIn("unflowed_chi_m", data.files)
                    self.assertIn("unflowed_chi_t_Q_z", data.files)
                    self.assertIn("unflowed_chi_t_Q_U", data.files)
                    self.assertFalse(bool(data["unflowed_Q_s_applicable"]))
            chunk = next((run_dir / "observations").glob("flow_*.npz"))
            with np.load(chunk, allow_pickle=False) as data:
                self.assertEqual(data["flow_kinds"].tolist(), ["model", "covariant"])
                self.assertIn("model__E_action", data.files)
                self.assertIn("covariant__E_action", data.files)
            self.assertIn(manifests[0]["status"], ("ok", "complete_with_warning"))
            self.assertIsNone(manifests[0]["acceptance_rate_s"])
            self.assertEqual(manifests[0]["completed_total_samples"], 12)
            self.assertEqual(manifests[0]["committed_chunks"], 1)
            self.assertEqual(manifests[0]["run_seed"], _run_seed(load_config(config), 2.0))
            resumed = resume_run(experiment)
            self.assertEqual(resumed["status"], "complete")
            self.assertEqual(resumed["runs"][0]["action"], "skipped")

    def test_automatic_flow_buffer_targets(self):
        cfg = {"flow": {"buffer_configurations": 0},
               "compute": {"device": "cpu", "max_vram_fraction": 0.70}}
        self.assertEqual(_flow_buffer_configurations(cfg, 48, 64), 384)
        self.assertEqual(_flow_buffer_configurations(cfg, 96, 64), 128)
        self.assertEqual(_flow_buffer_configurations(cfg, 160, 64), 64)

    def test_flow_kinds_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            for value, message in (("[]", "non-empty"),
                                   ('["model", "model"]', "duplicates"),
                                   ('["unknown"]', "unsupported")):
                path.write_text(f"[flow]\nkinds = {value}\n", encoding="utf-8")
                with self.assertRaisesRegex((ValueError, TypeError), message):
                    load_config(path)
            path.write_text('[flow]\nkinds = ["model", "covariant"]\n',
                            encoding="utf-8")
            self.assertEqual(load_config(path)["flow"]["kinds"],
                             ["model", "covariant"])

    def test_experiment_resume_prioritizes_checkpoint_then_starts_added_mul(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = Path(directory) / "experiment"
            experiment.mkdir()
            root_config = experiment / "config.toml"
            old_config = Path(directory) / "old.toml"
            self._write_config(root_config, "[0.7, 0.8, 0.9]", Path(directory) / "runs")
            self._write_config(old_config, "[0.8, 0.9]", Path(directory) / "runs")
            with root_config.open("a", encoding="utf-8") as fh:
                fh.write("\n[analysis]\nmin_t_over_a2_for_fit = 2.0\n")
            for mul, phase in ((0.8, "production"), (0.9, "complete")):
                run_dir = experiment / f"mul_{mul:g}"
                run_dir.mkdir()
                shutil.copy2(old_config, run_dir / "config.toml")
                (run_dir / "manifest.json").write_text(json.dumps({
                    "phase": phase, "status": "ok" if phase == "complete" else "running",
                    "model": {"mul": mul}}), encoding="utf-8")

            events, restored_thresholds = [], []

            def restore(cfg, run_dir):
                events.append(("restore", run_dir.name))
                restored_thresholds.append(cfg["analysis"]["min_t_over_a2_for_fit"])
                mul = float(run_dir.name.removeprefix("mul_"))
                return object(), {"model": {"mul": mul}}, 0, 0, 0

            def prepare(cfg, model, run_dir, seed):
                events.append(("prepare", run_dir.name))
                return object(), {"model": model}, 0, 0, 0

            def production(cfg, run_dir, engine, manifest, samples, chunks, streak):
                events.append(("production", run_dir.name))
                return {"status": "ok"}

            with patch("cpn_gf.runner._restore", side_effect=restore), \
                    patch("cpn_gf.runner._prepare_new", side_effect=prepare), \
                    patch("cpn_gf.runner._production", side_effect=production):
                result = resume_run(experiment)

            self.assertEqual(events, [
                ("restore", "mul_0.8"), ("production", "mul_0.8"),
                ("prepare", "mul_0.7"), ("production", "mul_0.7")])
            self.assertEqual(restored_thresholds, [2.0])
            self.assertEqual([(x["mul"], x["action"]) for x in result["runs"]],
                             [(0.8, "resumed"), (0.7, "started"), (0.9, "skipped")])
            self.assertTrue((experiment / "mul_0.7" / "config.toml").is_file())

    def test_run_seed_does_not_depend_on_mul_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first.toml", root / "second.toml"
            self._write_config(first, "[0.7, 0.8]", root / "runs")
            self._write_config(second, "[0.8, 0.7]", root / "runs")
            self.assertEqual(_run_seed(load_config(first), 0.7),
                             _run_seed(load_config(second), 0.7))
            self.assertNotEqual(_run_seed(load_config(first), 0.7),
                                _run_seed(load_config(first), 0.8))

    def test_experiment_may_resume_old_chains_then_start_new_chains(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = root / "experiment"
            experiment.mkdir()
            root_config = experiment / "config.toml"
            old_config = root / "old.toml"
            self._write_config(root_config, "[0.8, 0.9]", root / "runs", chains=128)
            self._write_config(old_config, "[0.8]", root / "runs")
            run_dir = experiment / "mul_0.8"
            run_dir.mkdir()
            shutil.copy2(old_config, run_dir / "config.toml")
            (run_dir / "manifest.json").write_text(json.dumps({
                "phase": "production", "status": "running",
                "model": {"mul": 0.8}}), encoding="utf-8")
            seen = []

            def restore(cfg, path):
                seen.append(("restore", cfg["hmc"]["chains"]))
                return object(), {"model": {"mul": 0.8}}, 0, 0, 0

            def prepare(cfg, model, path, seed):
                seen.append(("prepare", cfg["hmc"]["chains"]))
                return object(), {"model": model}, 0, 0, 0

            with patch("cpn_gf.runner._restore", side_effect=restore), \
                    patch("cpn_gf.runner._prepare_new", side_effect=prepare), \
                    patch("cpn_gf.runner._production", return_value={"status": "ok"}):
                resume_run(experiment)
            self.assertEqual(seen, [("restore", 64), ("prepare", 128)])
            self.assertEqual(load_config(experiment / "mul_0.9" / "config.toml")
                             ["hmc"]["chains"], 128)

    def test_run_family_allows_only_chain_count_difference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first.toml", root / "second.toml"
            self._write_config(first, "[0.8]", root / "runs")
            self._write_config(second, "[0.8]", root / "runs", chains=128)
            self.assertEqual(run_family_fingerprint(load_config(first)),
                             run_family_fingerprint(load_config(second)))

    def test_run_family_allows_relative_error_difference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first.toml", root / "second.toml"
            self._write_config(first, "[0.8]", root / "runs")
            self._write_config(second, "[0.8]", root / "runs")
            with first.open("a", encoding="utf-8") as fh:
                fh.write("\n[sampling]\nrelative_error = 0.05\n")
            with second.open("a", encoding="utf-8") as fh:
                fh.write("\n[sampling]\nrelative_error = 0.02\n")
            self.assertEqual(run_family_fingerprint(load_config(first)),
                             run_family_fingerprint(load_config(second)))

    def test_experiment_resume_uses_new_convergence_batch_for_unfinished_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = root / "experiment"
            experiment.mkdir()
            root_config, child_config = experiment / "config.toml", root / "child.toml"
            self._write_config(root_config, "[0.8, 0.9]", root / "runs")
            self._write_config(child_config, "[0.8, 0.9]", root / "runs")
            with root_config.open("a", encoding="utf-8") as fh:
                fh.write("\n[sampling]\nconvergence_batch_total_samples = 12800\n"
                         "relative_error = 0.05\n")
            with child_config.open("a", encoding="utf-8") as fh:
                fh.write("\n[sampling]\nconvergence_batch_total_samples = 6400\n"
                         "relative_error = 0.02\n")
            for mul, phase in ((0.8, "production"), (0.9, "complete")):
                run_dir = experiment / f"mul_{mul:g}"
                run_dir.mkdir()
                shutil.copy2(child_config, run_dir / "config.toml")
                (run_dir / "manifest.json").write_text(json.dumps({
                    "phase": phase, "status": "running", "model": {"mul": mul}}),
                    encoding="utf-8")

            seen = []

            def restore(cfg, run_dir):
                seen.append((run_dir.name,
                             cfg["sampling"]["convergence_batch_total_samples"],
                             cfg["sampling"]["relative_error"]))
                return object(), {"model": {"mul": 0.8}}, 0, 0, 0

            with patch("cpn_gf.runner._restore", side_effect=restore), \
                    patch("cpn_gf.runner._production", return_value={"status": "ok"}):
                result = resume_run(experiment)

            self.assertEqual(seen, [("mul_0.8", 12800, 0.05)])
            self.assertEqual([(item["mul"], item["action"]) for item in result["runs"]],
                             [(0.8, "resumed"), (0.9, "skipped")])

    def test_restore_allows_convergence_batch_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_config, new_config = root / "old.toml", root / "new.toml"
            self._write_config(old_config, "[0.8]", root / "runs")
            self._write_config(new_config, "[0.8]", root / "runs")
            with old_config.open("a", encoding="utf-8") as fh:
                fh.write("\n[sampling]\nconvergence_batch_total_samples = 6400\n")
            with new_config.open("a", encoding="utf-8") as fh:
                fh.write("\n[sampling]\nconvergence_batch_total_samples = 12800\n")
            old_cfg, new_cfg = load_config(old_config), load_config(new_config)
            run_dir = root / "mul_0.8"
            run_dir.mkdir()
            manifest = {"schema_version": SCHEMA, "config_fingerprint": fingerprint(old_cfg),
                        "phase": "production", "status": "running",
                        "model": {"mul": 0.8}, "L": 8, "chains": 4,
                        "sampling": old_cfg["sampling"], "analysis": old_cfg["analysis"]}
            (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            class Engine:
                def load_state_dict(self, state):
                    pass

            checkpoint = {"phase": "production", "engine": {}, "samples": 10,
                          "chunks": 2, "convergence_streak": 3}
            with patch("cpn_gf.runner._new_engine", return_value=Engine()), \
                    patch("cpn_gf.runner.load_checkpoint", return_value=checkpoint):
                _, restored, _, _, streak = _restore(new_cfg, run_dir)

            self.assertEqual(streak, 3)
            self.assertEqual(restored["sampling"]["convergence_batch_total_samples"], 12800)
            self.assertEqual(restored["config_fingerprint"], fingerprint(new_cfg))

    def test_restore_reopens_complete_checkpoint_for_stricter_relative_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_config, new_config = root / "old.toml", root / "new.toml"
            self._write_config(old_config, "[0.8]", root / "runs")
            self._write_config(new_config, "[0.8]", root / "runs")
            with old_config.open("a", encoding="utf-8") as fh:
                fh.write("\n[sampling]\nrelative_error = 0.05\n")
            with new_config.open("a", encoding="utf-8") as fh:
                fh.write("\n[sampling]\nrelative_error = 0.02\n")
            old_cfg, new_cfg = load_config(old_config), load_config(new_config)
            run_dir = root / "mul_0.8"
            run_dir.mkdir()
            manifest = {"schema_version": SCHEMA, "config_fingerprint": fingerprint(old_cfg),
                        "phase": "complete", "status": "ok",
                        "model": {"mul": 0.8}, "L": 8, "chains": 4,
                        "sampling": old_cfg["sampling"], "analysis": old_cfg["analysis"]}
            (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            class Engine:
                def load_state_dict(self, state):
                    self.state = state

            checkpoint = {"phase": "complete", "engine": {"state": 1},
                          "samples": 10, "chunks": 2, "convergence_streak": 3}
            with patch("cpn_gf.runner._new_engine", return_value=Engine()), \
                    patch("cpn_gf.runner.load_checkpoint", return_value=checkpoint):
                _, restored, samples, chunks, streak = _restore(new_cfg, run_dir)

            self.assertEqual((samples, chunks, streak), (10, 2, 0))
            self.assertEqual(restored["phase"], "production")
            self.assertEqual(restored["sampling"]["relative_error"], 0.02)

    def test_relaxed_relative_error_leaves_completed_run_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = root / "experiment"
            experiment.mkdir()
            root_config, child_config = experiment / "config.toml", root / "child.toml"
            self._write_config(root_config, "[0.8]", root / "runs")
            self._write_config(child_config, "[0.8]", root / "runs")
            with root_config.open("a", encoding="utf-8") as fh:
                fh.write("\n[sampling]\nrelative_error = 0.05\n")
            with child_config.open("a", encoding="utf-8") as fh:
                fh.write("\n[sampling]\nrelative_error = 0.02\n")
            run_dir = experiment / "mul_0.8"
            run_dir.mkdir()
            shutil.copy2(child_config, run_dir / "config.toml")
            manifest = {"phase": "complete", "status": "complete_with_warning",
                        "model": {"mul": 0.8},
                        "sampling": {"relative_error": 0.02}}
            manifest_path = run_dir / "manifest.json"
            original = json.dumps(manifest)
            manifest_path.write_text(original, encoding="utf-8")

            with patch("cpn_gf.runner._restore") as restore:
                result = resume_run(experiment)

            restore.assert_not_called()
            self.assertEqual(result["runs"][0]["action"], "skipped")
            self.assertEqual(manifest_path.read_text(encoding="utf-8"), original)

    def test_stricter_relative_error_resumes_completed_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = root / "experiment"
            experiment.mkdir()
            root_config, child_config = experiment / "config.toml", root / "child.toml"
            self._write_config(root_config, "[0.8]", root / "runs")
            self._write_config(child_config, "[0.8]", root / "runs")
            with root_config.open("a", encoding="utf-8") as fh:
                fh.write("\n[sampling]\nrelative_error = 0.02\n")
            with child_config.open("a", encoding="utf-8") as fh:
                fh.write("\n[sampling]\nrelative_error = 0.05\n")
            run_dir = experiment / "mul_0.8"
            run_dir.mkdir()
            shutil.copy2(child_config, run_dir / "config.toml")
            (run_dir / "manifest.json").write_text(json.dumps({
                "phase": "complete", "status": "ok", "model": {"mul": 0.8},
                "chains": 64, "sampling": {"relative_error": 0.05}}), encoding="utf-8")
            (run_dir / "checkpoint.pt").touch()
            checkpoint = {"phase": "complete", "samples": 10}
            seen = []

            def restore(cfg, path):
                seen.append(cfg["sampling"]["relative_error"])
                return object(), {"model": {"mul": 0.8}}, 10, 1, 0

            with patch("cpn_gf.runner.load_checkpoint", return_value=checkpoint), \
                    patch("cpn_gf.runner._restore", side_effect=restore), \
                    patch("cpn_gf.runner._production", return_value={"status": "ok"}):
                result = resume_run(experiment)

            self.assertEqual(seen, [0.02])
            self.assertEqual(result["runs"][0]["action"], "resumed")

    def test_stricter_relative_error_preflight_rejects_all_unresumable_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = root / "experiment"
            experiment.mkdir()
            root_config, child_config = experiment / "config.toml", root / "child.toml"
            self._write_config(root_config, "[0.8, 0.9, 1.0]", root / "runs")
            self._write_config(child_config, "[0.8, 0.9]", root / "runs")
            with root_config.open("a", encoding="utf-8") as fh:
                fh.write("\n[sampling]\nrelative_error = 0.02\n")
            with child_config.open("a", encoding="utf-8") as fh:
                fh.write("\n[sampling]\nrelative_error = 0.05\n")
            for mul in (0.8, 0.9):
                run_dir = experiment / f"mul_{mul:g}"
                run_dir.mkdir()
                shutil.copy2(child_config, run_dir / "config.toml")
                (run_dir / "manifest.json").write_text(json.dumps({
                    "phase": "complete", "status": "ok", "model": {"mul": mul},
                    "chains": 64, "sampling": {"relative_error": 0.05}}), encoding="utf-8")
            (experiment / "mul_0.9" / "checkpoint.pt").touch()

            with patch("cpn_gf.runner.load_checkpoint", return_value={
                    "phase": "complete", "samples": 1563}), \
                    patch("cpn_gf.runner._restore") as restore, \
                    patch("cpn_gf.runner._prepare_new") as prepare:
                with self.assertRaisesRegex(
                        RuntimeError, "(?s)final checkpoint.*flow_max_total_samples"):
                    resume_run(experiment)

            restore.assert_not_called()
            prepare.assert_not_called()
            self.assertFalse((experiment / "mul_1").exists())

    def test_mul_directory_collisions_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "collision.toml"
            self._write_config(path, "[1.0000001, 1.0000002]", Path(directory) / "runs")
            with self.assertRaisesRegex(ValueError, "distinct mul_\\* directory names"):
                load_config(path)

    def test_pilot_lattice_defaults_preserve_single_attempt_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            self._write_config(path, "[1.0]", Path(directory) / "runs")
            with path.open("a", encoding="utf-8") as fh:
                fh.write("\n[lattice]\nL0 = 37\n")
            lattice = load_config(path)["lattice"]
            self.assertEqual(lattice["L0"], 37)
            self.assertEqual(lattice["max_L0"], 37)
            self.assertEqual(lattice["target_L0_over_xi"], 2.0)

    def test_invalid_pilot_lattice_growth_config_is_rejected(self):
        cases = (("target_L0_over_xi = 1.9\nmax_L0 = 50", "at least 2"),
                 ("target_L0_over_xi = 2\nmax_L0 = 19", "greater than or equal"),
                 ("L0 = 20.5\ntarget_L0_over_xi = 2\nmax_L0 = 50", "positive integer"))
        for values, message in cases:
            with self.subTest(values=values), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.toml"
                self._write_config(path, "[1.0]", Path(directory) / "runs")
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(f"\n[lattice]\n{values}\n")
                with self.assertRaisesRegex(ValueError, message):
                    load_config(path)

    def test_pilot_retries_with_larger_lattice_until_target_ratio(self):
        cfg = {
            "lattice": {"L0": 20, "target_L0_over_xi": 2.0, "max_L0": 100,
                        "pilot_warmup": 3, "pilot_total_samples": 8,
                        "minimum_L": 6, "L_multiple": 6},
            "hmc": {"chains": 4, "target_accept": 0.8},
            "sampling": {"tau_window_c": 5.0},
        }

        class Engine:
            step_size = 0.025

        with patch("cpn_gf.runner._new_engine", side_effect=[Engine(), Engine()]) as create, \
                patch("cpn_gf.runner.tune"), \
                patch("cpn_gf.runner._collect_modes", return_value=object()), \
                patch("cpn_gf.runner.scale_statistics", side_effect=[
                    (12.2, np.asarray([12.0, 12.4]), 1.0),
                    (10.0, np.asarray([9.8, 10.2]), 1.5)]):
            result = _run_pilot(cfg, {"mul": 1.0}, 100)

        self.assertEqual([call.args[2:5:2] for call in create.call_args_list],
                         [(20, 100), (25, 101)])
        self.assertEqual(result["L0"], 25)
        self.assertEqual(result["attempt_count"], 2)
        self.assertEqual([attempt["L0"] for attempt in result["attempts"]], [20, 25])
        self.assertEqual(result["recommended_L"], 60)

    def test_pilot_reports_failure_at_maximum_lattice(self):
        cfg = {
            "lattice": {"L0": 20, "target_L0_over_xi": 2.0, "max_L0": 25,
                        "pilot_warmup": 3, "pilot_total_samples": 8,
                        "minimum_L": 6, "L_multiple": 6},
            "hmc": {"chains": 4, "target_accept": 0.8},
            "sampling": {"tau_window_c": 5.0},
        }

        class Engine:
            step_size = 0.025

        with patch("cpn_gf.runner._new_engine", side_effect=[Engine(), Engine(), Engine()]) as create, \
                patch("cpn_gf.runner.tune"), \
                patch("cpn_gf.runner._collect_modes", return_value=object()), \
                patch("cpn_gf.runner.scale_statistics", side_effect=[
                    (12.0, np.asarray([12.0, 12.0]), 1.0),
                    (13.0, np.asarray([13.0, 13.0]), 1.0),
                    (13.0, np.asarray([13.0, 13.0]), 1.0)]):
            with self.assertRaisesRegex(RuntimeError, "max_L0=25.*below target 2"):
                _run_pilot(cfg, {"mul": 1.0}, 100)
        self.assertEqual([call.args[2] for call in create.call_args_list], [20, 24, 25])

    def test_analysis_uses_only_muls_listed_in_root_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            root.mkdir()
            self._write_config(root / "config.toml", "[0.8, 0.9]", root)
            for mul in (0.8, 0.9, 1.0):
                run = root / f"mul_{mul:g}"
                run.mkdir()
                (run / "manifest.json").write_text(json.dumps({
                    "phase": "complete", "status": "ok",
                    "model": {"mul": mul}}), encoding="utf-8")

            def fake_analyze(run, **kwargs):
                return {}, {"model": {"run": Path(run).name}}

            with patch("cpn_gf.analysis.analyze_run", side_effect=fake_analyze) as analyze, \
                    patch("cpn_gf.analysis._continuum_analysis") as continuum:
                result = analyze_path(root)

            self.assertEqual(list(result), ["mul_0.8", "mul_0.9"])
            self.assertEqual([call.args[0].name for call in analyze.call_args_list],
                             ["mul_0.8", "mul_0.9"])
            continuum.assert_called_once()

    def test_aggregate_only_reuses_existing_mul_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            root.mkdir()
            self._write_config(root / "config.toml", "[0.8, 0.9]", root)
            for mul in (0.8, 0.9):
                run = root / f"mul_{mul:g}"
                run.mkdir()
                (run / "manifest.json").write_text(json.dumps({
                    "phase": "complete", "status": "ok",
                    "model": {"mul": mul}}), encoding="utf-8")
                (run / "results").mkdir()
                (run / "results" / "model.json").write_text(json.dumps({
                    "xi_scale": mul, "xi_scale_error": 0.1}), encoding="utf-8")
                np.savez(run / "results" / "model.npz", rho=np.asarray([0.1]),
                         target_times=np.asarray([1.0]),
                         tE_action=np.asarray([0.2]),
                         tE_action_error=np.asarray([0.01]))

            with patch("cpn_gf.analysis.analyze_run") as analyze, \
                    patch("cpn_gf.analysis._continuum_analysis") as continuum:
                result = analyze_path(root, aggregate_only=True)

            analyze.assert_not_called()
            self.assertEqual(list(result), ["mul_0.8", "mul_0.9"])
            continuum.assert_called_once()

    def test_aggregate_only_requires_existing_mul_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            root.mkdir()
            self._write_config(root / "config.toml", "[0.8]", root)
            run = root / "mul_0.8"
            run.mkdir()
            (run / "manifest.json").write_text(json.dumps({
                "phase": "complete", "status": "ok", "model": {"mul": 0.8}}),
                encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "existing results.*mul_0.8"):
                analyze_path(root, aggregate_only=True)

    def test_experiment_resume_rejects_non_mul_config_changes_before_work(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = Path(directory) / "experiment"
            experiment.mkdir()
            root_config = experiment / "config.toml"
            child_source = Path(directory) / "child.toml"
            output = Path(directory) / "runs"
            self._write_config(root_config, "[0.8]", output, warmup=3000)
            self._write_config(child_source, "[0.8]", output, warmup=2000)
            run_dir = experiment / "mul_0.8"
            run_dir.mkdir()
            shutil.copy2(child_source, run_dir / "config.toml")
            (run_dir / "manifest.json").write_text(json.dumps({
                "phase": "production", "status": "running", "model": {"mul": 0.8}}),
                encoding="utf-8")
            with patch("cpn_gf.runner._restore") as restore:
                with self.assertRaisesRegex(ValueError, "settings other than"):
                    resume_run(experiment)
                restore.assert_not_called()

    def test_pilot_only_writes_summaries_and_honors_fixed_production_lattice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "pilot.toml"
            self._write_config(config, "[0.7, 0.8]", root / "runs")
            with config.open("a", encoding="utf-8") as fh:
                fh.write("\n[lattice]\nL = 12\nL0 = 20\n")

            def pilot_result(cfg, model, seed):
                xi = float(model["mul"]) * 2
                return {"xi": xi, "xi_error": 0.1, "tau_max": 1.5, "L0": 20,
                        "L": 30, "recommended_L": 30, "step_size": 0.02,
                        "samples_per_chain": 2, "total_samples": 128,
                        "attempt_count": 1,
                        "attempts": [{"L0": 20, "xi": xi, "L0_over_xi": 20 / xi}]}

            with patch("cpn_gf.runner._run_pilot", side_effect=pilot_result):
                result = pilot_config(config)

            experiment = Path(result["experiment"])
            self.assertTrue((experiment / "pilot_results.json").is_file())
            self.assertTrue((experiment / "pilot_results.csv").is_file())
            summary = json.loads((experiment / "pilot_results.json")
                                 .read_text(encoding="utf-8"))
            self.assertEqual(summary["runs"][0]["attempt_count"], 1)
            self.assertEqual(summary["runs"][0]["attempts"][0]["L0"], 20)
            csv_header = (experiment / "pilot_results.csv").read_text(
                encoding="utf-8").splitlines()[0]
            self.assertIn("attempt_count", csv_header)
            self.assertEqual([item["action"] for item in result["runs"]],
                             ["piloted", "piloted"])
            for mul in (0.7, 0.8):
                run_dir = experiment / f"mul_{mul:g}"
                manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["phase"], "pilot_complete")
                self.assertEqual(manifest["L"], 12)
                self.assertEqual(manifest["pilot"]["recommended_L"], 30)
                self.assertFalse((run_dir / "checkpoint.pt").exists())
                self.assertFalse((run_dir / "scale.npz").exists())
            with self.assertRaisesRegex(RuntimeError, "no production runs"):
                analyze_path(experiment)

    def test_restore_from_pilot_uses_saved_result_without_rerunning_pilot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.toml"
            self._write_config(config, "[0.8]", root / "runs")
            cfg = load_config(config)
            run_dir = root / "mul_0.8"
            run_dir.mkdir()
            shutil.copy2(config, run_dir / "config.toml")
            manifest = {"schema_version": SCHEMA, "config_fingerprint": fingerprint(cfg),
                "phase": "pilot_complete", "status": "pilot_complete",
                "model": {"mul": 0.8}, "analysis": cfg["analysis"],
                "run_seed": _run_seed(cfg, 0.8)}
            (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            expected = (object(), manifest, 0, 0, 0)
            with patch("cpn_gf.runner._start_after_pilot", return_value=expected) as start, \
                    patch("cpn_gf.runner._run_pilot") as pilot:
                self.assertIs(_restore(cfg, run_dir), expected)
                start.assert_called_once()
                pilot.assert_not_called()

    def test_restore_resets_convergence_streak_when_flow_minimum_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.toml"
            self._write_config(config, "[0.8]", root / "runs")
            with config.open("a", encoding="utf-8") as fh:
                fh.write("\n[analysis]\nmin_t_over_a2_for_fit = 2.0\n")
            cfg = load_config(config)
            run_dir = root / "mul_0.8"
            run_dir.mkdir()
            manifest = {"schema_version": SCHEMA, "config_fingerprint": fingerprint(cfg),
                        "phase": "production", "status": "running",
                        "model": {"mul": 0.8}, "L": 8, "chains": 4,
                        "analysis": {"min_t_over_a2_for_fit": 1.0}}
            (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            class Engine:
                def load_state_dict(self, state):
                    self.state = state

            checkpoint = {"phase": "production", "engine": {"state": 1},
                          "samples": 10, "chunks": 2, "convergence_streak": 3}
            with patch("cpn_gf.runner._new_engine", return_value=Engine()), \
                    patch("cpn_gf.runner.load_checkpoint", return_value=checkpoint):
                _, restored_manifest, samples, chunks, streak = _restore(cfg, run_dir)

            self.assertEqual((samples, chunks, streak), (10, 2, 0))
            self.assertEqual(restored_manifest["analysis"]["min_t_over_a2_for_fit"], 2.0)
            saved = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["analysis"]["min_t_over_a2_for_fit"], 2.0)

    def test_failed_pilot_is_persisted_and_stops_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "pilot.toml"
            self._write_config(config, "[0.7, 0.8]", root / "runs")
            with patch("cpn_gf.runner._run_pilot", side_effect=RuntimeError("bad xi")):
                with self.assertRaisesRegex(RuntimeError, "pilot failed in"):
                    pilot_config(config)
            experiment = next((root / "runs").iterdir())
            manifest = json.loads((experiment / "mul_0.7" / "manifest.json")
                                  .read_text(encoding="utf-8"))
            self.assertEqual(manifest["phase"], "pilot_error")
            self.assertIn("bad xi", manifest["error"])
            self.assertTrue((experiment / "pilot_results.json").is_file())
            self.assertFalse((experiment / "mul_0.8").exists())


if __name__ == "__main__":
    unittest.main()
