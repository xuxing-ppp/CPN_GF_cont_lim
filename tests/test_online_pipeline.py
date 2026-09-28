import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from cpn_gf.analysis import (_continuum_analysis, _relative_error_summary,
                             analyze_run)
from cpn_gf.__main__ import _print_analysis_summary
from cpn_gf.io import atomic_json, atomic_npz
from cpn_gf.stats import _interp


class OnlineAnalysisTests(unittest.TestCase):
    def test_cli_analysis_summary_is_compact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root / "continuum_fits.json", {
                "rho_0.1": {"fit": {"continuum": 1.0}},
                "rho_0.2": {"fit": None}})
            output = io.StringIO()
            summaries = {"mul_0.8": {"flowed": {"large": list(range(100))}},
                         "mul_0.9": {"flowed": {"large": list(range(100))}}}

            with redirect_stdout(output):
                _print_analysis_summary(root, summaries, aggregate_only=True)

            text = output.getvalue()
            self.assertIn("mode: aggregate only", text)
            self.assertIn("mul runs: mul_0.8, mul_0.9", text)
            self.assertIn("continuum fits: 1/2 rho values", text)
            self.assertNotIn("large", text)

    def test_continuum_uses_quadratic_fit_and_writes_mul_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            muls = [0.8, 0.9, 1.0, 1.1, 1.2]
            (root / "config.toml").write_text(
                "[model]\nmul = [0.8, 0.9, 1.0, 1.1, 1.2]\n"
                "[analysis]\nmin_t_over_a2_for_fit = 0.0\n", encoding="utf-8")
            runs, summaries = [], {}
            for index, mul in enumerate(muls):
                run = root / f"mul_{mul:g}"
                run.mkdir()
                atomic_json(run / "manifest.json", {
                    "model": {"mul": mul},
                    "analysis": {"min_t_over_a2_for_fit": 0.0}})
                xi = float(index + 2)
                inverse_xi2 = 1.0 / xi ** 2
                observable = 1.0 + 2.0 * inverse_xi2 + 3.0 * inverse_xi2 ** 2
                atomic_npz(run / "results.npz", rho=np.asarray([0.1]),
                           target_times=np.asarray([5.0]),
                           tE_action=np.asarray([observable]),
                           tE_action_error=np.asarray([0.01]))
                runs.append(run)
                summaries[run.name] = {"xi_scale": xi, "xi_scale_error": 0.01}

            _continuum_analysis(root, runs, summaries)

            fits = json.loads((root / "continuum_fits.json").read_text(encoding="utf-8"))
            fit = fits["rho_0.100000"]["fit"]
            self.assertEqual(fit["model"], "quadratic_in_inverse_xi2")
            self.assertAlmostEqual(fit["quadratic"], 3.0, places=6)
            self.assertAlmostEqual(fit["slope"], 2.0, places=6)
            self.assertAlmostEqual(fit["continuum"], 1.0, places=6)
            self.assertTrue((root / "plots" / "tE_action_vs_rho_by_mul.png").is_file())

    def test_vectorized_interpolation_matches_numpy(self):
        rng = np.random.default_rng(14)
        values = rng.normal(size=(5, 4, 3))
        times = np.asarray([0.0, 0.2, 0.7, 1.0])
        targets = np.asarray([-0.1, 0.0, 0.1, 0.7, 0.9, 1.0, 1.2])
        expected = np.empty((5, len(targets), 3))
        for sample in range(5):
            for observable in range(3):
                expected[sample, :, observable] = np.interp(
                    targets, times, values[sample, :, observable])
        np.testing.assert_allclose(_interp(values, times, targets), expected,
                                   rtol=1e-14, atol=1e-14)

    def test_analysis_reads_chunked_observables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rho = [0.05, 0.1]
            manifest = {"flow": {"rho": rho}, "L": 8, "chains": 3,
                        "sampling": {"relative_error": 1.0},
                        "analysis": {"min_t_over_a2_for_fit": 0.0}}
            atomic_json(root / "manifest.json", manifest)
            atomic_npz(root / "scale.npz", xi=np.asarray(2.0),
                       xi_loo=np.asarray([2.0, 2.0, 2.0]))
            rng = np.random.default_rng(2)
            shape = (3, 4, 4)
            values = np.stack((rng.random(shape) + 1.0,
                               rng.random(shape) + 4.0,
                               rng.random(shape) + 1.0,
                               rng.random(shape) + 1.0,
                               rng.integers(-2, 3, shape),
                               rng.integers(-2, 3, shape)), axis=-1)
            q_s = rng.integers(-3, 4, (3, 4))
            atomic_npz(root / "observations" / "flow_00000000.npz", values=values, Q_s=q_s,
                       output_steps=np.asarray([0, 10, 20, 40]),
                       times=np.asarray([0.0, 0.1, 0.2, 0.4]))
            result, summary = analyze_run(root)
            self.assertEqual(result["tE_action"].shape, (2,))
            self.assertEqual(result["n_samples_per_chain"].tolist(), [4, 4, 4])
            self.assertTrue(np.isfinite(summary["maximum_tE_relative_error"]))
            self.assertIn("unflowed_chi_t_Q_U", result)
            self.assertTrue(bool(result["unflowed_Q_s_applicable"]))

    def test_relative_error_ignores_times_below_fit_minimum(self):
        summary = _relative_error_summary(
            [0.9, 0.03, 0.01], [0.5, 1.0, 2.0], threshold=0.02,
            minimum_flow_time=1.0)
        self.assertEqual(summary["maximum_tE_relative_error"], 0.03)
        self.assertEqual(summary["maximum_tE_relative_error_flow_time"], 1.0)
        self.assertFalse(summary["converged"])

    def test_relative_error_with_no_eligible_times_is_vacuously_converged(self):
        summary = _relative_error_summary(
            [0.9, 0.8], [0.1, 0.2], threshold=0.02, minimum_flow_time=1.0)
        self.assertIsNone(summary["maximum_tE_relative_error"])
        self.assertIsNone(summary["maximum_tE_relative_error_flow_time"])
        self.assertTrue(summary["converged"])

    def test_nonfinite_eligible_error_does_not_converge(self):
        summary = _relative_error_summary(
            [0.01, np.nan], [1.0, 2.0], threshold=0.02, minimum_flow_time=1.0)
        self.assertIsNone(summary["maximum_tE_relative_error"])
        self.assertFalse(summary["converged"])


if __name__ == "__main__":
    unittest.main()
