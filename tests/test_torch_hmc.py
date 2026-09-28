import math
import unittest

import numpy as np
import torch

from cpn_gf.hmc import BatchedHMC
from legacy.func.func_CPN_RefVil_HMC import CPN_RefVil_HMCSampler


class BatchedHMCTests(unittest.TestCase):
    def _engine(self, mod=0, beta1=0.0, alpha=0.2):
        model = {"N": 2, "beta": 0.8, "beta1": beta1, "alpha": alpha,
                 "alpha1": 0.1, "mod": mod}
        settings = {"mass_a": 1.0, "mass_z": 1.0, "s_step": 1.0,
                    "s_updates": 2, "s_max": 20, "initial_step_size": 0.005,
                    "trajectory_length": 0.02, "trajectory_jitter": 0.0}
        return BatchedHMC(2, 4, model, settings, "cpu", 123)

    def test_action_and_forces_match_numpy_mod0(self):
        engine = self._engine()
        for chain in range(engine.chains):
            ref = CPN_RefVil_HMCSampler(2, 4, 4, engine.beta, engine.beta1,
                                        engine.alpha, engine.alpha1, seed=1)
            ref.z = engine.z[chain].numpy().copy()
            ref.a = engine.a[chain].numpy().copy()
            ref.s = engine.s[chain].numpy().copy()
            ref._sync_U_from_a()
            self.assertAlmostEqual(float(engine.action()[chain]), ref.action(mod=0), places=10)
            np.testing.assert_allclose(engine._a_force()[chain].numpy(), ref._a_force(),
                                       rtol=1e-11, atol=1e-11)
            np.testing.assert_allclose(engine._z_force()[chain].numpy(), ref._z_force(mod=0),
                                       rtol=1e-11, atol=1e-11)

    def test_constraints_and_wrap_invariant(self):
        engine = self._engine()
        for _ in range(3):
            engine.trajectory()
        torch.testing.assert_close(torch.linalg.vector_norm(engine.z, dim=-1),
                                   torch.ones((2, 4, 4), dtype=torch.float64),
                                   rtol=1e-11, atol=1e-11)
        self.assertTrue(bool(torch.all(engine.a >= -math.pi)))
        self.assertTrue(bool(torch.all(engine.a < math.pi)))

    def test_action_and_forces_match_numpy_mod1(self):
        engine = self._engine(mod=1, beta1=-0.35, alpha=0.2)
        for chain in range(engine.chains):
            ref = CPN_RefVil_HMCSampler(2, 4, 4, engine.beta, engine.beta1,
                                        engine.alpha, engine.alpha1, seed=2)
            ref.z = engine.z[chain].numpy().copy()
            ref.a = engine.a[chain].numpy().copy()
            ref.s = engine.s[chain].numpy().copy()
            ref._sync_U_from_a()
            self.assertAlmostEqual(float(engine.action()[chain]), ref.action(mod=1), places=10)
            np.testing.assert_allclose(engine._a_force()[chain].numpy(), ref._a_force(),
                                       rtol=1e-11, atol=1e-11)
            np.testing.assert_allclose(engine._z_force()[chain].numpy(), ref._z_force(mod=1),
                                       rtol=1e-10, atol=1e-10)

    def test_topological_charges_match_numpy(self):
        engine = self._engine(alpha=0.2)
        qz, qu, qs = engine.topological_charges()
        for chain in range(engine.chains):
            ref = CPN_RefVil_HMCSampler(2, 4, 4, engine.beta, engine.beta1,
                                        engine.alpha, engine.alpha1, seed=3)
            ref.z = engine.z[chain].numpy().copy()
            ref.a = engine.a[chain].numpy().copy()
            ref.s = engine.s[chain].numpy().copy()
            ref._sync_U_from_a()
            expected = ref.topo_charge()
            self.assertEqual(float(qu[chain]), float(expected[0]))
            self.assertEqual(float(qz[chain]), float(expected[1]))
            self.assertEqual(float(qs[chain]), float(expected[2]))

    def test_checkpoint_restores_rng_exactly(self):
        first = self._engine()
        first.trajectory()
        state = first.state_dict()
        second = self._engine()
        second.load_state_dict(state)
        first.trajectory()
        second.trajectory()
        torch.testing.assert_close(first.z, second.z, rtol=0, atol=0)
        torch.testing.assert_close(first.a, second.a, rtol=0, atol=0)
        torch.testing.assert_close(first.s, second.s, rtol=0, atol=0)

    def test_alpha_zero_has_no_integer_field(self):
        engine = self._engine(alpha=0.0)
        self.assertIsNone(engine.s)
        before = engine.plaquette().clone()
        engine.trajectory()
        self.assertIsNone(engine.s)
        self.assertNotIn("s", engine.state_dict())
        self.assertEqual(before.shape, engine.plaquette().shape)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_action_and_forces_match_cpu(self):
        cpu = self._engine(mod=1, beta1=-0.35, alpha=0.2)
        gpu = BatchedHMC(2, 4,
                         {"N": 2, "beta": cpu.beta, "beta1": cpu.beta1,
                          "alpha": cpu.alpha, "alpha1": cpu.alpha1, "mod": cpu.mod},
                         {"mass_a": 1.0, "mass_z": 1.0, "s_step": 1.0,
                          "s_updates": 2, "s_max": 20, "initial_step_size": 0.005,
                          "trajectory_length": 0.02, "trajectory_jitter": 0.0},
                         "cuda:0", 456)
        gpu.z = cpu.z.cuda()
        gpu.a = cpu.a.cuda()
        gpu.s = cpu.s.cuda()
        torch.testing.assert_close(gpu.action().cpu(), cpu.action(), rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(gpu._a_force().cpu(), cpu._a_force(), rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(gpu._z_force().cpu(), cpu._z_force(), rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
