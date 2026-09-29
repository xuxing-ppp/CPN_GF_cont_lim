"""Numerical parity tests for the optional PyTorch flow backend."""

import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from legacy.func.func_CPN_RefVil_flow import CPN_RefVil_flow_fix_s, CPN_halfRefVil_flow


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TorchFlowParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from cpn_gf.flow_engine import TorchCPNFlowBatch
        cls.torch_flow = TorchCPNFlowBatch
        cls.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def _states(self, batch=2, L=5, N=2):
        rng = np.random.default_rng(20260921)
        z = rng.normal(size=(batch, L, L, N)) + 1j * rng.normal(size=(batch, L, L, N))
        z /= np.linalg.norm(z, axis=-1, keepdims=True)
        a = rng.uniform(-np.pi, np.pi, size=(batch, L, L, 2))
        s = rng.integers(-2, 3, size=(batch, L, L), dtype=np.int64)
        return z, a, s

    @staticmethod
    def _params(beta, beta1, alpha, alpha1):
        return {"beta": beta, "beta1": beta1, "alpha": alpha, "alpha1": alpha1,
                "flow_epsilon": 0.002, "mass_a": 1.0, "mass_z": 1.0}

    def _cpu_flows(self, kind, z, a, s, params):
        flows = []
        for index in range(len(z)):
            if kind == "RefVil_fix_s":
                flow = CPN_RefVil_flow_fix_s(
                    z.shape[-1], z.shape[1], z.shape[2], params["beta"], params["beta1"],
                    params["alpha"], params["alpha1"], epsilon=params["flow_epsilon"], n_step=1)
                flow.s = s[index].copy()
            else:
                beta, beta1, alpha1 = ((1.0, 0.0, 0.0) if kind == "covariant" else
                                        (params["beta"], params["beta1"], params["alpha1"]))
                flow = CPN_halfRefVil_flow(
                    z.shape[1], z.shape[2], z.shape[-1], beta, beta1, alpha1,
                    epsilon=params["flow_epsilon"], n_step=1)
            flow.z = z[index].copy()
            flow.a = a[index].copy()
            flow._sync_U_from_a()
            flows.append(flow)
        return flows

    @staticmethod
    def _cpu_measure(flow, mod):
        modes = flow.PP_corr_k()
        plaquette = (flow._plaquette() if hasattr(flow, "_plaquette") else
                     np.exp(1j * flow._plaquette_da()))
        q_u = np.sum(np.angle(plaquette)) / (2.0 * np.pi)
        return np.asarray((flow.action(mod=mod) / flow.V, modes[0, 0], modes[1, 0],
                           modes[0, 1], flow.topo_charge_z(), q_u), dtype=float)

    def _check_kind(self, kind, mod):
        z, a, s = self._states()
        params = self._params(1.15, -0.15, 0.35, 0.22)
        cpu = self._cpu_flows(kind, z, a, s, params)
        torch_flow = self.torch_flow(
            z, a, s if kind == "RefVil_fix_s" else None, params, kind, self.device)

        expected = np.stack([self._cpu_measure(flow, mod) for flow in cpu])
        initial_action = expected[:, 0].copy()
        np.testing.assert_allclose(torch_flow.measure(mod).cpu().numpy(), expected,
                                   rtol=1e-11, atol=1e-12)
        for flow in cpu:
            flow.flow_step(mod=mod, n_step=1)
        torch_flow.flow_step(mod=mod)
        np.testing.assert_allclose(torch_flow.z.cpu().numpy(), np.stack([x.z for x in cpu]),
                                   rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(torch_flow.a.cpu().numpy(), np.stack([x.a for x in cpu]),
                                   rtol=1e-11, atol=1e-12)
        if kind == "RefVil_fix_s":
            np.testing.assert_array_equal(torch_flow.s.cpu().numpy(), np.stack([x.s for x in cpu]))

        for _ in range(99):
            for flow in cpu:
                flow.flow_step(mod=mod, n_step=1)
            torch_flow.flow_step(mod=mod)
        expected = np.stack([self._cpu_measure(flow, mod) for flow in cpu])
        np.testing.assert_allclose(torch_flow.measure(mod).cpu().numpy(), expected,
                                   rtol=1e-9, atol=1e-11)
        self.assertTrue(np.all(expected[:, 0] <= initial_action + 1e-10 * np.abs(initial_action)))
        z_norm = torch.linalg.vector_norm(torch_flow.z, dim=-1)
        u_norm = torch.abs(torch_flow.U)
        self.assertLess(float(torch.max(torch.abs(z_norm - 1)).cpu()), 1e-12)
        self.assertLess(float(torch.max(torch.abs(u_norm - 1)).cpu()), 1e-12)

    def test_covariant_mod0(self):
        self._check_kind("covariant", 0)

    def test_half_refvil_mod1(self):
        self._check_kind("halfRefVil", 1)

    def test_fixed_s_mod0(self):
        self._check_kind("RefVil_fix_s", 0)

    def test_fixed_s_mod1(self):
        self._check_kind("RefVil_fix_s", 1)

    def test_fixed_s_wrap_preserves_physical_plaquette(self):
        z, a, s = self._states(batch=2)
        params = self._params(1.15, -0.15, 0.35, 0.22)
        flow = self.torch_flow(z, a, s, params, "RefVil_fix_s", self.device)
        shifts = torch.zeros_like(flow.a)
        shifts[..., 0] = 2.0 * np.pi
        shifts[:, ::2, :, 1] = -4.0 * np.pi
        flow.a = flow.a + shifts
        before = flow._plaquette_da().clone()
        flow._apply_periodicity_wrap()
        after = flow._plaquette_da()
        torch.testing.assert_close(after, before, rtol=1e-12, atol=1e-12)

    def test_flow_batch_runner_preserves_order_and_shape(self):
        from cpn_gf.flow_engine import flow_batch

        z, a, s = self._states(batch=3, L=4)
        params = self._params(1.15, -0.15, 0.35, 0.22)
        params.update(do_covariant=False, mod=1, output_steps=np.asarray([0, 1, 5]))
        values, kind, violations, max_increase = flow_batch(
            z, a, s, params, self.device)
        self.assertEqual(values.shape, (3, 3, 6))
        self.assertEqual(kind, "RefVil_fix_s")
        self.assertEqual(violations, 0)
        self.assertEqual(max_increase, 0.0)
        cpu = self._cpu_flows(kind, z, a, s, params)
        expected_initial = np.stack([self._cpu_measure(flow, 1) for flow in cpu])
        np.testing.assert_allclose(values[:, 0], expected_initial, rtol=1e-11, atol=1e-12)

    def test_model_and_covariant_flows_share_unchanged_initial_states(self):
        from cpn_gf.online_flow import flow_observables

        z, a, s = self._states(batch=3, L=4)
        z_tensor = torch.as_tensor(z, device=self.device)
        a_tensor = torch.as_tensor(a, device=self.device)
        s_tensor = torch.as_tensor(s, device=self.device)
        z_before, a_before, s_before = (z_tensor.clone(), a_tensor.clone(), s_tensor.clone())
        model = {"N": 2, "beta": 1.15, "beta1": -0.15,
                 "alpha": 0.35, "alpha1": 0.22, "mod": 1}
        flow_cfg = {"epsilon": 0.002, "mass_a": 1.0, "mass_z": 1.0}
        model_values, _, _ = flow_observables(
            z_tensor, a_tensor, s_tensor, model, flow_cfg,
            np.asarray([0, 1]), self.device, "model")
        covariant_values, _, _ = flow_observables(
            z_tensor, a_tensor, s_tensor, model, flow_cfg,
            np.asarray([0, 1]), self.device, "covariant")

        np.testing.assert_allclose(model_values[:, 0, 1:],
                                   covariant_values[:, 0, 1:], rtol=0, atol=0)
        torch.testing.assert_close(z_tensor, z_before)
        torch.testing.assert_close(a_tensor, a_before)
        torch.testing.assert_close(s_tensor, s_before)


if __name__ == "__main__":
    unittest.main()
