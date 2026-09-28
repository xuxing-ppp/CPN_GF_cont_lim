from __future__ import annotations

import math

import torch


def _roll(x, shift, dim):
    if isinstance(dim, tuple) and not isinstance(shift, tuple):
        shift = (shift,) * len(dim)
    return torch.roll(x, shifts=shift, dims=dim)


def _log_i0(x):
    return torch.log(torch.special.i0e(x)) + torch.abs(x)


class BatchedHMC:
    """Independent constrained CPN HMC chains sharing a Torch device."""

    def __init__(self, chains, L, model, hmc, device, seed):
        self.chains, self.L, self.N = int(chains), int(L), int(model["N"])
        self.V, self.device = self.L * self.L, torch.device(device)
        self.beta, self.beta1 = float(model["beta"]), float(model["beta1"])
        self.alpha, self.alpha1, self.mod = float(model["alpha"]), float(model["alpha1"]), int(model["mod"])
        self.mass_a, self.mass_z = float(hmc["mass_a"]), float(hmc["mass_z"])
        self.s_step, self.s_updates, self.s_max = float(hmc["s_step"]), int(hmc["s_updates"]), int(hmc["s_max"])
        self.step_size = float(hmc["initial_step_size"])
        self.trajectory_length = float(hmc["trajectory_length"])
        self.trajectory_jitter = float(hmc["trajectory_jitter"])
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(seed))
        shape = (self.chains, self.L, self.L, self.N)
        z = torch.randn(shape, dtype=torch.float64, device=self.device, generator=self.generator)
        z = z + 1j * torch.randn(shape, dtype=torch.float64, device=self.device, generator=self.generator)
        self.z = z / torch.linalg.vector_norm(z, dim=-1, keepdim=True)
        self.a = (2 * math.pi * torch.rand((self.chains, self.L, self.L, 2), dtype=torch.float64,
                                           device=self.device, generator=self.generator) - math.pi)
        self.s = (torch.zeros((self.chains, self.L, self.L), dtype=torch.int64, device=self.device)
                  if abs(self.alpha) > 1e-8 else None)
        self.accepted_hmc = torch.zeros(self.chains, dtype=torch.int64, device=self.device)
        self.attempted_hmc = torch.zeros_like(self.accepted_hmc)
        self.accepted_metro = torch.zeros_like(self.accepted_hmc)
        self.attempted_metro = torch.zeros_like(self.accepted_hmc)

    @property
    def U(self):
        return torch.exp(1j * self.a)

    def _project(self, z, field):
        overlap = torch.sum(torch.conj(z) * field, dim=-1)
        return field - torch.real(overlap)[..., None] * z

    def _inner_x(self):
        return torch.sum(torch.conj(self.z) * _roll(self.z, -1, 1), dim=-1)

    def _inner_y(self):
        return torch.sum(torch.conj(self.z) * _roll(self.z, -1, 2), dim=-1)

    def plaquette(self):
        a = self.a
        result = a[..., 0] + _roll(a, -1, 1)[..., 1] - _roll(a, -1, 2)[..., 0] - a[..., 1]
        return result if self.s is None else result + 2 * math.pi * self.s

    def action(self):
        U, ix, iy = self.U, self._inner_x(), self._inner_y()
        cp = torch.real(U[..., 0] * torch.conj(ix) + U[..., 1] * torch.conj(iy))
        if self.mod == 0:
            improved = self.N * self.beta1 * (torch.abs(ix) ** 2 + torch.abs(iy) ** 2 - 2)
        else:
            arg = 2 * self.N * self.beta1
            improved = torch.sign(torch.as_tensor(self.beta1, device=self.device)) * (
                _log_i0(arg * torch.abs(ix)) + _log_i0(arg * torch.abs(iy)))
        f = self.plaquette()
        density = (-2 * self.N * self.beta * (cp - 2) - improved
                   + 0.5 * self.alpha * f * f - self.alpha1 * (torch.cos(f) - 1))
        return density.sum(dim=(1, 2))

    def _z_force(self):
        z, U, ix, iy = self.z, self.U, self._inner_x(), self._inner_y()
        F, F1 = torch.zeros_like(z), torch.zeros_like(z)
        F += _roll(U[..., 0], 1, 1)[..., None] * _roll(z, 1, 1)
        F += torch.conj(U[..., 0])[..., None] * _roll(z, -1, 1)
        F += _roll(U[..., 1], 1, 2)[..., None] * _roll(z, 1, 2)
        F += torch.conj(U[..., 1])[..., None] * _roll(z, -1, 2)
        if self.mod == 0:
            F1 += _roll(ix, 1, 1)[..., None] * _roll(z, 1, 1) * self.beta1
            F1 += torch.conj(ix)[..., None] * _roll(z, -1, 1) * self.beta1
            F1 += _roll(iy, 1, 2)[..., None] * _roll(z, 1, 2) * self.beta1
            F1 += torch.conj(iy)[..., None] * _roll(z, -1, 2) * self.beta1
        else:
            for inner, dim in ((ix, 1), (iy, 2)):
                mag = torch.abs(inner)
                arg = 2 * self.N * self.beta1 * mag
                ratio = torch.special.i1e(arg) / torch.special.i0e(arg)
                phase = torch.where(mag > 0, inner / mag, torch.zeros_like(inner))
                temp = ratio * phase
                F1 += _roll(temp, 1, dim)[..., None] * _roll(z, 1, dim) * abs(self.beta1)
                F1 += torch.conj(temp)[..., None] * _roll(z, -1, dim) * abs(self.beta1)
        return 2 * self.N * (self.beta * F + F1)

    def _a_force(self):
        U, ix, iy, f = self.U, self._inner_x(), self._inner_y(), self.plaquette()
        F = torch.zeros_like(self.a)
        F[..., 0] += 2 * self.N * self.beta * torch.imag(torch.conj(U[..., 0]) * ix)
        F[..., 1] += 2 * self.N * self.beta * torch.imag(torch.conj(U[..., 1]) * iy)
        sinf = torch.sin(f)
        F[..., 0] -= self.alpha * (f - _roll(f, 1, 2)) + self.alpha1 * (sinf - _roll(sinf, 1, 2))
        F[..., 1] -= self.alpha * (_roll(f, 1, 1) - f) + self.alpha1 * (_roll(sinf, 1, 1) - sinf)
        return F

    def _momenta(self):
        pa = math.sqrt(self.mass_a) * torch.randn(self.a.shape, dtype=torch.float64,
                                                   device=self.device, generator=self.generator)
        pz = math.sqrt(self.mass_z) * torch.randn(self.z.shape, dtype=torch.float64,
                                                   device=self.device, generator=self.generator)
        pz = pz + 1j * math.sqrt(self.mass_z) * torch.randn(self.z.shape, dtype=torch.float64,
                                                                  device=self.device, generator=self.generator)
        return pa, self._project(self.z, pz)

    def _kinetic(self, pa, pz):
        return (0.5 * (pa * pa).sum(dim=(1, 2, 3)) / self.mass_a
                + 0.5 * (torch.abs(pz) ** 2).sum(dim=(1, 2, 3)) / self.mass_z)

    def _wrap(self):
        wrapped = torch.remainder(self.a + math.pi, 2 * math.pi) - math.pi
        delta = torch.round((self.a - wrapped) / (2 * math.pi)).to(torch.int64)
        self.a = wrapped
        if self.s is not None:
            self.s += (delta[..., 0] + _roll(delta, -1, 1)[..., 1]
                       - _roll(delta, -1, 2)[..., 0] - delta[..., 1])

    @torch.no_grad()
    def trajectory(self, step_size=None, n_steps=None):
        eps = self.step_size if step_size is None else float(step_size)
        if n_steps is None:
            jitter = 1 + self.trajectory_jitter * (2 * torch.rand((), generator=self.generator,
                                                                   device=self.device).item() - 1)
            n_steps = max(1, round(self.trajectory_length * jitter / eps))
        old = (self.z.clone(), self.a.clone(), None if self.s is None else self.s.clone())
        pa, pz = self._momenta()
        old_h = self.action() + self._kinetic(pa, pz)
        pa += 0.5 * eps * self._a_force()
        pz += 0.5 * eps * self._project(self.z, self._z_force())
        for step in range(int(n_steps)):
            self.a += eps * pa / self.mass_a
            norm = torch.linalg.vector_norm(pz, dim=-1)
            safe = torch.where(norm > 0, norm, torch.ones_like(norm))
            angle = eps * norm / self.mass_z
            z_old = self.z
            self.z = torch.cos(angle)[..., None] * z_old + torch.sin(angle)[..., None] * pz / safe[..., None]
            pz = torch.cos(angle)[..., None] * pz - torch.sin(angle)[..., None] * z_old * norm[..., None]
            self.z /= torch.linalg.vector_norm(self.z, dim=-1, keepdim=True)
            pz = self._project(self.z, pz)
            coeff = 0.5 if step == int(n_steps) - 1 else 1.0
            pa += coeff * eps * self._a_force()
            pz += coeff * eps * self._project(self.z, self._z_force())
            pz = self._project(self.z, pz)
        self._wrap()
        delta_h = self.action() + self._kinetic(pa, pz) - old_h
        accept = torch.log(torch.rand(self.chains, device=self.device, generator=self.generator)) < -delta_h
        shape_z, shape_a = accept[:, None, None, None], accept[:, None, None, None]
        self.z = torch.where(shape_z, self.z, old[0])
        self.a = torch.where(shape_a, self.a, old[1])
        if self.s is not None:
            self.s = torch.where(accept[:, None, None], self.s, old[2])
        self.attempted_hmc += 1
        self.accepted_hmc += accept.to(torch.int64)
        if abs(self.alpha) > 1e-8:
            self._metro_s()
        return accept, delta_h

    @torch.no_grad()
    def _metro_s(self):
        if self.s is None:
            return
        base_f = self.plaquette()
        for _ in range(self.s_updates):
            mag = torch.floor(torch.abs(self.s_step * torch.randn(self.s.shape, dtype=torch.float64,
                                                                  device=self.device, generator=self.generator))).to(torch.int64) + 1
            sign = torch.where(torch.rand(self.s.shape, device=self.device, generator=self.generator) < 0.5, -1, 1)
            proposal = self.s + mag * sign
            in_range = torch.abs(proposal) <= self.s_max
            new_f = base_f + 2 * math.pi * (proposal - self.s)
            dS = 0.5 * self.alpha * (new_f * new_f - base_f * base_f)
            accept = in_range & (torch.log(torch.rand(self.s.shape, device=self.device,
                                                       generator=self.generator)) < -dS)
            self.s = torch.where(accept, proposal, self.s)
            base_f = torch.where(accept, new_f, base_f)
            self.attempted_metro += self.L * self.L
            self.accepted_metro += accept.sum(dim=(1, 2))

    def structure_modes(self):
        projector = self.z[..., :, None] * torch.conj(self.z[..., None, :])
        eye = torch.eye(self.N, dtype=torch.complex128, device=self.device)
        transformed = torch.fft.fftn(projector - eye / self.N, dim=(1, 2))
        modes = (torch.abs(transformed) ** 2).sum(dim=(3, 4)) / self.V
        return torch.stack((modes[:, 0, 0], modes[:, 1, 0], modes[:, 0, 1]), dim=1)

    def topological_charges(self):
        """Return Q_z, Q_U and, when present, Q_s for every chain."""
        z = self.z
        def dot(left, right): return torch.sum(torch.conj(left) * right, dim=-1)
        temp0 = dot(z, _roll(z, -1, 1)) * dot(_roll(z, -1, 1), _roll(z, -1, (1, 2))) * dot(_roll(z, -1, (1, 2)), z)
        temp1 = dot(z, _roll(z, -1, (1, 2))) * dot(_roll(z, -1, (1, 2)), _roll(z, -1, 2)) * dot(_roll(z, -1, 2), z)
        qz = torch.round((torch.angle(temp0) + torch.angle(temp1)).sum(dim=(1, 2)) / (2 * math.pi))
        U = self.U
        plaquette = U[..., 0] * _roll(U, -1, 1)[..., 1] * torch.conj(
            _roll(U, -1, 2)[..., 0] * U[..., 1])
        qu = torch.round(torch.angle(plaquette).sum(dim=(1, 2)) / (2 * math.pi))
        qs = None if self.s is None else self.s.sum(dim=(1, 2)).to(torch.float64)
        return qz, qu, qs

    def state_dict(self):
        state = {"z": self.z.detach().cpu().clone(), "a": self.a.detach().cpu().clone(),
                "accepted_hmc": self.accepted_hmc.cpu().clone(),
                "attempted_hmc": self.attempted_hmc.cpu().clone(),
                "accepted_metro": self.accepted_metro.cpu().clone(),
                "attempted_metro": self.attempted_metro.cpu().clone(),
                "step_size": self.step_size, "rng_state": self.generator.get_state().cpu().clone()}
        if self.s is not None:
            state["s"] = self.s.detach().cpu().clone()
        return state

    def load_state_dict(self, state):
        for key in ("z", "a", "accepted_hmc", "attempted_hmc", "accepted_metro", "attempted_metro"):
            setattr(self, key, state[key].to(self.device))
        self.s = state["s"].to(self.device) if "s" in state else None
        self.step_size = float(state["step_size"])
        # Generator state is serialized as a CPU ByteTensor.  Recent PyTorch
        # releases require set_state() to receive that CPU tensor even when
        # the generator itself is a CUDA generator.
        rng_state = state["rng_state"].detach().to(device="cpu", dtype=torch.uint8)
        self.generator.set_state(rng_state)


def tune(engine, trajectories, target=0.8, progress=None):
    """Dual-averaging step-size adaptation with a frozen averaged endpoint."""
    initial = engine.step_size
    mu, log_eps, log_bar, hbar = math.log(10 * initial), math.log(initial), math.log(initial), 0.0
    gamma, t0, kappa = 0.05, 10.0, 0.75
    for index in range(1, int(trajectories) + 1):
        _, delta_h = engine.trajectory(step_size=math.exp(log_eps))
        rate = float(torch.exp(-delta_h).clamp(max=1).mean().cpu())
        eta = 1.0 / (index + t0)
        hbar = (1 - eta) * hbar + eta * (target - rate)
        log_eps = min(0.0, max(math.log(1e-5), mu - math.sqrt(index) / gamma * hbar))
        weight = index ** -kappa
        log_bar = weight * log_eps + (1 - weight) * log_bar
        if progress is not None:
            progress.update()
    engine.step_size = math.exp(log_bar)
    return engine.step_size
