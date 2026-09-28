from __future__ import annotations

import numpy as np
import torch

from .flow_engine import TorchCPNFlowBatch


@torch.no_grad()
def flow_observables(z, a, s, model, flow_cfg, output_steps, device):
    """Flow snapshots without modifying the live HMC tensors."""
    kind = ("covariant" if flow_cfg["covariant"] else
            "halfRefVil" if abs(model["alpha"]) < 1e-8 else "RefVil_fix_s")
    params = {**model, "flow_epsilon": float(flow_cfg["epsilon"]),
              "mass_a": float(flow_cfg["mass_a"]), "mass_z": float(flow_cfg["mass_z"])}
    total = len(z)
    requested = total
    batch_size, start, pieces = min(requested, total), 0, []
    while start < total:
        stop = min(total, start + batch_size)
        try:
            batch_s = None if s is None else s[start:stop]
            engine = TorchCPNFlowBatch(z[start:stop], a[start:stop], batch_s,
                                       params, kind, device)
            values = torch.empty((stop - start, len(output_steps), 6), dtype=torch.float64,
                                 device=device)
            values[:, 0] = engine.measure(mod=model["mod"])
            previous = int(output_steps[0])
            for out_index, target in enumerate(output_steps[1:], start=1):
                for _ in range(int(target) - previous):
                    engine.flow_step(mod=model["mod"])
                values[:, out_index] = engine.measure(mod=model["mod"])
                previous = int(target)
            pieces.append(values.cpu().numpy())
            start = stop
        except torch.cuda.OutOfMemoryError:
            if torch.device(device).type != "cuda" or batch_size <= 1:
                raise
            torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)
    return np.concatenate(pieces, axis=0), kind, batch_size
