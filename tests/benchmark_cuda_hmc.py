"""Representative batched-HMC CUDA/CPU throughput benchmark."""

import time
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cpn_gf.hmc import BatchedHMC


L = 48
CHAINS = 64
TRAJECTORIES = 20


def make(device):
    model = {"N": 2, "beta": 1.2, "beta1": 0.0, "alpha": 0.0,
             "alpha1": 0.0, "mod": 0}
    hmc = {"mass_a": 1.0, "mass_z": 1.0, "s_step": 1.0, "s_updates": 3,
           "s_max": 100, "initial_step_size": 0.01,
           "trajectory_length": 0.2, "trajectory_jitter": 0.0}
    return BatchedHMC(CHAINS, L, model, hmc, device, 1234)


def measure(device):
    engine = make(device)
    engine.trajectory()
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(TRAJECTORIES):
        engine.trajectory()
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return CHAINS * TRAJECTORIES / elapsed, elapsed


def main():
    cpu_rate, cpu_time = measure("cpu")
    print(f"CPU: {cpu_rate:.2f} chain-trajectories/s ({cpu_time:.3f}s)")
    if not torch.cuda.is_available():
        print("CUDA unavailable")
        return
    gpu_rate, gpu_time = measure("cuda:0")
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {gpu_rate:.2f} chain-trajectories/s ({gpu_time:.3f}s), peak={peak:.1f} MiB")
    print(f"speedup: {gpu_rate / cpu_rate:.2f}x")


if __name__ == "__main__":
    main()
