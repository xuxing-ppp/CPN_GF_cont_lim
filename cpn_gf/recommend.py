from __future__ import annotations

import gc
import math
import time
from copy import deepcopy

import numpy as np
import torch

from .config import load_config, scaled_model
from .hmc import BatchedHMC
from .online_flow import flow_observables
from .runner import _device, _run_pilot


def _cuda_budget(device, fraction):
    if device.type != "cuda":
        return None
    props = torch.cuda.get_device_properties(device)
    free, _ = torch.cuda.mem_get_info(device)
    return min(int(props.total_memory * fraction), int(free * 0.80))


def _temporary_pilot(cfg):
    """Pilot only the largest mul, retrying with fewer chains after CUDA OOM."""
    mul = max(float(value) for value in cfg["model"]["mul"])
    probe = deepcopy(cfg)
    chains = int(probe["hmc"]["chains"])
    while True:
        probe["hmc"]["chains"] = chains
        try:
            result = _run_pilot(probe, scaled_model(probe, mul), int(cfg["compute"]["seed"]) + 1)
            return int(result["recommended_L"]), mul, chains, result
        except torch.cuda.OutOfMemoryError:
            if chains <= 2:
                raise
            chains = max(2, chains // 2)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _measure_candidate(cfg, model, L, chains, repeats=3, trajectories=3):
    device = _device(cfg)
    budget = _cuda_budget(device, float(cfg["compute"]["max_vram_fraction"]))
    engine = None
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        engine = BatchedHMC(chains, L, model, cfg["hmc"], device,
                            int(cfg["compute"]["seed"]) + chains)
        engine.trajectory()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(trajectories):
                engine.trajectory()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append(time.perf_counter() - start)
        rate = chains * trajectories / float(np.median(timings))

        # One production event is the minimum buffer imposed by the chain
        # count.  The real flow can still split this into smaller micro-batches.
        z, a = engine.z.clone(), engine.a.clone()
        s = None if engine.s is None else engine.s.clone()
        _, _, micro_batch = flow_observables(
            z, a, s, model, cfg["flow"], np.asarray([0, 1]), device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak = int(torch.cuda.max_memory_allocated(device))
        else:
            peak = 0
        within_budget = budget is None or peak <= budget
        return {"chains": chains, "status": "ok" if within_budget else "over_budget",
                "rate": rate, "peak_bytes": peak, "budget_bytes": budget,
                "flow_micro_batch": int(micro_batch)}
    except torch.cuda.OutOfMemoryError:
        return {"chains": chains, "status": "oom", "rate": None,
                "peak_bytes": None, "budget_bytes": budget, "flow_micro_batch": None}
    finally:
        del engine
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def recommend_chains(config_path, lattice_size=None, run_pilot=False, max_chains=1024):
    """Benchmark power-of-two chain counts without changing configuration files."""
    cfg = load_config(config_path)
    fixed = int(cfg["lattice"]["L"])
    if lattice_size is not None and run_pilot:
        raise ValueError("--lattice-size and --pilot are mutually exclusive")
    pilot = None
    if lattice_size is not None:
        L, source = int(lattice_size), "explicit"
        mul = max(float(value) for value in cfg["model"]["mul"])
    elif run_pilot:
        if fixed > 0:
            raise ValueError("--pilot is unnecessary when lattice.L is fixed")
        L, mul, pilot_chains, pilot = _temporary_pilot(cfg)
        source = f"temporary pilot ({pilot_chains} chains)"
    elif fixed > 0:
        L, source = fixed, "lattice.L"
        mul = max(float(value) for value in cfg["model"]["mul"])
    else:
        raise ValueError("lattice.L=0; choose --lattice-size L or --pilot")
    if L <= 0:
        raise ValueError("lattice size must be positive")
    if int(max_chains) < 2:
        raise ValueError("--max-chains must be at least 2")

    model = scaled_model(cfg, mul)
    rows, chains, slow_gains = [], 2, 0
    while chains <= int(max_chains):
        row = _measure_candidate(cfg, model, L, chains)
        rows.append(row)
        if row["status"] != "ok":
            break
        if len(rows) >= 2 and rows[-2]["status"] == "ok":
            previous = rows[-2]["rate"]
            gain = row["rate"] / previous - 1.0
            slow_gains = slow_gains + 1 if gain < 0.05 else 0
            if slow_gains >= 2:
                break
        chains *= 2
    eligible = [row for row in rows if row["status"] == "ok"]
    if not eligible:
        raise RuntimeError(f"no chain count fitted the device at L={L}")
    best = max(row["rate"] for row in eligible)
    recommended = min(row["chains"] for row in eligible if row["rate"] >= 0.95 * best)
    return {"recommended_chains": recommended, "L": L, "L_source": source,
            "mul": mul, "device": str(cfg["compute"]["device"]),
            "best_rate": best, "candidates": rows, "pilot": pilot}


def print_recommendation(result):
    print(f"device={result['device']}  L={result['L']} ({result['L_source']})  "
          f"mul={result['mul']:g}")
    print("chains  status        chain-trajectories/s  peak MiB  flow micro-batch")
    for row in result["candidates"]:
        rate = "-" if row["rate"] is None else f"{row['rate']:.2f}"
        peak = "-" if row["peak_bytes"] is None else f"{row['peak_bytes'] / 2**20:.1f}"
        micro = "-" if row["flow_micro_batch"] is None else str(row["flow_micro_batch"])
        print(f"{row['chains']:>6}  {row['status']:<12}  {rate:>20}  {peak:>8}  {micro:>16}")
    print(f"recommended chains = {result['recommended_chains']} "
          "(smallest candidate within 95% of measured best throughput)")
