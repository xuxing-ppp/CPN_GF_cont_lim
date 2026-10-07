from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .config import (config_for_mul, load_config, mul_directory_name,
                     run_family_fingerprint, scaled_model)
from .io import atomic_json
from .hmc import BatchedHMC
from .online_flow import flow_observables
from .runner import _device, pilot_experiment


def _cuda_budget(device, fraction):
    if device.type != "cuda":
        return None
    props = torch.cuda.get_device_properties(device)
    free, _ = torch.cuda.mem_get_info(device)
    return min(int(props.total_memory * fraction), int(free * 0.80))


def _measure_candidate(cfg, model, L, chains, repeats=3, trajectories=3):
    device = _device(cfg)
    if device.type == "cuda":
        # Release the preceding candidate's cached allocations before deriving
        # this candidate's free-memory budget, including across mul values.
        torch.cuda.empty_cache()
    budget = _cuda_budget(device, float(cfg["compute"]["max_vram_fraction"]))
    engine = z = a = s = None
    try:
        if device.type == "cuda":
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
        micro_batches = []
        for flow_kind in cfg["flow"]["kinds"]:
            _, _, micro_batch = flow_observables(
                z, a, s, model, cfg["flow"], np.asarray([0, 1]), device, flow_kind)
            micro_batches.append(micro_batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak = int(torch.cuda.max_memory_allocated(device))
        else:
            peak = 0
        within_budget = budget is None or peak <= budget
        return {"chains": chains, "status": "ok" if within_budget else "over_budget",
                "rate": rate, "peak_bytes": peak, "budget_bytes": budget,
                "flow_micro_batch": int(min(micro_batches))}
    except torch.cuda.OutOfMemoryError:
        return {"chains": chains, "status": "oom", "rate": None,
                "peak_bytes": None, "budget_bytes": budget, "flow_micro_batch": None}
    finally:
        del engine, z, a, s
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def recommend_chains(experiment_dir, max_chains=1024):
    """Persist per-mul recommendations, reusing or completing scale pilots."""
    directory = Path(experiment_dir)
    if (directory / "manifest.json").is_file():
        raise ValueError("recommend-chains requires an experiment directory, not a mul run")
    cfg = load_config(directory / "config.toml")
    if int(max_chains) < 2:
        raise ValueError("--max-chains must be at least 2")
    # Validate all existing children before any expensive or mutating work.
    for mul in cfg["model"]["mul"]:
        child = directory / mul_directory_name(mul)
        if (child / "config.toml").exists():
            frozen = load_config(child / "config.toml")
            if run_family_fingerprint(frozen) != run_family_fingerprint(cfg):
                raise ValueError(f"{child}: configuration incompatible with experiment")
        if (child / "manifest.json").exists():
            manifest = json.loads((child / "manifest.json").read_text())
            if float(manifest.get("model", {}).get("mul", float("nan"))) != float(mul):
                raise ValueError(f"{child}: manifest mul does not match configuration")
            if not (child / "config.toml").is_file():
                raise ValueError(f"{child}: missing frozen config.toml")
            if int(cfg["lattice"]["L"]) == 0 and manifest.get("phase") != "pilot_error":
                pilot = manifest.get("pilot") or {}
                try:
                    valid = (math.isfinite(float(pilot["xi"])) and float(pilot["xi"]) > 0
                             and int(manifest["L"]) > 0)
                except (KeyError, TypeError, ValueError, OverflowError):
                    valid = False
                if not valid:
                    raise ValueError(f"{child}: invalid saved pilot")
    if int(cfg["lattice"]["L"]) == 0:
        pilot_experiment(directory, retry_oom=True)
    result = {"experiment": str(directory), "device": cfg["compute"]["device"],
              "mul": list(cfg["model"]["mul"]), "runs": [], "recommended_chains": []}
    for mul in cfg["model"]["mul"]:
        probe = config_for_mul(cfg, mul)
        L, source, pilot = int(cfg["lattice"]["L"]), "lattice.L", None
        if L == 0:
            manifest = json.loads((directory / mul_directory_name(mul) / "manifest.json").read_text())
            pilot = manifest.get("pilot")
            if (not pilot or pilot.get("xi") is None
                    or not math.isfinite(float(pilot["xi"])) or float(pilot["xi"]) <= 0):
                raise ValueError(f"mul={mul}: invalid saved pilot")
            L, source = int(manifest["L"]), "saved pilot"
        if L <= 0:
            raise ValueError(f"mul={mul}: lattice size must be positive")
        row = _recommend_mul(probe, float(mul), L, source, pilot, max_chains)
        result["runs"].append(row)
        result["recommended_chains"].append(row["recommended_chains"])
        atomic_json(directory / "chains_recommendations.json", result)
    return result


def _recommend_mul(cfg, mul, L, source, pilot, max_chains):

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
    if "runs" in result:
        for row in result["runs"]:
            print_recommendation(row)
        print("hmc.chains = " + json.dumps(result["recommended_chains"]))
        return
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
