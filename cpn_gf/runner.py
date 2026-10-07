from __future__ import annotations

import csv
import json
import hashlib
import math
import os
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .analysis import analyze_run
from .config import (config_for_mul, fingerprint, load_config, mul_directory_name,
                     run_family_fingerprint, scaled_model, write_config)
from .hmc import BatchedHMC, tune
from .io import atomic_json, atomic_npz, atomic_torch, load_checkpoint
from .online_flow import flow_observables
from .stats import jackknife_error, scale_statistics


# First public on-disk format of the refactored online pipeline. Increment only
# when a future change makes existing manifests/checkpoints unsafe to resume.
SCHEMA = 2


def _device(cfg):
    device = torch.device(cfg["compute"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("a CUDA device was requested but torch.cuda.is_available() is false")
    if cfg["compute"]["deterministic"]:
        torch.use_deterministic_algorithms(True)
    return device


def _advance(engine, count, desc=None):
    iterator = range(int(count))
    if desc:
        iterator = tqdm(iterator, desc=desc, leave=False)
    for _ in iterator:
        engine.trajectory()


def _collect_modes(engine, samples, stride, desc):
    values = []
    for _ in tqdm(range(int(samples)), desc=desc, leave=False):
        _advance(engine, stride)
        values.append(engine.structure_modes().cpu().numpy())
    return np.stack(values, axis=1)  # chain,sample,observable


def _new_engine(cfg, model, L, chains, seed):
    return BatchedHMC(chains, L, model, cfg["hmc"], _device(cfg), seed)


def _run_pilot(cfg, model, seed):
    lat, hmc = cfg["lattice"], cfg["hmc"]
    chains = int(hmc["chains"])
    current_L0 = int(lat["L0"])
    maximum_L0 = int(lat["max_L0"])
    target_ratio = float(lat["target_L0_over_xi"])
    per_chain = math.ceil(int(lat["pilot_total_samples"]) / chains)
    attempts = []

    while True:
        attempt_index = len(attempts)
        engine = _new_engine(cfg, model, current_L0, chains, seed + attempt_index)
        with tqdm(total=int(lat["pilot_warmup"]), desc=f"pilot warmup L0={current_L0}",
                  leave=False) as bar:
            tune(engine, lat["pilot_warmup"], hmc["target_accept"], bar)
        modes = _collect_modes(engine, per_chain, 1, f"pilot scale L0={current_L0}")
        xi, xi_loo, tau = scale_statistics(
            modes, current_L0, cfg["sampling"]["tau_window_c"])
        if not np.isfinite(xi) or not np.all(np.isfinite(xi_loo)) or xi <= 0:
            raise RuntimeError(
                f"pilot produced invalid xi={xi} at L0={current_L0}; "
                "increase pilot statistics")

        xi_error = float(jackknife_error(xi_loo))
        ratio = current_L0 / float(xi)
        attempts.append({"L0": current_L0, "xi": float(xi), "xi_error": xi_error,
                         "tau_max": float(tau), "L0_over_xi": ratio,
                         "step_size": float(engine.step_size),
                         "samples_per_chain": per_chain,
                         "total_samples": per_chain * chains})
        if ratio >= target_ratio:
            break
        if current_L0 >= maximum_L0:
            raise RuntimeError(
                f"pilot reached lattice.max_L0={maximum_L0} with xi={xi:.8g} "
                f"and L0/xi={ratio:.8g}, below target {target_ratio:.8g}")
        current_L0 = min(maximum_L0, math.ceil(target_ratio * float(xi)))

    L = max(int(lat["minimum_L"]), int(lat["L_multiple"]) * math.ceil(xi))
    return {"xi": float(xi), "xi_error": xi_error, "tau_max": float(tau),
            "L0": current_L0, "L": L, "recommended_L": L,
            "step_size": float(engine.step_size), "samples_per_chain": per_chain,
            "total_samples": per_chain * chains, "chains": chains, "attempt_count": len(attempts),
            "attempts": attempts}


def _flow_steps(rho, xi, xi_loo, epsilon):
    scaled = np.asarray(rho)[:, None] * np.concatenate(([xi], xi_loo))[None, :] ** 2 / epsilon
    steps = np.unique(np.concatenate(([0], np.floor(scaled).ravel(), np.ceil(scaled).ravel())).astype(int))
    return steps[steps >= 0]


def _base_manifest(cfg, model, run_dir, pilot, L, run_seed,
                   phase="warmup", status="running"):
    device = _device(cfg)
    execution = {"device": str(device), "torch_version": torch.__version__, "dtype": "float64"}
    if device.type == "cuda":
        execution.update(gpu=torch.cuda.get_device_name(device), cuda=torch.version.cuda)
    return {"schema_version": SCHEMA, "status": status, "phase": phase,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "config_fingerprint": fingerprint(cfg), "model": model, "L": L,
            "chains": int(cfg["hmc"]["chains"]), "pilot": pilot,
            "sampling": cfg["sampling"], "flow": cfg["flow"], "hmc": cfg["hmc"],
            "analysis": cfg["analysis"],
            "execution": execution, "run_dir": str(run_dir), "run_seed": int(run_seed)}


def _write_checkpoint(run_dir, engine, phase, **extra):
    atomic_torch(Path(run_dir) / "checkpoint.pt",
                 {"schema_version": SCHEMA, "phase": phase,
                  "engine": engine.state_dict(), **extra})


def _prepare_new(cfg, model, run_dir, seed):
    fixed_L = int(cfg["lattice"].get("L", 0))
    pilot = ({"xi": None, "L": fixed_L, "step_size": None, "skipped": "fixed lattice"}
             if fixed_L > 0 else _run_pilot(cfg, model, seed + 1))
    L = fixed_L if fixed_L > 0 else int(pilot["L"])
    manifest = _base_manifest(cfg, model, run_dir, pilot, L, seed)
    atomic_json(run_dir / "manifest.json", manifest)
    return _start_after_pilot(cfg, run_dir, manifest)


def _start_after_pilot(cfg, run_dir, manifest):
    """Start the independent production chain after a completed pilot."""
    seed = int(manifest["run_seed"])
    engine = _new_engine(cfg, manifest["model"], manifest["L"],
                         manifest["chains"], seed + 2)
    _write_checkpoint(run_dir, engine, "warmup")
    manifest.update(status="running", phase="warmup")
    atomic_json(run_dir / "manifest.json", manifest)
    with tqdm(total=int(cfg["hmc"]["warmup"]), desc="production warmup", leave=False) as bar:
        tune(engine, cfg["hmc"]["warmup"], cfg["hmc"]["target_accept"], bar)
    manifest.update(phase="scale_setting", tuned_step_size=engine.step_size)
    atomic_json(run_dir / "manifest.json", manifest)
    _write_checkpoint(run_dir, engine, "scale_setting")
    return _finish_scale(cfg, run_dir, engine, manifest)


def _prepare_pilot(cfg, model, run_dir, seed, retry_oom=False):
    """Run and persist only the scale-selection pilot."""
    probe = deepcopy(cfg)
    while True:
        try:
            pilot = _run_pilot(probe, model, seed + 1)
            pilot["chains"] = probe["hmc"]["chains"]
            break
        except torch.cuda.OutOfMemoryError:
            if not retry_oom or probe["hmc"]["chains"] <= 2:
                raise
            probe["hmc"]["chains"] = max(2, probe["hmc"]["chains"] // 2)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    fixed_L = int(cfg["lattice"].get("L", 0))
    production_L = fixed_L if fixed_L > 0 else int(pilot["recommended_L"])
    manifest = _base_manifest(cfg, model, run_dir, pilot, production_L, seed,
                              phase="pilot_complete", status="pilot_complete")
    atomic_json(run_dir / "manifest.json", manifest)
    return manifest


def _finish_scale(cfg, run_dir, engine, manifest):
    """Run scale setting from its stage-start checkpoint and enter production."""
    L = int(manifest["L"])
    chains = int(manifest["chains"])
    scale_per_chain = math.ceil(int(cfg["sampling"]["scale_total_samples"]) / chains)
    scale = _collect_modes(engine, scale_per_chain,
                           cfg["sampling"]["scale_stride"], "scale setting")
    xi, xi_loo, tau = scale_statistics(scale, L, cfg["sampling"]["tau_window_c"])
    if not np.isfinite(xi) or not np.all(np.isfinite(xi_loo)):
        raise RuntimeError("scale-setting xi or a jackknife xi is non-finite")
    stride = max(int(cfg["sampling"]["minimum_flow_stride"]),
                 int(math.ceil(2 * tau) * int(cfg["sampling"]["scale_stride"])))
    steps = _flow_steps(cfg["flow"]["rho"], xi, xi_loo, cfg["flow"]["epsilon"])
    atomic_npz(run_dir / "scale.npz", modes=scale, xi=np.asarray(xi), xi_loo=xi_loo,
               tau_max=np.asarray(tau), flow_stride=np.asarray(stride), output_steps=steps)
    manifest.update(phase="production", xi_scale=xi, xi_jackknife=xi_loo,
                    tau_max=tau, flow_stride=stride, output_steps=steps,
                    scale_samples_per_chain=scale_per_chain,
                    scale_total_samples=scale_per_chain * chains,
                    completed_samples_per_chain=0, completed_total_samples=0,
                    committed_chunks=0)
    atomic_json(run_dir / "manifest.json", manifest)
    _write_checkpoint(run_dir, engine, "production", samples=0, chunks=0,
                      convergence_streak=0)
    return engine, manifest, 0, 0, 0


def _restore(cfg, run_dir):
    with (run_dir / "manifest.json").open(encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest["schema_version"] != SCHEMA:
        raise ValueError(f"run schema {manifest['schema_version']} cannot be resumed by schema {SCHEMA}; "
                         "completed older runs remain analyzable")
    previous_sampling = manifest.get("sampling", {})
    previous_relative_error = previous_sampling.get("relative_error")
    if manifest["config_fingerprint"] != fingerprint(cfg):
        # Preserve the old on-disk fingerprint while permitting purely
        # stopping/scheduling-related convergence settings to change.
        prior_batch = previous_sampling.get("convergence_batch_total_samples")
        compatible_cfg = deepcopy(cfg)
        if prior_batch is not None:
            compatible_cfg["sampling"]["convergence_batch_total_samples"] = prior_batch
        if previous_relative_error is not None:
            compatible_cfg["sampling"]["relative_error"] = previous_relative_error
        if ((prior_batch is None and previous_relative_error is None)
                or manifest["config_fingerprint"] != fingerprint(compatible_cfg)):
            raise ValueError("configuration does not match this run")
    # Analysis settings are checkpoint-safe, but changing the online error
    # selection invalidates a convergence streak accumulated under the old rule.
    previous_minimum = manifest.get("analysis", {}).get("min_t_over_a2_for_fit")
    current_minimum = float(cfg["analysis"]["min_t_over_a2_for_fit"])
    current_relative_error = float(cfg["sampling"]["relative_error"])
    threshold_changed = (previous_minimum is None
                         or float(previous_minimum) != current_minimum
                         or previous_relative_error is None
                         or float(previous_relative_error) != current_relative_error)
    stricter_relative_error = (previous_relative_error is not None
                               and current_relative_error < float(previous_relative_error))
    manifest["analysis"] = cfg["analysis"]
    manifest["sampling"] = cfg["sampling"]
    manifest["config_fingerprint"] = fingerprint(cfg)
    manifest.pop("error", None)
    manifest["status"] = "running"
    if manifest.get("phase") == "pilot_complete":
        return _start_after_pilot(cfg, run_dir, manifest)
    if manifest.get("phase") == "pilot_error":
        raise RuntimeError("pilot failed; rerun it with 'python -m cpn_gf pilot --run <experiment>'")
    checkpoint = load_checkpoint(run_dir / "checkpoint.pt")
    engine = _new_engine(cfg, manifest["model"], manifest["L"], manifest["chains"], 0)
    engine.load_state_dict(checkpoint["engine"])
    if checkpoint["phase"] == "warmup":
        with tqdm(total=int(cfg["hmc"]["warmup"]), desc="production warmup", leave=False) as bar:
            tune(engine, cfg["hmc"]["warmup"], cfg["hmc"]["target_accept"], bar)
        manifest.update(phase="scale_setting", tuned_step_size=engine.step_size)
        atomic_json(run_dir / "manifest.json", manifest)
        _write_checkpoint(run_dir, engine, "scale_setting")
        return _finish_scale(cfg, run_dir, engine, manifest)
    if checkpoint["phase"] == "scale_setting":
        return _finish_scale(cfg, run_dir, engine, manifest)
    if checkpoint["phase"] == "complete":
        if not stricter_relative_error:
            raise RuntimeError("this run is already complete")
        manifest["phase"] = "production"
    elif checkpoint["phase"] != "production":
        raise RuntimeError(f"unsupported checkpoint phase: {checkpoint['phase']}")
    streak = 0 if threshold_changed else int(checkpoint.get("convergence_streak", 0))
    atomic_json(run_dir / "manifest.json", manifest)
    return (engine, manifest, int(checkpoint["samples"]), int(checkpoint["chunks"]), streak)


def _production(cfg, run_dir, engine, manifest, samples, chunks, streak):
    sampling, flow = cfg["sampling"], cfg["flow"]
    flow_kinds = list(flow["kinds"])
    steps = np.asarray(manifest["output_steps"], dtype=int)
    times = steps * float(flow["epsilon"])
    chains = int(manifest["chains"])
    maximum = math.ceil(int(sampling["flow_max_total_samples"]) / chains)
    minimum = math.ceil(int(sampling["flow_min_total_samples"]) / chains)
    check_every = max(1, math.ceil(int(sampling["convergence_batch_total_samples"]) / chains))
    buffer_configs = _flow_buffer_configurations(cfg, int(manifest["L"]), chains)
    chunk_size = max(1, buffer_configs // chains)
    needed = int(sampling["consecutive_checks"])
    manifest.update(requested_flow_buffer_configurations=buffer_configs,
                    flow_events_per_chunk=chunk_size,
                    effective_flow_min_total_samples=minimum * chains,
                    effective_flow_max_total_samples=maximum * chains,
                    effective_convergence_batch_total_samples=check_every * chains)
    converged = False
    progress = tqdm(total=maximum, initial=samples, desc=f"online flow mul={manifest['model']['mul']:g}")
    while samples < maximum:
        take = min(chunk_size, maximum - samples)
        z_buffer, a_buffer, s_buffer = [], [], []
        if engine.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(engine.device)
        for _ in range(take):
            _advance(engine, manifest["flow_stride"])
            z_buffer.append(engine.z.clone())
            a_buffer.append(engine.a.clone())
            if engine.s is not None:
                s_buffer.append(engine.s.clone())
        z_batch, a_batch = torch.cat(z_buffer), torch.cat(a_buffer)
        s_batch = torch.cat(s_buffer) if s_buffer else None
        blocks, engine_kinds, micro_batches = {}, {}, {}
        for flow_kind in flow_kinds:
            values, engine_kind, used = flow_observables(
                z_batch, a_batch, s_batch, manifest["model"], flow, steps,
                engine.device, flow_kind)
            blocks[flow_kind] = values.reshape(
                take, chains, len(steps), 6).transpose(1, 0, 2, 3)
            engine_kinds[flow_kind] = engine_kind
            micro_batches[flow_kind] = used
        q_s = (None if s_batch is None else
               s_batch.sum(dim=(1, 2)).reshape(take, chains).T.cpu().numpy())
        peak_mib = (torch.cuda.max_memory_allocated(engine.device) / 2 ** 20
                    if engine.device.type == "cuda" else 0.0)
        chunk_path = run_dir / "observations" / f"flow_{chunks:08d}.npz"
        archive = {"output_steps": steps, "times": times,
                   "observable_schema": np.asarray(2),
                   "first_sample": np.asarray(samples),
                   "flow_kinds": np.asarray(flow_kinds),
                   "peak_memory_mib": np.asarray(peak_mib)}
        observable_names = ("E_action", "S00", "S10", "S01", "Q_z", "Q_U")
        for flow_kind, block in blocks.items():
            for index, name in enumerate(observable_names):
                archive[f"{flow_kind}__{name}"] = block[..., index]
            archive[f"{flow_kind}__engine_kind"] = np.asarray(engine_kinds[flow_kind])
            archive[f"{flow_kind}__micro_batch_used"] = np.asarray(
                micro_batches[flow_kind])
        if q_s is not None:
            archive["Q_s"] = q_s
        atomic_npz(chunk_path, **archive)
        samples += take
        chunks += 1
        progress.update(take)
        _write_checkpoint(run_dir, engine, "production", samples=samples, chunks=chunks,
                          convergence_streak=streak)
        manifest.update(completed_samples_per_chain=samples,
                        completed_total_samples=samples * chains,
                        committed_chunks=chunks,
                        effective_flow_micro_batch=micro_batches,
                        peak_memory_mib=max(float(manifest.get("peak_memory_mib", 0)), peak_mib))
        atomic_json(run_dir / "manifest.json", manifest)
        if samples >= minimum and samples // check_every > (samples - take) // check_every:
            _, summaries = analyze_run(
                run_dir, write=True,
                minimum_flow_time=cfg["analysis"]["min_t_over_a2_for_fit"])
            eligible = [(kind, summary["maximum_tE_relative_error"],
                         summary["maximum_tE_relative_error_flow_time"])
                        for kind, summary in summaries.items()
                        if summary["maximum_tE_relative_error"] is not None]
            worst = max(eligible, key=lambda item: item[1]) if eligible else None
            maximum_error = None if worst is None else worst[1]
            maximum_time = None if worst is None else worst[2]
            if maximum_error is None:
                progress.set_postfix_str("max rel err=N/A @ t/a^2=N/A")
            else:
                progress.set_postfix_str(
                    f"max rel err={maximum_error:.3e} ({worst[0]}) @ t/a^2={maximum_time:g}")
            all_converged = all(summary["converged"] for summary in summaries.values())
            streak = streak + 1 if all_converged else 0
            _write_checkpoint(run_dir, engine, "production", samples=samples, chunks=chunks,
                              convergence_streak=streak)
            if streak >= needed:
                converged = True
                break
    progress.close()
    _, summaries = analyze_run(
        run_dir, write=True,
        minimum_flow_time=cfg["analysis"]["min_t_over_a2_for_fit"])
    all_converged = all(summary["converged"] for summary in summaries.values())
    converged = converged or (all_converged and streak >= needed)
    eligible = [(kind, summary["maximum_tE_relative_error"],
                 summary["maximum_tE_relative_error_flow_time"])
                for kind, summary in summaries.items()
                if summary["maximum_tE_relative_error"] is not None]
    worst = max(eligible, key=lambda item: item[1]) if eligible else None
    s_acceptance = (None if engine.s is None else
                    (engine.accepted_metro.double()
                     / torch.clamp(engine.attempted_metro, min=1)).cpu().numpy())
    manifest.update(status="ok" if converged else "complete_with_warning", phase="complete",
                    converged=converged, completed_samples_per_chain=samples,
                    completed_total_samples=samples * chains,
                    completed_at=datetime.now().isoformat(timespec="seconds"),
                    flow_convergence={kind: {
                        "converged": summary["converged"],
                        "maximum_tE_relative_error": summary["maximum_tE_relative_error"],
                        "maximum_tE_relative_error_flow_time":
                            summary["maximum_tE_relative_error_flow_time"]}
                        for kind, summary in summaries.items()},
                    maximum_tE_relative_error=None if worst is None else worst[1],
                    maximum_tE_relative_error_flow_time=None if worst is None else worst[2],
                    maximum_tE_relative_error_flow_kind=None if worst is None else worst[0],
                    acceptance_rate_hmc=(engine.accepted_hmc.double()
                                         / torch.clamp(engine.attempted_hmc, min=1)).cpu().numpy(),
                    acceptance_rate_s=s_acceptance)
    atomic_json(run_dir / "manifest.json", manifest)
    if cfg["output"]["keep_final_checkpoint"]:
        _write_checkpoint(run_dir, engine, "complete", samples=samples, chunks=chunks,
                          convergence_streak=streak)
    else:
        os.remove(run_dir / "checkpoint.pt")
    return manifest


def _flow_buffer_configurations(cfg, L, chains):
    """Choose a near-saturation flow batch while keeping ample display headroom."""
    requested = int(cfg["flow"]["buffer_configurations"])
    target = requested or (384 if L <= 64 else 128 if L <= 128 else 64)
    target = max(chains, math.ceil(target / chains) * chains)
    device = torch.device(cfg["compute"]["device"])
    if device.type == "cuda":
        total_bytes = torch.cuda.get_device_properties(device).total_memory
        free_bytes, _ = torch.cuda.mem_get_info(device)
        budget = min(total_bytes * float(cfg["compute"]["max_vram_fraction"]),
                     free_bytes * 0.80)
        # Measured flow peak is about 400 bytes/site/config; use 640 as a safety estimate.
        safe = int(budget / (640 * L * L))
        safe = max(chains, safe // chains * chains)
        target = min(target, safe)
    return target


def _run_seed(cfg, mul):
    """Derive an order-independent seed for one mul run."""
    material = f"{int(cfg['compute']['seed'])}:{float(mul).hex()}".encode()
    derived = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    # _prepare_new reserves the following two integers for pilot and production.
    return derived % (2 ** 63 - 2)


def _create_experiment(config_path):
    cfg = load_config(config_path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment = Path(cfg["output"]["root"]) / f"{stamp}_{fingerprint(cfg)[:8]}"
    experiment.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, experiment / "config.toml")
    return cfg, experiment


def _pilot_summary_row(manifest):
    pilot = manifest.get("pilot") or {}
    return {"mul": manifest.get("model", {}).get("mul"),
            "status": manifest.get("status"),
            "chains": pilot.get("chains"),
            "xi": pilot.get("xi"), "xi_error": pilot.get("xi_error"),
            "tau_max": pilot.get("tau_max"), "L0": pilot.get("L0"),
            "recommended_L": pilot.get("recommended_L", pilot.get("L")),
            "production_L": manifest.get("L"),
            "step_size": pilot.get("step_size"),
            "samples_per_chain": pilot.get("samples_per_chain"),
            "total_samples": pilot.get("total_samples"),
            "attempt_count": pilot.get("attempt_count"),
            "attempts": pilot.get("attempts"),
            "error": manifest.get("error")}


def _write_pilot_summary(experiment_dir):
    rows = []
    for path in Path(experiment_dir).glob("mul_*/manifest.json"):
        with path.open(encoding="utf-8") as fh:
            manifest = json.load(fh)
        if "pilot" in manifest:
            rows.append(_pilot_summary_row(manifest))
    rows.sort(key=lambda row: (row["mul"] is None,
                               float(row["mul"]) if row["mul"] is not None else 0.0))
    payload = {"schema_version": SCHEMA, "experiment": str(experiment_dir),
               "updated_at": datetime.now().isoformat(timespec="seconds"), "runs": rows}
    atomic_json(Path(experiment_dir) / "pilot_results.json", payload)
    csv_path = Path(experiment_dir) / "pilot_results.csv"
    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    fields = ("mul", "status", "chains", "xi", "xi_error", "tau_max", "L0", "recommended_L",
              "production_L", "step_size", "samples_per_chain", "total_samples",
              "attempt_count", "error")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: "" if row[key] is None else row[key] for key in fields}
                         for row in rows)
    os.replace(tmp, csv_path)
    return rows


def pilot_config(config_path):
    """Create an experiment and run only its pilots."""
    _, experiment = _create_experiment(config_path)
    try:
        return pilot_experiment(experiment)
    except Exception as exc:
        raise RuntimeError(f"pilot failed in {experiment}: {exc}") from exc


def pilot_experiment(experiment_dir, retry_oom=False):
    """Run missing or previously failed pilots in an experiment."""
    experiment_dir = Path(experiment_dir)
    root_cfg = load_config(experiment_dir / "config.toml")
    pending, outputs = [], []
    for mul in root_cfg["model"]["mul"]:
        mul = float(mul)
        run_dir = experiment_dir / mul_directory_name(mul)
        manifest_path = run_dir / "manifest.json"
        cfg = config_for_mul(root_cfg, mul)
        manifest = None
        child_config = run_dir / "config.toml"
        if child_config.is_file():
            cfg = load_config(child_config)
            if run_family_fingerprint(cfg) != run_family_fingerprint(root_cfg):
                raise ValueError(
                    f"{run_dir} differs from the experiment config in settings other than "
                    "model.mul, hmc.chains, sampling.convergence_batch_total_samples, "
                    "sampling.relative_error, or analysis")
        if manifest_path.is_file():
            with manifest_path.open(encoding="utf-8") as fh:
                manifest = json.load(fh)
            recorded_mul = float(manifest.get("model", {}).get("mul", float("nan")))
            if recorded_mul != mul:
                raise ValueError(f"{run_dir} contains mul={recorded_mul}, expected {mul}")
            if not child_config.is_file():
                raise RuntimeError(f"existing run has no frozen config: {child_config}")
        if manifest is not None and manifest.get("phase") != "pilot_error":
            outputs.append({"mul": mul, "status": manifest.get("status", "existing"),
                            "action": "skipped", "pilot": manifest.get("pilot")})
        else:
            cfg["hmc"]["chains"] = config_for_mul(root_cfg, mul)["hmc"]["chains"]
            pending.append((mul, run_dir, cfg))

    for mul, run_dir, cfg in pending:
        run_dir.mkdir(exist_ok=True)
        config_copy = run_dir / "config.toml"
        write_config(config_copy, cfg)
        model = scaled_model(cfg, mul)
        seed = _run_seed(cfg, mul)
        try:
            manifest = (_prepare_pilot(cfg, model, run_dir, seed, retry_oom=True)
                        if retry_oom else _prepare_pilot(cfg, model, run_dir, seed))
        except Exception as exc:
            fixed_L = int(cfg["lattice"].get("L", 0))
            manifest = _base_manifest(
                cfg, model, run_dir, None, fixed_L if fixed_L > 0 else None, seed,
                phase="pilot_error", status="error")
            manifest.update(error=str(exc), failed_at=datetime.now().isoformat(timespec="seconds"))
            atomic_json(run_dir / "manifest.json", manifest)
            _write_pilot_summary(experiment_dir)
            raise
        outputs.append({"mul": mul, "status": manifest["status"],
                        "action": "piloted", "pilot": manifest["pilot"]})
        _write_pilot_summary(experiment_dir)
    rows = _write_pilot_summary(experiment_dir)
    return {"status": "complete", "experiment": str(experiment_dir),
            "runs": outputs, "pilot_results": rows}


def run_config(config_path):
    cfg, experiment = _create_experiment(config_path)
    outputs = []
    for mul in cfg["model"]["mul"]:
        child_cfg = config_for_mul(cfg, mul)
        model = scaled_model(child_cfg, float(mul))
        run_dir = experiment / mul_directory_name(mul)
        run_dir.mkdir()
        write_config(run_dir / "config.toml", child_cfg)
        try:
            engine, manifest, samples, chunks, streak = _prepare_new(
                child_cfg, model, run_dir, _run_seed(child_cfg, mul))
            outputs.append(_production(child_cfg, run_dir, engine, manifest, samples, chunks, streak))
        except Exception as exc:
            path = run_dir / "manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            manifest.update(status="error", error=str(exc))
            atomic_json(path, manifest)
            raise
    return experiment, outputs


def resume_run(run_dir):
    run_dir = Path(run_dir)
    if not (run_dir / "manifest.json").is_file() and (run_dir / "config.toml").is_file():
        return resume_experiment(run_dir)
    cfg = load_config(run_dir / "config.toml")
    engine, manifest, samples, chunks, streak = _restore(cfg, run_dir)
    return _production(cfg, run_dir, engine, manifest, samples, chunks, streak)


def resume_experiment(experiment_dir):
    """Finish every configured mul in an existing experiment directory."""
    experiment_dir = Path(experiment_dir)
    root_cfg = load_config(experiment_dir / "config.toml")
    outputs = []
    pending, missing, complete, preflight_errors = [], [], [], []

    # Preflight every requested existing run before performing expensive work.
    for mul in root_cfg["model"]["mul"]:
        mul = float(mul)
        run_dir = experiment_dir / mul_directory_name(mul)
        manifest_path = run_dir / "manifest.json"
        if manifest_path.is_file():
            with manifest_path.open(encoding="utf-8") as fh:
                manifest = json.load(fh)
            recorded_mul = float(manifest.get("model", {}).get("mul", float("nan")))
            if recorded_mul != mul:
                raise ValueError(f"{run_dir} contains mul={recorded_mul}, expected {mul}")
            child_config = run_dir / "config.toml"
            if not child_config.is_file():
                raise RuntimeError(f"existing run has no frozen config: {child_config}")
            child_cfg = load_config(child_config)
            if run_family_fingerprint(child_cfg) != run_family_fingerprint(root_cfg):
                raise ValueError(
                    f"{run_dir} differs from the experiment config in settings other than "
                    "model.mul, hmc.chains, sampling.convergence_batch_total_samples, "
                    "sampling.relative_error, or analysis")
            previous_relative_error = float(
                manifest.get("sampling", {}).get(
                    "relative_error", child_cfg["sampling"]["relative_error"]))
            current_relative_error = float(root_cfg["sampling"]["relative_error"])
            child_cfg["analysis"] = dict(root_cfg["analysis"])
            child_cfg["sampling"]["convergence_batch_total_samples"] = int(
                root_cfg["sampling"]["convergence_batch_total_samples"])
            child_cfg["sampling"]["relative_error"] = current_relative_error
            if manifest.get("phase") == "pilot_complete":
                if (run_dir / "checkpoint.pt").exists():
                    raise RuntimeError(f"{run_dir}: pilot_complete has a production checkpoint")
                child_cfg["hmc"]["chains"] = config_for_mul(root_cfg, mul)["hmc"]["chains"]
            if manifest.get("phase") == "complete":
                if current_relative_error < previous_relative_error:
                    checkpoint_path = run_dir / "checkpoint.pt"
                    if not checkpoint_path.is_file():
                        preflight_errors.append(
                            f"{run_dir}: stricter relative_error requires the final checkpoint")
                    else:
                        try:
                            checkpoint = load_checkpoint(checkpoint_path)
                        except Exception as exc:
                            preflight_errors.append(
                                f"{run_dir}: final checkpoint cannot be loaded ({exc})")
                        else:
                            chains = int(manifest.get("chains", child_cfg["hmc"]["chains"]))
                            maximum = math.ceil(
                                int(child_cfg["sampling"]["flow_max_total_samples"]) / chains)
                            samples = int(checkpoint.get(
                                "samples", manifest.get("completed_samples_per_chain", 0)))
                            if checkpoint.get("phase") != "complete":
                                preflight_errors.append(
                                    f"{run_dir}: final checkpoint is not in the complete phase")
                            elif samples >= maximum:
                                preflight_errors.append(
                                    f"{run_dir}: flow_max_total_samples has already been reached")
                    pending.append((mul, run_dir, manifest_path, manifest, child_cfg))
                else:
                    complete.append((mul, manifest))
            else:
                pending.append((mul, run_dir, manifest_path, manifest, child_cfg))
        else:
            missing.append((mul, run_dir))

    if preflight_errors:
        details = "\n".join(f"- {message}" for message in preflight_errors)
        raise RuntimeError(f"cannot resume completed runs under stricter relative_error:\n{details}")

    for mul, run_dir, manifest_path, manifest, cfg in pending:
        try:
            if manifest.get("phase") == "pilot_complete":
                write_config(run_dir / "config.toml", cfg)
                manifest.update(chains=cfg["hmc"]["chains"], hmc=cfg["hmc"],
                                config_fingerprint=fingerprint(cfg))
                atomic_json(manifest_path, manifest)
            engine, manifest, samples, chunks, streak = _restore(cfg, run_dir)
            finished = _production(cfg, run_dir, engine, manifest, samples, chunks, streak)
            outputs.append({"mul": mul, "status": finished["status"], "action": "resumed"})
        except Exception as exc:
            manifest.update(status="error", error=str(exc))
            atomic_json(manifest_path, manifest)
            raise

    for mul, run_dir in missing:
        cfg = config_for_mul(root_cfg, mul)
        model = scaled_model(cfg, mul)
        run_dir.mkdir(exist_ok=True)
        config_copy = run_dir / "config.toml"
        if config_copy.exists():
            cfg = load_config(config_copy)
            if run_family_fingerprint(cfg) != run_family_fingerprint(root_cfg):
                raise ValueError(
                    f"{run_dir} differs from the experiment config in settings other than "
                    "model.mul, hmc.chains, sampling.convergence_batch_total_samples, "
                    "sampling.relative_error, or analysis")
            cfg["sampling"]["convergence_batch_total_samples"] = int(
                root_cfg["sampling"]["convergence_batch_total_samples"])
            cfg["sampling"]["relative_error"] = float(
                root_cfg["sampling"]["relative_error"])
            cfg["hmc"]["chains"] = config_for_mul(root_cfg, mul)["hmc"]["chains"]
            write_config(config_copy, cfg)
            model = scaled_model(cfg, mul)
        else:
            write_config(config_copy, cfg)
        try:
            engine, manifest, samples, chunks, streak = _prepare_new(
                cfg, model, run_dir, _run_seed(cfg, mul))
            finished = _production(cfg, run_dir, engine, manifest, samples, chunks, streak)
            outputs.append({"mul": mul, "status": finished["status"], "action": "started"})
        except Exception as exc:
            manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                        if manifest_path.exists() else {})
            manifest.update(status="error", error=str(exc))
            atomic_json(manifest_path, manifest)
            raise
    for mul, manifest in complete:
        outputs.append({"mul": mul, "status": manifest.get("status", "complete"),
                        "action": "skipped"})
    return {"status": "complete", "experiment": str(experiment_dir), "runs": outputs}
