from __future__ import annotations

import hashlib
import json
import math
import tomllib
from copy import deepcopy
from pathlib import Path


DEFAULTS = {
    "model": {"N": 2, "beta": 1.0, "beta1": 0.0, "alpha": 0.0,
              "alpha1": 0.0, "mod": 0, "mul": [1.0], "normalize": True},
    "lattice": {"L": 0, "L0": 50, "target_L0_over_xi": 2.0, "max_L0": None,
                "L_multiple": 6, "minimum_L": 6,
                "pilot_warmup": 400, "pilot_total_samples": 10000},
    "hmc": {"chains": 64, "warmup": 2000, "target_accept": 0.8,
            "initial_step_size": 0.01, "trajectory_length": 0.4,
            "trajectory_jitter": 0.1, "mass_a": 1.0, "mass_z": 1.0,
            "s_step": 1.0, "s_updates": 3, "s_max": 100},
    "sampling": {"scale_total_samples": 50000, "scale_stride": 4,
                 "flow_min_total_samples": 10000,
                 "flow_max_total_samples": 100000,
                 "convergence_batch_total_samples": 6400,
                 "relative_error": 0.02, "consecutive_checks": 2,
                 "minimum_flow_stride": 20, "tau_window_c": 5.0},
    "flow": {"rho": [0.01, 0.05, 0.09, 0.13, 0.17, 0.21, 0.25],
             "epsilon": 0.01, "mass_a": 1.0, "mass_z": 1.0,
             "kinds": ["model"], "buffer_configurations": 0},
    "analysis": {"min_t_over_a2_for_fit": 1.0},
    "compute": {"device": "cuda:0", "dtype": "float64", "seed": 12345,
                "deterministic": True, "max_vram_fraction": 0.70},
    "output": {"root": "runs", "keep_final_checkpoint": True},
}


def _merge(base, update, prefix=""):
    for key, value in update.items():
        if key not in base:
            raise ValueError(f"unknown configuration key: {prefix}{key}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise TypeError(f"{prefix}{key} must be a table")
            _merge(base[key], value, f"{prefix}{key}.")
        else:
            base[key] = value


def validate(cfg):
    if cfg["compute"]["dtype"] != "float64":
        raise ValueError("production runs require compute.dtype='float64'")
    if int(cfg["model"]["N"]) <= 1:
        raise ValueError("model.N must be greater than one")
    if int(cfg["hmc"]["chains"]) < 2:
        raise ValueError("at least two independent HMC chains are required")
    lat = cfg["lattice"]
    try:
        initial_L0 = float(lat["L0"])
    except (TypeError, ValueError) as exc:
        raise TypeError("lattice.L0 must be an integer") from exc
    if not math.isfinite(initial_L0) or initial_L0 <= 0 or not initial_L0.is_integer():
        raise ValueError("lattice.L0 must be a positive integer")
    lat["L0"] = int(initial_L0)
    try:
        target_ratio = float(lat["target_L0_over_xi"])
    except (TypeError, ValueError) as exc:
        raise TypeError("lattice.target_L0_over_xi must be a number") from exc
    if not math.isfinite(target_ratio) or target_ratio < 2:
        raise ValueError("lattice.target_L0_over_xi must be finite and at least 2")
    lat["target_L0_over_xi"] = target_ratio
    try:
        maximum = initial_L0 if lat["max_L0"] is None else float(lat["max_L0"])
    except (TypeError, ValueError) as exc:
        raise TypeError("lattice.max_L0 must be an integer") from exc
    if not math.isfinite(maximum) or maximum <= 0 or not maximum.is_integer():
        raise ValueError("lattice.max_L0 must be a positive integer")
    if maximum < initial_L0:
        raise ValueError("lattice.max_L0 must be greater than or equal to lattice.L0")
    lat["max_L0"] = int(maximum)
    try:
        mul = [float(x) for x in cfg["model"]["mul"]]
    except (TypeError, ValueError) as exc:
        raise TypeError("model.mul must be a list of numbers") from exc
    if not mul or any(not math.isfinite(x) for x in mul):
        raise ValueError("model.mul must be non-empty and contain only finite values")
    names = [mul_directory_name(x) for x in mul]
    if len(set(mul)) != len(mul):
        raise ValueError("model.mul values must be unique")
    if len(set(names)) != len(names):
        raise ValueError("model.mul values must map to distinct mul_* directory names")
    rho = [float(x) for x in cfg["flow"]["rho"]]
    if not rho or any(x <= 0 for x in rho) or any(b <= a for a, b in zip(rho, rho[1:])):
        raise ValueError("flow.rho must be positive and strictly increasing")
    kinds = cfg["flow"]["kinds"]
    if not isinstance(kinds, list) or not kinds:
        raise ValueError("flow.kinds must be a non-empty list")
    if any(not isinstance(kind, str) for kind in kinds):
        raise TypeError("flow.kinds entries must be strings")
    if len(set(kinds)) != len(kinds):
        raise ValueError("flow.kinds must not contain duplicates")
    unknown_kinds = set(kinds) - {"model", "covariant"}
    if unknown_kinds:
        raise ValueError(f"unsupported flow.kinds: {sorted(unknown_kinds)}")
    s = cfg["sampling"]
    if not 0 < float(s["relative_error"]) < 1:
        raise ValueError("sampling.relative_error must lie in (0, 1)")
    if int(s["flow_max_total_samples"]) < int(s["flow_min_total_samples"]):
        raise ValueError("flow_max_total_samples must be >= flow_min_total_samples")
    for key in ("scale_total_samples", "flow_min_total_samples",
                "flow_max_total_samples", "convergence_batch_total_samples"):
        if int(s[key]) <= 0:
            raise ValueError(f"sampling.{key} must be positive")
    if not 0 < float(cfg["compute"]["max_vram_fraction"]) < 1:
        raise ValueError("compute.max_vram_fraction must lie in (0, 1)")
    minimum_flow_time = float(cfg["analysis"]["min_t_over_a2_for_fit"])
    if not math.isfinite(minimum_flow_time) or minimum_flow_time < 0:
        raise ValueError("analysis.min_t_over_a2_for_fit must be finite and non-negative")
    return cfg


def load_config(path):
    path = Path(path)
    with path.open("rb") as fh:
        supplied = tomllib.load(fh)
    cfg = deepcopy(DEFAULTS)
    _merge(cfg, supplied)
    cfg["_source"] = str(path.resolve())
    return validate(cfg)


def canonical(cfg):
    # Analysis-only choices do not alter a Markov chain or its checkpoint. In
    # particular, excluding this table keeps older runs resumable.
    clean = {k: v for k, v in cfg.items()
             if not k.startswith("_") and k != "analysis"}
    return json.dumps(clean, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(cfg):
    return hashlib.sha256(canonical(cfg).encode()).hexdigest()


def run_family_fingerprint(cfg):
    """Identify settings that must agree across all mul runs in an experiment."""
    clean = deepcopy(cfg)
    clean.pop("_source", None)
    clean.pop("analysis", None)
    clean["model"].pop("mul", None)
    # A chain count belongs to one independently restartable mul run.  Existing
    # children keep their frozen value while newly added mul values may use a
    # new experiment-level value.
    clean["hmc"].pop("chains", None)
    # This only controls how often accumulated production observations are
    # checked for convergence. It does not alter the chain or saved data.
    clean["sampling"].pop("convergence_batch_total_samples", None)
    # This is an online stopping rule. Existing observations and checkpoints
    # remain valid when the experiment-level target changes.
    clean["sampling"].pop("relative_error", None)
    payload = json.dumps(clean, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def mul_directory_name(mul):
    return f"mul_{float(mul):.6g}"


def scaled_model(cfg, mul):
    m = cfg["model"]
    beta, beta1 = float(m["beta"]), float(m["beta1"])
    alpha, alpha1 = float(m["alpha"]), float(m["alpha1"])
    if m["normalize"]:
        norm = beta + beta1
        if abs(norm) < 1e-15:
            raise ValueError("beta + beta1 cannot vanish when normalize=true")
        beta, beta1, alpha, alpha1 = (x / norm for x in (beta, beta1, alpha, alpha1))
    return {"N": int(m["N"]), "beta": beta * mul, "beta1": beta1 * mul,
            "alpha": alpha * mul, "alpha1": alpha1 * mul,
            "mod": 0 if abs(beta1) < 1e-8 else int(m["mod"]), "mul": float(mul)}
