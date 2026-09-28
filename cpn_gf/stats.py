from __future__ import annotations

import numpy as np

from .autocorr import integrated_autocorr_time


def xi_from_modes(s00, s10, s01, L):
    m00, m10, m01 = map(lambda x: float(np.mean(x)), (s00, s10, s01))
    denom = 2.0 * np.sin(np.pi / L)
    rx, ry = m00 / m10 - 1.0, m00 / m01 - 1.0
    if m10 <= 0 or m01 <= 0 or rx <= 0 or ry <= 0:
        return float("nan")
    return float(0.5 * (np.sqrt(rx) + np.sqrt(ry)) / denom)


def scale_statistics(modes, L, c=5.0):
    # modes: chain, sample, (S00,S10,S01)
    central = xi_from_modes(modes[..., 0], modes[..., 1], modes[..., 2], L)
    loo = np.asarray([xi_from_modes(np.delete(modes, i, 0)[..., 0],
                                    np.delete(modes, i, 0)[..., 1],
                                    np.delete(modes, i, 0)[..., 2], L)
                      for i in range(len(modes))])
    taus = []
    for chain in modes:
        for col in range(3):
            series = chain[:, col]
            tau = 0.5 if np.all(series == series[0]) else integrated_autocorr_time(series, c=c)
            if np.isfinite(tau):
                taus.append(float(tau))
    return central, loo, max(taus, default=0.5)


def jackknife_error(replicas):
    replicas = np.asarray(replicas, dtype=float)
    center = replicas.mean(axis=0)
    return np.sqrt((len(replicas) - 1.0) / len(replicas)
                   * np.sum((replicas - center) ** 2, axis=0))


def _interp(values, times, targets):
    """Interpolate every sample/observable on one shared time grid."""
    values = np.asarray(values)
    times = np.asarray(times)
    targets = np.asarray(targets)
    out = np.empty((values.shape[0], len(targets), values.shape[2]), dtype=float)
    if len(times) == 1:
        out[...] = values[:, :1, :]
        return out

    upper = np.searchsorted(times, targets, side="right")
    below = upper == 0
    above = upper == len(times)
    middle = ~(below | above)
    if np.any(below):
        out[:, below, :] = values[:, :1, :]
    if np.any(above):
        out[:, above, :] = values[:, -1:, :]
    if np.any(middle):
        hi = upper[middle]
        lo = hi - 1
        weight = ((targets[middle] - times[lo]) / (times[hi] - times[lo]))
        out[:, middle, :] = (values[:, lo, :] * (1.0 - weight[None, :, None])
                             + values[:, hi, :] * weight[None, :, None])
    return out


def _progress(iterable, enabled, desc, total=None):
    if not enabled:
        return iterable
    from tqdm import tqdm
    return tqdm(iterable, total=total, desc=desc, leave=False)


def _observables(values, L):
    # values: sample,rho,(E,S00,S10,S01,Qz,QU)
    E = values[..., 0].mean(axis=0)
    S00, S10, S01 = (values[..., i].mean(axis=0) for i in (1, 2, 3))
    denom = 2.0 * np.sin(np.pi / L)
    with np.errstate(invalid="ignore", divide="ignore"):
        xi = 0.5 * (np.sqrt(S00 / S10 - 1.0) + np.sqrt(S00 / S01 - 1.0)) / denom
    qz, qu = values[..., 4], values[..., 5]
    qz_mean, qu_mean = qz.mean(axis=0), qu.mean(axis=0)
    chit_z = ((qz - qz_mean) ** 2).mean(axis=0) / float(L * L)
    chit_u = ((qu - qu_mean) ** 2).mean(axis=0) / float(L * L)
    return np.stack((E, S00, xi, qz_mean, chit_z, qu_mean, chit_u), axis=-1)


def analyze_flow(chain_values, times, rho, xi, xi_loo, L, progress=False,
                 progress_prefix=""):
    target = np.asarray(rho) * xi * xi
    selected = np.concatenate([_interp(c, times, target) for c in chain_values], axis=0)
    central = _observables(selected, L)
    replicas = []
    omitted_chains = _progress(range(len(chain_values)), progress,
                               f"{progress_prefix}flow jackknife",
                               total=len(chain_values))
    for omitted in omitted_chains:
        jt = np.asarray(rho) * xi_loo[omitted] ** 2
        sample = np.concatenate([_interp(c, times, jt) for i, c in enumerate(chain_values)
                                 if i != omitted], axis=0)
        replicas.append(_observables(sample, L))
    errors = jackknife_error(replicas)
    return central, errors, target


def analyze_unflowed(chain_values, chain_qs, L, progress=False,
                     progress_prefix=""):
    """Analyze the t=0 row of production configurations with chain jackknife."""
    selected = np.concatenate([c[:, 0, :] for c in chain_values], axis=0)[:, None, :]
    central = _observables(selected, L)[0]
    replicas = []
    omitted_chains = _progress(range(len(chain_values)), progress,
                               f"{progress_prefix}unflowed jackknife",
                               total=len(chain_values))
    for omitted in omitted_chains:
        sample = np.concatenate([c[:, 0, :] for i, c in enumerate(chain_values)
                                 if i != omitted], axis=0)[:, None, :]
        replicas.append(_observables(sample, L)[0])
    errors = jackknife_error(replicas)
    qs = None
    if chain_qs is not None:
        flat = np.concatenate(chain_qs)
        mean = float(flat.mean())
        chi = float(np.mean((flat - mean) ** 2) / (L * L))
        qrep, chirep = [], []
        omitted_chains = _progress(range(len(chain_qs)), progress,
                                   f"{progress_prefix}Q_s jackknife",
                                   total=len(chain_qs))
        for omitted in omitted_chains:
            sample = np.concatenate([q for i, q in enumerate(chain_qs) if i != omitted])
            sample_mean = float(sample.mean())
            qrep.append(sample_mean)
            chirep.append(float(np.mean((sample - sample_mean) ** 2) / (L * L)))
        qs = {"mean": mean, "mean_error": float(jackknife_error(qrep)),
              "chi_t": chi, "chi_t_error": float(jackknife_error(chirep))}
    return central, errors, qs
