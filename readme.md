# CPN Gradient Flow: batched HMC and online flow

This repository uses one PyTorch implementation for CPU and CUDA. Every
parameter set is sampled with constrained HMC; heatbath is retained only as a
legacy statistical reference. Production configurations are flowed immediately
and are never archived. The only full configuration on disk is the latest
atomic restart checkpoint.

## Quick start

Copy `experiment.example.toml`, edit it, then run:

```powershell
python -m cpn_gf run --config experiment.toml
```

To scan a broad `mul` range cheaply before choosing production points, run only
the pilots:

```powershell
python -m cpn_gf pilot --config experiment.toml
```

The experiment directory receives `pilot_results.json` and
`pilot_results.csv`, containing each `mul`, its pilot xi and jackknife error,
the recommended lattice size, and the production lattice size. Edit only that
experiment's root `config.toml` so `model.mul` contains the points selected for
production, then continue with:

```powershell
python -m cpn_gf resume --run runs/<experiment>
```

The selected runs reuse their pilot scale estimates and begin independent
production warmup; unselected pilot directories remain available. Adding more
values to the root config later and running
`python -m cpn_gf pilot --run runs/<experiment>` fills in their pilots without
advancing existing runs into production. A failed pilot is recorded in both
the manifest and summary and stops the scan; remove that value or rerun the
pilot command to retry it.

Resume one interrupted `mul` run with:

```powershell
python -m cpn_gf resume --run runs/<experiment>/mul_<value>
```

Or pass the experiment directory to finish the complete `mul` list from its
`config.toml`. Completed runs are skipped, an interrupted run is restored, and
not-yet-created later runs are started automatically:

```powershell
python -m cpn_gf resume --run runs/<experiment>
```

To add parameter points to an existing experiment, edit only the experiment-level
`config.toml` and add, insert, or reorder values in `model.mul`, then run the same
`resume` command. Existing interrupted runs are resumed first, newly added values
are run afterward, and completed values are skipped. Each `mul_*` directory keeps
an automatically managed frozen config for exact restart and provenance; do not
edit those copies. Settings other than `model.mul`, `[analysis]`, and the
documented resumable sampling settings below must remain unchanged within one
experiment directory.

Rebuild results and plots for one `mul` or a whole experiment with:

```powershell
python -m cpn_gf analyze --run runs/<experiment>
```

Experiment-level analysis considers only the `model.mul` values listed in the
experiment root `config.toml`; extra `mul_*` directories are ignored. The
command displays progress while loading flow chunks, computing the per-chain
jackknife estimates, and building the continuum fits.

Each completed production run already writes its own `results.json` and
`results.npz`. To reuse those files and rebuild only the experiment-level
summaries and plots, run:

```powershell
python -m cpn_gf analyze --run runs/<experiment> --aggregate-only
```

This mode does not read the flow chunks, repeat the per-`mul` jackknife
analysis, or rebuild the individual `mul` plots. It requires both result files
for every configured production run and reports any that are missing. The CLI
prints only a compact completion summary and output paths; complete numerical
results remain in the JSON and NPZ files.

At each fixed `rho=t/xi^2`, continuum extrapolation uses an error-in-both-axes
quadratic fit

```text
t<E(t)> = c0 + c1/xi^2 + c2/xi^4
```

with at least four eligible `mul` points; `c0` is the continuum value. The
numeric fits are written to `continuum_fits.json`, the fixed-`rho` fit plots to
`plots/continuum/`, and the overlay of all configured `mul` curves to
`plots/tE_action_vs_rho_by_mul.png`. That overlay also shows the fitted
continuum values and their errors at every `rho` with enough eligible points.

`analysis.min_t_over_a2_for_fit` controls the minimum lattice flow time used
by continuum fits and by the online relative-error stopping test. The flow
progress bar reports the largest included relative error and its `t/a^2` at
each convergence check. Smaller-flow-time points are still stored and remain
visible as excluded points in the plot. Changing this option does not
invalidate a checkpoint; unfinished runs use the latest experiment-level
value and restart their consecutive-convergence count.

Set `compute.device = "cpu"` for CPU. Production calculations use
`float64/complex128` on both devices. `lattice.L = 0` enables the pilot that
chooses `L`; a positive value uses that lattice size directly. The pilot starts
at `lattice.L0`. If its central estimate does not satisfy
`L0 / xi >= lattice.target_L0_over_xi`, it reruns on a larger trial lattice up
to `lattice.max_L0`. Omitting `max_L0` makes it equal to `L0`, preserving the
single-attempt behavior of older configurations.

## Choosing the chain count

Measure the current machine instead of guessing a chain count. For a known
production lattice size, run:

```powershell
python -m cpn_gf recommend-chains --config experiment.toml --lattice-size 72
```

If `lattice.L=0`, the command can temporarily pilot only the largest configured
`mul` and benchmark its recommended lattice size:

```powershell
python -m cpn_gf recommend-chains --config experiment.toml --pilot
```

The temporary pilot and benchmark do not create a run or edit the TOML. The
reported recommendation is the smallest power-of-two chain count within 95%
of the best measured throughput and within `compute.max_vram_fraction`. Use
`--max-chains N` to change the default search limit of 1024.

An experiment may contain different chain counts for different `mul` values.
To change the count for later values, edit only the experiment-level
`config.toml`, add the new `mul` values if needed, and run `resume` on the
experiment directory. Existing incomplete runs resume from their frozen child
config and checkpoint with their original count; newly created runs use the
new count. Never edit a `mul_*` child config to change an existing run's chain
count.

`sampling.convergence_batch_total_samples` may also be changed in the
experiment-level `config.toml`. Completed `mul` runs remain untouched, while
incomplete and newly created runs use the new convergence-check batch size.

`sampling.relative_error` may be changed there as well. Incomplete and newly
created runs use the new target and restart their consecutive-convergence count.
Completed runs are left untouched when the target is unchanged or relaxed. When
the target is tightened, a completed run resumes production from its final
checkpoint and accumulates fresh convergence checks. This requires
`output.keep_final_checkpoint=true` and unused `flow_max_total_samples` capacity;
resume reports all runs that fail either requirement before starting any work.

## Run phases

1. A pilot estimates the correlation length and chooses the production volume.
2. HMC warmup tunes the step size to the requested acceptance probability.
3. Scale setting measures only `S00/S10/S01` to determine pooled and
   leave-one-chain-out `xi`, autocorrelation time, and the flow sampling interval.
4. The configured chains continue in production. Selected states accumulate only in a
   GPU-memory buffer (typically 384 configurations at L<=64 or 128 at L<=128),
   then flow as one batch. Only `E_action`, `S00`, `S10`, `S01`, and `Q_z` are
   written; buffered configurations never reach disk.
5. Sampling continues until every requested rho has the configured relative
   error in `tE_action`, or the maximum sample count is reached.

Outputs live below `runs/`. Each `mul` directory contains `manifest.json`,
`scale.npz`, atomically committed `observations/flow_*.npz`, `results.npz`, a
small restart checkpoint, and plots. Existing `gf_data/`, `gf_results/`, and
`gf_plots/` are not modified or deleted.

All sample budgets in the example TOML are totals across chains. The program
rounds them upward so every chain remains the same length. When `alpha=0`, the
integer plaquette field `s` is not allocated, updated, checkpointed, or flowed.

Production results are separated into unflowed and flowed observables. The
unflowed block contains `chi_m`, pooled `xi`, and connected susceptibilities for
`Q_z`, `Q_U`, and (only when alpha is nonzero) `Q_s`. The flowed block contains
`tE`, `chi_m`, pooled `xi`, and connected susceptibilities for `Q_z` and `Q_U`;
`Q_s` is not repeated because fixed-s flow leaves it unchanged.

## Verification

```powershell
python -m unittest discover -s tests -p "test_*.py"
python tests/benchmark_cuda_hmc.py
python tests/benchmark_cuda_flow.py
```

Tests cover NumPy/Torch action and force parity for both model modes,
constraint preservation, exact checkpoint/RNG restoration, flow parity, online
analysis, and a complete small CPU run.
