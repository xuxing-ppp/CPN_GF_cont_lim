# CPN Gradient Flow

## Common commands

Copy `config.example.toml` to `config.toml`, edit it, and run from the repository root:
If using the provided conda setup, activate it first with `conda activate pytorch`.

```bash
python -m cpn_gf run --config config.toml
python -m cpn_gf pilot --config config.toml
python -m cpn_gf recommend-chains --run runs/<experiment>
python -m cpn_gf resume --run runs/<experiment>
python -m cpn_gf analyze --run runs/<experiment>
```

`run` creates a timestamped experiment under `output.root`. `pilot --config`
creates an experiment with only scale-selection pilots. Both print its path.
The example enables `hmc.auto_chains = true`: `run` automatically runs or reuses
each mul's pilot, selects production chains, prints the chosen count, and starts
production. With a fixed lattice it selects chains directly, without a pilot.
For a named experiment, create `runs/my_experiment/`, put your configuration
there as `config.toml`, and use `resume --run runs/my_experiment` for the same
automatic workflow. The separate pilot and recommendation commands are optional.

`--run` takes a directory, not a TOML file or the general `runs/` directory.
Resume and analyze also accept an individual `mul_*` directory. Use
`pilot --run runs/<experiment>` to fill in missing or failed pilots, or
`analyze --run runs/<experiment> --aggregate-only` to rebuild experiment plots
and summaries using existing per-mul JSON/NPZ results.

## Configuration and restart

`hmc.auto_chains` defaults to `false` for compatibility with old configs; the
example explicitly enables it. When enabled, `hmc.chains` supplies pilot initial
chain counts, and production counts are measured automatically before warmup.
With the switch disabled, `hmc.chains` sets production counts directly.
It accepts an integer for all mul values or an integer array matching
`model.mul` in length and order. Every count must be at least 2:

```toml
model.mul = [0.8, 1.0, 1.2]
hmc.chains = [64, 32, 16]
```

Update the chains array when adding, removing, or reordering mul values. Removing
a mul excludes its directory from experiment resume and analysis without deleting
it. Edit only the experiment root `config.toml`; child configs are managed
automatically and contain only `model.mul = [current_value]` and one integer
chain count. New children and children with only a completed pilot use the
latest root auto/manual setting. Once production warmup starts, chains remain frozen for
exact checkpoint/RNG restoration. Resume handles interrupted children first,
starts missing children, and skips completed children.

Other root settings that can be changed on resume:

- `sampling.convergence_batch_total_samples`: convergence-check interval.
- `sampling.relative_error`: stopping target; resets convergence streaks.
  Tightening it can reopen completed runs if the final checkpoint and unused
  maximum sample budget remain. These requirements are checked before work starts.
- `analysis.min_t_over_a2_for_fit`: flow-time threshold for fits and online
  stopping; resets convergence streaks.

Other model, lattice, HMC, sampling, flow, compute, and output settings must
remain unchanged after a child is created. Use a new experiment for changes.

## Pilots and chain recommendations

`lattice.L = 0` selects production size from each mul's pilot; a positive value
fixes production size. Pilots start at `L0` and increase it until the central
estimate satisfies `L0/xi >= target_L0_over_xi`, bounded by `max_L0`.
Omitting `max_L0` makes it equal to `L0`.

Pilots measure structure modes only. Results and actual pilot chains are saved
in child manifests and experiment-level `pilot_results.json`/`.csv`. Failed
pilots can be retried. Scan broadly with `pilot`, select mul values in the root
config, and `resume` to reuse their estimates with independent production warmup.

`recommend-chains` benchmarks each mul at its own production size. Automatic
size completes missing pilots; fixed size needs no pilot. CUDA pilot OOM retries
halve chains down to 2. The benchmark searches powers of two, selecting the
smallest candidate within 95% of best measured HMC throughput that meets the
VRAM budget and flow feasibility check. It stops after two consecutive gains
below 5%, OOM, or the memory limit. `--max-chains N` sets the ceiling (default
1024). Details are saved incrementally in `chains_recommendations.json`.
Rerunning benchmarks the current machine again. Root TOML and production
checkpoints are never changed by this command.

Automatic selection uses the same search and default ceiling of 1024, saving
per-child diagnostics in `chains_recommendation.json` and the chosen result in
the manifest. A matching completed result is reused before production starts;
changed configuration or device information triggers a new benchmark. Existing
production checkpoints always restore their frozen chains without benchmarking.
`pilot` remains pilot-only even when automatic chains are enabled.

## Sampling, storage, and analysis

CPU and CUDA share PyTorch HMC with production `float64/complex128`; set
`compute.device = "cpu"` for CPU. Production warmup tunes the step size, then
scale setting measures `S00/S10/S01`, pooled and leave-one-chain-out xi,
autocorrelation time, and the flow sampling interval. Flow targets are then fixed.

Sample budgets are totals across chains, rounded upward for equal chain lengths.
Full z/a/s histories are never archived. Configurations may be buffered within
one transactional flow chunk; only the atomic current checkpoint stores full
configurations. Chunks retain chain identity and named `E_action`, `S00`, `S10`,
`S01`, `Q_z`, and `Q_U`. Second-moment xi comes from pooled structure means,
never averaged per configuration. With `alpha=0`, s is absent throughout;
otherwise `Q_s` is stored only as an unflowed observable.

`flow.kinds = ["model", "covariant"]` runs paired flows sequentially from the
same sampled states, committing a chunk only after both succeed. Sampling stops
when every included rho in every kind meets the relative error target in tE,
or the maximum sample budget is reached.

Children contain manifests, scale statistics, observation chunks, checkpoints,
per-kind results and plots. At fixed rho, at least four eligible mul points are
needed for the error-in-both-axes fit `t<E(t)> = c0 + c1/xi^2 + c2/xi^4`.
Experiment outputs are under `analysis/`, `continuum_fits/`, and `plots/`.
The analysis threshold excludes small flow times from fitting and stopping but
preserves their stored data and plots. Legacy `gf_data/`, `gf_results/`,
`gf_plots/`, and scripts below `legacy/` remain untouched.

## Verification

```bash
python -m unittest discover -s tests -p "test_*.py"
python tests/benchmark_cuda_hmc.py
python tests/benchmark_cuda_flow.py
```

Tests cover action/force parity, constraints and reversibility, exact restart,
flow monotonicity/parity, online analysis, and a small complete CPU run.
