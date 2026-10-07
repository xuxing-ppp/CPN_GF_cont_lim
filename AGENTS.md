# Repository Guide

## Active workflow

- Configure runs with TOML; start with `config.example.toml`.
- Run `python -m cpn_gf run --config <file.toml>` from the repository root.
- Resume a mul run with `python -m cpn_gf resume --run <run-directory>`.
- Rebuild results/plots with `python -m cpn_gf analyze --run <mul-or-experiment-directory>`.
- Active sampling is HMC-only. CPU and CUDA both use the same PyTorch code in
  `cpn_gf/`; production precision is float64/complex128.
- Sample budgets are totals across all chains. Flow snapshots may be buffered
  in VRAM within one transactional chunk but must never be archived.
- The old three-stage scripts are read-only references below `legacy/`.

## Data and statistics

- Never archive production `z/a/s` histories. Only the atomic current-state
  checkpoint may contain full configurations.
- Online flow chunks contain named per-configuration `E_action`, `S00`, `S10`,
  `S01`, `Q_z`, and `Q_U`, retaining chain identity. `Q_s` is stored only as an
  unflowed observable when `alpha!=0`.
- Compute second-moment xi from pooled means of S00/S10/S01; never average
  per-sample xi values.
- Flow targets are fixed after scale setting using central and
  leave-one-chain-out xi estimates.
- Pilot and scale setting measure structure modes only; topology is measured
  on production configurations.
- Existing `gf_data/`, `gf_results/`, and `gf_plots/` are legacy user data and
  must not be changed or removed.
- `alpha=0` means the integer field `s` is absent throughout the active pipeline.

## Verification

```powershell
python -m unittest discover -s tests -p "test_*.py"
python tests/benchmark_cuda_hmc.py
python tests/benchmark_cuda_flow.py
```

For HMC changes, verify NumPy/Torch action and force parity, reversibility,
unit-normalized z, wrapped phases, invariant Villain plaquettes, independent
per-chain acceptance, and exact restart/RNG restoration. For flow changes,
retain action monotonicity and CPU/CUDA parity tests.
