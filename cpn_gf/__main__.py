from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import analyze_path
from .recommend import print_recommendation, recommend_chains
from .runner import pilot_config, pilot_experiment, resume_run, run_config


def _print_analysis_summary(path, summaries, aggregate_only=False):
    path = Path(path)
    print(f"Analysis complete: {path}")
    if (path / "manifest.json").is_file():
        summary = next(iter(summaries.values()))
        xi = summary.get("xi_scale")
        xi_error = summary.get("xi_scale_error")
        if xi is not None and xi_error is not None:
            print(f"xi_scale: {float(xi):.8g} +/- {float(xi_error):.3g}")
        samples = summary.get("n_samples_per_chain")
        if samples is not None and len(samples):
            low, high = int(min(samples)), int(max(samples))
            value = str(low) if low == high else f"{low}-{high}"
            print(f"samples per chain: {value}")
        if "converged" in summary:
            print(f"converged: {bool(summary['converged'])}")
        print(f"outputs: {path / 'results.json'}, {path / 'results.npz'}, {path / 'plots'}")
        return

    mode = "aggregate only" if aggregate_only else "per-mul and aggregate"
    print(f"mode: {mode}")
    print("mul runs: " + ", ".join(summaries))
    fits_path = path / "continuum_fits.json"
    if fits_path.is_file():
        with fits_path.open(encoding="utf-8") as fh:
            fits = json.load(fh)
        fitted = sum(item.get("fit") is not None for item in fits.values())
        print(f"continuum fits: {fitted}/{len(fits)} rho values")
    print(f"outputs: {path / 'analysis.json'}, {fits_path}, {path / 'plots'}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m cpn_gf")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="start a new experiment")
    run.add_argument("--config", required=True)
    pilot = sub.add_parser("pilot", help="run only scale-selection pilots")
    pilot_source = pilot.add_mutually_exclusive_group(required=True)
    pilot_source.add_argument("--config")
    pilot_source.add_argument("--run")
    resume = sub.add_parser("resume", help="resume a mul run or extend an experiment")
    resume.add_argument("--run", required=True)
    analyze = sub.add_parser("analyze", help="rebuild aggregate results")
    analyze.add_argument("--run", required=True)
    analyze.add_argument("--aggregate-only", action="store_true",
                         help="reuse per-mul results and rebuild only experiment summaries")
    recommend = sub.add_parser("recommend-chains", help="benchmark and recommend HMC chains")
    recommend.add_argument("--config", required=True)
    size = recommend.add_mutually_exclusive_group()
    size.add_argument("--lattice-size", type=int)
    size.add_argument("--pilot", action="store_true")
    recommend.add_argument("--max-chains", type=int, default=1024)
    args = parser.parse_args(argv)
    if args.command == "run":
        path, _ = run_config(args.config)
        print(path)
    elif args.command == "pilot":
        result = (pilot_config(args.config) if args.config
                  else pilot_experiment(args.run))
        print(result["experiment"])
        for item in result["runs"]:
            values = item.get("pilot") or {}
            xi = values.get("xi")
            detail = ("" if xi is None else
                      f" xi={xi:.8g} +/- {values.get('xi_error', float('nan')):.3g}"
                      f" recommended_L={values.get('recommended_L', values.get('L'))}")
            print(f"mul={item['mul']:g}: {item['action']} ({item['status']}){detail}")
    elif args.command == "resume":
        result = resume_run(args.run)
        if "runs" in result:
            for item in result["runs"]:
                print(f"mul={item['mul']:g}: {item['action']} ({item['status']})")
        else:
            print(result["status"])
    elif args.command == "analyze":
        summaries = analyze_path(args.run, progress=True,
                                 aggregate_only=args.aggregate_only)
        _print_analysis_summary(args.run, summaries,
                                aggregate_only=args.aggregate_only)
    else:
        print_recommendation(recommend_chains(
            args.config, lattice_size=args.lattice_size,
            run_pilot=args.pilot, max_chains=args.max_chains))


if __name__ == "__main__":
    main()
