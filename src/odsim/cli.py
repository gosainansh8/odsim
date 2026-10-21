"""Command-line interface: ``odsim run | sweep | compare``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .config import SimConfig
from .calibrate import abc_smc, load_spec
from .experiments import compare_scenarios, housing_cox, run_replicates, run_sweep
from .model import BACKENDS


def _parse_value(text: str) -> Any:
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _parse_params(items: list[str]) -> dict[str, list[Any]]:
    """Turn ["p_enroll=0.02,0.05", "frac_housed=0.5"] into a sweep grid."""
    grid: dict[str, list[Any]] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--param must look like name=v1,v2 (got {item!r})")
        name, values = item.split("=", 1)
        grid[name.strip()] = [_parse_value(v.strip()) for v in values.split(",") if v.strip()]
    return grid


def _cmd_run(args: argparse.Namespace) -> None:
    config = SimConfig.from_json(args.config)
    results = run_replicates(config, args.reps, args.seed, args.workers, args.backend)
    df = pd.DataFrame([{"rep": i, **r.summary()} for i, r in enumerate(results)])
    _emit(df, args.out)


def _cmd_sweep(args: argparse.Namespace) -> None:
    config = SimConfig.from_json(args.config)
    grid = _parse_params(args.param)
    df = run_sweep(config, grid, args.reps, args.seed, args.workers, args.backend)
    if args.heatmap:
        if len(grid) != 2:
            raise ValueError("--heatmap needs exactly two --param sweeps")
        from .report import plot_sweep_heatmap

        x, y = list(grid)
        fig = plot_sweep_heatmap(df, x, y, args.metric)
        fig.savefig(args.heatmap, dpi=150)
        print(f"wrote heatmap to {args.heatmap}")
    if args.aggregate and grid:
        metrics = [c for c in df.columns if c not in grid and c != "rep"]
        df = df.groupby(list(grid), as_index=False)[metrics].mean()
    _emit(df, args.out)


def _load_comparison(args: argparse.Namespace):
    config = SimConfig.from_json(args.config)
    scenarios = json.loads(Path(args.scenarios).read_text())
    comparison = compare_scenarios(
        config, scenarios, args.reps, args.seed, args.workers, args.baseline, args.backend
    )
    return config, comparison


def _cmd_compare(args: argparse.Namespace) -> None:
    _, comparison = _load_comparison(args)
    _emit(comparison.table, args.out)
    if args.curves:
        rows = [
            {"scenario": name, "time": t, "survival": s, "at_risk": n}
            for name, km in comparison.curves.items()
            for t, s, n in zip(km.times, km.survival, km.at_risk)
        ]
        pd.DataFrame(rows).to_csv(args.curves, index=False)


def _cmd_report(args: argparse.Namespace) -> None:
    from .report import build_report

    config, comparison = _load_comparison(args)
    housing = housing_cox(comparison.pooled[comparison.baseline])
    Path(args.html).write_text(build_report(config, comparison, housing, args.reps))
    print(f"wrote report to {args.html}")


def _cmd_calibrate(args: argparse.Namespace) -> None:
    config = SimConfig.from_json(args.config)
    priors, targets = load_spec(json.loads(Path(args.spec).read_text()))

    def progress(h: dict) -> None:
        means = ", ".join(f"{k[5:]}={v:.4g}" for k, v in h.items() if k.startswith("mean_"))
        print(
            f"generation {h['generation']}: tolerance {h['epsilon']:.3g}, "
            f"acceptance {h['acceptance']:.0%}, {h['simulations']} sims, {means}",
            file=sys.stderr,
        )

    result = abc_smc(
        config, priors, targets,
        n_particles=args.particles, generations=args.generations, reps=args.reps,
        quantile=args.quantile, seed=args.seed, workers=args.workers, backend=args.backend,
        progress=progress,
    )  # fmt: skip
    _emit(result.summary(), None)
    if args.out:
        result.posterior_frame().to_csv(args.out, index=False)
        print(f"wrote {len(result.particles)} posterior samples to {args.out}")


def _cmd_serve(args: argparse.Namespace) -> None:
    from .web.app import serve

    serve(args.host, args.port)


def _emit(df: pd.DataFrame, out: str | None) -> None:
    if out:
        df.to_csv(out, index=False)
        print(f"wrote {len(df)} rows to {out}")
    else:
        with pd.option_context("display.width", 160, "display.max_columns", 20):
            print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odsim", description="Agent-based simulation of overdose risk and treatment access."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("config", help="path to a JSON config file")
        p.add_argument("--reps", type=int, default=10, help="replicates per configuration")
        p.add_argument("--seed", type=int, default=0, help="master random seed")
        p.add_argument(
            "--workers", type=int, default=os.cpu_count() or 1, help="parallel processes"
        )
        p.add_argument("--out", help="write results to this CSV instead of printing")
        p.add_argument("--backend", choices=BACKENDS, default="numpy", help="simulation engine")

    p_run = sub.add_parser("run", help="run replicates of one configuration")
    common(p_run)
    p_run.set_defaults(func=_cmd_run)

    p_sweep = sub.add_parser("sweep", help="grid search over parameters")
    common(p_sweep)
    p_sweep.add_argument(
        "--param", action="append", default=[], help="name=v1,v2,... (repeatable)"
    )
    p_sweep.add_argument(
        "--aggregate", action="store_true", help="average metrics over replicates"
    )
    p_sweep.add_argument("--heatmap", help="save a heatmap PNG (needs exactly two --param)")
    p_sweep.add_argument("--metric", default="mortality", help="metric for --heatmap")
    p_sweep.set_defaults(func=_cmd_sweep)

    p_cmp = sub.add_parser("compare", help="compare intervention scenarios")
    common(p_cmp)
    p_cmp.add_argument("scenarios", help="JSON file mapping scenario name -> overrides")
    p_cmp.add_argument("--baseline", help="scenario to test others against (default: first)")
    p_cmp.add_argument("--curves", help="write Kaplan-Meier curves to this CSV")
    p_cmp.set_defaults(func=_cmd_compare)

    p_rep = sub.add_parser("report", help="compare scenarios and write an HTML report")
    common(p_rep)
    p_rep.add_argument("scenarios", help="JSON file mapping scenario name -> overrides")
    p_rep.add_argument("--baseline", help="scenario to test others against (default: first)")
    p_rep.add_argument("--html", default="report.html", help="output HTML file")
    p_rep.set_defaults(func=_cmd_report)

    p_cal = sub.add_parser("calibrate", help="infer parameters from targets with ABC-SMC")
    common(p_cal)
    p_cal.add_argument("spec", help="JSON file with 'priors' and 'targets'")
    p_cal.add_argument("--particles", type=int, default=200)
    p_cal.add_argument("--generations", type=int, default=5)
    p_cal.add_argument("--quantile", type=float, default=0.5, help="tolerance shrink quantile")
    p_cal.set_defaults(func=_cmd_calibrate, reps=1)

    p_srv = sub.add_parser("serve", help="start the web dashboard")
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8000)
    p_srv.set_defaults(func=_cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (ValueError, FileNotFoundError, KeyError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
