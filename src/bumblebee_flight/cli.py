import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import (
    DIRECTIONS,
    TRIM_STATE,
    ControllerConfig,
    SimulationConfig,
    milliseconds_to_steps,
)
from .controllers import build_model
from .results import load_results, metrics, save_results, write_csv
from .simulation import simulate


def simulation_arguments(parser):
    parser.add_argument("--controller", choices=["predictor", "scheduled"], default="predictor")
    parser.add_argument(
        "--directions", nargs="+", choices=DIRECTIONS, default=["plus_y", "minus_y"]
    )
    parser.add_argument(
        "--delays",
        type=int,
        nargs="+",
        default=[0, 20, 50, 100],
        help="Measurement delay in milliseconds",
    )
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--beta", type=float, default=0.79)
    parser.add_argument(
        "--predictor-steps", type=int, default=None, help="Default: match the measurement delay"
    )
    parser.add_argument("--output", type=Path, default=Path("results"))


def run_cases(args, config, controller, model):
    cases = {}
    for direction in dict.fromkeys(args.directions):
        for delay in dict.fromkeys(args.delays):
            print(f"{direction}: delay {delay} ms, beta {config.beta:g}", flush=True)
            cases[(direction, delay)] = simulate(
                DIRECTIONS[direction],
                delay,
                config,
                controller,
                model=model,
                predictor_steps=args.predictor_steps,
            )
    return cases


def make_plots(cases, config, trim_state, directory):
    from .plots import plot_all

    plot_all(cases, config, trim_state, directory)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bumblebee flight dynamics and delayed feedback")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("simulate", help="Run a delay comparison")
    simulation_arguments(run)
    run.add_argument("--no-plots", action="store_true")
    sweep = commands.add_parser("sweep", help="Compare gust strengths")
    simulation_arguments(sweep)
    sweep.add_argument("--betas", nargs="+", type=float, default=[0.5, 0.6, 0.79, 1.0])
    plot = commands.add_parser("plot", help="Regenerate figures from saved arrays")
    plot.add_argument("--data", type=Path, default=Path("results"))
    video = commands.add_parser("video", help="Render a flight video and attitude sequence")
    video.add_argument("--data", type=Path, default=Path("results"))
    video.add_argument("--delay", type=int, default=20)
    video.add_argument("--direction", choices=DIRECTIONS, default="plus_y")
    video.add_argument("--view", choices=["perspective", "top"], default="perspective")
    video.add_argument("--fps", type=int, default=20)
    video.add_argument("--frames", type=int, default=120)
    trim = commands.add_parser("trim", help="Recompute the nominal steady-flight state")
    trim.add_argument("--wing-samples", type=int, default=20)
    args = parser.parse_args(argv)
    if args.command == "trim":
        from .dynamics import find_trim
        from .rotations import quat_to_euler

        state = find_trim(SimulationConfig(wing_samples=args.wing_samples))
        print("Pitch (degrees):", np.rad2deg(quat_to_euler(state[3:7])[1]))
        print("Wing controls (degrees):", state[13:18].tolist())
        return
    if args.command in ("simulate", "sweep"):
        config = SimulationConfig(dt=args.dt, duration=args.duration, beta=args.beta)
        controller = ControllerConfig()
        for delay in args.delays:
            milliseconds_to_steps(delay, config.dt)
        if args.predictor_steps is not None and args.predictor_steps < 0:
            parser.error("Predictor horizon must be non-negative")
        if args.controller == "scheduled" and args.predictor_steps not in (None, 0):
            parser.error("Scheduled LQR does not use a predictor horizon")
        model = build_model(config, controller, algorithm=args.controller)
        if args.command == "simulate":
            cases = run_cases(args, config, controller, model)
            save_results(args.output, cases, config, controller, model=model)
            if not args.no_plots:
                make_plots(cases, config, TRIM_STATE, args.output / "figures")
        else:
            from .coefficients import MODEL

            rows = []
            for beta in dict.fromkeys(args.betas):
                sweep_config = SimulationConfig(**(asdict(config) | {"beta": beta}))
                cases = run_cases(args, sweep_config, controller, model)
                save_results(
                    args.output / f"beta_{beta!r}", cases, sweep_config, controller, model=model
                )
                for (direction, delay), result in cases.items():
                    row = metrics(direction, delay, result, sweep_config)
                    row["force_mN"] = beta * float(MODEL.wbody) * 1000
                    rows.append(row)
            write_csv(args.output / "sweep_metrics.csv", rows)
        print(args.output.resolve())
        return
    cases, metadata = load_results(args.data)
    config = SimulationConfig(**metadata["simulation"])
    trim_state = np.asarray(metadata["trim_state"])
    if args.command == "plot":
        make_plots(cases, config, trim_state, args.data / "figures")
        print((args.data / "figures").resolve())
    else:
        from .visualization import render_flight

        key = (args.direction, args.delay)
        if key not in cases:
            parser.error(f"No saved case for {args.direction}, {args.delay} ms")
        for path in render_flight(
            cases[key],
            args.data / "video" / args.direction,
            args.delay,
            direction=DIRECTIONS[args.direction],
            view=args.view,
            fps=args.fps,
            frames=args.frames,
            wind_start=config.wind_start,
            wind_end=config.wind_end,
        ):
            print(path.resolve())


if __name__ == "__main__":
    main()
