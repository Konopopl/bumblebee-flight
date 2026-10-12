import csv
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from platform import python_version

import numpy as np

from .config import TRIM_STATE


def crossing_time(time, x, boundary, initial_x=0.0):
    time, x = np.r_[0.0, time], np.r_[initial_x, x]
    crossings = np.flatnonzero((x[:-1] < boundary) & (x[1:] >= boundary))
    if not len(crossings):
        return None
    index = crossings[0]
    fraction = (boundary - x[index]) / (x[index + 1] - x[index])
    return float(time[index] + fraction * (time[index + 1] - time[index]))


def event_times(result, config, trim_state=TRIM_STATE):
    time, x = result["state_time"], result["states"][:, 0]
    start = (
        0.0
        if config.wind_start <= trim_state[0] < config.wind_end
        else crossing_time(time, x, config.wind_start, trim_state[0])
    )
    return (
        start,
        crossing_time(time, x, config.wind_end, trim_state[0]),
    )


def settling_time(time, speed_error, gust_end, threshold=0.03, minimum_tail=1.0):
    """First time below threshold for the rest of the record, with >=1 s remaining."""
    if gust_end is None:
        return None
    inside = speed_error < threshold
    inside_to_end = np.logical_and.accumulate(inside[::-1])[::-1]
    valid = inside_to_end & (time >= gust_end) & (time <= time[-1] - minimum_tail)
    indices = np.flatnonzero(valid)
    return float(time[indices[0]]) if len(indices) else None


def metrics(direction, delay_ms, result, config, trim_state=TRIM_STATE):
    states, time = result["states"], result["state_time"]
    error = np.linalg.norm(states[:, 7:10] - trim_state[7:10], axis=1)
    gust_start, gust_end = event_times(result, config, trim_state)
    stable = settling_time(time, error, gust_end) if config.beta > 0 else None
    dy, dz = (states[:, 1:3] - trim_state[1:3]).T
    return {
        "direction": direction,
        "delay_ms": delay_ms,
        "predictor_steps": int(result["predictor_steps"]),
        "beta": config.beta,
        "gust_reached": gust_start is not None and config.beta > 0,
        "gust_start_s": gust_start,
        "gust_end_s": gust_end,
        "settling_time_after_onset_s": None if stable is None else stable - gust_start,
        "final_speed_error_mps": float(error[-1]),
        "peak_dy_m": float(dy[np.argmax(np.abs(dy))]),
        "min_dy_m": float(dy.min()),
        "max_dy_m": float(dy.max()),
        "final_dy_m": float(dy[-1]),
        "peak_dz_m": float(dz[np.argmax(np.abs(dz))]),
        "final_dz_m": float(dz[-1]),
    }


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_results(directory, cases, config, controller, trim_state=TRIM_STATE, model=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {"trim_state": trim_state, "dt": np.asarray(config.dt)}
    rows = []
    for (direction, delay_ms), result in cases.items():
        prefix = f"{direction}_delay_{delay_ms}ms__"
        arrays.update({prefix + key: value for key, value in result.items()})
        rows.append(metrics(direction, delay_ms, result, config, trim_state))
    np.savez_compressed(directory / "results.npz", **arrays)
    if model is not None:
        np.savez_compressed(
            directory / "linear_model.npz",
            A=model.A,
            B=model.B,
            Ad=model.Ad,
            Bd=model.Bd,
            K=model.K,
        )
    controller_settings = asdict(controller)
    for key in ("q_weights", "r_weights", "control_range"):
        controller_settings[key] = np.asarray(controller_settings[key], dtype=float).tolist()
    metadata = {
        "simulation": asdict(config),
        "controller": controller_settings,
        "trim_state": trim_state.tolist(),
        "algorithm": "predictor" if model is None else model.algorithm,
        "software": {
            "python": python_version(),
            **{name: version(name) for name in ("numpy", "scipy", "jax", "jaxlib")},
        },
        "cases": [
            {
                "direction": direction,
                "delay_ms": delay,
                "predictor_steps": int(result["predictor_steps"]),
            }
            for (direction, delay), result in cases.items()
        ],
    }
    if model is not None and model.algorithm == "scheduled":
        metadata["controller"] = {
            "q_weights": [30, 20, 8, 30, 35, 12, 4, 12, 0.8, 0.08, 0.8],
            "r_weights": list(controller.r_weights),
            "gain_min": 0.07,
            "gain_max": 0.14,
            "ky": 2.5,
            "kz": 2.5,
            "vy_limit": 1.2,
            "vz_limit": 1.5,
            "trim_pitch_deg": 53.79295161,
            "trim_control": [74.87955567, 22.48168998, 7.24077107, 0, 0],
        }
    (directory / "config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write_csv(directory / "metrics.csv", rows)


def load_results(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    cases = {}
    with np.load(directory / "results.npz", allow_pickle=False) as data:
        for item in metadata["cases"]:
            key = (item["direction"], item["delay_ms"])
            prefix = f"{key[0]}_delay_{key[1]}ms__"
            cases[key] = {
                name[len(prefix) :]: data[name].copy()
                for name in data.files
                if name.startswith(prefix)
            }
    return cases, metadata
