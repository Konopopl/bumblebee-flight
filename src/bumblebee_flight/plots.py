from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .results import event_times
from .rotations import quat_to_euler

COLORS = {0: "#333333", 20: "#0072B2", 50: "#D55E00", 100: "#CC79A7"}
STYLES = {0: "--", 20: "-", 50: "-.", 100: ":"}


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.25,
            "lines.linewidth": 1.7,
            "pdf.fonttype": 42,
        }
    )


def save_figure(figure, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(directory / f"{name}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def curve_style(delay, index):
    return {"color": COLORS.get(delay, f"C{index % 10}"), "linestyle": STYLES.get(delay, "-")}


def direction_label(name):
    return name.replace("plus_", "+").replace("minus_", "-").replace("_", " ")


def legend(delays, gust=True):
    handles = [
        Line2D(
            [],
            [],
            **curve_style(delay, index),
            label="без задержки" if delay == 0 else f"задержка {delay} мс",
        )
        for index, delay in enumerate(delays)
    ]
    if gust:
        handles.append(Patch(facecolor="#dddddd", label="область воздействия"))
    return handles


def plot_components(cases, config, trim_state, output_dir, kind):
    directions = list(dict.fromkeys(direction for direction, _ in cases))
    delays = sorted({delay for _, delay in cases})
    is_error = kind == "speed_error"
    rows = 1 if is_error else 3
    align = config.beta > 0 and all(
        event_times(result, config, trim_state)[0] is not None for result in cases.values()
    )
    labels = {
        "coordinates": ("$x$, м", "$y$, м", "$z$, м"),
        "velocities": ("$v_x$, м/с", "$v_y$, м/с", "$v_z$, м/с"),
        "euler_angles": ("крен, град", "тангаж, град", "рыскание, град"),
        "speed_error": (r"$\|\mathbf{v}-\mathbf{v}^{*}\|_2$, м/с",),
    }[kind]
    figure, axes = plt.subplots(
        rows,
        len(directions),
        figsize=(6 * len(directions), 2.7 * rows + 0.6),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    for column, direction in enumerate(directions):
        spans = []
        for index, delay in enumerate(delays):
            result = cases.get((direction, delay))
            if result is None:
                continue
            states = result["states"]
            start, end = event_times(result, config, trim_state)
            time = result["state_time"] - (start if align else 0)
            if start is not None and end is not None and config.beta > 0:
                spans.append((0, end - start) if align else (start, end))
            if kind == "coordinates":
                values = states[:, :3]
            elif kind == "velocities":
                values = states[:, 7:10]
            elif kind == "euler_angles":
                values = np.rad2deg([quat_to_euler(q) for q in states[:, 3:7]])
            else:
                values = np.linalg.norm(states[:, 7:10] - trim_state[7:10], axis=1)[:, None]
            for row in range(rows):
                axes[row, column].plot(time, values[:, row], **curve_style(delay, index))
        for row in range(rows):
            axis = axes[row, column]
            if spans:
                axis.axvspan(
                    min(span[0] for span in spans),
                    max(span[1] for span in spans),
                    color="#dddddd",
                    linewidth=0,
                    zorder=0,
                )
            axis.set_ylabel(labels[row])
            if is_error:
                axis.axhline(0.03, color="#666666", linestyle=":", linewidth=1)
            if align:
                axis.set_xlim(-min(4, config.wind_start / trim_state[7]), min(6, config.duration))
        axes[0, column].text(
            0.97,
            0.94,
            direction_label(direction),
            transform=axes[0, column].transAxes,
            ha="right",
            va="top",
            fontweight="bold",
        )
        axes[-1, column].set_xlabel("время от начала воздействия, с" if align else "время, с")
    figure.legend(
        handles=legend(delays, gust=config.beta > 0),
        loc="upper center",
        ncol=len(delays) + 1,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94 if rows > 1 else 0.85))
    save_figure(figure, output_dir, kind)


def plot_trajectories(cases, config, trim_state, output_dir):
    directions = list(dict.fromkeys(direction for direction, _ in cases))
    delays = sorted({delay for _, delay in cases})
    figure, axes = plt.subplots(
        2,
        len(directions),
        figsize=(6 * len(directions), 6.5),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    for column, direction in enumerate(directions):
        for index, delay in enumerate(delays):
            result = cases.get((direction, delay))
            if result is None:
                continue
            states = result["states"]
            shown = (states[:, 0] >= config.wind_start - 0.5) & (
                states[:, 0] <= config.wind_start + 0.5
            )
            for row, coordinate in enumerate((1, 2)):
                axes[row, column].plot(
                    states[shown, 0],
                    states[shown, coordinate] - trim_state[coordinate],
                    **curve_style(delay, index),
                )
        for row, label in enumerate((r"$\Delta y$, м", r"$\Delta z$, м")):
            axis = axes[row, column]
            axis.set_aspect("equal", adjustable="box")
            if config.beta > 0:
                axis.axvspan(config.wind_start, config.wind_end, color="#dddddd", zorder=0)
            axis.set_xlim(config.wind_start - 0.5, config.wind_start + 0.5)
            axis.set_ylim(-0.15, 0.15)
            axis.set_ylabel(label)
            axis.set_xlabel("$x$, м")
        axes[0, column].text(
            0.97,
            0.94,
            direction_label(direction),
            transform=axes[0, column].transAxes,
            ha="right",
            va="top",
            fontweight="bold",
        )
    figure.legend(
        handles=legend(delays, gust=config.beta > 0),
        loc="upper center",
        ncol=len(delays) + 1,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(figure, output_dir, "trajectory_projections")
    figure = plt.figure(figsize=(8, 5))
    axis = figure.add_subplot(111, projection="3d")
    for index, ((direction, delay), result) in enumerate(cases.items()):
        states = result["states"]
        axis.plot(
            *states[:, :3].T,
            label=f"{direction_label(direction)}, {delay} мс",
            **curve_style(delay, index),
        )
    axis.set(xlabel="$x$, м", ylabel="$y$, м", zlabel="$z$, м")
    span = np.ptp(np.concatenate([result["states"][:, :3] for result in cases.values()]), axis=0)
    axis.set_box_aspect(np.maximum(span, 0.05))
    axis.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    save_figure(figure, output_dir, "trajectory_3d")


def plot_controls(cases, config, trim_state, output_dir):
    delays = sorted({delay for _, delay in cases})
    align = config.beta > 0 and all(
        event_times(result, config, trim_state)[0] is not None for result in cases.values()
    )
    labels = (
        r"$a^{\phi}_{1}$, град",
        r"$a^{\phi}_{0}$, град",
        r"$a^{\alpha}_{0}$, град",
        r"$\Delta a^{\phi}_{1,R}$, град",
        r"$\Delta a^{\phi}_{1,L}$, град",
    )
    for direction in dict.fromkeys(direction for direction, _ in cases):
        figure, axes = plt.subplots(5, 1, figsize=(8, 8), sharex=True)
        spans = []
        for index, delay in enumerate(delays):
            result = cases.get((direction, delay))
            if result is None:
                continue
            start, end = event_times(result, config, trim_state)
            time = result["state_time"] - (start if align else 0)
            if config.beta > 0 and start is not None and end is not None:
                spans.append((0, end - start) if align else (start, end))
            for row, axis in enumerate(axes):
                axis.plot(time, result["controls"][:, row], **curve_style(delay, index))
                axis.set_ylabel(labels[row])
        if spans:
            for axis in axes:
                axis.axvspan(
                    min(span[0] for span in spans),
                    max(span[1] for span in spans),
                    color="#dddddd",
                    zorder=0,
                )
        axes[-1].set_xlabel("время от начала воздействия, с" if align else "время, с")
        figure.legend(
            handles=legend(delays, gust=config.beta > 0),
            loc="upper center",
            ncol=len(delays) + 1,
            frameon=False,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.94))
        save_figure(figure, output_dir, f"controls_{direction}")


def plot_all(cases, config, trim_state, output_dir):
    configure_style()
    for kind in ("coordinates", "velocities", "euler_angles", "speed_error"):
        plot_components(cases, config, trim_state, output_dir, kind)
    plot_trajectories(cases, config, trim_state, output_dir)
    plot_controls(cases, config, trim_state, output_dir)
