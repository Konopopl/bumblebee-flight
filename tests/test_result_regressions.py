import json

import matplotlib.pyplot as plt
import numpy as np
import pytest

from bumblebee_flight import cli, plots, results
from bumblebee_flight.config import TRIM_STATE, ControllerConfig, SimulationConfig

ABSOLUTE_TIME_LABEL = "\u0432\u0440\u0435\u043c\u044f, \u0441"


@pytest.fixture
def result_factory():
    def make_result(config, trim=TRIM_STATE, speed=0.5):
        time = np.arange(1, round(config.duration / config.dt) + 1) * config.dt
        states = np.tile(trim, (len(time), 1))
        states[:, 0] += speed * time
        return {
            "state_time": time,
            "states": states,
            "controls": states[:, 13:18],
            "predictor_steps": np.asarray(0),
        }

    return make_result


@pytest.fixture
def captured_figures(monkeypatch):
    figures = []
    monkeypatch.setattr(plots, "save_figure", lambda figure, *args: figures.append(figure))
    yield figures
    for figure in figures:
        plt.close(figure)


def test_metrics_for_gust_active_at_start(result_factory):
    config = SimulationConfig(duration=1.5)
    trim = TRIM_STATE.copy()
    trim[0] = 2.02
    result = result_factory(config, trim=trim)
    start, end = results.event_times(result, config, trim)
    assert start == 0
    assert end == pytest.approx((config.wind_end - trim[0]) / trim[7])
    row = results.metrics("plus_y", 0, result, config, trim)
    assert row["gust_reached"]
    assert row["gust_start_s"] == 0
    expected_settled = result["state_time"][result["state_time"] >= end][0]
    assert row["settling_time_after_onset_s"] == expected_settled


def test_numpy_controller_weights_save_and_load_as_json_lists(
    tmp_path, result_factory, linear_model
):
    config = SimulationConfig(duration=0.01)
    controller = ControllerConfig(
        q_weights=np.arange(1, 10, dtype=float),
        r_weights=np.arange(1, 6, dtype=float) / 10,
        control_range=np.arange(1, 6, dtype=float),
    )
    original = result_factory(config)
    cases = {("plus_y", 0): original}
    results.save_results(tmp_path, cases, config, controller, model=linear_model)
    metadata = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    for field in ("q_weights", "r_weights", "control_range"):
        assert isinstance(metadata["controller"][field], list)
        np.testing.assert_array_equal(metadata["controller"][field], getattr(controller, field))
    loaded, loaded_metadata = results.load_results(tmp_path)
    assert loaded_metadata == metadata
    for field in original:
        np.testing.assert_array_equal(loaded[("plus_y", 0)][field], original[field])


def test_mixed_reached_and_unreached_cases_use_absolute_time(result_factory, captured_figures):
    config = SimulationConfig(duration=6)
    reached = result_factory(config)
    unreached = result_factory(config, speed=0.1)
    assert results.event_times(reached, config)[0] is not None
    assert results.event_times(unreached, config)[0] is None
    cases = {("plus_y", 0): reached, ("plus_y", 20): unreached}
    plots.plot_components(cases, config, TRIM_STATE, ".", "coordinates")
    plots.plot_controls(cases, config, TRIM_STATE, ".")
    assert len(captured_figures) == 2
    for figure in captured_figures:
        assert figure.axes[-1].get_xlabel() == ABSOLUTE_TIME_LABEL
        for axis in figure.axes:
            assert len(axis.lines) == 2
            for line in axis.lines:
                np.testing.assert_array_equal(line.get_xdata(), reached["state_time"])


def test_zero_beta_has_no_gust_spans_or_event_alignment(result_factory, captured_figures):
    config = SimulationConfig(beta=0, duration=6)
    result = result_factory(config)
    assert results.event_times(result, config)[0] is not None
    cases = {("plus_y", 0): result}
    plots.plot_components(cases, config, TRIM_STATE, ".", "coordinates")
    plots.plot_controls(cases, config, TRIM_STATE, ".")
    plots.plot_trajectories(cases, config, TRIM_STATE, ".")
    assert len(captured_figures) == 4
    for figure in captured_figures:
        assert all(not axis.patches for axis in figure.axes)
        assert all(len(legend.get_texts()) == 1 for legend in figure.legends)
    for figure in captured_figures[:2]:
        assert figure.axes[-1].get_xlabel() == ABSOLUTE_TIME_LABEL
        for axis in figure.axes:
            np.testing.assert_array_equal(axis.lines[0].get_xdata(), result["state_time"])


def test_sweep_preserves_distinct_close_beta_output_directories(
    monkeypatch, tmp_path, result_factory, linear_model
):
    betas = [0.7900001, 0.7900002]
    saved, simulated, csv_rows = [], [], []
    monkeypatch.setattr(cli, "build_model", lambda *args, **kwargs: linear_model)

    def fake_simulate(direction, delay_ms, config, controller, **kwargs):
        simulated.append(config.beta)
        return result_factory(config)

    def capture_save(directory, cases, config, controller, **kwargs):
        saved.append((directory, config.beta))

    monkeypatch.setattr(cli, "simulate", fake_simulate)
    monkeypatch.setattr(cli, "save_results", capture_save)
    monkeypatch.setattr(cli, "write_csv", lambda path, rows: csv_rows.extend(rows))
    cli.main(
        [
            "sweep",
            "--betas",
            *map(str, betas),
            "--directions",
            "plus_y",
            "--delays",
            "0",
            "--duration",
            "0.01",
            "--output",
            str(tmp_path),
        ]
    )
    assert simulated == betas
    assert len(saved) == 2
    assert len({directory for directory, _ in saved}) == 2
    assert [float(directory.name.removeprefix("beta_")) for directory, _ in saved] == betas
    assert [beta for _, beta in saved] == betas
    assert [row["beta"] for row in csv_rows] == betas
