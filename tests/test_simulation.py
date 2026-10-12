from dataclasses import replace

import numpy as np
import pytest

from bumblebee_flight import simulation
from bumblebee_flight.config import (
    CONTROL_HIGH,
    CONTROL_LOW,
    CONTROL_STEP,
    TRIM_STATE,
    SimulationConfig,
)


@pytest.fixture
def fake_flight(monkeypatch, linear_model):
    class Flight:
        def __init__(self, config, direction):
            self.config = config

        def step(self, state, control):
            updated = state.copy()
            updated[:3] += self.config.dt * np.array([1, 0.1, -0.2])
            updated[7:10] += [0.001, -0.002, 0.003]
            return updated

        def wind_active(self, x):
            return self.config.wind_start < x < self.config.wind_end

    monkeypatch.setattr(simulation, "FlightModel", Flight)
    monkeypatch.setattr(simulation, "build_model", lambda *args: linear_model)


@pytest.mark.parametrize("delay_ms", [0, 20, 50, 100])
def test_sensor_is_causal_including_zero_delay(fake_flight, delay_ms):
    config = SimulationConfig(duration=0.16)
    trim = TRIM_STATE.copy()
    result = simulation.simulate(delay_ms=delay_ms, config=config, trim_state=trim)
    count, lag = len(result["states"]), delay_ms // 5
    assert result["states"].shape == (32, 18)
    assert int(result["delay_steps"]) == int(result["predictor_steps"]) == lag
    expected = np.vstack([np.tile(trim, (lag + 1, 1)), result["states"][: count - lag - 1]])
    np.testing.assert_allclose(result["delayed_states"], expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        result["controller_time"], np.arange(count) * config.dt, rtol=0, atol=1e-14
    )
    np.testing.assert_allclose(
        result["state_time"], np.arange(1, count + 1) * config.dt, rtol=0, atol=1e-14
    )
    np.testing.assert_allclose(
        result["observed_time"],
        np.maximum(0, result["controller_time"] - lag * config.dt),
        rtol=0,
        atol=1e-14,
    )
    np.testing.assert_array_equal(trim, TRIM_STATE)


@pytest.mark.parametrize("delay_ms", [0, 20, 50, 100])
def test_history_contains_applied_not_requested_controls_in_time_order(
    monkeypatch, fake_flight, delay_ms
):
    calls = []

    class ScriptedController:
        def __init__(self, *args):
            pass

        def command(self, observed, history, applied, horizon):
            i = len(calls)
            requested = TRIM_STATE[13:18] + [
                100 - i,
                -50 + 7 * i,
                90 - 3 * i,
                60 - 2 * i,
                -60 + 2 * i,
            ]
            calls.append(
                (
                    np.asarray(list(history)).reshape(-1, 5).copy(),
                    applied.copy(),
                    requested,
                    horizon,
                )
            )
            return requested, requested.copy(), requested.copy()

    monkeypatch.setattr(simulation, "PredictorLQR", ScriptedController)
    result = simulation.simulate(delay_ms=delay_ms, config=SimulationConfig(duration=0.16))
    lag = delay_ms // 5
    controls = result["controls"]
    for i, (history, applied, requested, horizon) in enumerate(calls):
        padded = np.vstack([np.zeros((lag, 5)), controls[:i] - TRIM_STATE[13:18]])
        np.testing.assert_array_equal(history, padded[i : i + lag])
        np.testing.assert_array_equal(applied, TRIM_STATE[13:18] if i == 0 else controls[i - 1])
        assert horizon == lag
        bounded = np.clip(requested, CONTROL_LOW, CONTROL_HIGH)
        demanded = bounded - applied
        action = np.clip(demanded, -CONTROL_STEP, CONTROL_STEP)
        np.testing.assert_array_equal(
            controls[i], np.clip(applied + action, CONTROL_LOW, CONTROL_HIGH)
        )
        np.testing.assert_array_equal(result["command_clipped"][i], requested != bounded)
        np.testing.assert_array_equal(result["rate_limited"][i], demanded != action)
    assert result["command_clipped"].any() and result["rate_limited"].any()


@pytest.mark.parametrize("horizon", [0, 2, 4, 7])
def test_simulation_with_explicit_horizon_is_deterministic(fake_flight, horizon):
    options = dict(delay_ms=20, config=SimulationConfig(duration=0.06), predictor_steps=horizon)
    first, second = simulation.simulate(**options), simulation.simulate(**options)
    for field in first:
        np.testing.assert_array_equal(first[field], second[field])
    assert int(first["predictor_steps"]) == horizon


@pytest.mark.parametrize("horizon", [-1, 1.5])
def test_invalid_horizon_is_rejected(fake_flight, horizon):
    with pytest.raises(ValueError):
        simulation.simulate(predictor_steps=horizon)


@pytest.mark.parametrize("kind", ["shape", "nonfinite", "zero_quaternion"])
def test_invalid_initial_state_is_rejected_before_integration(fake_flight, kind):
    trim = TRIM_STATE.copy()
    if kind == "shape":
        trim = trim[:-1]
    elif kind == "nonfinite":
        trim[0] = np.nan
    else:
        trim[3:7] = 0
    with pytest.raises(ValueError):
        simulation.simulate(config=SimulationConfig(duration=0.01), trim_state=trim)


@pytest.mark.parametrize("horizon", [None, 0, 1, 4])
def test_scheduled_simulation_disables_prediction_and_rejects_nonzero_horizon(
    fake_flight, linear_model, horizon
):
    model = replace(linear_model, K=np.zeros((5, 11)), algorithm="scheduled")
    options = dict(
        model=model, delay_ms=20, predictor_steps=horizon, config=SimulationConfig(duration=0.01)
    )
    if horizon not in (None, 0):
        with pytest.raises(ValueError, match="predictor horizon"):
            simulation.simulate(**options)
    else:
        result = simulation.simulate(**options)
        assert int(result["predictor_steps"]) == 0
        assert int(result["delay_steps"]) == 4
        np.testing.assert_allclose(
            result["delayed_states"], np.tile(TRIM_STATE, (2, 1)), rtol=0, atol=1e-12
        )
