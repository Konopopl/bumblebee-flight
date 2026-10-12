import numpy as np
import pytest

from bumblebee_flight.config import (
    ControllerConfig,
    SimulationConfig,
    milliseconds_to_steps,
)


@pytest.mark.parametrize(
    "delay_ms,dt,expected",
    [
        (0, 0.005, 0),
        (20, 0.005, 4),
        (50, 0.005, 10),
        (100, 0.005, 20),
        (20, 0.001, 20),
        (100, 0.02, 5),
        (30, 0.003, 10),
    ],
)
def test_delay_is_an_exact_integer_number_of_steps(delay_ms, dt, expected):
    actual = milliseconds_to_steps(delay_ms, dt)
    assert isinstance(actual, int)
    assert actual == expected


@pytest.mark.parametrize("delay_ms", [-5, -0.001, 1, 20.5, 20.00001, np.nan, np.inf])
def test_invalid_delay_is_rejected(delay_ms):
    with pytest.raises(ValueError):
        milliseconds_to_steps(delay_ms, 0.005)


@pytest.mark.parametrize("dt", [0, -0.005, np.nan, np.inf, -np.inf])
def test_delay_conversion_rejects_invalid_dt_even_for_zero_delay(dt):
    with pytest.raises(ValueError):
        milliseconds_to_steps(0, dt)


@pytest.mark.parametrize(
    "changes",
    [
        {"dt": 0},
        {"dt": -0.005},
        {"dt": np.nan},
        {"duration": 0},
        {"duration": 0.006},
        {"duration": np.inf},
        {"beta": -1},
        {"beta": np.nan},
        {"wind_start": 2.05},
        {"wind_end": 1.0},
        {"wind_end": np.inf},
        {"wing_samples": 1},
        {"wing_samples": 2.5},
        {"flap_frequency": 0},
    ],
)
def test_invalid_simulation_config_is_rejected(changes):
    with pytest.raises(ValueError):
        SimulationConfig(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"gain": 0},
        {"ky": np.nan},
        {"tau_control": -1},
        {"tau_y": np.inf},
        {"q_weights": (1,) * 8},
        {"r_weights": (1, 1, 0, 1, 1)},
        {"control_range": (1, 1, np.nan, 1, 1)},
    ],
)
def test_invalid_controller_config_is_rejected(changes):
    with pytest.raises(ValueError):
        ControllerConfig(**changes)
