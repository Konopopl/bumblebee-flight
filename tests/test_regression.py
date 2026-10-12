from pathlib import Path

import numpy as np
import pytest

from bumblebee_flight.config import (
    CONTROL_HIGH,
    CONTROL_LOW,
    CONTROL_STEP,
    TRIM_STATE,
    SimulationConfig,
)
from bumblebee_flight.results import load_results


@pytest.fixture(scope="module")
def reference():
    with np.load(Path(__file__).parent / "data" / "reference.npz", allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def test_reference_is_complete_and_trim_unchanged(reference):
    assert int(reference["format_version"]) == 1
    assert reference["states"].shape == (8, 13, 18)
    assert reference["directions"].shape == (8, 3)
    assert set(zip(reference["case_names"], reference["delays_ms"])) == {
        (name, delay) for name in ("plus_y", "minus_y") for delay in (0, 20, 50, 100)
    }
    np.testing.assert_array_equal(
        reference["indices"], [0, 199, 399, 599, 799, 809, 819, 839, 879, 999, 1199, 1599, 1999]
    )
    np.testing.assert_array_equal(reference["trim_state"], TRIM_STATE)
    assert float(reference["dt"]) == SimulationConfig().dt
    assert float(reference["duration"]) == SimulationConfig().duration
    assert np.isfinite(reference["states"]).all()


@pytest.fixture(scope="module")
def regression_output(request, reference):
    directory = request.config.getoption("--regression-output")
    if directory is None:
        pytest.skip("Supply --regression-output with the completed full simulation directory")
    cases, metadata = load_results(directory)
    config = SimulationConfig(**metadata["simulation"])
    assert config == SimulationConfig()
    np.testing.assert_allclose(metadata["trim_state"], reference["trim_state"], rtol=0, atol=1e-12)
    return cases, config


@pytest.mark.slow
@pytest.mark.parametrize(
    "case",
    range(8),
    ids=[f"{name}-{delay}ms" for name in ("plus_y", "minus_y") for delay in (0, 20, 50, 100)],
)
def test_nonlinear_trajectory_matches_legacy_reference(reference, regression_output, case):
    cases, config = regression_output
    trim = reference["trim_state"]
    delay_ms = int(reference["delays_ms"][case])
    result = cases[(str(reference["case_names"][case]), delay_ms)]
    assert int(result["delay_steps"]) == int(result["predictor_steps"]) == delay_ms // 5
    states = result["states"]
    assert states.shape == (2000, 18) and np.isfinite(states).all()
    np.testing.assert_allclose(
        states[reference["indices"]], reference["states"][case], rtol=1e-7, atol=1e-8
    )
    np.testing.assert_allclose(np.linalg.norm(states[:, 3:7], axis=1), 1, rtol=0, atol=1e-12)
    assert np.all((states[:, 13:18] >= CONTROL_LOW) & (states[:, 13:18] <= CONTROL_HIGH))
    changes = np.diff(np.vstack([trim[13:18], states[:, 13:18]]), axis=0)
    assert np.all(np.abs(changes) <= CONTROL_STEP + 1e-12)
    lag = int(result["delay_steps"]) + 1
    np.testing.assert_allclose(
        result["delayed_states"][:lag], np.tile(trim, (lag, 1)), rtol=0, atol=1e-12
    )
    np.testing.assert_array_equal(result["delayed_states"][lag:], states[:-lag])
    np.testing.assert_array_equal(
        result["wind_active"], (states[:, 0] > config.wind_start) & (states[:, 0] < config.wind_end)
    )
