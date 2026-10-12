import numpy as np
import pytest

from bumblebee_flight.config import SimulationConfig
from bumblebee_flight.dynamics import FlightModel


def test_wind_interval_is_strictly_open():
    config = SimulationConfig()
    flight = FlightModel(config)
    start, end = config.wind_start, config.wind_end
    points = [
        np.nextafter(start, -np.inf),
        start,
        np.nextafter(start, end),
        (start + end) / 2,
        np.nextafter(end, start),
        end,
        np.nextafter(end, np.inf),
    ]
    assert [bool(flight.wind_active(x)) for x in points] == [
        False,
        False,
        True,
        True,
        True,
        False,
        False,
    ]
    assert not FlightModel(SimulationConfig(beta=0)).wind_active((start + end) / 2)


@pytest.mark.parametrize("direction", [(0, 7, 0), (3, -4, 0), (1, 1, -1)])
def test_wind_direction_is_normalized_without_changing_sign(direction):
    vector = np.asarray(direction, dtype=float)
    flight = FlightModel(direction=vector)
    np.testing.assert_allclose(flight.direction, vector / np.linalg.norm(vector), atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(flight.direction), 1, atol=1e-12)
    np.testing.assert_array_equal(vector, direction)


@pytest.mark.parametrize("direction", [(0, 0, 0), (1, 0), (1, np.nan, 0), (0, np.inf, 0)])
def test_invalid_wind_direction_is_rejected(direction):
    with pytest.raises(ValueError):
        FlightModel(direction=direction)
