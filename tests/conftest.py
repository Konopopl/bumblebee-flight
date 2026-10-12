import numpy as np
import pytest

from bumblebee_flight.controllers import LinearModel


@pytest.fixture
def linear_model():
    rng = np.random.default_rng(42)
    Ad = np.diag(np.linspace(0.95, 1.05, 11))
    Ad[0, 3], Ad[1, 4] = 0.02, 0.03
    return LinearModel(
        A=np.zeros((11, 11)),
        B=np.zeros((11, 5)),
        Ad=Ad,
        Bd=rng.normal(scale=0.01, size=(11, 5)),
        K=rng.normal(size=(5, 9)),
    )
