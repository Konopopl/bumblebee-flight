from dataclasses import dataclass

import numpy as np

TARGET_VELOCITY = np.array([0.5, 0.0, 0.0])
TRIM_STATE = np.array(
    [
        0.0,
        0.0,
        0.5,
        0.8918259523688917,
        0.0,
        0.45237868062201964,
        0.0,
        0.5,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        74.88059032384642,
        22.494892064641316,
        7.258747655223729,
        0.0,
        0.0,
    ]
)
TRIM_CONTROL = TRIM_STATE[13:18].copy()
CONTROL_LOW = np.array([0.0, 0.0, -88.0, -40.0, -40.0])
CONTROL_HIGH = np.array([88.0, 88.0, 88.0, 40.0, 40.0])
CONTROL_STEP = np.array([0.5, 5.0, 1.0, 1.0, 1.0])
DIRECTIONS = {
    "plus_x": (1, 0, 0),
    "minus_x": (-1, 0, 0),
    "plus_y": (0, 1, 0),
    "minus_y": (0, -1, 0),
    "plus_z": (0, 0, 1),
    "minus_z": (0, 0, -1),
    "plus_xy": (1, 1, 0),
    "plus_x_minus_y": (1, -1, 0),
    "plus_xz": (1, 0, 1),
    "minus_xz": (-1, 0, -1),
    "plus_yz": (0, 1, 1),
    "plus_y_minus_z": (0, 1, -1),
    "plus_xyz": (1, 1, 1),
    "minus_xy_plus_z": (-1, -1, 1),
}


@dataclass(frozen=True)
class SimulationConfig:
    dt: float = 0.005
    duration: float = 10.0
    beta: float = 0.79
    wind_start: float = 2.0
    wind_end: float = 2.05
    wing_samples: int = 20
    flap_frequency: float = 123.0

    def __post_init__(self):
        values = (
            self.dt,
            self.duration,
            self.beta,
            self.wind_start,
            self.wind_end,
            self.flap_frequency,
        )
        if not np.isfinite(values).all():
            raise ValueError("Simulation parameters must be finite")
        if self.dt <= 0 or self.duration <= 0 or self.flap_frequency <= 0:
            raise ValueError("Time step, duration and frequency must be positive")
        if self.beta < 0 or self.wind_end <= self.wind_start:
            raise ValueError("Invalid gust strength or spatial interval")
        if not isinstance(self.wing_samples, int) or self.wing_samples < 2:
            raise ValueError("wing_samples must be an integer >= 2")
        if not np.isclose(
            self.duration / self.dt, round(self.duration / self.dt), rtol=0, atol=1e-8
        ):
            raise ValueError("Duration must be a multiple of the time step")


@dataclass(frozen=True)
class ControllerConfig:
    gain: float = 0.0125
    q_weights: tuple = (8.0, 30.0, 16000.0, 12.0, 4.0, 12.0, 0.8, 0.08, 0.8)
    r_weights: tuple = (0.08, 0.08, 0.10, 0.20, 0.20)
    control_range: tuple = (4.0, 15.0, 8.0, 8.0, 8.0)
    ky: float = 0.15
    kz: float = 1.0
    vy_limit: float = 0.06
    vz_limit: float = 0.12
    tau_y: float = 0.45
    tau_z: float = 0.40
    tau_control: float = 0.04

    def __post_init__(self):
        for weights, length in ((self.q_weights, 9), (self.r_weights, 5), (self.control_range, 5)):
            if len(weights) != length or not np.isfinite(weights).all() or min(weights) <= 0:
                raise ValueError("Invalid controller weights or control range")
        values = (
            self.gain,
            self.ky,
            self.kz,
            self.vy_limit,
            self.vz_limit,
            self.tau_y,
            self.tau_z,
            self.tau_control,
        )
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Controller gains, limits and time constants must be positive")


def milliseconds_to_steps(delay_ms, dt):
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Time step must be finite and positive")
    steps = delay_ms / (1000.0 * dt)
    if (
        not np.isfinite(steps)
        or steps < 0
        or not np.isclose(steps, round(steps), atol=1e-9, rtol=0)
    ):
        raise ValueError("Delay must be non-negative and a multiple of the time step")
    return round(steps)
