from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm, solve_continuous_are

from .config import TRIM_STATE, ControllerConfig, SimulationConfig
from .dynamics import FlightModel
from .rotations import euler_to_quat, quat_to_euler


@dataclass(frozen=True)
class LinearModel:
    A: np.ndarray
    B: np.ndarray
    Ad: np.ndarray
    Bd: np.ndarray
    K: np.ndarray
    algorithm: str = "predictor"


def state_error(state, trim_state=TRIM_STATE):
    angles = np.rad2deg(quat_to_euler(state[3:7]))
    trim_angles = np.rad2deg(quat_to_euler(trim_state[3:7]))
    angle_error = (angles - trim_angles + 180.0) % 360.0 - 180.0
    return np.r_[
        state[1:3] - trim_state[1:3],
        state[7:10] - trim_state[7:10],
        angle_error,
        np.rad2deg(state[10:13] - trim_state[10:13]),
    ]


def linearize(config=SimulationConfig(), trim_state=TRIM_STATE):
    model = FlightModel(config)
    trim = trim_state[:13].copy()
    control = trim_state[13:18].copy()

    def rhs(state, command):
        derivative = model.derivatives(state, command, gust=False)
        roll, pitch, _ = quat_to_euler(state[3:7])
        p, q, r = state[10:13]
        cosine = np.cos(pitch)
        if abs(cosine) < 1e-6:
            cosine = np.copysign(1e-6, cosine)
        tangent = np.sin(pitch) / cosine
        euler_rate = [
            p + np.sin(roll) * tangent * q + np.cos(roll) * tangent * r,
            np.cos(roll) * q - np.sin(roll) * r,
            (np.sin(roll) * q + np.cos(roll) * r) / cosine,
        ]
        return np.r_[
            derivative[1:3], derivative[7:10], np.rad2deg(euler_rate), np.rad2deg(derivative[10:13])
        ]

    indices = (1, 2, 7, 8, 9, None, None, None, 10, 11, 12)
    steps = (0.001,) * 5 + (0.2,) * 6
    A = np.empty((11, 11))
    B = np.empty((11, 5))
    for column, (index, step) in enumerate(zip(indices, steps)):
        plus, minus = trim.copy(), trim.copy()
        if index is None:
            angles = np.array(quat_to_euler(trim[3:7]))
            angle_shift = np.zeros(3)
            angle_shift[column - 5] = np.deg2rad(step)
            plus[3:7] = euler_to_quat(*(angles + angle_shift))
            minus[3:7] = euler_to_quat(*(angles - angle_shift))
        else:
            shift = step if column < 5 else np.deg2rad(step)
            plus[index] += shift
            minus[index] -= shift
        A[:, column] = (rhs(plus, control) - rhs(minus, control)) / (2 * step)
    for column in range(5):
        plus, minus = control.copy(), control.copy()
        plus[column] += 0.5
        minus[column] -= 0.5
        B[:, column] = rhs(trim, plus) - rhs(trim, minus)
    return A, B


def build_model(
    config=SimulationConfig(),
    controller=ControllerConfig(),
    trim_state=TRIM_STATE,
    algorithm="predictor",
):
    if algorithm not in ("predictor", "scheduled"):
        raise ValueError("Unknown controller algorithm")
    linearization_state = trim_state.copy()
    if algorithm == "scheduled":
        linearization_state[3:7] = euler_to_quat(0, np.deg2rad(53.79295161), 0)
        linearization_state[13:18] = [74.87955567, 22.48168998, 7.24077107, 0, 0]
    A, B = linearize(config, linearization_state)
    block = np.zeros((16, 16))
    block[:11, :11], block[:11, 11:] = A, B
    discrete = expm(block * config.dt)
    Q, R = np.diag(controller.q_weights), np.diag(controller.r_weights)
    if algorithm == "scheduled":
        Q = np.diag([30, 20, 8, 30, 35, 12, 4, 12, 0.8, 0.08, 0.8])
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.solve(R, B.T @ P)
    else:
        P = solve_continuous_are(A[2:, 2:], B[2:], Q, R)
        K = np.linalg.solve(R, B[2:].T @ P)
    return LinearModel(A, B, discrete[:11, :11], discrete[:11, 11:], K, algorithm)


class PredictorLQR:
    def __init__(self, model, dt=0.005, config=ControllerConfig(), trim_state=TRIM_STATE):
        self.model = model
        self.dt = dt
        self.config = config
        self.trim_state = trim_state
        self.trim_control = trim_state[13:18]
        self.velocity_command = np.zeros(2)

    def command(self, delayed_state, control_history, applied_control, predictor_steps=None):
        estimate = state_error(delayed_state, self.trim_state)
        history = list(control_history)
        horizon = len(history) if predictor_steps is None else predictor_steps
        held = applied_control - self.trim_control
        for step in range(horizon):
            delta = history[step] if step < len(history) else held
            estimate = self.model.Ad @ estimate + self.model.Bd @ delta
        cfg = self.config
        target = np.clip(
            -np.array([cfg.ky, cfg.kz]) * estimate[:2],
            -np.array([cfg.vy_limit, cfg.vz_limit]),
            np.array([cfg.vy_limit, cfg.vz_limit]),
        )
        smoothing = 1 - np.exp(-self.dt / np.array([cfg.tau_y, cfg.tau_z]))
        self.velocity_command += smoothing * (target - self.velocity_command)
        inner_error = estimate[2:].copy()
        inner_error[1:3] -= self.velocity_command
        raw = self.trim_control - cfg.gain * (self.model.K @ inner_error)
        limited = np.clip(
            raw, self.trim_control - cfg.control_range, self.trim_control + cfg.control_range
        )
        requested = applied_control + (1 - np.exp(-self.dt / cfg.tau_control)) * (
            limited - applied_control
        )
        return requested, raw, limited


class ScheduledLQR:
    """Earlier full-state LQR with error-dependent gain and direct measurements."""

    def __init__(self, model, dt=0.005, config=ControllerConfig(), trim_state=TRIM_STATE):
        self.model = model
        self.trim_state = trim_state.copy()
        self.trim_state[3:7] = euler_to_quat(0, np.deg2rad(53.79295161), 0)
        self.trim_state[13:18] = [74.87955567, 22.48168998, 7.24077107, 0, 0]

    def command(self, observed_state, control_history, applied_control, predictor_steps=None):
        error = state_error(observed_state, self.trim_state)
        error[3] -= np.clip(-2.5 * error[0], -1.2, 1.2)
        error[4] -= np.clip(-2.5 * error[1], -1.5, 1.5)
        strength = np.clip(0.8 * np.linalg.norm(error[2:5]) + 0.4 * np.linalg.norm(error[:2]), 0, 1)
        gain = 0.07 + (0.14 - 0.07) * strength
        raw = self.trim_state[13:18] - gain * (self.model.K @ error)
        return raw, raw, raw
