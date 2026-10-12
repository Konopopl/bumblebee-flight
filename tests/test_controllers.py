from collections import deque
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad_vec
from scipy.linalg import expm

from bumblebee_flight import controllers
from bumblebee_flight.config import TRIM_STATE, ControllerConfig, SimulationConfig
from bumblebee_flight.controllers import PredictorLQR, ScheduledLQR, state_error
from bumblebee_flight.rotations import euler_to_quat


@pytest.mark.parametrize("dt", [0.005, 0.02])
def test_discretization_shapes_and_zero_order_hold(monkeypatch, dt):
    A = np.diag(np.r_[0, 0, -np.arange(1, 10)])
    A[0, 3], A[1, 4] = 1, 1
    B = np.arange(55, dtype=float).reshape(11, 5) / 100
    monkeypatch.setattr(controllers, "linearize", lambda *args: (A.copy(), B.copy()))
    model = controllers.build_model(SimulationConfig(dt=dt))
    assert model.A.shape == model.Ad.shape == (11, 11)
    assert model.B.shape == model.Bd.shape == (11, 5)
    assert model.K.shape == (5, 9)
    assert model.algorithm == "predictor"
    np.testing.assert_array_equal(model.A, A)
    np.testing.assert_array_equal(model.B, B)
    block = np.block([[A, B], [np.zeros((5, 11)), np.zeros((5, 5))]])
    expected = expm(block * dt)
    np.testing.assert_allclose(model.Ad, expected[:11, :11], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(model.Bd, expected[:11, 11:], rtol=1e-12, atol=1e-14)
    integral, _ = quad_vec(lambda t: expm(A * t) @ B, 0, dt)
    np.testing.assert_allclose(model.Bd, integral, rtol=1e-12, atol=1e-14)
    assert np.isfinite(model.K).all()


def test_state_error_order_units_and_angle_wrapping():
    trim = TRIM_STATE.copy()
    trim[3:7] = euler_to_quat(0, 0, np.deg2rad(179))
    state = trim.copy()
    state[1:3] += [0.1, -0.2]
    state[7:10] += [0.3, -0.4, 0.5]
    state[3:7] = euler_to_quat(np.deg2rad(10), np.deg2rad(-20), np.deg2rad(-179))
    state[10:13] += np.deg2rad([1, -2, 3])
    expected = [0.1, -0.2, 0.3, -0.4, 0.5, 10, -20, 2, 1, -2, 3]
    np.testing.assert_allclose(state_error(state, trim), expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "horizon", [0, 2, 4, None, 7], ids=["zero", "short", "matched", "default", "long"]
)
def test_predictor_uses_chronological_history_and_held_control(linear_model, horizon):
    cfg, dt = ControllerConfig(), 0.005
    observed = TRIM_STATE.copy()
    observed[1:3] += [0.2, -0.1]
    observed[7:10] += [0.1, 0.2, -0.3]
    history = deque([np.arange(5, dtype=float) * value for value in (0.3, -0.2, 0.7, 0.1)])
    saved_history = np.array(history)
    applied = TRIM_STATE[13:18] + np.array([1, -2, 3, -4, 5])
    count = len(history) if horizon is None else horizon
    initial = np.array([0.2, -0.1, 0.1, 0.2, -0.3, 0, 0, 0, 0, 0, 0])
    estimate = np.linalg.matrix_power(linear_model.Ad, count) @ initial
    for i in range(count):
        delta = saved_history[i] if i < len(history) else applied - TRIM_STATE[13:18]
        estimate += np.linalg.matrix_power(linear_model.Ad, count - i - 1) @ linear_model.Bd @ delta
    first = PredictorLQR(linear_model, dt, cfg)
    second = PredictorLQR(linear_model, dt, cfg)
    velocity_command = np.zeros(2)
    for _ in range(2):
        target = np.clip(
            -np.array([cfg.ky, cfg.kz]) * estimate[:2],
            -np.array([cfg.vy_limit, cfg.vz_limit]),
            [cfg.vy_limit, cfg.vz_limit],
        )
        velocity_command += (1 - np.exp(-dt / np.array([cfg.tau_y, cfg.tau_z]))) * (
            target - velocity_command
        )
        inner = estimate[2:].copy()
        inner[1:3] -= velocity_command
        raw = TRIM_STATE[13:18] - cfg.gain * (linear_model.K @ inner)
        limited = np.clip(
            raw, TRIM_STATE[13:18] - cfg.control_range, TRIM_STATE[13:18] + cfg.control_range
        )
        requested = applied + (1 - np.exp(-dt / cfg.tau_control)) * (limited - applied)
        actual = first.command(observed, history, applied, horizon)
        np.testing.assert_allclose(actual, (requested, raw, limited), rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(actual, second.command(observed, history, applied, horizon))
    np.testing.assert_array_equal(np.array(history), saved_history)


def test_tuple_control_ranges_work_with_numpy_and_saturate_both_signs(linear_model):
    cfg = ControllerConfig(gain=1, control_range=(0.5, 1.0, 1.5, 2.0, 2.5))
    K = np.zeros((5, 9))
    K[:, 0] = [100, -100, 100, -100, 100]
    controller = PredictorLQR(replace(linear_model, K=K), config=cfg)
    observed = TRIM_STATE.copy()
    observed[7] += 1
    applied = TRIM_STATE[13:18].copy()
    requested, raw, limited = controller.command(observed, [], applied, predictor_steps=0)
    expected = applied + np.array([-0.5, 1.0, -1.5, 2.0, -2.5])
    np.testing.assert_allclose(limited, expected, rtol=0, atol=1e-12)
    assert np.all(np.abs(raw - applied) > np.asarray(cfg.control_range))
    assert np.all(np.abs(requested - applied) < np.asarray(cfg.control_range))
    assert np.diag(cfg.q_weights).shape == (9, 9)
    assert np.diag(cfg.r_weights).shape == (5, 5)


@pytest.mark.parametrize("speed_error,gain", [(0, 0.07), (0.5, 0.098), (2, 0.14)])
def test_scheduled_lqr_preserves_legacy_trim_and_gain(linear_model, speed_error, gain):
    controls = np.array([74.87955567, 22.48168998, 7.24077107, 0, 0])
    coefficients = np.array([1, -2, 3, -4, 5])
    K = np.zeros((5, 11))
    K[:, 2], K[:, 5] = coefficients, 1
    controller = ScheduledLQR(replace(linear_model, K=K, algorithm="scheduled"))
    observed = TRIM_STATE.copy()
    observed[3:7] = euler_to_quat(np.deg2rad(1), np.deg2rad(53.79295161), 0)
    observed[7] += speed_error
    expected = controls - gain * (coefficients * speed_error + 1)
    actual = controller.command(observed, [np.ones(5)], np.zeros(5), predictor_steps=0)
    np.testing.assert_allclose(actual, np.tile(expected, (3, 1)), rtol=0, atol=1e-12)
    np.testing.assert_array_equal(controller.trim_state[13:18], controls)
    np.testing.assert_allclose(
        controller.trim_state[3:7], euler_to_quat(0, np.deg2rad(53.79295161), 0), atol=1e-12
    )


@pytest.mark.parametrize("dy,dz", [(0.1, -0.2), (0.8, -1.0)])
def test_scheduled_lqr_full_state_law_and_velocity_command_limits(linear_model, dy, dz):
    K = np.arange(55, dtype=float).reshape(5, 11) / 10
    model = replace(linear_model, K=K, algorithm="scheduled")
    observed = TRIM_STATE.copy()
    observed[1:3] += [dy, dz]
    observed[7:10] += [0.3, 0.2, -0.1]
    observed[3:7] = euler_to_quat(*np.deg2rad([10, 53.79295161 - 3, 20]))
    observed[10:13] = np.deg2rad([2, -4, 6])
    error = np.array(
        [
            dy,
            dz,
            0.3,
            0.2 + np.clip(2.5 * dy, -1.2, 1.2),
            -0.1 + np.clip(2.5 * dz, -1.5, 1.5),
            10,
            -3,
            20,
            2,
            -4,
            6,
        ]
    )
    gain = 0.07 + 0.07 * min(1, 0.8 * np.linalg.norm(error[2:5]) + 0.4 * np.linalg.norm(error[:2]))
    expected = np.array([74.87955567, 22.48168998, 7.24077107, 0, 0]) - gain * (K @ error)
    controller = ScheduledLQR(model)
    actual = controller.command(observed, [], np.zeros(5), predictor_steps=0)
    np.testing.assert_allclose(actual, np.tile(expected, (3, 1)), rtol=0, atol=1e-12)
