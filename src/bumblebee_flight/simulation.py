from collections import deque

import numpy as np

from .config import (
    CONTROL_HIGH,
    CONTROL_LOW,
    CONTROL_STEP,
    TRIM_STATE,
    ControllerConfig,
    SimulationConfig,
    milliseconds_to_steps,
)
from .controllers import PredictorLQR, ScheduledLQR, build_model
from .dynamics import FlightModel


def simulate(
    direction=(0, 1, 0),
    delay_ms=0,
    config=SimulationConfig(),
    controller_config=ControllerConfig(),
    model=None,
    predictor_steps=None,
    trim_state=TRIM_STATE,
):
    delay_steps = milliseconds_to_steps(delay_ms, config.dt)
    horizon = delay_steps if predictor_steps is None else predictor_steps
    if not isinstance(horizon, int) or horizon < 0:
        raise ValueError("Predictor horizon must be a non-negative integer")
    trim_state = np.asarray(trim_state, dtype=float).copy()
    if trim_state.shape != (18,) or not np.isfinite(trim_state).all():
        raise ValueError("Initial trim state must be a finite 18-vector")
    quaternion_norm = np.linalg.norm(trim_state[3:7])
    if quaternion_norm < 1e-12:
        raise ValueError("Initial quaternion must be nonzero")
    trim_state[3:7] /= quaternion_norm
    flight = FlightModel(config, direction)
    if model is None:
        model = build_model(config, controller_config, trim_state)
    if model.algorithm == "scheduled":
        if predictor_steps not in (None, 0):
            raise ValueError("Scheduled LQR does not use a predictor horizon")
        horizon = 0
        controller = ScheduledLQR(model, config.dt, controller_config, trim_state)
    else:
        controller = PredictorLQR(model, config.dt, controller_config, trim_state)
    sensor = deque([trim_state.copy() for _ in range(delay_steps + 1)], maxlen=delay_steps + 1)
    commands = deque([np.zeros(5) for _ in range(delay_steps)], maxlen=delay_steps)
    count = round(config.duration / config.dt)
    states = np.empty((count, 18))
    delayed_states = np.empty_like(states)
    raw_controls = np.empty((count, 5))
    limited_controls = np.empty_like(raw_controls)
    command_clipped = np.empty((count, 5), dtype=bool)
    rate_limited = np.empty_like(command_clipped)
    state = trim_state.copy()
    for step in range(count):
        observed = sensor[0].copy()
        applied = state[13:18].copy()
        requested, raw, limited = controller.command(observed, commands, applied, horizon)
        bounded = np.clip(requested, CONTROL_LOW, CONTROL_HIGH)
        demanded_step = bounded - applied
        action = np.clip(demanded_step, -CONTROL_STEP, CONTROL_STEP)
        control = np.clip(applied + action, CONTROL_LOW, CONTROL_HIGH)
        state = np.r_[flight.step(state[:13], control), control]
        if not np.isfinite(state).all():
            raise RuntimeError(f"Non-finite flight state at t={(step + 1) * config.dt:.3f} s")
        states[step], delayed_states[step] = state, observed
        raw_controls[step], limited_controls[step] = raw, limited
        command_clipped[step], rate_limited[step] = requested != bounded, demanded_step != action
        sensor.append(state.copy())
        commands.append(control - trim_state[13:18])
    controller_time = np.arange(count) * config.dt
    return {
        "state_time": controller_time + config.dt,
        "controller_time": controller_time,
        "observed_time": np.maximum(0, controller_time - delay_steps * config.dt),
        "states": states,
        "delayed_states": delayed_states,
        "controls": states[:, 13:18],
        "raw_controls": raw_controls,
        "limited_controls": limited_controls,
        "command_clipped": command_clipped,
        "rate_limited": rate_limited,
        "wind_active": np.array([flight.wind_active(x) for x in states[:, 0]]),
        "delay_steps": np.asarray(delay_steps),
        "predictor_steps": np.asarray(horizon),
    }
