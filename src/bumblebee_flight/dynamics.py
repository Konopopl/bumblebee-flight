import numpy as np
from scipy.optimize import least_squares

from .aerodynamics import forces_and_moments
from .coefficients import MODEL
from .config import CONTROL_HIGH, CONTROL_LOW, TRIM_STATE, SimulationConfig
from .rotations import euler_to_quat, quat_multiply, quat_normalize


class FlightModel:
    def __init__(self, config=SimulationConfig(), direction=(0, 1, 0), inertia=None):
        self.config = config
        direction = np.asarray(direction, dtype=float)
        if (
            direction.shape != (3,)
            or not np.isfinite(direction).all()
            or np.linalg.norm(direction) == 0
        ):
            raise ValueError("Gust direction must be a finite, nonzero 3-vector")
        self.direction = direction / np.linalg.norm(direction)
        self.inertia = (
            np.eye(3) * float(MODEL.Jbody) if inertia is None else np.asarray(inertia, dtype=float)
        )
        if (
            self.inertia.shape != (3, 3)
            or not np.isfinite(self.inertia).all()
            or not np.allclose(self.inertia, self.inertia.T)
            or np.min(np.linalg.eigvalsh(self.inertia)) <= 0
        ):
            raise ValueError("Inertia tensor must be symmetric positive definite")
        self.inertia_inv = np.linalg.inv(self.inertia)

    def wind_active(self, x):
        return self.config.beta > 0 and self.config.wind_start < x < self.config.wind_end

    def derivatives(self, state, control, gust=True):
        quaternion = quat_normalize(state[3:7])
        omega = state[10:13]
        force, torque = forces_and_moments(
            state,
            control,
            frequency=self.config.flap_frequency,
            ntr=self.config.wing_samples,
        )
        force = np.asarray(force, dtype=float).copy()
        torque = np.asarray(torque, dtype=float)
        if gust and self.wind_active(state[0]):
            force += self.config.beta * float(MODEL.wbody) * self.direction
        acceleration = force / float(MODEL.mbody)
        acceleration[2] -= float(MODEL.wbody) / float(MODEL.mbody)
        angular_acceleration = self.inertia_inv @ (torque - np.cross(omega, self.inertia @ omega))
        quaternion_rate = 0.5 * quat_multiply(quaternion, np.r_[0.0, omega])
        return np.r_[state[7:10], quaternion_rate, acceleration, angular_acceleration]

    def step(self, state, control):
        dt = self.config.dt
        k1 = self.derivatives(state, control)
        k2 = self.derivatives(state + 0.5 * dt * k1, control)
        k3 = self.derivatives(state + 0.5 * dt * k2, control)
        k4 = self.derivatives(state + dt * k3, control)
        state = state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        state[3:7] = quat_normalize(state[3:7])
        return state


def find_trim(config=SimulationConfig()):
    """Find pitch and symmetric wing controls for steady flight at 0.5 m/s."""
    model = FlightModel(config)

    def state_from_parameters(parameters):
        state = TRIM_STATE.copy()
        state[3:7] = euler_to_quat(0, np.deg2rad(parameters[0]), 0)
        state[13:16] = parameters[1:]
        return state

    def residual(parameters):
        state = state_from_parameters(parameters)
        derivative = model.derivatives(state[:13], state[13:18], gust=False)
        return np.r_[derivative[7], derivative[9], derivative[11]]

    result = least_squares(
        residual,
        [53.8, *TRIM_STATE[13:16]],
        bounds=([0, *CONTROL_LOW[:3]], [85, *CONTROL_HIGH[:3]]),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=2000,
    )
    if not result.success or np.linalg.norm(result.fun) > 1e-6:
        raise RuntimeError(f"Trim solution failed: residual={np.linalg.norm(result.fun):.3g}")
    return state_from_parameters(result.x)
