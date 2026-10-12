"""Wingbeat-averaged forces in ground axes and torques in body axes."""

from functools import partial
from math import pi

import jax
import jax.numpy as np

from .coefficients import MODEL, predict_coefficients, wing_geometry
from .kinematics import wing_kinematics
from .rotations import Rx, Ry, Rz, dRx, dRy, dRz, quat_normalize, quat_to_matrix


def _section_force_parts(c, v, alpha, sign_CT, dfeath, m22, a, tail, lead, domega_wing):
    norm_v = np.sqrt((v**2).sum(-1))
    sin_a, cos_a = np.sin(alpha), np.cos(alpha)
    c, m22 = c.reshape(1, -1), m22.reshape(1, -1)
    dfeath = dfeath.reshape(-1, 1)
    tail, lead = tail.reshape(1, -1), lead.reshape(1, -1)
    rho = MODEL.rou_f
    return (
        np.stack(
            (
                -0.5 * rho * c * c * dfeath * v[..., 0],
                -0.5 * rho * c * norm_v * np.sin(2.0 * alpha) * sign_CT * v[..., 0],
                -0.5 * rho * c * (cos_a**2) * norm_v * v[..., 2],
                -0.5 * rho * c * (sin_a**2) * norm_v * v[..., 2],
                -m22 * a[..., 2],
                -0.5 * (tail + lead) * m22 * domega_wing[:, None, 1],
                -1.0 / 6.0 * rho * (lead**3 + tail**3) * dfeath * np.abs(dfeath),
            ),
            axis=-1,
        )
        * 1e-9
    )


def _body_force(v_ground, E_ground2body):
    u = v_ground @ E_ground2body.T
    norm_u = np.linalg.norm(u, axis=-1, keepdims=True)
    low_speed = norm_u <= MODEL.u_ref * 0.03
    safe_norm_u = np.where(norm_u > 0.0, norm_u, 1.0)
    cos_alpha = np.clip(-u[..., [0]] / safe_norm_u, -1.0, 1.0)
    alpha = low_speed * pi / 2.0 + (~low_speed) * np.arccos(cos_alpha)
    cross_u_x = np.cross(u, np.cross(u, np.array([-1.0, 0.0, 0.0])))
    norm_cross = np.linalg.norm(cross_u_x, axis=-1, keepdims=True)
    lift = (
        -(MODEL.R**2)
        * MODEL.rou_f
        * np.sin(2 * alpha)
        * norm_u**2
        * cross_u_x
        / np.maximum(norm_cross, 1e-9)
    ) * 1e-9
    drag0 = (
        -(MODEL.R**2)
        * MODEL.rou_f
        * (np.cos(alpha) ** 2)
        * norm_u**2
        * u
        / np.maximum(norm_u, 1e-9)
    ) * 1e-9
    drag1 = (
        -(MODEL.R**2)
        * MODEL.rou_f
        * (np.sin(alpha) ** 2)
        * norm_u**2
        * u
        / np.maximum(norm_u, 1e-9)
    ) * 1e-9
    active = norm_u > MODEL.u_ref * 1e-7
    lift = (active * lift) @ E_ground2body
    drag0 = (active * drag0) @ E_ground2body
    drag1 = (active * drag1) @ E_ground2body
    C = MODEL.C_body_force
    return C[0] * lift + C[1] * drag0 + C[2] * drag1


def _body_torque(body_force):
    tx = MODEL.R * np.cross(np.array([1.0, 0.0, 0.0]), body_force)
    tz = MODEL.R * np.cross(np.array([0.0, 0.0, 1.0]), body_force)
    C = MODEL.C_body_torque
    return C[0] * tx + C[1] * tz


def _wing_output(v_ground, omega_body, E_ini2ground, E_ground2ini, E_body2ini, E_ini2body, wing):
    r, c, m22, lead, tail = wing_geometry()
    flap, elev, feath = (wing["bb_flap_vals"], wing["bb_elev_vals"], wing["bb_feath_vals"])
    dflap, delev, dfeath = (wing["bb_dflap_vals"], wing["bb_delev_vals"], wing["bb_dfeath_vals"])
    ddflap, ddelev, ddfeath = (
        wing["bb_ddflap_vals"],
        wing["bb_ddelev_vals"],
        wing["bb_ddfeath_vals"],
    )

    E_ini2flap, E_flap2elev, E_elev2feath = Rx(flap), Rz(elev), Ry(feath)
    E_ini2wing = Ry(feath) @ Rz(elev) @ Rx(flap)
    E_wing2ini = E_ini2wing.mT
    v_per = (E_ground2ini @ v_ground[..., None]).squeeze(-1)
    omega_ini = (E_body2ini @ omega_body[..., None]).squeeze(-1)
    pivw_ini = (E_body2ini @ MODEL.pivw_body[..., None]).squeeze(-1)

    E_ini2elev = E_ini2flap.mT @ E_flap2elev.mT
    E_ini2feath = E_ini2elev @ E_elev2feath.mT
    omega = (
        E_ini2flap.mT[..., 0] * dflap.reshape(-1, 1)
        + E_ini2elev[..., 2] * delev.reshape(-1, 1)
        + E_ini2feath[..., 1] * dfeath.reshape(-1, 1)
        + omega_ini
    )
    dRx_flap, dRz_elev, dRy_feath = dRx(flap, dflap), dRz(elev, delev), dRy(feath, dfeath)
    domega = (
        dRx_flap.mT[..., 0] * dflap.reshape(-1, 1)
        + E_ini2flap.mT[..., 0] * ddflap.reshape(-1, 1)
        + (dRx_flap.mT @ E_flap2elev.mT)[..., 2] * delev.reshape(-1, 1)
        + (E_ini2flap.mT @ dRz_elev.mT)[..., 2] * delev.reshape(-1, 1)
        + (E_ini2flap.mT @ E_flap2elev.mT)[..., 2] * ddelev.reshape(-1, 1)
        + (dRx_flap.mT @ E_flap2elev.mT @ E_elev2feath.mT)[..., 1] * dfeath.reshape(-1, 1)
        + (E_ini2flap.mT @ dRz_elev.mT @ E_elev2feath.mT)[..., 1] * dfeath.reshape(-1, 1)
        + (E_ini2flap.mT @ E_flap2elev.mT @ dRy_feath.mT)[..., 1] * dfeath.reshape(-1, 1)
        + (E_ini2flap.mT @ E_flap2elev.mT @ E_elev2feath.mT)[..., 1] * ddfeath.reshape(-1, 1)
    )
    domega_wing = np.einsum("nij, nj -> ni", E_ini2wing, domega)

    rot_axes = np.vstack((np.zeros_like(r), r, np.zeros_like(r)))
    vec_body = E_wing2ini @ rot_axes
    cross1 = np.cross(omega_ini[:, None, :], vec_body.mT)
    cross2 = np.cross(omega_ini, pivw_ini)[:, None, :]
    v_temp = np.einsum("ij, nkj -> nki", Ry(pi / 2).T, cross1 + v_per[:, None, :] + cross2)
    para = predict_coefficients((v_temp / 1005.0).reshape(-1, 3))
    para = para.reshape(*v_temp.shape[:2], -1)

    rot_axis_global = (E_wing2ini @ rot_axes).mT
    v_rot = (
        np.cross(omega.reshape(-1, 1, 3), rot_axis_global, axis=-1)
        + v_per[:, None]
        + np.cross(omega_ini, pivw_ini, axis=-1)[:, None]
    )
    a_rot = np.cross(domega[:, None], rot_axis_global, axis=-1) + np.cross(
        omega[:, None], v_rot, axis=-1
    )
    v_wing = np.einsum("nij, nmj -> nmi", E_ini2wing, v_rot).at[..., 1].set(0.0)
    a_wing = np.einsum("nij, nmj -> nmi", E_ini2wing, a_rot)
    norm_v = np.sqrt(np.sum(v_wing**2, axis=-1))
    alpha = np.where(
        norm_v <= MODEL.u_ref * 0.03,
        np.pi / 2,
        np.arccos(np.clip(-v_wing[..., 0] / norm_v, -1.0, 1.0)),
    )
    sign_CT = np.where(v_wing[..., 2] < 0.0, 1.0, -1.0)
    parts = _section_force_parts(
        c, v_wing, alpha, sign_CT, dfeath, m22, a_wing, tail, lead, domega_wing
    )
    dF_z = np.sum(para * parts, axis=-1)
    dM = np.stack((r.reshape(1, -1) * dF_z, np.zeros_like(dF_z), np.zeros_like(dF_z)), axis=-1)
    step = MODEL.R / (r.shape[0] - 1)

    # Force uses tip coefficients, whereas torque uses local coefficients.
    integrated_parts = np.sum(parts[:, 1:], axis=1) * step
    F_z = np.einsum("ni, ni -> n", para[:, -1], integrated_parts)
    M_quasi_wing = dM[:, 1:].sum(axis=1) * step
    E_wing2body = E_ini2body @ E_wing2ini
    F_wing = np.stack(np.broadcast_arrays(0.0, 0.0, F_z), axis=-1)
    F_body = np.einsum("nij, nj -> ni", E_wing2body, F_wing)
    return np.concatenate(
        (
            np.einsum("nij, nj -> ni", E_ini2ground @ E_wing2ini, F_wing),
            np.einsum("nij, nj -> ni", E_wing2body, M_quasi_wing)
            + np.cross(MODEL.pivw_body, F_body),
        ),
        axis=-1,
    )


@partial(jax.jit, static_argnames=("ntr",))
def _forces_and_moments(state13, control5, frequency, ntr, wind_velocity):
    wings = wing_kinematics(frequency, ntr, control5)
    ep_R = quat_normalize(state13[3:7])
    E_body2ini = Ry(pi / 2.0 - 45.0 * pi / 180.0)
    E_ini2body = E_body2ini.T
    E_body2ground_R = quat_to_matrix(ep_R)
    E_ground2body_R = E_body2ground_R.T
    E_ground2ini_R = E_body2ini @ E_ground2body_R
    v_ground = np.tile((state13[7:10] - wind_velocity) * 1000.0, (ntr, 1))
    omega_body = np.tile(state13[10:13], (ntr, 1))
    right = _wing_output(
        v_ground,
        omega_body,
        E_ground2ini_R.T,
        E_ground2ini_R,
        E_body2ini,
        E_ini2body,
        wings["right"],
    )

    ep_L = ep_R * np.array([1.0, -1.0, 1.0, -1.0])
    E_ground2ini_L = E_body2ini @ quat_to_matrix(ep_L).T
    left = _wing_output(
        v_ground * np.array([1.0, -1.0, 1.0]),
        omega_body * np.array([-1.0, 1.0, -1.0]),
        E_ground2ini_L.T,
        E_ground2ini_L,
        E_body2ini,
        E_ini2body,
        wings["left"],
    )
    body_F = _body_force(v_ground, E_ground2body_R)
    body_T = _body_torque(body_F)
    F = np.stack((body_F, right[:, :3], left[:, :3] * np.array([1.0, -1.0, 1.0])), axis=-1)
    T = np.stack((body_T, right[:, 3:], left[:, 3:] * np.array([-1.0, 1.0, -1.0])), axis=-1)
    return F.mean(axis=0).sum(axis=-1) * 1e-3, T.mean(axis=0).sum(axis=-1) * 1e-6


def forces_and_moments(state13, control5, frequency=123.0, ntr=20, wind_velocity=None):
    """Return (force_N, torque_Nm), each a JAX array of shape (3,).

    state13 = [x, y, z, qw, qx, qy, qz, vx, vy, vz, p, q, r], in SI.
    Velocity and wind are in ground axes; angular rates and torque are
    in body axes. Controls are [ai_phi0, a0_phi, a0_alpha, delta_R, delta_L]
    in degrees. Gravity is not included. ntr is a static positive integer.
    """
    state13, control5 = np.asarray(state13, dtype=float), np.asarray(control5, dtype=float)
    wind_velocity = np.zeros(3) if wind_velocity is None else np.asarray(wind_velocity, dtype=float)
    if state13.shape != (13,) or control5.shape != (5,) or wind_velocity.shape != (3,):
        raise ValueError("Expected state13 (13,), control5 (5,) and wind_velocity (3,).")
    if ntr < 1:
        raise ValueError("ntr must be positive.")
    return _forces_and_moments(state13, control5, frequency, ntr, wind_velocity)
