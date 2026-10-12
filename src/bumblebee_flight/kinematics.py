"""Five-control wing kinematics; controls are in degrees."""

from functools import partial
from math import pi

import jax
import jax.numpy as np


def _fourier(a, b, f, t):
    w = 2 * np.pi * f
    i = np.arange(len(a))[:, None]
    cos_phase, sin_phase = np.cos(i * w * t), np.sin(i * w * t)
    values = a[:, None] * cos_phase + b[:, None] * sin_phase
    first = w * np.sum(i * (-a[:, None] * sin_phase + b[:, None] * cos_phase), axis=0)
    second = -np.sum((i * w) ** 2 * values, axis=0)
    return np.sum(values, axis=0), first, second


def _single_wing(f, ntr, a0_phi_deg, ai_phi0_deg, a0_alpha_deg):
    t = np.arange(1, ntr + 1) / float(ntr) / f
    a0_phi = np.array([a0_phi_deg])
    ai_phi = np.array([ai_phi0_deg, 2.9459409402, 0.7147237016])
    bi_phi = np.array([-3.6338653612, -0.0528668264, -0.6925250832])
    a0_theta = np.array([-24.7305876269])
    ai_theta = np.array([-0.1043130156, -2.1636568640, 0.9756511338])
    bi_theta = np.array([-0.4552237989, 0.2672709143, 0.9443299389])
    a0_alpha = np.array([a0_alpha_deg])
    ai_alpha = np.array([5.3866912157, 0.3691997080, 0.3956823556, 0.2085039852])
    bi_alpha = np.array([68.8546370037, 3.2757439902, 8.7337985159, 0.2535964724])

    a_flap = np.concatenate((a0_phi / 2.0, ai_phi)) * (pi / 180.0)
    b_flap = np.concatenate((np.array([0.0]), bi_phi)) * (pi / 180.0)
    a_elev = -np.concatenate((a0_theta / 2.0, ai_theta)) * (pi / 180.0)
    b_elev = -np.concatenate((np.array([0.0]), bi_theta)) * (pi / 180.0)
    a_feath = -np.concatenate((a0_alpha / 2.0, ai_alpha)) * (pi / 180.0)
    b_feath = -np.concatenate((np.array([0.0]), bi_alpha)) * (pi / 180.0)

    flap, dflap, ddflap = _fourier(a_flap, b_flap, f, t)
    elev, delev, ddelev = _fourier(a_elev, b_elev, f, t)
    feath, dfeath, ddfeath = _fourier(a_feath, b_feath, f, t)
    return {
        "bb_flap_vals": flap,
        "bb_elev_vals": elev,
        "bb_feath_vals": feath,
        "bb_dflap_vals": dflap,
        "bb_delev_vals": delev,
        "bb_dfeath_vals": dfeath,
        "bb_ddflap_vals": ddflap,
        "bb_ddelev_vals": ddelev,
        "bb_ddfeath_vals": ddfeath,
    }


@partial(jax.jit, static_argnames=("ntr",))
def wing_kinematics(f, ntr, controls):
    """Return right/left angle arrays and derivatives for one wingbeat.

    controls = [ai_phi0, a0_phi, a0_alpha, d_ai_phi0_right, d_ai_phi0_left].
    Samples are 1/ntr through 1 of the period, as in the force model.
    """
    ai_phi0, a0_phi, a0_alpha, delta_right, delta_left = np.asarray(controls)
    right_amplitude = np.clip(ai_phi0 + delta_right, 0.0, 88.0)
    left_amplitude = np.clip(ai_phi0 + delta_left, 0.0, 88.0)
    return {
        "right": _single_wing(f, ntr, a0_phi, right_amplitude, a0_alpha),
        "left": _single_wing(f, ntr, a0_phi, left_amplitude, a0_alpha),
    }
