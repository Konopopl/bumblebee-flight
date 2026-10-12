"""Aerodynamic coefficients and geometry; internal lengths are in millimetres."""

from math import pi

import jax.numpy as np


class Model:
    """Parameters used by the force model; mass, weight and inertia are in SI."""

    def __init__(self):
        self.Ysc = np.array(
            [
                [2.9452, 2.0113, 0, 3.2026, 0.3790, 0.6259, 1.1273],
                [0.7766, 0.3437, 1, 0.7463, 0.4047, 0.7554, 2.6112],
            ]
        )
        self.Ssc = np.array([[0, 0, 1.8462], [4, 3.2660, 2.0755]])
        self.S = np.array(
            [
                [0, 0, -0.8895],
                [-1, 0, -0.8895],
                [-1, 0, 1.0377],
                [-1, 1.2247, -0.8895],
                [-1, -1.2247, -0.8895],
                [-1, 1.2247, 1.0377],
                [-1, -1.2247, 1.0377],
                [1, 0, -0.8895],
                [1, 0, 1.0377],
                [1, 1.2247, -0.8895],
                [1, -1.2247, -0.8895],
                [1, 1.2247, 1.0377],
                [1, -1.2247, 1.0377],
            ]
        )
        self.gamma = np.array(
            [
                [
                    1.2433,
                    -0.0290,
                    0.0411,
                    -0.3482,
                    -0.5574,
                    0.1821,
                    0.0897,
                    0.9266,
                    0.4834,
                    -0.5490,
                    -0.6863,
                    -0.6176,
                    -0.1788,
                ],
                [
                    -2.2642,
                    0.2462,
                    -0.0395,
                    0.6751,
                    0.8419,
                    0.0512,
                    -0.6428,
                    0.4084,
                    0.7411,
                    0.1396,
                    -0.0469,
                    -0.4119,
                    0.3018,
                ],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [
                    1.2980,
                    0.8558,
                    0.9351,
                    -0.7315,
                    -1.7829,
                    0.1070,
                    -0.0326,
                    0.6189,
                    -0.5472,
                    -0.3309,
                    0.0725,
                    -0.6250,
                    0.1627,
                ],
                [
                    -0.5325,
                    0.8951,
                    0.1372,
                    0.2615,
                    -0.3826,
                    -0.6672,
                    0.0223,
                    0.5909,
                    0.6306,
                    -0.2026,
                    -0.6298,
                    -0.2524,
                    0.1295,
                ],
                [
                    0.0080,
                    0.1769,
                    0.4958,
                    0.0706,
                    -0.1345,
                    -0.6793,
                    0.0666,
                    -0.1708,
                    0.2055,
                    0.8480,
                    -0.7981,
                    -0.5969,
                    0.5084,
                ],
                [
                    -1.6712,
                    0.6275,
                    0.3906,
                    -0.1479,
                    1.3795,
                    -0.5682,
                    -0.8459,
                    0.4838,
                    0.8658,
                    0.3468,
                    -1.0185,
                    0.0210,
                    0.1367,
                ],
            ]
        )
        self.beta = np.array(
            [
                [-5.3656e-10, 6.1418e-10, 0, -6.4542e-10, -6.6781e-11, -3.4955e-12, 3.5370e-10],
                [0.2674, -0.3525, 0, -0.0415, 0.0355, -0.2765, -0.5185],
                [0.4786, -0.4778, 0, -0.5290, 0.8549, -0.5871, -0.0899],
                [-0.5841, -0.0138, 0, 0.1800, -0.1244, -0.5873, 0.1335],
            ]
        )
        self.theta = 20
        self.C_body_force = np.array([0.0701, 0.0465, 0.1604])
        self.C_body_torque = np.array([-0.3118, -0.0081])
        self.g = 9800.0
        self.rou_f = 1.225
        self.r_1 = 0.33
        self.R = 15.20
        self.r_2 = self.R + self.r_1
        self.u_ref = 2 * (139.36 * pi / 180.0) * self.R * 135.9619
        self.c_mean = 4.1
        self.pivw_body = self.c_mean * np.array([-1.6641, 0.6414, 0.6735])
        self.mbody = 418e-6
        self.Jbody = self.mbody * (8.4761 / 64.65) * (self.R / 1000.0) ** 2
        self.wbody = self.mbody * (self.g / 1000.0)
        self.coeff_lead = np.array(
            [
                -3.81438909765279e-08,
                2.96994268846037e-06,
                -9.96190443137349e-05,
                1.88340033260014e-03,
                -2.20521668689716e-02,
                1.65517729061739e-01,
                -7.97813087520728e-01,
                2.40871328870580e00,
                -4.28073966461681e00,
                3.97904181072880e00,
                -6.86589535684637e-01,
            ]
        )
        self.coeff_tail = np.array(
            [
                8.02276679956668e-08,
                -6.33801124035311e-06,
                2.13809320302790e-04,
                -4.01551403387463e-03,
                4.58746414489226e-02,
                -3.26798181618711e-01,
                1.42691164697543e00,
                -3.57730252903570e00,
                4.48766806503114e00,
                -2.87136696460492e00,
                -1.16604897405664e-01,
            ]
        )


MODEL = Model()


def predict_coefficients(x):
    """Return seven DACE coefficients per normalized-velocity input row."""
    x = np.atleast_2d(x)
    x_norm = (x - MODEL.Ssc[0]) / MODEL.Ssc[1]
    regression = np.hstack((np.ones((x.shape[0], 1)), x_norm))
    dx = (x_norm[:, None, :] - MODEL.S[None, :, :]).reshape(-1, 3)
    correlation = np.exp(-np.sum(MODEL.theta * dx**2, axis=1))
    correlation = correlation.reshape(x.shape[0], MODEL.S.shape[0])
    prediction = regression @ MODEL.beta + (MODEL.gamma @ correlation.T).T
    return MODEL.Ysc[0] + MODEL.Ysc[1] * prediction


def wing_geometry(num_space=100):
    """Return span, chord, added mass, leading edge and trailing edge."""
    r = np.linspace(MODEL.r_1, MODEL.R + MODEL.r_1, num_space)
    x = np.linspace(MODEL.r_1, 15.2 + MODEL.r_1, num_space)
    lead = np.polyval(MODEL.coeff_lead, x)
    tail = np.polyval(MODEL.coeff_tail, x)
    chord = (lead - tail) / 4.1 * MODEL.c_mean
    m22 = np.pi * MODEL.rou_f * chord * chord / 4.0
    return r, chord, m22, lead, tail
