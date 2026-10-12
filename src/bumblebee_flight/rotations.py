"""Scalar-first quaternions and passive wing rotations; angles are radians."""

import jax.numpy as np


def quat_normalize(quat):
    """Normalize [..., w, x, y, z]; a zero quaternion becomes identity."""
    quat = np.asarray(quat, dtype=float)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return np.where(
        norm > 0.0, quat / np.where(norm > 0.0, norm, 1.0), np.array([1.0, 0.0, 0.0, 0.0])
    )


def quat_multiply(left, right):
    """Hamilton product of scalar-first quaternions, with broadcasting."""
    w1, x1, y1, z1 = np.moveaxis(np.asarray(left), -1, 0)
    w2, x2, y2, z2 = np.moveaxis(np.asarray(right), -1, 0)
    return np.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        axis=-1,
    )


def quat_to_euler(quat):
    """Return [..., roll, pitch, yaw] for the XYZ Euler convention."""
    w, x, y, z = np.moveaxis(quat_normalize(quat), -1, 0)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.stack((roll, pitch, yaw), axis=-1)


def euler_to_quat(roll, pitch, yaw):
    """Convert XYZ Euler angles to [..., w, x, y, z]."""
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    return np.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        axis=-1,
    )


def quat_to_matrix(quat):
    """Return the body-to-ground rotation for a unit [w, x, y, z]."""
    quat = np.asarray(quat, dtype=float)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    return np.stack(
        (
            1 - 2 * (yy + zz),
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            1 - 2 * (xx + zz),
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            1 - 2 * (xx + yy),
        ),
        axis=-1,
    ).reshape(*quat.shape[:-1], 3, 3)


def Rx(angle):
    """Passive rotation about X, supporting scalar and array angles."""
    angle = np.asarray(angle)
    s, c = np.sin(angle), np.cos(angle)
    return np.stack(
        np.broadcast_arrays(
            1.0,
            0.0,
            0.0,
            0.0,
            c,
            s,
            0.0,
            -s,
            c,
        ),
        axis=-1,
    ).reshape(*angle.shape, 3, 3)


def Ry(angle):
    """Passive rotation about Y, supporting scalar and array angles."""
    angle = np.asarray(angle)
    s, c = np.sin(angle), np.cos(angle)
    return np.stack(
        np.broadcast_arrays(
            c,
            0.0,
            -s,
            0.0,
            1.0,
            0.0,
            s,
            0.0,
            c,
        ),
        axis=-1,
    ).reshape(*angle.shape, 3, 3)


def Rz(angle):
    """Passive rotation about Z, supporting scalar and array angles."""
    angle = np.asarray(angle)
    s, c = np.sin(angle), np.cos(angle)
    return np.stack(
        np.broadcast_arrays(
            c,
            s,
            0.0,
            -s,
            c,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        axis=-1,
    ).reshape(*angle.shape, 3, 3)


def dRx(angle, dangle):
    """Time derivative of Rx(angle)."""
    angle, dangle = np.broadcast_arrays(angle, dangle)
    s, c = np.sin(angle), np.cos(angle)
    return np.stack(
        np.broadcast_arrays(
            0.0,
            0.0,
            0.0,
            0.0,
            -s * dangle,
            c * dangle,
            0.0,
            -c * dangle,
            -s * dangle,
        ),
        axis=-1,
    ).reshape(*angle.shape, 3, 3)


def dRy(angle, dangle):
    """Time derivative of Ry(angle)."""
    angle, dangle = np.broadcast_arrays(angle, dangle)
    s, c = np.sin(angle), np.cos(angle)
    return np.stack(
        np.broadcast_arrays(
            -s * dangle,
            0.0,
            -c * dangle,
            0.0,
            0.0,
            0.0,
            c * dangle,
            0.0,
            -s * dangle,
        ),
        axis=-1,
    ).reshape(*angle.shape, 3, 3)


def dRz(angle, dangle):
    """Time derivative of Rz(angle)."""
    angle, dangle = np.broadcast_arrays(angle, dangle)
    s, c = np.sin(angle), np.cos(angle)
    return np.stack(
        np.broadcast_arrays(
            -s * dangle,
            c * dangle,
            0.0,
            -c * dangle,
            -s * dangle,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        axis=-1,
    ).reshape(*angle.shape, 3, 3)
