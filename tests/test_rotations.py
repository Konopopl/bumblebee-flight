import numpy as np
import pytest

from bumblebee_flight.rotations import (
    euler_to_quat,
    quat_normalize,
    quat_to_euler,
    quat_to_matrix,
)


@pytest.mark.parametrize(
    "angles",
    [
        (0, 0, 0),
        (np.pi / 2, 0, 0),
        (0, np.pi / 4, 0),
        (0, 0, -np.pi / 2),
        (0.3, -0.8, 1.2),
        (-2.0, 1.3, -2.4),
    ],
)
def test_euler_quaternion_roundtrip_in_radians(angles):
    quaternion = np.asarray(euler_to_quat(*angles))
    assert quaternion.shape == (4,)
    np.testing.assert_allclose(np.linalg.norm(quaternion), 1, rtol=0, atol=1e-12)
    for sign in (1, -1):
        np.testing.assert_allclose(quat_to_euler(sign * quaternion), angles, rtol=0, atol=1e-12)
    matrix = np.asarray(quat_to_matrix(quaternion))
    np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(matrix), 1, rtol=0, atol=1e-12)


def test_quaternion_scalar_first_and_rotation_direction():
    quaternion = euler_to_quat(np.pi / 2, 0, 0)
    np.testing.assert_allclose(quaternion, [np.sqrt(0.5), np.sqrt(0.5), 0, 0], atol=1e-12)
    yaw = euler_to_quat(0, 0, np.pi / 2)
    np.testing.assert_allclose(quat_to_matrix(yaw) @ np.array([1, 0, 0]), [0, 1, 0], atol=1e-12)


def test_quaternion_normalization_preserves_orientation():
    quaternion = np.asarray(euler_to_quat(0.3, -0.8, 1.2))
    np.testing.assert_allclose(quat_normalize(7 * quaternion), quaternion, atol=1e-12)
