"""Tilt compensation stage for offline traversability pipeline."""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation


def _yaw_from_quaternion_xyzw(quaternion: np.ndarray) -> float:
    """Extract yaw (rotation about Z) from quaternion [x, y, z, w]."""
    x, y, z, w = quaternion
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def tilt_compensate(points: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """Remove yaw from orientation and apply pitch/roll rotation to cloud XYZ.

    Args:
        points: (N, C) array where C >= 3; first three columns are [x, y, z].
        quaternion: (4,) odom orientation [qx, qy, qz, qw].

    Returns:
        Corrected (N, C) array with rotated XYZ and untouched remaining columns.
    """
    points = np.asarray(points)
    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(-1)

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"`points` must be 2D with at least 3 columns, got {points.shape}")
    if quaternion.shape != (4,):
        raise ValueError(f"`quaternion` must have shape (4,), got {quaternion.shape}")

    norm = np.linalg.norm(quaternion)
    if norm == 0.0:
        raise ValueError("`quaternion` must be non-zero.")
    q_world_cam = quaternion / norm

    # Mirror C++ logic:
    # yaw = getYaw(q_world_cam)
    # q_yaw = Rz(yaw)
    # q_pr  = q_world_cam * q_yaw^-1   (pitch+roll only)
    yaw = _yaw_from_quaternion_xyzw(q_world_cam)
    q_pr = Rotation.from_quat(q_world_cam) * Rotation.from_euler("z", yaw).inv()

    corrected = np.array(points, copy=True)
    corrected[:, :3] = q_pr.apply(points[:, :3])
    return corrected
