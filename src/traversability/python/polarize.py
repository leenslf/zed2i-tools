"""Cartesian-to-polar stage for offline traversability pipeline."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def polarize(points: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """Convert XYZ points to [r, theta, z] with z/r filtering.

    Args:
        points: (N, 3) float array [x, y, z].
        config: Polarization config mapping with:
            - z_threshold
            - min_range

    Returns:
        (M, 3) float32 array [r, theta, z].
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"`points` must be 2D with at least 3 columns, got {points.shape}")

    xyz = np.asarray(points[:, :3], dtype=np.float32)
    if xyz.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    z_threshold = float(config.get("z_threshold", 1.5))
    min_range = float(config.get("min_range", 0.1))

    if z_threshold < 0.0:
        raise ValueError("`z_threshold` must be >= 0.")
    if min_range < 0.0:
        raise ValueError("`min_range` must be >= 0.")

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    z_mask = np.abs(z) < z_threshold
    x = x[z_mask]
    y = y[z_mask]
    z = z[z_mask]

    if x.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    r = np.sqrt(x * x + y * y).astype(np.float32, copy=False)
    theta = np.arctan2(y, x)

    range_mask = r > min_range
    if not np.any(range_mask):
        return np.empty((0, 3), dtype=np.float32)

    polarized = np.column_stack((r[range_mask], theta[range_mask], z[range_mask]))
    return polarized.astype(np.float32, copy=False)
