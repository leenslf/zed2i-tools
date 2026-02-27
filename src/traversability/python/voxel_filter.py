"""Voxel filtering stage for offline traversability pipeline."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def voxel_filter(points: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """Downsample XYZ points by voxel occupancy and center snapping.

    Args:
        points: (N, C) float array where C >= 3, first three columns are [x, y, z].
        config: Voxel filter config mapping with:
            - voxel_size_x, voxel_size_y, voxel_size_z
            - min_points_per_voxel

    Returns:
        (M, 3) float32 array of voxel-center XYZ points.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"`points` must be 2D with at least 3 columns, got {points.shape}")

    # Drop non-geometric channels immediately and operate on XYZ only.
    xyz = np.asarray(points[:, :3], dtype=np.float32)
    if xyz.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    vx = float(config.get("voxel_size_x", 0.05))
    vy = float(config.get("voxel_size_y", 0.05))
    vz = float(config.get("voxel_size_z", 0.05))
    min_points = int(config.get("min_points_per_voxel", 8))

    if vx <= 0.0 or vy <= 0.0 or vz <= 0.0:
        raise ValueError("Voxel sizes must be positive.")
    if min_points < 1:
        raise ValueError("`min_points_per_voxel` must be >= 1.")

    voxel_size = np.asarray([vx, vy, vz], dtype=np.float32)

    # Match floor-based C++ voxel indexing for both positive and negative coords.
    voxel_indices = np.floor(xyz / voxel_size).astype(np.int64)

    _unique_voxels, inverse, counts = np.unique(
        voxel_indices, axis=0, return_inverse=True, return_counts=True
    )
    keep_voxel_mask = counts >= min_points
    keep_point_mask = keep_voxel_mask[inverse]

    if not np.any(keep_point_mask):
        return np.empty((0, 3), dtype=np.float32)

    kept_voxel_indices = voxel_indices[keep_point_mask]
    centered_xyz = (kept_voxel_indices.astype(np.float32) + 0.5) * voxel_size
    return centered_xyz.astype(np.float32, copy=False)
