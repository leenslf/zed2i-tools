"""Traversability stage for offline traversability pipeline."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from scipy.ndimage import generic_filter, maximum_filter, minimum_filter
from scipy.stats import binned_statistic_2d


def _estimate_danger_value(
    terrain: np.ndarray,
    polar_grid_size_r: float,
    polar_grid_size_theta: float,
    scrit_deg: float,
    rcrit_m: float,
    hcrit_m: float,
) -> np.ndarray:
    """Mirror the ROS traversability danger computation on a polar height map."""
    scrit = np.deg2rad(scrit_deg)

    # Slope: gradient computed in metric units (r in meters, theta in radians).
    dzdx, dzdy = np.gradient(terrain, polar_grid_size_r, polar_grid_size_theta)
    slope = np.arctan(np.sqrt(dzdx * dzdx + dzdy * dzdy))
    slope[slope > scrit] = np.inf

    # Roughness: std deviation over 3x3 neighborhood.
    roughness = generic_filter(terrain.astype(float), np.std, size=3, mode="reflect")
    roughness[roughness > rcrit_m] = np.inf

    # Step height: max absolute height difference to any neighbor in 5x5 window.
    max_z = maximum_filter(terrain, size=5, mode="reflect")
    min_z = minimum_filter(terrain, size=5, mode="reflect")
    step_height = np.maximum(np.abs(terrain - max_z), np.abs(terrain - min_z))

    # Count how many neighbors differ from center by more than hcrit.
    n_crit = 24  # 5x5 window minus center cell
    st_mask = np.zeros_like(terrain, dtype=int)
    padded = np.pad(terrain, 2, mode="reflect")
    for i in range(5):
        for j in range(5):
            if i == 2 and j == 2:
                continue
            shifted = padded[i : i + terrain.shape[0], j : j + terrain.shape[1]]
            st_mask += (np.abs(terrain - shifted) > hcrit_m).astype(int)

    step_height = np.minimum(step_height, step_height * st_mask / n_crit)
    step_height[step_height > hcrit_m] = np.inf

    danger_value = 0.3 * slope / scrit + 0.3 * roughness / rcrit_m + 0.4 * step_height / hcrit_m
    return danger_value


def compute_traversability(
    points: np.ndarray,
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute polar traversability danger grid from [r, theta, z] points."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"`points` must be 2D with at least 3 columns, got {points.shape}")

    polar_points = np.asarray(points[:, :3], dtype=np.float32)
    if polar_points.size == 0:
        empty_f32 = np.empty((0,), dtype=np.float32)
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0, 0), dtype=bool),
            np.empty((0, 0), dtype=bool),
            empty_f32,
            empty_f32,
        )

    danger_threshold = float(config.get("danger_threshold", 0.3))
    scrit_deg = float(config.get("scrit_deg", 30.0))
    rcrit_m = float(config.get("rcrit_m", 0.10))
    hcrit_m = float(config.get("hcrit_m", 0.20))
    polar_grid_size_r = float(config.get("polar_grid_size_r_m", 0.10))
    polar_grid_size_theta_deg = float(config.get("polar_grid_size_theta_deg", 1.0))
    polar_grid_size_theta = float(np.deg2rad(polar_grid_size_theta_deg))

    if polar_grid_size_r <= 0.0 or polar_grid_size_theta <= 0.0:
        raise ValueError("Polar grid sizes must be positive.")
    if rcrit_m <= 0.0 or hcrit_m <= 0.0:
        raise ValueError("`rcrit_m` and `hcrit_m` must be positive.")

    r = polar_points[:, 0]
    theta = polar_points[:, 1]
    z = polar_points[:, 2]

    r_min = float(config.get("r_min_m", r.min()))
    r_max = float(config.get("r_max_m", r.max()))
    theta_min = float(np.deg2rad(config.get("theta_min_deg", float(np.degrees(theta.min())))))
    theta_max = float(np.deg2rad(config.get("theta_max_deg", float(np.degrees(theta.max())))))
    r_edges = np.arange(r_min, r_max + polar_grid_size_r, polar_grid_size_r, dtype=np.float32)
    theta_edges = np.arange(
        theta_min,
        theta_max + polar_grid_size_theta,
        polar_grid_size_theta,
        dtype=np.float32,
    )

    # Match ROS callback behavior: skip if not enough bins to build a 2D map.
    if r_edges.size < 2 or theta_edges.size < 2:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0, 0), dtype=bool),
            np.empty((0, 0), dtype=bool),
            r_edges.astype(np.float32, copy=False),
            theta_edges.astype(np.float32, copy=False),
        )

    height_map, _, _, _ = binned_statistic_2d(
        r,
        theta,
        z,
        statistic="max",
        bins=[r_edges, theta_edges],
    )
    valid_mask = ~np.isnan(height_map)

    # Preserve original algorithm: fill missing cells with 0 before local filters.
    height_map = np.asarray(height_map, dtype=np.float32)
    height_map[~valid_mask] = 0.0

    danger_grid = _estimate_danger_value(
        terrain=height_map,
        polar_grid_size_r=polar_grid_size_r,
        polar_grid_size_theta=polar_grid_size_theta,
        scrit_deg=scrit_deg,
        rcrit_m=rcrit_m,
        hcrit_m=hcrit_m,
    ).astype(np.float32, copy=False)

    # Critical behavior from ROS node: invalidate danger where input bins were empty.
    danger_grid[~valid_mask] = np.nan
    nontraversable = (danger_grid > danger_threshold) & valid_mask

    return (
        danger_grid,
        valid_mask.astype(bool, copy=False),
        nontraversable.astype(bool, copy=False),
        r_edges.astype(np.float32, copy=False),
        theta_edges.astype(np.float32, copy=False),
    )
