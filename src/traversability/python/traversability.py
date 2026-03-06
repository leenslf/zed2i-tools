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


def _compute_ray_cast_mask(valid_mask: np.ndarray, nontraversable: np.ndarray) -> np.ndarray:
    """Infer line-of-sight free-space bins along each theta column.

    Depth points only mark bins where returns landed, leaving NaNs between the
    sensor and the furthest return in a column. This helper applies occupancy-grid
    style ray casting: for each theta column, all radial bins closer than the ray
    cast limit are marked as observed free space.

    The ray cast limit is the closer of:
      - the first non-traversable bin in the column, OR
      - the furthest valid return (if no obstacle is present).

    Stopping at the first non-traversable bin is critical: without it, a
    beam that passes through a gap in an obstacle (e.g. a hole in a wall) would
    cause all intermediate bins, including those behind the obstacle, to be
    incorrectly marked as traversable free space. The obstacle itself blocks the
    ray, so nothing beyond it can be inferred as observed.
    """
    if valid_mask.ndim != 2:
        raise ValueError(f"`valid_mask` must be 2D, got shape {valid_mask.shape}")

    n_r, _ = valid_mask.shape

    # Furthest valid return per column — used as the limit when no obstacle is present.
    has_hit = valid_mask.any(axis=0)
    furthest_hit_idx = n_r - 1 - np.argmax(valid_mask[::-1], axis=0)
    furthest_hit_idx = np.where(has_hit, furthest_hit_idx, -1)

    # Closest non-traversable bin per column — this is where the ray is blocked.
    # nontraversable only flags bins with actual returns, so phantom occlusion
    # from NaN cells is not possible.
    has_obstacle = nontraversable.any(axis=0)
    closest_obstacle_idx = np.argmax(nontraversable, axis=0)

    # Use the obstacle as the limit where present; fall back to furthest valid return.
    ray_limit = np.where(has_obstacle, closest_obstacle_idx, furthest_hit_idx)

    radial_idx = np.arange(n_r, dtype=np.int32)[:, None]
    return radial_idx < ray_limit[None, :]


def compute_traversability(
    points: np.ndarray,
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute polar traversability outputs from [r, theta, z] points.

    Returns:
        trav_grid: float32 traversability map using sentinels:
            NaN unknown/unobserved, 0.0 traversable observed space, 1.0 non-traversable.
        r_edges: radial bin edges in meters.
        theta_edges: angular bin edges in radians.
        height_map: max-height polar map with missing bins filled by 0.0 for filtering.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"`points` must be 2D with at least 3 columns, got {points.shape}")

    polar_points = np.asarray(points[:, :3], dtype=np.float32)
    if polar_points.size == 0:
        empty_f32 = np.empty((0,), dtype=np.float32)
        return (
            np.empty((0, 0), dtype=np.float32),
            empty_f32,
            empty_f32,
            np.empty((0, 0), dtype=np.float32),
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

    # currently fixing the range, may not the be the best choice
    r_min = float(config.get("r_min_m", r.min()))
    r_max = float(config.get("r_max_m", r.max()))
    theta_min = float(np.deg2rad(config.get("theta_min_deg", float(np.degrees(theta.min())))))
    theta_max = float(np.deg2rad(config.get("theta_max_deg", float(np.degrees(theta.max())))))
    # r_min, r_max = float(r.min()), float(r.max())
    # theta_min, theta_max = float(theta.min()), float(theta.max())
    r_edges = np.arange(r_min, r_max + polar_grid_size_r, polar_grid_size_r, dtype=np.float32)
    theta_edges = np.arange(
        theta_min,
        theta_max + polar_grid_size_theta,
        polar_grid_size_theta,
        dtype=np.float32,
    )

    # skip if not enough bins to build a 2D map.
    if r_edges.size < 2 or theta_edges.size < 2:
        return (
            np.empty((0, 0), dtype=np.float32),
            r_edges.astype(np.float32, copy=False),
            theta_edges.astype(np.float32, copy=False),
            np.empty((0, 0), dtype=np.float32),
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

    danger_grid[~valid_mask] = np.nan
    nontraversable = (danger_grid > danger_threshold) & valid_mask
    ray_cast_mask = _compute_ray_cast_mask(valid_mask, nontraversable)
    observed_mask = valid_mask | ray_cast_mask
    trav_grid = np.full(valid_mask.shape, np.nan, dtype=np.float32)
    trav_grid[observed_mask & ~nontraversable] = 0.0
    trav_grid[valid_mask & (danger_grid > danger_threshold)] = 1.0

    return (
        trav_grid,
        r_edges.astype(np.float32, copy=False),
        theta_edges.astype(np.float32, copy=False),
        height_map.astype(np.float32, copy=False),
    )
