"""Traversability stage for offline traversability pipeline."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from scipy.ndimage import generic_filter
from scipy.stats import binned_statistic_2d


def _estimate_danger_value(
    terrain: np.ndarray,
    r_min: float,
    theta_min: float,
    polar_grid_size_r: float,
    polar_grid_size_theta: float,
    scrit_deg: float,
    rcrit_m: float,
    hcrit_m: float,
) -> np.ndarray:
    """Compute a per-cell danger score from a polar height map.

    Three hazard signals are computed and combined into a weighted danger value:

      danger = 0.3 * (slope / scrit) + 0.3 * (roughness / rcrit) + 0.4 * (step / hcrit)

    Any signal that exceeds its threshold is clipped to inf, guaranteeing the
    cell is classified non-traversable regardless of the other terms.

    Slope
        Finite differences of the height map in the radial (∂z/∂r) and angular
        (∂z/∂θ) directions. The angular gradient is divided by the radial bin
        centres to convert from z/rad to z/m — without this, slope would be
        overestimated by a factor of r in the tangential direction, growing
        unboundedly with range.

    Roughness
        Standard deviation of heights in a 3×3 neighborhood. Measures local
        surface irregularity independent of overall tilt.

    Step height
        For each cell, every neighbor in a 5×5 window is tested. A neighbor
        qualifies as a step only if the height difference exceeds hcrit AND
        the pairwise slope (arctan2(dz, dxy)) exceeds scrit. The second
        condition prevents gradual ramps from being mis-classified as steps.
        Distances (dxy) are computed in true Cartesian space from the polar
        bin centres, so cells at different radii are treated fairly. The final
        step score scales the worst-case qualifying height by the fraction of
        neighbors that qualified, penalising confirmed ledges while suppressing
        isolated noise.
    """
    scrit = np.deg2rad(scrit_deg)

    # Slope: finite differences in the radial (r) and angular (θ) directions.
    # dzdx → ∂z/∂r [z/m]:   rate of height change per meter radially.
    # dzdy → ∂z/∂θ [z/rad]: rate of height change per radian angularly.
    dzdx, dzdy = np.gradient(terrain, polar_grid_size_r, polar_grid_size_theta)

    # The true metric tangential slope is ∂z/∂(arc) = (∂z/∂θ) / r, because arc
    # length = r·dθ. Dividing dzdy by the radial bin centres converts it from
    # z/rad to z/m, fixing the overestimation that otherwise grows with range.
    r_centres = r_min + (np.arange(terrain.shape[0]) + 0.5) * polar_grid_size_r
    dzdy_metric = dzdy / r_centres[:, None]

    slope = np.arctan(np.sqrt(dzdx * dzdx + dzdy_metric * dzdy_metric))
    slope[slope > scrit] = np.inf

    # Roughness: std deviation over 3x3 neighborhood.
    roughness = generic_filter(terrain.astype(float), np.std, size=3, mode="reflect")
    roughness[roughness > rcrit_m] = np.inf

    # Step height: Distances are computed in true Cartesian metric space (not grid indices), so cells at different radii are treated fairly.
    nr, nc = terrain.shape
    t_centres = theta_min + (np.arange(nc) + 0.5) * polar_grid_size_theta
    # r_centres already computed above for the slope fix.
    R, T = np.meshgrid(r_centres, t_centres, indexing="ij")
    Xgrid = R * np.cos(T)
    Ygrid = R * np.sin(T)

    half = 2  # 5x5 window
    n_crit = (2 * half + 1) ** 2 - 1  # 24 neighbors
    st_mask  = np.zeros((nr, nc), dtype=int)
    h_max    = np.zeros((nr, nc), dtype=float)

    padded_z = np.pad(terrain, half, mode="reflect")
    padded_x = np.pad(Xgrid,   half, mode="reflect")
    padded_y = np.pad(Ygrid,   half, mode="reflect")

    for i in range(2 * half + 1):
        for j in range(2 * half + 1):
            if i == half and j == half:
                continue
            shifted_z = padded_z[i : i + nr, j : j + nc]
            shifted_x = padded_x[i : i + nr, j : j + nc]
            shifted_y = padded_y[i : i + nr, j : j + nc]

            dz  = np.abs(terrain - shifted_z)
            # True horizontal distance between the cell centres in Cartesian space.
            dxy = np.hypot(Xgrid - shifted_x, Ygrid - shifted_y)

            # Pairwise slope from centre cell to each neighbor.
            pair_slope = np.arctan2(dz, np.where(dxy == 0, np.nan, dxy))

            qualifies = (dz > hcrit_m) & (pair_slope > scrit)
            st_mask  += qualifies.astype(int)
            h_max     = np.maximum(h_max, np.where(qualifies, dz, 0.0))

    # Weigh h_max by what fraction of the 24 neighbors confirmed the step.
    # step_height = h_max * (st_mask / n_crit), capped at h_max.
    # e.g. 1/24 qualifying → step_height ≈ h_max/24 (noise suppressed)
    #      24/24 qualifying → step_height = h_max   (confirmed ledge)
    step_height = np.minimum(h_max, h_max * st_mask / n_crit)
    step_height[step_height > hcrit_m] = np.inf

    danger_value = 0.3 * slope / scrit + 0.3 * roughness / rcrit_m + 0.4 * step_height / hcrit_m
    return danger_value


def _compute_ray_cast_mask(nontraversable: np.ndarray) -> np.ndarray:
    """Return a boolean mask of cells inferred as free space via ray casting.
    For each angular column, finds the closest non-traversable cell radially.
    All cells between the sensor origin and that obstacle are marked True.
    """
    n_r, _ = nontraversable.shape

    # Closest non-traversable bin per column 
    has_obstacle = nontraversable.any(axis=0)
    closest_obstacle_idx = np.argmax(nontraversable, axis=0)

    # Use the obstacle as the limit where present; fall back to furthest valid return.
    ray_limit = np.where(has_obstacle, closest_obstacle_idx, -1)

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
    polar_grid_size_theta_deg = float(config.get("polar_grid_size_theta_deg", 5.0))
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

  
    height_map = np.asarray(height_map, dtype=np.float32)
    height_map[~valid_mask] = -0.3  # fill missing bins with a low value, 
    # the value is chosen to represent terrain somewhat below the camera plane
    danger_grid = _estimate_danger_value(
        terrain=height_map,
        r_min=r_min,
        theta_min=theta_min,
        polar_grid_size_r=polar_grid_size_r,
        polar_grid_size_theta=polar_grid_size_theta,
        scrit_deg=scrit_deg,
        rcrit_m=rcrit_m,
        hcrit_m=hcrit_m,
    ).astype(np.float32, copy=False)
    # height_map[~valid_mask] = np.nan
    danger_grid[~valid_mask] = np.nan

    nontraversable = (danger_grid > danger_threshold) & valid_mask
    observed_mask = _compute_ray_cast_mask(nontraversable)
    trav_grid = np.full(valid_mask.shape, np.nan, dtype=np.float32)
    trav_grid[observed_mask & ~nontraversable] = 0.0
    trav_grid[valid_mask & (danger_grid > danger_threshold)] = 1.0

    return (
        trav_grid,
        r_edges.astype(np.float32, copy=False),
        theta_edges.astype(np.float32, copy=False),
        height_map.astype(np.float32, copy=False),
    )
