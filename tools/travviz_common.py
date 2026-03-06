from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.patches import Patch

FRAME_PATTERN = re.compile(r"^frame_(\d+)\.npz$")
DISPLAY_X_LIM = (-0.9, 0.9)
DISPLAY_Y_LIM = (0.0, 0.8)


def sorted_frame_paths(input_dir: Path) -> list[Path]:
    frames: list[tuple[int, Path]] = []
    for path in input_dir.glob("frame_*.npz"):
        match = FRAME_PATTERN.match(path.name)
        if match:
            frames.append((int(match.group(1)), path))
    frames.sort(key=lambda item: item[0])
    return [path for _, path in frames]


def frame_id_from_name(path: Path) -> int:
    match = FRAME_PATTERN.match(path.name)
    if not match:
        raise ValueError(f"Invalid frame filename: {path.name}")
    return int(match.group(1))


def pick_single_frame(input_path: Path, frame_id: int | None, frame_pos: int) -> Path:
    if input_path.is_file():
        return input_path

    frames = sorted_frame_paths(input_path)
    if not frames:
        raise FileNotFoundError(f"No frame_*.npz in {input_path}")

    if frame_id is not None:
        target = f"frame_{frame_id:05d}.npz"
        for frame in frames:
            if frame.name == target:
                return frame
        raise FileNotFoundError(f"Frame not found: {target}")

    if frame_pos < 0 or frame_pos >= len(frames):
        raise IndexError(f"frame-pos {frame_pos} out of range [0, {len(frames)-1}]")
    return frames[frame_pos]


def pick_frames_from_dir(
    input_dir: Path,
    frame_id: int | None,
    frame_pos: int | None,
    render_all: bool,
) -> list[Path]:
    frames = sorted_frame_paths(input_dir)
    if not frames:
        raise FileNotFoundError(f"No frame_*.npz files found in {input_dir}")

    if render_all or (frame_id is None and frame_pos is None):
        return frames

    if frame_id is not None:
        target = f"frame_{frame_id:05d}.npz"
        for frame in frames:
            if frame.name == target:
                return [frame]
        raise FileNotFoundError(f"Frame not found: {target}")

    assert frame_pos is not None
    if frame_pos < 0 or frame_pos >= len(frames):
        raise IndexError(f"frame-pos {frame_pos} out of range [0, {len(frames)-1}]")
    return [frames[frame_pos]]


def _pick_key(data: np.lib.npyio.NpzFile, key: str, source: Path) -> str:
    if key in data:
        return key
    prefixed = f"traversability_{key}"
    if prefixed in data:
        return prefixed
    raise KeyError(f"Missing `{key}` or `{prefixed}` in {source}")


def load_npz_frame(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        if "tilt_points" not in data:
            raise KeyError(
                f"{path} missing `tilt_points`. Enable tilt_compensate.write_output in config."
            )

        if "traversability_trav_grid" in data:
            trav_grid = np.asarray(data["traversability_trav_grid"], dtype=np.float32)
        elif "trav_grid" in data:
            trav_grid = np.asarray(data["trav_grid"], dtype=np.float32)
        else:
            # Backward compatibility for frames that predate trav_grid export.
            valid_mask = np.asarray(data[_pick_key(data, "valid_mask", path)], dtype=bool)
            nontraversable = np.asarray(data[_pick_key(data, "nontraversable", path)], dtype=bool)
            trav_grid = np.full(valid_mask.shape, np.nan, dtype=np.float32)
            trav_grid[valid_mask & ~nontraversable] = 0.0
            trav_grid[valid_mask & nontraversable] = 1.0

        r_edges = np.asarray(data[_pick_key(data, "r_edges", path)], dtype=np.float32)
        theta_edges = np.asarray(data[_pick_key(data, "theta_edges", path)], dtype=np.float32)
        if "height_map" in data or "traversability_height_map" in data:
            height_map = np.asarray(data[_pick_key(data, "height_map", path)], dtype=np.float32)
        else:
            # Older NPZ dumps may only contain danger_grid; use it as a plotting fallback.
            height_map = np.asarray(data[_pick_key(data, "danger_grid", path)], dtype=np.float32)
        tilt_points = np.asarray(data["tilt_points"], dtype=np.float32)

    if trav_grid.shape != (r_edges.size - 1, theta_edges.size - 1):
        raise ValueError(f"Grid/edge shape mismatch in {path}")
    if height_map.shape != trav_grid.shape:
        raise ValueError(f"Height map shape mismatch in {path}")

    return trav_grid, r_edges, theta_edges, height_map, tilt_points


def build_trav_grid(trav_grid: np.ndarray) -> np.ndarray:
    finite = np.isfinite(trav_grid)
    traversable = finite & np.isclose(trav_grid, 0.0)

    out = np.full(trav_grid.shape, np.nan, dtype=np.float32)
    out[traversable] = 0.0
    out[finite & ~traversable] = 1.0
    return out


def build_nontraversable_grid(trav_grid: np.ndarray) -> np.ndarray:
    """Build cartesian input with only non-traversable bins retained."""
    out = np.full(trav_grid.shape, np.nan, dtype=np.float32)
    out[np.isfinite(trav_grid) & np.isclose(trav_grid, 1.0)] = 1.0
    return out


def build_height_grid(height_map: np.ndarray, trav_grid: np.ndarray) -> np.ndarray:
    out = np.array(height_map, copy=True, dtype=np.float32)
    out[~np.isfinite(trav_grid)] = np.nan
    return out


def build_polar_status_overlay(trav_grid: np.ndarray) -> np.ndarray:
    finite = np.isfinite(trav_grid)
    traversable = finite & np.isclose(trav_grid, 0.0)

    overlay = np.full(trav_grid.shape, np.nan, dtype=np.float32)
    overlay[~finite] = 0.0
    overlay[traversable] = 1.0
    return overlay


def prepare_plot_layers(height_map: np.ndarray, trav_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    trav_classes = build_trav_grid(trav_grid)
    return build_height_grid(height_map, trav_classes), build_polar_status_overlay(trav_classes)


def polar_to_cartesian(
    polar_grid: np.ndarray,
    r_edges: np.ndarray,
    theta_edges: np.ndarray,
    cart_resolution: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_coords = np.arange(DISPLAY_X_LIM[0], DISPLAY_X_LIM[1], cart_resolution, dtype=np.float32)
    y_coords = np.arange(DISPLAY_Y_LIM[0], DISPLAY_Y_LIM[1], cart_resolution, dtype=np.float32)
    cart_grid = np.full((y_coords.size, x_coords.size), np.nan, dtype=np.float32)

    xx, yy = np.meshgrid(x_coords + 0.5 * cart_resolution, y_coords + 0.5 * cart_resolution)
    world_x = yy
    world_y = -xx

    rr = np.sqrt(world_x * world_x + world_y * world_y)
    tt = np.arctan2(world_y, world_x)

    ir = np.searchsorted(r_edges, rr, side="right") - 1
    it = np.searchsorted(theta_edges, tt, side="right") - 1

    nr, nt = polar_grid.shape
    inside = (ir >= 0) & (ir < nr) & (it >= 0) & (it < nt)
    ir_safe = ir.clip(0, nr - 1)
    it_safe = it.clip(0, nt - 1)
    valid = inside & np.isfinite(polar_grid[ir_safe, it_safe])
    cart_grid[valid] = polar_grid[ir[valid], it[valid]]
    return cart_grid, x_coords, y_coords


def rasterize_polar_to_cartesian(
    values: np.ndarray,
    source_valid_mask: np.ndarray,
    r_edges: np.ndarray,
    theta_edges: np.ndarray,
    x_min: float = 0.0,
    x_max: float = 1.5,
    y_min: float = -0.75,
    y_max: float = 0.75,
    grid_res_m: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_edges = np.arange(x_min, x_max + grid_res_m, grid_res_m, dtype=np.float32)
    y_edges = np.arange(y_min, y_max + grid_res_m, grid_res_m, dtype=np.float32)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="xy")
    rr = np.hypot(xx, yy)
    tt = np.arctan2(yy, xx)

    i_r = np.searchsorted(r_edges, rr, side="right") - 1
    i_t = np.searchsorted(theta_edges, tt, side="right") - 1

    in_bounds = (
        (i_r >= 0)
        & (i_r < (r_edges.size - 1))
        & (i_t >= 0)
        & (i_t < (theta_edges.size - 1))
    )
    cart = np.full(rr.shape, np.nan, dtype=np.float32)
    i_r_safe = np.clip(i_r, 0, r_edges.size - 2)
    i_t_safe = np.clip(i_t, 0, theta_edges.size - 2)
    src_ok = in_bounds & source_valid_mask[i_r_safe, i_t_safe]
    cart[src_ok] = values[i_r[src_ok], i_t[src_ok]]
    return cart, x_edges, y_edges


def draw_polar(
    ax: plt.Axes,
    polar_grid: np.ndarray,
    r_edges: np.ndarray,
    theta_edges: np.ndarray,
    title: str,
):
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#808080")
    mesh = ax.pcolormesh(
        np.degrees(theta_edges),
        r_edges,
        polar_grid,
        cmap=cmap,
        shading="auto",
        edgecolors="#202020",
        linewidth=0.2,
    )
    ax.set_xlabel("Theta (deg)")
    ax.set_ylabel("Range (m)")
    ax.set_title(title)
    return mesh


def draw_polar_with_trav_overlay(
    ax: plt.Axes,
    height_grid: np.ndarray,
    status_overlay: np.ndarray,
    r_edges: np.ndarray,
    theta_edges: np.ndarray,
    title: str,
):
    base_mesh = draw_polar(ax, height_grid, r_edges, theta_edges, title=title)

    status_cmap = ListedColormap(["#9e9e9e", "#517E51"])
    status_cmap.set_bad((0, 0, 0, 0))
    status_norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5], ncolors=status_cmap.N)
    ax.pcolormesh(
        np.degrees(theta_edges),
        r_edges,
        status_overlay,
        cmap=status_cmap,
        norm=status_norm,
        shading="auto",
        edgecolors="none",
        alpha=0.95,
    )


    return base_mesh


def draw_cartesian_overlay(
    ax: plt.Axes,
    cart_grid: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    tilt_points: np.ndarray,
    max_points: int,
    show_pc_overlay: bool,
    title: str,
):
    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad("#808080")
    cmap.set_over("black")
    norm = Normalize(vmin=0.0, vmax=1.0, clip=False)

    mesh = ax.imshow(
        cart_grid,
        origin="lower",
        extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]],
        interpolation="nearest",
        aspect="equal",
        cmap=cmap,
        norm=norm,
    )
    ax.set_xlim(*DISPLAY_X_LIM)
    ax.set_ylim(*DISPLAY_Y_LIM)

    scatter = None
    if show_pc_overlay and tilt_points.size:
        x_all = -tilt_points[:, 1]
        y_all = tilt_points[:, 0]
        z_all = tilt_points[:, 2]
        in_view = (
            np.isfinite(x_all)
            & np.isfinite(y_all)
            & np.isfinite(z_all)
            & (x_all >= DISPLAY_X_LIM[0])
            & (x_all <= DISPLAY_X_LIM[1])
            & (y_all >= DISPLAY_Y_LIM[0])
            & (y_all <= DISPLAY_Y_LIM[1])
        )
        x = x_all[in_view]
        y = y_all[in_view]
        z = z_all[in_view]
        if max_points > 0 and x.size > max_points:
            idx = np.random.default_rng(0).choice(x.size, size=max_points, replace=False)
            x, y, z = x[idx], y[idx], z[idx]

        scatter = ax.scatter(
            x,
            y,
            s=2.0,
            c=z,
            cmap="viridis",
            alpha=0.05,
            linewidths=0,
            label="tilt_points",
            zorder=5,
        )
        ax.legend(loc="upper right")

    ax.set_xlabel("Y (m)")
    ax.set_ylabel("X (m)")
    ax.set_title(title)
    return mesh, scatter
