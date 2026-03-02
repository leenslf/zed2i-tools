#!/usr/bin/env python3
"""Remap traversability polar grid from NPZ to Cartesian for visualization."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


FRAME_PATTERN = re.compile(r"^frame_(\d+)\.npz$")
DISPLAY_X_LIM = (-0.9, 0.9)
DISPLAY_Y_LIM = (0.0, 0.8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polar -> Cartesian remap from traversability NPZ.")
    parser.add_argument("input_path", type=Path, help="NPZ file or folder containing frame_*.npz.")
    parser.add_argument("--frame-id", type=int, default=None, help="Frame id to pick from folder.")
    parser.add_argument("--frame-pos", type=int, default=0, help="Frame position when --frame-id unset.")
    parser.add_argument("--cart-resolution", type=float, default=0.05, help="Cartesian cell size in meters.")
    parser.add_argument(
        "--max-points",
        type=int,
        default=120000,
        help="Max tilt_points to draw in overlay (default: 120000).",
    )
    parser.add_argument(
        "--no-pc-overlay",
        action="store_true",
        help="Disable tilt point-cloud overlay in the Cartesian panel.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path.")
    parser.add_argument("--show", action="store_true", help="Show interactive window.")
    return parser.parse_args()


def sorted_frame_paths(input_dir: Path) -> list[Path]:
    frames: list[tuple[int, Path]] = []
    for path in input_dir.glob("frame_*.npz"):
        match = FRAME_PATTERN.match(path.name)
        if match:
            frames.append((int(match.group(1)), path))
    frames.sort(key=lambda item: item[0])
    return [path for _, path in frames]


def pick_frame(input_path: Path, frame_id: int | None, frame_pos: int) -> Path:
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


def _pick_key(data: np.lib.npyio.NpzFile, key: str) -> str:
    if key in data:
        return key
    prefixed = f"traversability_{key}"
    if prefixed in data:
        return prefixed
    raise KeyError(f"Missing `{key}` or `{prefixed}` in {data.filename}")


def load_npz_frame(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        if "tilt_points" not in data:
            raise KeyError(
                f"{path} missing `tilt_points`. Enable tilt_compensate.write_output in config."
            )
        danger_grid = np.asarray(data[_pick_key(data, "danger_grid")], dtype=np.float32)
        valid_mask = np.asarray(data[_pick_key(data, "valid_mask")], dtype=bool)
        nontraversable = np.asarray(data[_pick_key(data, "nontraversable")], dtype=bool)
        r_edges = np.asarray(data[_pick_key(data, "r_edges")], dtype=np.float32)
        theta_edges = np.asarray(data[_pick_key(data, "theta_edges")], dtype=np.float32)
        tilt_points = np.asarray(data["tilt_points"], dtype=np.float32)
    if danger_grid.shape != valid_mask.shape or danger_grid.shape != nontraversable.shape:
        raise ValueError(f"Inconsistent traversability grid shapes in {path}")
    if danger_grid.shape != (r_edges.size - 1, theta_edges.size - 1):
        raise ValueError(f"Grid/edge shape mismatch in {path}")
    return danger_grid, valid_mask, nontraversable, r_edges, theta_edges, tilt_points


def build_polar_grid(
    danger_grid: np.ndarray,
    valid_mask: np.ndarray,
    nontraversable: np.ndarray,
) -> np.ndarray:
    """Build display-ready polar grid directly from NPZ traversability output."""
    polar_grid = np.array(danger_grid, copy=True, dtype=np.float32)
    polar_grid[~valid_mask] = np.nan
    polar_grid[valid_mask & nontraversable] = 1.1
    return polar_grid


def polar_to_cartesian(
    polar_grid: np.ndarray,
    valid_mask: np.ndarray,
    r_edges: np.ndarray,
    theta_edges: np.ndarray,
    cart_resolution: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_coords = np.arange(DISPLAY_X_LIM[0], DISPLAY_X_LIM[1], cart_resolution, dtype=np.float32)
    y_coords = np.arange(DISPLAY_Y_LIM[0], DISPLAY_Y_LIM[1], cart_resolution, dtype=np.float32)
    cart_grid = np.full((y_coords.size, x_coords.size), np.nan, dtype=np.float32)

    xx, yy = np.meshgrid(x_coords + 0.5 * cart_resolution, y_coords + 0.5 * cart_resolution)

    # World frame view rotated 90 deg CCW in display space: X'=-y, Y'=x.
    world_x = yy
    world_y = -xx

    rr = np.sqrt(world_x * world_x + world_y * world_y)
    tt = np.arctan2(world_y, world_x)

    ir = np.searchsorted(r_edges, rr, side="right") - 1
    it = np.searchsorted(theta_edges, tt, side="right") - 1

    nr, nt = polar_grid.shape
    inside = (ir >= 0) & (ir < nr) & (it >= 0) & (it < nt)
    valid = inside & valid_mask[ir.clip(0, nr - 1), it.clip(0, nt - 1)]
    cart_grid[valid] = polar_grid[ir[valid], it[valid]]
    return cart_grid, x_coords, y_coords


def draw_polar(
    ax: plt.Axes,
    polar_grid: np.ndarray,
    r_edges: np.ndarray,
    theta_edges: np.ndarray,
    title: str = "Polar Grid",
):
    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_over("black")
    norm = Normalize(vmin=0.0, vmax=1.0, clip=False)
    theta_edges_deg = np.degrees(theta_edges)
    mesh = ax.pcolormesh(theta_edges_deg, r_edges, polar_grid, cmap=cmap, norm=norm, shading="auto")
    ax.set_xlabel("Theta (deg)")
    ax.set_ylabel("Range (m)")
    ax.set_title(title)
    return mesh


def draw_cartesian_overlay(
    ax: plt.Axes,
    cart_grid: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    tilt_points: np.ndarray,
    max_points: int,
    show_pc_overlay: bool,
    title: str | None = None,
):
    cmap = plt.get_cmap("RdYlGn_r").copy()
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

    pc_scatter = None
    if show_pc_overlay and tilt_points.size:
        pc_x_all = -tilt_points[:, 1]
        pc_y_all = tilt_points[:, 0]
        pc_z_all = tilt_points[:, 2]
        in_view = (
            np.isfinite(pc_x_all)
            & np.isfinite(pc_y_all)
            & np.isfinite(pc_z_all)
            & (pc_x_all >= DISPLAY_X_LIM[0])
            & (pc_x_all <= DISPLAY_X_LIM[1])
            & (pc_y_all >= DISPLAY_Y_LIM[0])
            & (pc_y_all <= DISPLAY_Y_LIM[1])
        )
        pc_x = pc_x_all[in_view]
        pc_y = pc_y_all[in_view]
        pc_z = pc_z_all[in_view]
        if max_points > 0 and pc_x.size > max_points:
            idx = np.random.default_rng(0).choice(pc_x.size, size=max_points, replace=False)
            pc_x = pc_x[idx]
            pc_y = pc_y[idx]
            pc_z = pc_z[idx]
        pc_scatter = ax.scatter(
            pc_x,
            pc_y,
            s=2.0,
            c=pc_z,
            cmap="viridis",
            alpha=0.05,
            linewidths=0,
            label="tilt_points",
            zorder=5,
        )
        ax.legend(loc="upper right")

    ax.set_xlabel("Y (m)")
    ax.set_ylabel("X (m)")
    if title is None:
        title = "Cartesian Grid + Overlay" if show_pc_overlay else "Cartesian Grid (overlay disabled)"
    ax.set_title(title)
    return mesh, pc_scatter


def plot_all(
    frame_path: Path,
    polar_grid: np.ndarray,
    r_edges: np.ndarray,
    theta_edges: np.ndarray,
    cart_grid: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    tilt_points: np.ndarray,
    max_points: int,
    show_pc_overlay: bool,
    out_path: Path | None,
    show: bool,
) -> None:
    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    m1 = draw_polar(ax1, polar_grid, r_edges, theta_edges, title="Polar Grid")
    fig.colorbar(m1, ax=ax1, orientation="horizontal", pad=0.14, label="Danger")

    ax2 = fig.add_subplot(1, 2, 2)
    m2, pc_scatter = draw_cartesian_overlay(
        ax=ax2,
        cart_grid=cart_grid,
        x_coords=x_coords,
        y_coords=y_coords,
        tilt_points=tilt_points,
        max_points=max_points,
        show_pc_overlay=show_pc_overlay,
    )
    fig.colorbar(m2, ax=ax2, orientation="horizontal", pad=0.14, label="Danger")
    if pc_scatter is not None:
        fig.colorbar(pc_scatter, ax=ax2, orientation="horizontal", pad=0.24, label="Point Z (m)")

    fig.suptitle(frame_path.name)
    plt.tight_layout()

    if out_path is None:
        out_path = frame_path.with_suffix(".cartesian.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out_path}")
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    frame_path = pick_frame(args.input_path, args.frame_id, args.frame_pos)
    print(f"Using frame: {frame_path}")

    danger_grid, valid_mask, nontraversable, r_edges, theta_edges, tilt_points = load_npz_frame(
        frame_path
    )
    polar_grid = build_polar_grid(danger_grid, valid_mask, nontraversable)
    cart_grid, x_coords, y_coords = polar_to_cartesian(
        polar_grid=polar_grid,
        valid_mask=valid_mask,
        r_edges=r_edges,
        theta_edges=theta_edges,
        cart_resolution=args.cart_resolution,
    )
    plot_all(
        frame_path=frame_path,
        polar_grid=polar_grid,
        r_edges=r_edges,
        theta_edges=theta_edges,
        cart_grid=cart_grid,
        x_coords=x_coords,
        y_coords=y_coords,
        tilt_points=tilt_points,
        max_points=args.max_points,
        show_pc_overlay=not args.no_pc_overlay,
        out_path=args.out,
        show=args.show,
    )
