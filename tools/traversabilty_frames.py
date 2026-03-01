#!/usr/bin/env python3
"""Batch-render traversability NPZ frames to PNG images."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


FRAME_PATTERN = re.compile(r"^frame_(\d+)\.npz$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render traversability NPZ frames as PNGs.")
    parser.add_argument("folder_name", help="Recording folder name under traversability_grid/.")
    parser.add_argument(
        "--view",
        choices=("2d", "polar", "cartesian", "cartesian_pc"),
        default="2d",
        help="Visualization mode: 2d, polar, cartesian, or cartesian_pc (default: 2d).",
    )
    return parser.parse_args()


def sorted_frame_paths(input_dir: Path) -> list[Path]:
    frames: list[tuple[int, Path]] = []
    for p in input_dir.glob("frame_*.npz"):
        m = FRAME_PATTERN.match(p.name)
        if m:
            frames.append((int(m.group(1)), p))
    frames.sort(key=lambda item: item[0])
    return [p for _, p in frames]


def _pick_key(data: np.lib.npyio.NpzFile, key: str) -> str:
    """Return either legacy key or traversability-prefixed key."""
    if key in data:
        return key
    prefixed = f"traversability_{key}"
    if prefixed in data:
        return prefixed
    raise KeyError(
        f"{data.filename} does not contain `{key}` or `{prefixed}`."
    )


def load_frame(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        danger_grid = np.asarray(data[_pick_key(data, "danger_grid")], dtype=np.float32)
        valid_mask = np.asarray(data[_pick_key(data, "valid_mask")], dtype=bool)
        nontraversable = np.asarray(data[_pick_key(data, "nontraversable")], dtype=bool)
        r_edges = np.asarray(data[_pick_key(data, "r_edges")], dtype=np.float32)
        theta_edges = np.asarray(data[_pick_key(data, "theta_edges")], dtype=np.float32)
        tilt_points = np.asarray(data["tilt_points"], dtype=np.float32)

    if danger_grid.shape != valid_mask.shape or danger_grid.shape != nontraversable.shape:
        raise ValueError(f"Inconsistent grid shapes in {path}.")
    if danger_grid.shape != (r_edges.size - 1, theta_edges.size - 1):
        raise ValueError(f"Grid/edge shape mismatch in {path}.")
    return danger_grid, valid_mask, nontraversable, r_edges, theta_edges, tilt_points


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    raw_input = Path(args.folder_name)
    if raw_input.is_absolute():
        input_dir = raw_input
        output_stem = raw_input.name
    else:
        direct_input = project_root / raw_input
        grid_input = project_root / "traversability_grid" / raw_input
        if direct_input.exists():
            input_dir = direct_input
        else:
            input_dir = grid_input
        output_stem = raw_input.name

    output_dir = project_root / "traversability_frames" / output_stem / args.view

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted_frame_paths(input_dir)
    if not frame_paths:
        raise FileNotFoundError(f"No frame_*.npz files found in: {input_dir}")

    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_over("black")
    norm = Normalize(vmin=0.0, vmax=1.0, clip=False)

    total = len(frame_paths)
    for idx, frame_path in enumerate(frame_paths, start=1):
        print(f"Processing frame {idx:04d}/{total:04d}")
        danger_grid, valid_mask, nontraversable, r_edges, theta_edges, tilt_points = load_frame(
            frame_path
        )

        # Same coloring logic as visualize_traversability.py.
        plot_values = np.array(danger_grid, copy=True, dtype=np.float32)
        plot_values[valid_mask & nontraversable] = 1.1

        fig = plt.figure(figsize=(9, 7))
        mesh = None
        if args.view == "polar":
            ax = fig.add_subplot(111, projection="polar")
            mesh = ax.pcolormesh(
                theta_edges,
                r_edges,
                plot_values,
                cmap=cmap,
                norm=norm,
                shading="auto",
            )
            theta_min_deg = float(np.degrees(theta_edges.min()))
            theta_max_deg = float(np.degrees(theta_edges.max()))
            ax.set_thetamin(theta_min_deg - 5.0)
            ax.set_thetamax(theta_max_deg + 5.0)
            title_view = "Polar View"
        elif args.view == "cartesian":
            ax = fig.add_subplot(111)
            n_r = nontraversable.shape[0]
            first_true = np.argmax(nontraversable, axis=0)
            has_true = np.any(nontraversable, axis=0)
            row_indices = np.where(has_true, first_true, n_r)

            theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
            obstacle_ranges = r_edges[row_indices][has_true]
            theta_centers = theta_centers[has_true]
            x_raw = obstacle_ranges * np.cos(theta_centers)
            y_raw = obstacle_ranges * np.sin(theta_centers)
            x = -y_raw
            y = x_raw

            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(-0.75, 0.75)
            ax.set_ylim(0.2, 0.8)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")
            title_view = "Cartesian View"
        elif args.view == "cartesian_pc":
            ax = fig.add_subplot(111)
            n_r = nontraversable.shape[0]
            first_true = np.argmax(nontraversable, axis=0)
            has_true = np.any(nontraversable, axis=0)
            row_indices = np.where(has_true, first_true, n_r)

            theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
            obstacle_ranges = r_edges[row_indices][has_true]
            theta_centers = theta_centers[has_true]
            x_raw = obstacle_ranges * np.cos(theta_centers)
            y_raw = obstacle_ranges * np.sin(theta_centers)
            x = -y_raw
            y = x_raw
            pc_x = -tilt_points[:, 1]
            pc_y = tilt_points[:, 0]

            pc_scatter = ax.scatter(
                pc_x,
                pc_y,
                c=tilt_points[:, 2],
                cmap="viridis",
                s=2,
                alpha=0.6,
                vmin=-0.3,
                vmax=0.3,
            )
            fig.colorbar(pc_scatter, ax=ax, label="Z (m)")
            ax.scatter(x, y, s=8, color="tab:red", label="Nearest obstacle boundary", zorder=3)
            ax.scatter([0.0], [0.0], color="tab:blue", s=35, marker="o", label="Robot", zorder=4)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(-0.75, 0.75)
            ax.set_ylim(0.2, 0.8)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")
            title_view = "Cartesian PC View"
        else:
            ax = fig.add_subplot(111)
            theta_edges_deg = np.degrees(theta_edges)
            mesh = ax.pcolormesh(
                theta_edges_deg,
                r_edges,
                plot_values,
                cmap=cmap,
                norm=norm,
                shading="auto",
            )
            theta_margin = 0.05 * float(theta_edges.max() - theta_edges.min())
            r_margin = 0.05 * float(r_edges.max() - r_edges.min())
            ax.set_xlim(
                float(np.degrees(theta_edges.min() - theta_margin)),
                float(np.degrees(theta_edges.max() + theta_margin)),
            )
            ax.set_ylim(float(r_edges.min() - r_margin), float(r_edges.max() + r_margin))
            ax.set_xlabel("Theta (deg)")
            ax.set_ylabel("Range (m)")
            title_view = "Range-Angle Grid"

        if mesh is not None:
            fig.colorbar(mesh, ax=ax, label="Danger Score")
        ax.set_title(f"{frame_path.name} | {title_view}")

        out_path = output_dir / f"{frame_path.stem}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
