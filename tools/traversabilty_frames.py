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
        choices=("2d", "polar"),
        default="2d",
        help="Visualization mode: 2d or polar (default: 2d).",
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


def load_frame(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        for key in ("danger_grid", "valid_mask", "nontraversable", "r_edges", "theta_edges"):
            if key not in data:
                raise KeyError(f"{path} does not contain `{key}`.")
        danger_grid = np.asarray(data["danger_grid"], dtype=np.float32)
        valid_mask = np.asarray(data["valid_mask"], dtype=bool)
        nontraversable = np.asarray(data["nontraversable"], dtype=bool)
        r_edges = np.asarray(data["r_edges"], dtype=np.float32)
        theta_edges = np.asarray(data["theta_edges"], dtype=np.float32)

    if danger_grid.shape != valid_mask.shape or danger_grid.shape != nontraversable.shape:
        raise ValueError(f"Inconsistent grid shapes in {path}.")
    if danger_grid.shape != (r_edges.size - 1, theta_edges.size - 1):
        raise ValueError(f"Grid/edge shape mismatch in {path}.")
    return danger_grid, valid_mask, nontraversable, r_edges, theta_edges


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    input_dir = project_root / "traversability_grid" / args.folder_name
    output_dir = project_root / "traversability_frames" / args.folder_name / args.view
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    frame_paths = sorted_frame_paths(input_dir)
    if not frame_paths:
        raise FileNotFoundError(f"No frame_*.npz files found in: {input_dir}")

    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_over("black")
    norm = Normalize(vmin=0.0, vmax=1.0, clip=False)

    total = len(frame_paths)
    for idx, frame_path in enumerate(frame_paths, start=1):
        print(f"Processing frame {idx:04d}/{total:04d}")
        danger_grid, valid_mask, nontraversable, r_edges, theta_edges = load_frame(frame_path)

        # Same coloring logic as visualize_traversability.py.
        plot_values = np.array(danger_grid, copy=True, dtype=np.float32)
        plot_values[valid_mask & nontraversable] = 1.1

        fig = plt.figure(figsize=(9, 7))
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

        fig.colorbar(mesh, ax=ax, label="Danger Score")
        ax.set_title(f"{frame_path.name} | {title_view}")

        out_path = output_dir / f"{frame_path.stem}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
