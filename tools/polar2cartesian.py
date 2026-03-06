#!/usr/bin/env python3
"""Remap traversability polar grid from NPZ to Cartesian for visualization."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from travviz_common import (
    build_nontraversable_grid,
    draw_cartesian_overlay,
    draw_polar_with_trav_overlay,
    load_npz_frame,
    pick_single_frame,
    polar_to_cartesian,
    prepare_plot_layers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polar -> Cartesian remap from traversability NPZ.")
    parser.add_argument("input_path", type=Path, help="NPZ file or folder containing frame_*.npz.")
    parser.add_argument("--frame-id", type=int, default=None, help="Frame id to pick from folder.")
    parser.add_argument("--frame-pos", type=int, default=0, help="Frame position when --frame-id unset.")
    parser.add_argument("--cart-resolution", type=float, default=0.05, help="Cartesian cell size in meters.")
    parser.add_argument("--max-points", type=int, default=120000, help="Max tilt_points drawn in overlay.")
    parser.add_argument(
        "--no-pc-overlay",
        action="store_true",
        help="Disable tilt point-cloud overlay in the Cartesian panel.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path.")
    parser.add_argument("--show", action="store_true", help="Show interactive window.")
    return parser.parse_args()


def render_frame(
    frame_path: Path,
    out_path: Path | None,
    cart_resolution: float,
    max_points: int,
    show_pc_overlay: bool,
    show: bool,
) -> None:
    trav_grid, r_edges, theta_edges, height_map, tilt_points = load_npz_frame(frame_path)
    nontraversable_grid = build_nontraversable_grid(trav_grid)
    height_polar_grid, status_overlay = prepare_plot_layers(height_map=height_map, trav_grid=trav_grid)
    cart_grid, x_coords, y_coords = polar_to_cartesian(
        polar_grid=nontraversable_grid,
        r_edges=r_edges,
        theta_edges=theta_edges,
        cart_resolution=cart_resolution,
    )

    fig = plt.figure(figsize=(12, 5))

    ax_polar = fig.add_subplot(1, 2, 1)
    polar_mesh = draw_polar_with_trav_overlay(
        ax=ax_polar,
        height_grid=height_polar_grid,
        status_overlay=status_overlay,
        r_edges=r_edges,
        theta_edges=theta_edges,
        title="Polar Grid",
    )


    ax_cart = fig.add_subplot(1, 2, 2)
    cart_mesh, pc_scatter = draw_cartesian_overlay(
        ax=ax_cart,
        cart_grid=cart_grid,
        x_coords=x_coords,
        y_coords=y_coords,
        tilt_points=tilt_points,
        max_points=max_points,
        show_pc_overlay=show_pc_overlay,
        title="Cartesian Grid + Overlay" if show_pc_overlay else "Cartesian Grid (overlay disabled)",
    )
    # fig.colorbar(cart_mesh, ax=ax_cart, orientation="horizontal", pad=0.14, label="Height (m)")
    if pc_scatter is not None:
        fig.colorbar(pc_scatter, ax=ax_cart, orientation="horizontal", pad=0.24, label="Point Z (m)")

    fig.suptitle(frame_path.name)
    plt.tight_layout()

    target = out_path or frame_path.with_suffix(".cartesian.png")
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=150, bbox_inches="tight")
    print(f"Saved -> {target}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    frame_path = pick_single_frame(args.input_path, args.frame_id, args.frame_pos)
    print(f"Using frame: {frame_path}")
    render_frame(
        frame_path=frame_path,
        out_path=args.out,
        cart_resolution=args.cart_resolution,
        max_points=args.max_points,
        show_pc_overlay=not args.no_pc_overlay,
        show=args.show,
    )


if __name__ == "__main__":
    main()
