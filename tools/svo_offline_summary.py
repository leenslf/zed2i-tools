#!/usr/bin/env python3
"""Render 2x2 offline summaries from NPZ traversability frames."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

from pointcloud_comparison import load_clouds_from_npz, plot_comparison
from travviz_common import (
    draw_polar,
    frame_id_from_name,
    draw_polar_with_trav_overlay,
    load_npz_frame,
    pick_frames_from_dir,
    prepare_plot_layers,
    rasterize_polar_to_cartesian,
)

IMAGE_KEYS = ("input_image_bgra", "image_rgb", "left_image_rgb", "image", "left_image")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render NPZ summaries with image, pointcloud comparison, polar, and cartesian plots."
    )
    parser.add_argument("npz_dir", type=Path, help="Folder containing frame_*.npz files.")
    parser.add_argument("--frame-id", type=int, default=None, help="Render only this frame id.")
    parser.add_argument("--frame-pos", type=int, default=None, help="Render only this frame index.")
    parser.add_argument("--all", action="store_true", help="Render all frames.")
    parser.add_argument("--xy-res", type=float, default=0.05, help="Legacy Cartesian grid resolution (m).")
    parser.add_argument(
        "--grid-res-m",
        type=float,
        default=0.05,
        help="Cartesian raster grid resolution (m).",
    )
    parser.add_argument("--max-points", type=int, default=100000, help="Max points per cloud plot.")
    parser.add_argument(
        "--pc-plane",
        choices=("xy", "xz", "yz"),
        default="xy",
        help="Projection plane for pointcloud comparison panel.",
    )
    parser.add_argument(
        "-pc_overlay",
        action="store_true",
        default=False,
        help="Overlay raw [r, theta, z] points on the cartesian raster.",
    )
    parser.add_argument(
        "--nt-height-overlay",
        action="store_true",
        default=False,
        help="Overlay non-traversable cartesian cells with height_map values (temporary/debug view).",
    )
    parser.add_argument(
        "--no-polar-trav-overlay",
        action="store_true",
        default=False,
        help="Show the polar panel as height_map only, without traversability overlay.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: traversability_frames/<npz_folder_name>).",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output file path for single-frame mode only.")
    return parser.parse_args()


def _load_image_from_npz(path: Path) -> np.ndarray | None:
    with np.load(path) as data:
        key = next((candidate for candidate in IMAGE_KEYS if candidate in data), None)
        if key is None:
            return None
        image = np.asarray(data[key])

    if image.ndim != 3:
        return None
    if key == "input_image_bgra":
        if image.shape[2] < 3:
            return None
        image = image[:, :, :3][:, :, ::-1]
    elif image.shape[2] == 4:
        image = image[:, :, :3]

    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _draw_image_panel(ax: plt.Axes, image: np.ndarray | None) -> None:
    if image is None:
        ax.set_facecolor("#111111")
        ax.text(
            0.5,
            0.5,
            "No image found in NPZ\n(expected key: input_image_bgra)",
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.imshow(image)
        ax.axis("off")
    ax.set_title("Image from the SVO")


def _with_nt_height_suffix(path: Path, enabled: bool) -> Path:
    if not enabled:
        return path
    if path.suffix:
        return path.with_name(f"{path.stem}.nth{path.suffix}")
    return path.with_name(f"{path.name}.nth")


def _draw_pointcloud_panel(
    ax: plt.Axes,
    frame_path: Path,
    max_points: int,
    pc_plane: str,
) -> None:
    try:
        input_points, tilt_points = load_clouds_from_npz(frame_path)
        plot_comparison(
            ax=ax,
            input_points=input_points,
            tilt_points=tilt_points,
            max_points=max_points,
            plane=pc_plane,
            title="original points vs tilt-compensated",
        )
        return
    except Exception as exc:
        error = str(exc)

    ax.set_facecolor("#111111")
    ax.text(
        0.5,
        0.5,
        f"pointcloud comparison unavailable\\n{error}",
        ha="center",
        va="center",
        color="white",
        fontsize=10,
        transform=ax.transAxes,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("original points vs tilt-compensated")


def render_frame(
    frame_path: Path,
    out_path: Path,
    grid_res_m: float,
    max_points: int,
    pc_plane: str,
    show_pc_overlay: bool,
    show_nt_height_overlay: bool,
    show_polar_trav_overlay: bool,
) -> None:
    trav_grid, r_edges, theta_edges, height_map, tilt_points = load_npz_frame(frame_path)
    height_polar_grid, status_overlay = prepare_plot_layers(height_map=height_map, trav_grid=trav_grid)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    _draw_image_panel(axes[0, 0], _load_image_from_npz(frame_path))
    _draw_pointcloud_panel(axes[0, 1], frame_path, max_points=max_points, pc_plane=pc_plane)

    if show_polar_trav_overlay:
        polar_mesh = draw_polar_with_trav_overlay(
            ax=axes[1, 0],
            height_grid=height_polar_grid,
            status_overlay=status_overlay,
            r_edges=r_edges,
            theta_edges=theta_edges,
            title="Traversability - Polar Bin",
        )
    else:
        polar_mesh = draw_polar(
            ax=axes[1, 0],
            polar_grid=height_polar_grid,
            r_edges=r_edges,
            theta_edges=theta_edges,
            title="Height Map - Polar Bin",
        )
    axes[1, 0].invert_xaxis()
    axes[1, 0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{-x:.4g}"))
    fig.colorbar(
        polar_mesh,
        ax=axes[1, 0],
        orientation="horizontal",
        pad=0.18,
        fraction=0.08,
        label="Height",
    )

    # Legacy: danger grid cartesian raster
    # cart_mesh, z_scatter = draw_cartesian_overlay(
    #     ax=axes[1, 1],
    #     cart_grid=legacy_cart_grid,
    #     x_coords=legacy_x_coords,
    #     y_coords=legacy_y_coords,
    #     tilt_points=tilt_points,
    #     max_points=max_points,
    #     show_pc_overlay=show_pc_overlay,
    #     title="Traversability - Cartesian Grid",
    # )

    # if z_scatter is not None:
    #     fig.colorbar(
    #         z_scatter,
    #         ax=axes[1, 1],
    #         orientation="horizontal",
    #         pad=0.30,
    #         fraction=0.08,
    #         label="Point Z (m)",
    #     )
    # New: trav_grid cartesian raster
    ax = axes[1, 1]
    cart_grid, x_edges, y_edges = rasterize_polar_to_cartesian(
        values=trav_grid,
        source_valid_mask=~np.isnan(trav_grid),
        r_edges=r_edges,
        theta_edges=theta_edges,
        x_min=0.0,
        x_max=2.0,
        y_min=-0.75,
        y_max=0.75,
        grid_res_m=float(grid_res_m),
    )

    cmap = colors.ListedColormap(["#4CAF50", "#F44336"])
    cmap.set_bad("#888888")
    norm = colors.BoundaryNorm(boundaries=[-0.5, 0.5, 1.5], ncolors=cmap.N)
    ax.pcolormesh(
        y_edges,
        x_edges,
        cart_grid.T,
        cmap=cmap,
        norm=norm,
        shading="auto",
    )
    nt_height_mappable = None
    # BEGIN TEMP OVERLAY (easy to remove): non-traversable cells colored by height_map.
    if show_nt_height_overlay:
        nt_mask = np.isfinite(trav_grid) & np.isclose(trav_grid, 1.0) & np.isfinite(height_map)
        height_cart, _, _ = rasterize_polar_to_cartesian(
            values=height_map,
            source_valid_mask=nt_mask,
            r_edges=r_edges,
            theta_edges=theta_edges,
            x_min=0.0,
            x_max=2.0,
            y_min=-0.75,
            y_max=0.75,
            grid_res_m=float(grid_res_m),
        )
        valid_h = np.isfinite(height_cart)
        if np.any(valid_h):
            h_lo, h_hi = np.percentile(height_cart[valid_h], [5.0, 95.0])
            if not np.isfinite(h_lo) or not np.isfinite(h_hi) or h_hi <= h_lo:
                h_lo, h_hi = float(np.nanmin(height_cart[valid_h])), float(np.nanmax(height_cart[valid_h]))
                if h_hi <= h_lo:
                    h_hi = h_lo + 1e-3
            h_cmap = plt.get_cmap("magma").copy()
            h_cmap.set_bad((0.0, 0.0, 0.0, 0.0))
            h_norm = colors.Normalize(vmin=h_lo, vmax=h_hi)
            nt_height_mappable = plt.cm.ScalarMappable(cmap=h_cmap, norm=h_norm)
            nt_height_mappable.set_array([])
            ax.pcolormesh(
                y_edges,
                x_edges,
                np.ma.masked_invalid(height_cart.T),
                cmap=h_cmap,
                norm=h_norm,
                shading="auto",
                alpha=0.65,
            )
    # END TEMP OVERLAY (easy to remove)

    if show_pc_overlay and tilt_points.size:
        r = tilt_points[:, 0]
        theta = tilt_points[:, 1]
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        finite = np.isfinite(x) & np.isfinite(y)
        in_view = finite & (x >= 0.0) & (x <= 2.0) & (y >= -0.75) & (y <= 0.75)
        x_plot = x[in_view]
        y_plot = y[in_view]
        if max_points > 0 and x_plot.size > max_points:
            idx = np.random.default_rng(0).choice(x_plot.size, size=max_points, replace=False)
            x_plot = x_plot[idx]
            y_plot = y_plot[idx]
        ax.scatter(
            y_plot,
            x_plot,
            s=2.0,
            alpha=0.08,
            c="#444444",
            linewidths=0.0,
            zorder=5,
        )

    ax.set_xlim(0.75, -0.75)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{-x:.4g}"))
    ax.set_ylim(0.0, 2.0)
    step = float(grid_res_m*2)
    major = 6 * step
    ax.set_xticks(np.arange(-0.75, 0.75 + 1e-6, major))
    ax.set_yticks(np.arange(0.0, 2.0 + 1e-6, major))
    ax.set_xticks(np.arange(-0.75, 0.75 + 1e-6, step), minor=True)
    ax.set_yticks(np.arange(0.0, 2.0 + 1e-6, step), minor=True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Y (m)")
    ax.set_ylabel("X (m)")
    ax.grid(True, which="major", color="0.80", linewidth=0.8, alpha=0.9)
    ax.grid(True, which="minor", color="0.92", linewidth=0.1, alpha=0.9)
    ax.set_title("Traversability - Cartesian Raster")

    bar_cmap = colors.ListedColormap(["#888888", "#4CAF50"])
    bar_norm = colors.BoundaryNorm(boundaries=[-1.5, -0.5, 0.5], ncolors=bar_cmap.N)
    bar_mappable = plt.cm.ScalarMappable(cmap=bar_cmap, norm=bar_norm)
    bar_mappable.set_array([])
    cbar = fig.colorbar(
        bar_mappable,
        ax=ax,
        orientation="horizontal",
        pad=0.20,
        fraction=0.08,
    )
    cbar.set_ticks([-1.0, 0.0])
    cbar.set_ticklabels(["Unknown", "Traversable"])

    frame_id = frame_id_from_name(frame_path)
    fig.suptitle(f"frame_{frame_id:05d} from {frame_path.parent.name}", fontsize=14, fontweight="bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    npz_dir = args.npz_dir.expanduser().resolve()
    if not npz_dir.is_dir():
        raise NotADirectoryError(f"Expected folder of NPZ files: {npz_dir}")

    selected = pick_frames_from_dir(
        input_dir=npz_dir,
        frame_id=args.frame_id,
        frame_pos=args.frame_pos,
        render_all=args.all,
    )

    if args.out is not None and len(selected) != 1:
        raise ValueError("--out can only be used when rendering exactly one frame")

    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir
        else Path(__file__).resolve().parents[1] / "traversability_frames" / npz_dir.name
    )

    for frame_path in selected:
        if args.out is not None:
            out_path = _with_nt_height_suffix(args.out.expanduser().resolve(), args.nt_height_overlay)
        else:
            out_path = _with_nt_height_suffix(out_dir / f"{frame_path.stem}.summary.png", args.nt_height_overlay)

        render_frame(
            frame_path=frame_path,
            out_path=out_path,
            grid_res_m=args.grid_res_m,
            max_points=args.max_points,
            pc_plane=args.pc_plane,
            show_pc_overlay=args.pc_overlay,
            show_nt_height_overlay=args.nt_height_overlay,
            show_polar_trav_overlay=not args.no_polar_trav_overlay,
        )
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
