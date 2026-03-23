#!/usr/bin/env python3
"""Render 2x2 offline summaries from NPZ traversability frames."""

from __future__ import annotations

import argparse
import functools
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRAME_PATTERN = re.compile(r"^frame_(\d+)\.npz$")

IMAGE_KEYS = ("input_image_bgra", "image_rgb", "left_image_rgb", "image", "left_image")

PLANE_AXES: dict[str, tuple[int, int, str, str, str]] = {
    "xy": (0, 1, "X (m)", "Y (m)", "XY"),
    "xz": (0, 2, "X (m)", "Z (m)", "XZ"),
    "yz": (1, 2, "Y (m)", "Z (m)", "YZ"),
}

# ---------------------------------------------------------------------------
# Frame utilities
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# NPZ loading
# ---------------------------------------------------------------------------

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


def _as_valid_xyz(arr: np.ndarray) -> np.ndarray:
    points = np.asarray(arr, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected Nx3 array, got shape={points.shape}")
    points = points[:, :3]
    valid = np.isfinite(points).all(axis=1)
    return points[valid]


def load_clouds_from_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        if "input_points" not in data:
            raise KeyError(f"{path} missing `input_points`")
        if "tilt_points" not in data:
            raise KeyError(f"{path} missing `tilt_points`")
        input_points = _as_valid_xyz(data["input_points"])
        tilt_points = _as_valid_xyz(data["tilt_points"])
    return input_points, tilt_points

# ---------------------------------------------------------------------------
# Polar / cartesian plotting utilities
# ---------------------------------------------------------------------------

def prepare_plot_layers(height_map: np.ndarray, trav_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(trav_grid)
    traversable = finite & np.isclose(trav_grid, 0.0)

    height_polar_grid = np.array(height_map, copy=True, dtype=np.float32)
    height_polar_grid[~finite] = np.nan

    # nan = non-traversable (shows height map behind), 0 = unknown, 1 = traversable
    status_overlay = np.full(trav_grid.shape, np.nan, dtype=np.float32)
    status_overlay[~finite] = 0.0
    status_overlay[traversable] = 1.0

    return height_polar_grid, status_overlay


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
    status_overlay: np.ndarray | None = None,
) -> plt.cm.ScalarMappable:
    """Plot a polar bin grid (theta × r) as a pcolormesh using magma height colours.

    If status_overlay is provided (values: 0=unknown, 1=traversable, nan=non-traversable),
    a second semi-transparent layer is drawn on top to show traversability class per bin.
    Returns the base mesh (useful for attaching a colorbar).
    """
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

    if status_overlay is not None:
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

    return mesh

# ---------------------------------------------------------------------------
# Pointcloud comparison plot
# ---------------------------------------------------------------------------

def _sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    idx = np.random.default_rng(0).choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def _project_points(points: np.ndarray, plane: str) -> tuple[np.ndarray, np.ndarray, str, str, str]:
    i, j, xlabel, ylabel, plane_name = PLANE_AXES[plane]
    return points[:, i], points[:, j], xlabel, ylabel, plane_name


def plot_comparison(
    ax: plt.Axes,
    input_points: np.ndarray,
    tilt_points: np.ndarray,
    max_points: int,
    plane: str = "xy",
    title: str = "Dense Pointcloud Comparison",
) -> dict[str, int]:
    input_color = "#1f77b4"
    tilt_color = "#ff7f0e"

    input_plot = _sample_points(input_points, max_points)
    tilt_plot = _sample_points(tilt_points, max_points)
    input_u, input_v, xlabel, ylabel, plane_name = _project_points(input_plot, plane)
    tilt_u, tilt_v, _, _, _ = _project_points(tilt_plot, plane)

    ax.scatter(input_u, input_v, s=1.0, c=input_color, alpha=0.35, linewidths=0)
    ax.scatter(tilt_u, tilt_v, s=1.0, c=tilt_color, alpha=0.35, linewidths=0)

    if plane == "yz":
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.6, 0.6)
    else:
        ax.set_xlim(0.0, 2.0)
        ax.set_ylim(-1.0, 1.0)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} ({plane_name} plane)")
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=input_color, markeredgecolor=input_color, markersize=6, label="input"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=tilt_color, markeredgecolor=tilt_color, markersize=6, label="tilt-compensated"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    return {"input": int(input_plot.shape[0]), "tilt": int(tilt_plot.shape[0])}

# ---------------------------------------------------------------------------
# Panel drawing
# ---------------------------------------------------------------------------

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
            0.5, 0.5,
            "No image found in NPZ\n(expected key: input_image_bgra)",
            ha="center", va="center", color="white", fontsize=10,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.imshow(image)
        ax.axis("off")
    ax.set_title("Image from the SVO")


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
        0.5, 0.5,
        f"pointcloud comparison unavailable\n{error}",
        ha="center", va="center", color="white", fontsize=10,
        transform=ax.transAxes,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("original points vs tilt-compensated")


def _draw_cartesian_panel(
    ax: plt.Axes,
    fig: plt.Figure,
    trav_grid: np.ndarray,
    height_map: np.ndarray,
    r_edges: np.ndarray,
    theta_edges: np.ndarray,
    tilt_points: np.ndarray,
    grid_res_m: float,
    show_nt_height_overlay: bool,
    show_pc_overlay: bool,
    max_points: int,
) -> None:
    rasterize = functools.partial(
        rasterize_polar_to_cartesian,
        r_edges=r_edges,
        theta_edges=theta_edges,
        x_min=0.0, x_max=2.0,
        y_min=-0.75, y_max=0.75,
        grid_res_m=grid_res_m,
    )

    # Base layer: traversability (0=traversable, 1=non-traversable, nan=unknown)
    cart_grid, x_edges, y_edges = rasterize(
        values=trav_grid,
        source_valid_mask=~np.isnan(trav_grid),
    )
    trav_cmap = colors.ListedColormap(["#4CAF50", "#F44336"])
    trav_cmap.set_bad("#888888")
    trav_norm = colors.BoundaryNorm(boundaries=[-0.5, 0.5, 1.5], ncolors=trav_cmap.N)
    ax.pcolormesh(y_edges, x_edges, cart_grid.T, cmap=trav_cmap, norm=trav_norm, shading="auto")

    # Debug overlay: non-traversable cells coloured by height
    if show_nt_height_overlay:
        nt_mask = np.isfinite(trav_grid) & np.isclose(trav_grid, 1.0) & np.isfinite(height_map)
        height_cart, _, _ = rasterize(values=height_map, source_valid_mask=nt_mask)
        valid_h = np.isfinite(height_cart)
        if np.any(valid_h):
            h_lo, h_hi = np.percentile(height_cart[valid_h], [5.0, 95.0])
            if not np.isfinite(h_lo) or not np.isfinite(h_hi) or h_hi <= h_lo:
                h_lo = float(np.nanmin(height_cart[valid_h]))
                h_hi = max(float(np.nanmax(height_cart[valid_h])), h_lo + 1e-3)
            h_cmap = plt.get_cmap("magma").copy()
            h_cmap.set_bad((0.0, 0.0, 0.0, 0.0))
            ax.pcolormesh(
                y_edges, x_edges,
                np.ma.masked_invalid(height_cart.T),
                cmap=h_cmap, norm=colors.Normalize(vmin=h_lo, vmax=h_hi),
                shading="auto", alpha=0.65,
            )

    # Debug overlay: raw point scatter
    if show_pc_overlay and tilt_points.size:
        x = tilt_points[:, 0] * np.cos(tilt_points[:, 1])
        y = tilt_points[:, 0] * np.sin(tilt_points[:, 1])
        in_view = np.isfinite(x) & np.isfinite(y) & (x >= 0.0) & (x <= 2.0) & (y >= -0.75) & (y <= 0.75)
        x_plot, y_plot = x[in_view], y[in_view]
        if max_points > 0 and x_plot.size > max_points:
            idx = np.random.default_rng(0).choice(x_plot.size, size=max_points, replace=False)
            x_plot, y_plot = x_plot[idx], y_plot[idx]
        ax.scatter(y_plot, x_plot, s=2.0, alpha=0.08, c="#444444", linewidths=0.0, zorder=5)

    step = float(grid_res_m * 2)
    major = 6 * step
    ax.set_xlim(0.75, -0.75)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{-x:.4g}"))
    ax.set_ylim(0.0, 2.0)
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

    # Colorbar showing unknown/traversable (non-traversable is visually obvious as red)
    bar_cmap = colors.ListedColormap(["#888888", "#4CAF50"])
    bar_norm = colors.BoundaryNorm(boundaries=[-1.5, -0.5, 0.5], ncolors=bar_cmap.N)
    bar_mappable = plt.cm.ScalarMappable(cmap=bar_cmap, norm=bar_norm)
    bar_mappable.set_array([])
    cbar = fig.colorbar(bar_mappable, ax=ax, orientation="horizontal", pad=0.20, fraction=0.08)
    cbar.set_ticks([-1.0, 0.0])
    cbar.set_ticklabels(["Unknown", "Traversable"])

# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def _with_nt_height_suffix(path: Path, enabled: bool) -> Path:
    if not enabled:
        return path
    if path.suffix:
        return path.with_name(f"{path.stem}.nth{path.suffix}")
    return path.with_name(f"{path.name}.nth")


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

    # [0,0] camera image  [0,1] point cloud comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    _draw_image_panel(axes[0, 0], _load_image_from_npz(frame_path))
    _draw_pointcloud_panel(axes[0, 1], frame_path, max_points=max_points, pc_plane=pc_plane)

    # [1,0] polar bin grid coloured by height, optionally with traversability overlay
    polar_mesh = draw_polar(
        ax=axes[1, 0],
        polar_grid=height_polar_grid,
        r_edges=r_edges,
        theta_edges=theta_edges,
        title="Traversability - Polar Bin" if show_polar_trav_overlay else "Height Map - Polar Bin",
        status_overlay=status_overlay if show_polar_trav_overlay else None,
    )
    axes[1, 0].invert_xaxis()  # pcolormesh plots theta left-to-right; invert so right side of robot is on the right
    axes[1, 0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{-x:.4g}"))  # restore true theta sign after inversion
    fig.colorbar(polar_mesh, ax=axes[1, 0], orientation="horizontal", pad=0.18, fraction=0.08, label="Height")

    # [1,1] traversability reprojected to cartesian
    _draw_cartesian_panel(
        ax=axes[1, 1],
        fig=fig,
        trav_grid=trav_grid,
        height_map=height_map,
        r_edges=r_edges,
        theta_edges=theta_edges,
        tilt_points=tilt_points,
        grid_res_m=float(grid_res_m),
        show_nt_height_overlay=show_nt_height_overlay,
        show_pc_overlay=show_pc_overlay,
        max_points=max_points,
    )

    frame_id = frame_id_from_name(frame_path)
    fig.suptitle(f"frame_{frame_id:05d} from {frame_path.parent.name}", fontsize=14, fontweight="bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render NPZ summaries with image, pointcloud comparison, polar, and cartesian plots."
    )
    parser.add_argument("npz_dir", type=Path, help="Folder containing frame_*.npz files.")
    parser.add_argument("--frame-id", type=int, default=None, help="Render only this frame id.")
    parser.add_argument("--frame-pos", type=int, default=None, help="Render only this frame index.")
    parser.add_argument("--all", action="store_true", help="Render all frames.")
    parser.add_argument("--xy-res", type=float, default=0.05, help="Legacy Cartesian grid resolution (m).")
    parser.add_argument("--grid-res-m", type=float, default=0.05, help="Cartesian raster grid resolution (m).")
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
