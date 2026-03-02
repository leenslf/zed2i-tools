#!/usr/bin/env python3
"""Render 2x2 offline summaries from NPZ traversability frames.

Layout per frame:
- Top-left: image from NPZ
- Top-right: dense pointcloud comparison (input vs tilt)
- Bottom-left: traversability polar
- Bottom-right: traversability Cartesian (+ optional pointcloud overlay)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pointcloud_comparison import load_clouds_from_npz, plot_comparison
from polar2cartesian import (
    build_polar_grid,
    draw_cartesian_overlay,
    draw_polar,
    load_npz_frame,
    polar_to_cartesian,
)


FRAME_PATTERN = re.compile(r"^frame_(\d+)\.npz$")
IMAGE_KEYS = ("input_image_bgra", "image_rgb", "left_image_rgb", "image", "left_image")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render NPZ summaries with image, pointcloud comparison, polar, and cartesian plots."
    )
    parser.add_argument("npz_dir", type=Path, help="Folder containing frame_*.npz files.")
    parser.add_argument("--frame-id", type=int, default=None, help="Render only this frame id.")
    parser.add_argument(
        "--frame-pos",
        type=int,
        default=None,
        help="Render only this index in sorted frame list.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Render all frames (default when no --frame-id/--frame-pos is provided).",
    )
    parser.add_argument("--xy-res", type=float, default=0.05, help="Cartesian grid resolution (m).")
    parser.add_argument("--max-points", type=int, default=100000, help="Max points per cloud plot.")
    parser.add_argument(
        "--pc-plane",
        choices=("xy", "xz", "yz"),
        default="xy",
        help="Projection plane for pointcloud_comparison panel (default: xy).",
    )
    parser.add_argument(
        "--no-pc-overlay",
        action="store_true",
        help="Disable tilt pointcloud overlay in traversability Cartesian panel.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: traversability_frames/<npz_folder_name>).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output file path for single-frame mode only.",
    )
    return parser.parse_args()


def sorted_frame_paths(input_dir: Path) -> list[Path]:
    frames: list[tuple[int, Path]] = []
    for path in input_dir.glob("frame_*.npz"):
        match = FRAME_PATTERN.match(path.name)
        if match:
            frames.append((int(match.group(1)), path))
    frames.sort(key=lambda item: item[0])
    return [path for _, path in frames]


def _pick_frame(frames: list[Path], frame_id: int | None, frame_pos: int | None) -> list[Path]:
    if frame_id is not None:
        target = f"frame_{frame_id:05d}.npz"
        for frame in frames:
            if frame.name == target:
                return [frame]
        raise FileNotFoundError(f"Frame not found: {target}")

    if frame_pos is not None:
        if frame_pos < 0 or frame_pos >= len(frames):
            raise IndexError(f"frame-pos {frame_pos} out of range [0, {len(frames)-1}]")
        return [frames[frame_pos]]

    return frames


def _load_image_from_npz(path: Path) -> np.ndarray | None:
    with np.load(path) as data:
        key = next((k for k in IMAGE_KEYS if k in data), None)
        if key is None:
            return None
        image = np.asarray(data[key])

    if image.ndim != 3:
        return None
    if key == "input_image_bgra":
        if image.shape[2] < 3:
            return None
        image = image[:, :, :3][:, :, ::-1]
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _frame_id_from_name(path: Path) -> int:
    m = FRAME_PATTERN.match(path.name)
    if not m:
        raise ValueError(f"Invalid frame filename: {path.name}")
    return int(m.group(1))


def _with_pc_suffix(path: Path, add_pc_suffix: bool) -> Path:
    if not add_pc_suffix:
        return path
    if path.suffix:
        return path.with_name(f"{path.stem}.pc{path.suffix}")
    return path.with_name(f"{path.name}.pc")


def render_frame(
    frame_path: Path,
    out_path: Path,
    xy_res: float,
    max_points: int,
    pc_plane: str,
    show_pc_overlay: bool,
) -> None:
    danger_grid, valid_mask, nontraversable, r_edges, theta_edges, tilt_points = load_npz_frame(
        frame_path
    )
    polar_grid = build_polar_grid(danger_grid, valid_mask, nontraversable)
    cart_grid, x_coords, y_coords = polar_to_cartesian(
        polar_grid=polar_grid,
        valid_mask=valid_mask,
        r_edges=r_edges,
        theta_edges=theta_edges,
        cart_resolution=xy_res,
    )

    pc_compare_error: str | None = None
    try:
        input_points, tilt_compare_points = load_clouds_from_npz(frame_path)
    except Exception as exc:
        input_points = np.empty((0, 3), dtype=np.float32)
        tilt_compare_points = np.empty((0, 3), dtype=np.float32)
        pc_compare_error = str(exc)
    image = _load_image_from_npz(frame_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    ax_img = axes[0, 0]
    ax_pc = axes[0, 1]
    ax_polar = axes[1, 0]
    ax_cart = axes[1, 1]

    if image is None:
        ax_img.set_facecolor("#111111")
        ax_img.text(
            0.5,
            0.5,
            "No image found in NPZ\n(expected key: input_image_bgra)",
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            transform=ax_img.transAxes,
        )
        ax_img.set_xticks([])
        ax_img.set_yticks([])
    else:
        ax_img.imshow(image)
        ax_img.axis("off")
    ax_img.set_title("Image from the SVO")

    if pc_compare_error is None:
        plot_comparison(
            ax=ax_pc,
            input_points=input_points,
            tilt_points=tilt_compare_points,
            max_points=max_points,
            plane=pc_plane,
            title="original points vs tilt-compensated",
        )
    else:
        ax_pc.set_facecolor("#111111")
        ax_pc.text(
            0.5,
            0.5,
            f"pointcloud comparison unavailable\\n{pc_compare_error}",
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            transform=ax_pc.transAxes,
        )
        ax_pc.set_xticks([])
        ax_pc.set_yticks([])
        ax_pc.set_title("original points vs tilt-compensated")

    polar_mesh = draw_polar(
        ax=ax_polar,
        polar_grid=polar_grid,
        r_edges=r_edges,
        theta_edges=theta_edges,
        title="Traversability - Polar Bin",
    )

    _, z_scatter = draw_cartesian_overlay(
        ax=ax_cart,
        cart_grid=cart_grid,
        x_coords=x_coords,
        y_coords=y_coords,
        tilt_points=tilt_points,
        max_points=max_points,
        show_pc_overlay=show_pc_overlay,
        title="Traversability - Cartesian Grid",
    )

    fig.colorbar(
        polar_mesh,
        ax=ax_polar,
        orientation="horizontal",
        pad=0.18,
        fraction=0.08,
        label="Danger",
    )
    if z_scatter is not None:
        fig.colorbar(
            z_scatter,
            ax=ax_cart,
            orientation="horizontal",
            pad=0.18,
            fraction=0.08,
            label="Point Z (m)",
        )

    frame_id = _frame_id_from_name(frame_path)
    folder_name = frame_path.parent.name
    fig.suptitle(f"frame_{frame_id:05d} from {folder_name}", fontsize=14, fontweight="bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    npz_dir = args.npz_dir.expanduser().resolve()
    if not npz_dir.is_dir():
        raise NotADirectoryError(f"Expected folder of NPZ files: {npz_dir}")

    frames = sorted_frame_paths(npz_dir)
    if not frames:
        raise FileNotFoundError(f"No frame_*.npz files found in {npz_dir}")

    single_mode = args.frame_id is not None or args.frame_pos is not None
    if args.all:
        selected = frames
    else:
        selected = _pick_frame(frames, args.frame_id, args.frame_pos)
        if not single_mode:
            selected = frames

    if args.out is not None and len(selected) != 1:
        raise ValueError("--out can only be used when rendering exactly one frame")

    if args.out_dir:
        out_dir = args.out_dir.expanduser().resolve()
    else:
        project_root = Path(__file__).resolve().parents[1]
        out_dir = project_root / "traversability_frames" / npz_dir.name

    for frame_path in selected:
        if args.out is not None:
            out_path = _with_pc_suffix(
                args.out.expanduser().resolve(), add_pc_suffix=args.no_pc_overlay
            )
        else:
            out_path = _with_pc_suffix(
                out_dir / f"{frame_path.stem}.summary.png",
                add_pc_suffix=args.no_pc_overlay,
            )

        render_frame(
            frame_path=frame_path,
            out_path=out_path,
            xy_res=args.xy_res,
            max_points=args.max_points,
            pc_plane=args.pc_plane,
            show_pc_overlay=not args.no_pc_overlay,
        )
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
