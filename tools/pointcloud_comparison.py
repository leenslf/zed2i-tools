#!/usr/bin/env python3
"""Compare input and tilt-compensated pointclouds for a frame.

Supported inputs:
- SVO/SVO2 path (or directory containing one):
  - resolves NPZ folder as temp-offline-outs/<recording_name>
  - reads input cloud from SVO frame
  - reads tilt cloud from matching NPZ frame
- NPZ file/folder:
  - reads both clouds from NPZ keys (input + tilt)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

FRAME_PATTERN = re.compile(r"^frame_(\d+)\.npz$")
INPUT_KEYS = (
    "input_points",
)
TILT_KEYS = ("tilt_points",)
PLANE_AXES: dict[str, tuple[int, int, str, str, str]] = {
    "xy": (0, 1, "X (m)", "Y (m)", "XY"),
    "xz": (0, 2, "X (m)", "Z (m)", "XZ"),
    "yz": (1, 2, "Y (m)", "Z (m)", "YZ"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot dense input-vs-tilt pointcloud comparison from SVO+NPZ or NPZ only."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="SVO/SVO2 path, recording directory, NPZ file, or folder containing frame_*.npz.",
    )
    parser.add_argument(
        "--npz-dir",
        type=Path,
        default=None,
        help="Override NPZ directory (useful with SVO input).",
    )
    parser.add_argument("--frame-id", type=int, default=None, help="Frame id to pick from folder.")
    parser.add_argument("--frame-pos", type=int, default=0, help="Frame position when --frame-id unset.")
    parser.add_argument("--max-points", type=int, default=100000, help="Max points per cloud to render.")
    parser.add_argument(
        "--plane",
        "--pc-plane",
        dest="plane",
        choices=("xy", "xz", "yz"),
        default="xy",
        help="Projection plane for plotting (default: xy).",
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


def frame_id_from_name(path: Path) -> int:
    m = FRAME_PATTERN.match(path.name)
    if not m:
        raise ValueError(f"Invalid frame filename: {path.name}")
    return int(m.group(1))


def _pick_first_key(data: np.lib.npyio.NpzFile, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key in data:
            return key
    return None


def _as_valid_xyz(arr: np.ndarray) -> np.ndarray:
    points = np.asarray(arr, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected Nx3 array, got shape={points.shape}")
    points = points[:, :3]
    valid = np.isfinite(points).all(axis=1)
    return points[valid]


def load_tilt_from_npz(path: Path) -> np.ndarray:
    with np.load(path) as data:
        tilt_key = _pick_first_key(data, TILT_KEYS)
        if tilt_key is None:
            raise KeyError(f"{path} missing any tilt key: {', '.join(TILT_KEYS)}")
        return _as_valid_xyz(data[tilt_key])


def load_clouds_from_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        if "input_points" not in data:
            raise KeyError(f"{path} missing `input_points`")
        if "tilt_points" not in data:
            raise KeyError(f"{path} missing `tilt_points`")
        input_points = _as_valid_xyz(data["input_points"])
        tilt_points = _as_valid_xyz(data["tilt_points"])
    return input_points, tilt_points


def _add_pipeline_module_path(project_root: Path) -> None:
    module_dir = project_root / "src" / "traversability" / "python"
    sys.path.insert(0, str(module_dir))


def _try_resolve_svo(input_path: Path) -> tuple[Path | None, Any | None]:
    project_root = Path(__file__).resolve().parents[1]
    _add_pipeline_module_path(project_root)

    import run_svo_pipeline as pipeline

    try:
        return pipeline.resolve_svo_path(input_path), pipeline
    except Exception:
        return None, pipeline


def fetch_image_and_raw_points(svo_path: Path, target_frame_id: int, pipeline: Any) -> tuple[np.ndarray, np.ndarray]:
    import pyzed.sl as sl

    zed = pipeline.open_svo(svo_path)
    image = sl.Mat()
    point_cloud = sl.Mat()
    frame_index = 0

    try:
        while True:
            err = zed.grab()
            if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            if err != sl.ERROR_CODE.SUCCESS:
                frame_index += 1
                continue

            if frame_index == target_frame_id:
                zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU)
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZ, sl.MEM.CPU)

                image_bgra = image.get_data()
                image_rgb = image_bgra[:, :, :3][:, :, ::-1].copy()

                pc_data = point_cloud.get_data()[:, :, :3].reshape(-1, 3)
                valid = np.isfinite(pc_data).all(axis=1)
                raw_points = pc_data[valid].astype(np.float32, copy=False)
                return image_rgb, raw_points

            frame_index += 1
    finally:
        zed.close()

    raise FileNotFoundError(
        f"Could not read SVO frame {target_frame_id}. "
        "This may happen if NPZ frame IDs do not align with the source recording."
    )


def sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    idx = np.random.default_rng(0).choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def project_points(points: np.ndarray, plane: str) -> tuple[np.ndarray, np.ndarray, str, str, str]:
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

    input_plot = sample_points(input_points, max_points)
    tilt_plot = sample_points(tilt_points, max_points)
    input_u, input_v, xlabel, ylabel, plane_name = project_points(input_plot, plane)
    tilt_u, tilt_v, _, _, _ = project_points(tilt_plot, plane)

    ax.scatter(
        input_u,
        input_v,
        s=1.0,
        c=input_color,
        alpha=0.35,
        linewidths=0,
    )
    ax.scatter(
        tilt_u,
        tilt_v,
        s=1.0,
        c=tilt_color,
        alpha=0.35,
        linewidths=0,
    )

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
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=input_color, markeredgecolor=input_color, markersize=6, label=f"input"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=tilt_color, markeredgecolor=tilt_color, markersize=6, label=f"tilt-compensated"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    return {
        "input": int(input_plot.shape[0]),
        "tilt": int(tilt_plot.shape[0]),
    }


def resolve_npz_root(input_path: Path, npz_override: Path | None) -> tuple[Path, Path | None, Any | None]:
    project_root = Path(__file__).resolve().parents[1]
    if npz_override is not None:
        npz_root = npz_override.expanduser().resolve()
        if not npz_root.exists():
            raise FileNotFoundError(f"NPZ path not found: {npz_root}")
        svo_path, pipeline = _try_resolve_svo(input_path)
        return npz_root, svo_path, pipeline

    svo_path, pipeline = _try_resolve_svo(input_path)
    if svo_path is not None:
        recording_name = svo_path.parent.parent.name
        npz_root = project_root / "temp-offline-outs" / recording_name
        if not npz_root.exists():
            raise FileNotFoundError(
                f"Expected NPZ folder not found for {svo_path}: {npz_root}"
            )
        return npz_root, svo_path, pipeline

    npz_root = input_path.expanduser().resolve()
    if not npz_root.exists():
        raise FileNotFoundError(f"NPZ path not found: {npz_root}")
    return npz_root, None, None


def main() -> None:
    args = parse_args()
    npz_root, _, _ = resolve_npz_root(args.input_path, args.npz_dir)

    frame_path = pick_frame(npz_root, args.frame_id, args.frame_pos)
    frame_id = frame_id_from_name(frame_path)
    input_points, tilt_points = load_clouds_from_npz(frame_path)

    fig, ax = plt.subplots(figsize=(8, 7))
    stats = plot_comparison(
        ax=ax,
        input_points=input_points,
        tilt_points=tilt_points,
        max_points=args.max_points,
        plane=args.plane,
    )

    if args.out is None:
        args.out = frame_path.with_suffix(".pointcloud_comparison.png")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(
        f"Saved -> {args.out} | frame={frame_id} | input={stats['input']} tilt={stats['tilt']}"
    )
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
