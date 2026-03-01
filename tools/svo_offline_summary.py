#!/usr/bin/env python3
"""Generate a single PNG summary from an SVO via the offline traversability pipeline.

Usage
-----
python3 tools/svo_offline_summary.py <svo_path> [options]

Examples
--------
1) Run pipeline + render first available NPZ frame:
   python3 tools/svo_offline_summary.py /data/run_001.svo2

2) Reuse existing NPZ outputs and render a specific frame id:
   python3 tools/svo_offline_summary.py /data/run_001.svo2 --use-existing --frame-id 24

3) Use YZ side projection and custom output path:
   python3 tools/svo_offline_summary.py /data/run_001.svo2 --side-plane yz --out /tmp/summary.png

Key arguments
-------------
- svo_path (positional): .svo/.svo2 file or a directory containing one.
- --config PATH: pipeline YAML (default: config/pipeline_config.yaml).
- --use-existing: skip running offline pipeline and reuse NPZ files.
- --frame-id N: choose frame_000NN.npz by id.
- --frame-pos N: choose frame by sorted position if --frame-id is not set.
- --side-plane {xz,yz}: side projection for raw/tilt point-cloud panels.
- --max-points N: downsample plotted points for readability/performance.
- --out PATH: output PNG path.

Output
------
A single PNG with 5 panels:
1) SVO image frame
2) Input point cloud side view
3) Tilt-compensated point cloud side view
4) Traversability polar grid
5) Traversability cartesian with tilt_points overlay
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


FRAME_PATTERN = re.compile(r"^frame_(\d+)\.npz$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run/reuse offline SVO traversability outputs and render a 5-panel summary PNG."
        )
    )
    parser.add_argument("svo_path", type=Path, help="Path to .svo/.svo2 or directory containing one.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Pipeline config YAML (default: config/pipeline_config.yaml).",
    )
    parser.add_argument(
        "--use-existing",
        action="store_true",
        help="Reuse existing NPZ outputs instead of running the offline pipeline.",
    )
    parser.add_argument(
        "--frame-id",
        type=int,
        default=None,
        help="Frame id from NPZ name (e.g. frame_00024.npz -> 24). Default: first available.",
    )
    parser.add_argument(
        "--frame-pos",
        type=int,
        default=0,
        help="Position in sorted NPZ list when --frame-id is not set (default: 0).",
    )
    parser.add_argument(
        "--side-plane",
        choices=("xz", "yz"),
        default="xz",
        help="Side projection plane for point clouds (default: xz).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=80000,
        help="Max points to plot per point-cloud panel (default: 80000).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: traversability_frames/<recording>/summary/frame_XXXXX.png).",
    )
    return parser.parse_args()


def _add_pipeline_module_path(project_root: Path) -> None:
    module_dir = project_root / "src" / "traversability" / "python"
    sys.path.insert(0, str(module_dir))


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
        raise ValueError(f"Invalid frame file name: {path.name}")
    return int(match.group(1))


def pick_frame(frame_paths: list[Path], frame_id: int | None, frame_pos: int) -> Path:
    if not frame_paths:
        raise FileNotFoundError("No frame_*.npz files found.")
    if frame_id is not None:
        target = f"frame_{frame_id:05d}.npz"
        for path in frame_paths:
            if path.name == target:
                return path
        raise FileNotFoundError(f"Requested frame id not found: {target}")
    if frame_pos < 0 or frame_pos >= len(frame_paths):
        raise IndexError(
            f"--frame-pos {frame_pos} out of range for {len(frame_paths)} frames."
        )
    return frame_paths[frame_pos]


def _pick_key(data: np.lib.npyio.NpzFile, key: str) -> str:
    if key in data:
        return key
    prefixed = f"traversability_{key}"
    if prefixed in data:
        return prefixed
    raise KeyError(f"{data.filename} missing `{key}` or `{prefixed}`.")


def load_npz_frame(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        if "tilt_points" not in data:
            raise KeyError(
                f"{path} missing `tilt_points`. Enable tilt_compensate.write_output in config."
            )
        out: dict[str, np.ndarray] = {
            "tilt_points": np.asarray(data["tilt_points"], dtype=np.float32),
            "danger_grid": np.asarray(data[_pick_key(data, "danger_grid")], dtype=np.float32),
            "valid_mask": np.asarray(data[_pick_key(data, "valid_mask")], dtype=bool),
            "nontraversable": np.asarray(data[_pick_key(data, "nontraversable")], dtype=bool),
            "r_edges": np.asarray(data[_pick_key(data, "r_edges")], dtype=np.float32),
            "theta_edges": np.asarray(data[_pick_key(data, "theta_edges")], dtype=np.float32),
        }
    return out


def sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    idx = np.random.default_rng(0).choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def side_projection(points: np.ndarray, plane: str) -> tuple[np.ndarray, np.ndarray, str, str]:
    if plane == "xz":
        return points[:, 0], points[:, 2], "X (m)", "Z (m)"
    return points[:, 1], points[:, 2], "Y (m)", "Z (m)"


def fetch_image_and_raw_points(svo_path: Path, target_frame_id: int) -> tuple[np.ndarray, np.ndarray]:
    import pyzed.sl as sl
    import run_svo_pipeline as pipeline

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


def draw_polar(ax: Any, danger_grid: np.ndarray, valid_mask: np.ndarray, nontrav: np.ndarray, r_edges: np.ndarray, theta_edges: np.ndarray) -> None:
    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_over("black")
    norm = Normalize(vmin=0.0, vmax=1.0, clip=False)
    plot_values = np.array(danger_grid, copy=True, dtype=np.float32)
    plot_values[valid_mask & nontrav] = 1.1
    mesh = ax.pcolormesh(theta_edges, r_edges, plot_values, cmap=cmap, norm=norm, shading="auto")
    ax.set_thetamin(float(np.degrees(theta_edges.min())) - 5.0)
    ax.set_thetamax(float(np.degrees(theta_edges.max())) + 5.0)
    ax.set_title("Traversability (Polar)")
    ax.figure.colorbar(mesh, ax=ax, fraction=0.046, pad=0.07, label="Danger")


def draw_cartesian_overlay(ax: Any, nontrav: np.ndarray, r_edges: np.ndarray, theta_edges: np.ndarray, tilt_points: np.ndarray) -> None:
    n_r = nontrav.shape[0]
    first_true = np.argmax(nontrav, axis=0)
    has_true = np.any(nontrav, axis=0)
    row_indices = np.where(has_true, first_true, n_r)

    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    obstacle_ranges = r_edges[row_indices][has_true]
    theta_obstacles = theta_centers[has_true]

    x_raw = obstacle_ranges * np.cos(theta_obstacles)
    y_raw = obstacle_ranges * np.sin(theta_obstacles)
    boundary_x = -y_raw
    boundary_y = x_raw

    pc_x = -tilt_points[:, 1]
    pc_y = tilt_points[:, 0]
    sc = ax.scatter(
        pc_x,
        pc_y,
        c=tilt_points[:, 2],
        cmap="viridis",
        s=2,
        alpha=0.6,
        linewidths=0,
        vmin=-0.3,
        vmax=0.3,
    )
    ax.scatter(boundary_x, boundary_y, s=7, color="tab:red", label="Obstacle boundary", zorder=3)
    ax.scatter([0.0], [0.0], color="tab:blue", s=30, marker="o", label="Robot", zorder=4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(0.2, 0.8)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_title("Traversability (Cartesian + tilt_points)")
    ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Z (m)")


def render_summary(
    image_rgb: np.ndarray,
    raw_points: np.ndarray,
    tilt_points: np.ndarray,
    danger_grid: np.ndarray,
    valid_mask: np.ndarray,
    nontrav: np.ndarray,
    r_edges: np.ndarray,
    theta_edges: np.ndarray,
    side_plane: str,
    out_path: Path,
    max_points: int,
) -> None:
    raw_points = sample_points(raw_points, max_points)
    tilt_points = sample_points(tilt_points, max_points)

    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image_rgb)
    ax1.axis("off")
    ax1.set_title("SVO Frame (LEFT)")

    ax2 = fig.add_subplot(gs[0, 1])
    sx, sz, sx_label, sz_label = side_projection(raw_points, side_plane)
    ax2.scatter(sx, sz, s=1, c=sz, cmap="viridis", alpha=0.6, linewidths=0)
    ax2.set_xlabel(sx_label)
    ax2.set_ylabel(sz_label)
    ax2.set_xlim(-0.2, 1.0)
    ax2.set_ylim(-0.4, 0.6)
    ax2.set_title(f"Input Point Cloud Side ({side_plane})")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    tx, tz, tx_label, tz_label = side_projection(tilt_points, side_plane)
    ax3.scatter(tx, tz, s=1, c=tz, cmap="viridis", alpha=0.6, linewidths=0)
    ax3.set_xlabel(tx_label)
    ax3.set_ylabel(tz_label)
    ax3.set_xlim(-0.2, 1.0)
    ax3.set_ylim(-0.4, 0.6)
    ax3.set_title(f"Tilt-Compensated Side ({side_plane})")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 0], projection="polar")
    draw_polar(ax4, danger_grid, valid_mask, nontrav, r_edges, theta_edges)

    ax5 = fig.add_subplot(gs[1, 1:])
    draw_cartesian_overlay(ax5, nontrav, r_edges, theta_edges, tilt_points)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    _add_pipeline_module_path(project_root)

    import run_svo_pipeline as pipeline

    svo_path = pipeline.resolve_svo_path(args.svo_path)
    recording_name = svo_path.parent.parent.name
    out_dir = project_root / "temp-offline-outs" / recording_name

    config_path = (
        args.config.expanduser().resolve()
        if args.config is not None
        else project_root / "config" / "pipeline_config.yaml"
    )
    config = pipeline.load_config(config_path)

    if not args.use_existing:
        pipeline.process_svo(project_root, svo_path, config)

    frame_paths = sorted_frame_paths(out_dir)
    if not frame_paths:
        raise FileNotFoundError(
            f"No frame_*.npz files under {out_dir}. "
            "Check write_output flags in config (tilt_compensate and traversability)."
        )

    frame_path = pick_frame(frame_paths, args.frame_id, args.frame_pos)
    frame_id = frame_id_from_name(frame_path)
    npz = load_npz_frame(frame_path)
    image_rgb, raw_points = fetch_image_and_raw_points(svo_path, frame_id)

    output_path = args.out
    if output_path is None:
        output_path = (
            project_root
            / "traversability_frames"
            / recording_name
            / "summary"
            / f"{frame_path.stem}.png"
        )

    render_summary(
        image_rgb=image_rgb,
        raw_points=raw_points,
        tilt_points=npz["tilt_points"],
        danger_grid=npz["danger_grid"],
        valid_mask=npz["valid_mask"],
        nontrav=npz["nontraversable"],
        r_edges=npz["r_edges"],
        theta_edges=npz["theta_edges"],
        side_plane=args.side_plane,
        out_path=output_path,
        max_points=args.max_points,
    )

    print(f"Wrote {output_path}")
    print(f"Used NPZ {frame_path}")


if __name__ == "__main__":
    main()
