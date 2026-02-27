#!/usr/bin/env python3
"""Batch-render traversability pipeline BIN frames to PNG images."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

FRAME_PATTERN = re.compile(r"^frame_(\d+)\.bin$")


class BinFormatError(ValueError):
    """Raised when a BIN frame payload is malformed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render traversability BIN frames as PNGs.")
    parser.add_argument(
        "folder_name",
        help=(
            "Input folder name under --base-dir (e.g., a recording/run id directory)."
        ),
    )
    parser.add_argument(
        "--base-dir",
        default="temp-offline-outs/traversability",
        help=(
            "Base directory containing frame_*.bin files or subfolders "
            "(default: temp-offline-outs/traversability)."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("auto", "traversability", "matrix"),
        default="auto",
        help="BIN format: auto-detect, traversability, or matrix (default: auto).",
    )
    parser.add_argument(
        "--view",
        choices=("2d", "polar"),
        default="2d",
        help="Visualization mode for traversability frames (default: 2d).",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=1.0,
        help="Marker size for matrix point-cloud visualization (default: 1.0).",
    )
    return parser.parse_args()


def sorted_frame_paths(input_dir: Path) -> list[Path]:
    frames: list[tuple[int, Path]] = []
    for p in input_dir.glob("frame_*.bin"):
        m = FRAME_PATTERN.match(p.name)
        if m:
            frames.append((int(m.group(1)), p))
    frames.sort(key=lambda item: item[0])
    return [p for _, p in frames]


def _read_i32(blob: bytes, offset: int) -> tuple[int, int]:
    if offset + 4 > len(blob):
        raise BinFormatError("Unexpected end of file while reading int32.")
    return int(np.frombuffer(blob, dtype=np.int32, count=1, offset=offset)[0]), offset + 4


def _read_f32_vec(blob: bytes, offset: int, count: int) -> tuple[np.ndarray, int]:
    nbytes = 4 * count
    if count < 0 or offset + nbytes > len(blob):
        raise BinFormatError("Unexpected end of file while reading float32 vector.")
    vec = np.frombuffer(blob, dtype=np.float32, count=count, offset=offset).copy()
    return vec, offset + nbytes


def _read_u8_vec(blob: bytes, offset: int, count: int) -> tuple[np.ndarray, int]:
    if count < 0 or offset + count > len(blob):
        raise BinFormatError("Unexpected end of file while reading uint8 vector.")
    vec = np.frombuffer(blob, dtype=np.uint8, count=count, offset=offset).copy()
    return vec, offset + count


def _read_matrix_f32(blob: bytes, offset: int) -> tuple[np.ndarray, int]:
    rows, offset = _read_i32(blob, offset)
    cols, offset = _read_i32(blob, offset)
    if rows < 0 or cols < 0:
        raise BinFormatError("Negative matrix shape is invalid.")
    flat, offset = _read_f32_vec(blob, offset, rows * cols)
    return flat.reshape(rows, cols), offset


def _read_mask(blob: bytes, offset: int) -> tuple[np.ndarray, int]:
    rows, offset = _read_i32(blob, offset)
    cols, offset = _read_i32(blob, offset)
    if rows < 0 or cols < 0:
        raise BinFormatError("Negative mask shape is invalid.")
    flat, offset = _read_u8_vec(blob, offset, rows * cols)
    return flat.reshape(rows, cols).astype(bool), offset


def _read_vec_f32_with_size(blob: bytes, offset: int) -> tuple[np.ndarray, int]:
    size, offset = _read_i32(blob, offset)
    if size < 0:
        raise BinFormatError("Negative vector size is invalid.")
    return _read_f32_vec(blob, offset, size)


def load_matrix_bin(path: Path) -> np.ndarray:
    blob = path.read_bytes()
    matrix, offset = _read_matrix_f32(blob, 0)
    if offset != len(blob):
        raise BinFormatError(f"Extra bytes in matrix BIN frame: {path}")
    return matrix


def load_traversability_bin(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    blob = path.read_bytes()
    offset = 0

    danger_grid, offset = _read_matrix_f32(blob, offset)
    valid_mask, offset = _read_mask(blob, offset)
    nontraversable, offset = _read_mask(blob, offset)
    r_edges, offset = _read_vec_f32_with_size(blob, offset)
    theta_edges, offset = _read_vec_f32_with_size(blob, offset)

    if offset != len(blob):
        raise BinFormatError(f"Extra bytes in traversability BIN frame: {path}")

    if danger_grid.shape != valid_mask.shape or danger_grid.shape != nontraversable.shape:
        raise BinFormatError(f"Inconsistent grid/mask shapes in {path}")
    if danger_grid.shape != (r_edges.size - 1, theta_edges.size - 1):
        raise BinFormatError(f"Grid/edge shape mismatch in {path}")

    return danger_grid, valid_mask, nontraversable, r_edges, theta_edges


def render_traversability(
    frame_path: Path,
    out_path: Path,
    view: str,
    cmap: plt.Colormap,
    norm: Normalize,
) -> None:
    danger_grid, valid_mask, nontraversable, r_edges, theta_edges = load_traversability_bin(frame_path)

    plot_values = np.array(danger_grid, copy=True, dtype=np.float32)
    plot_values[valid_mask & nontraversable] = 1.1

    fig = plt.figure(figsize=(9, 7))
    if view == "polar":
        ax = fig.add_subplot(111, projection="polar")
        mesh = ax.pcolormesh(theta_edges, r_edges, plot_values, cmap=cmap, norm=norm, shading="auto")
        ax.set_thetamin(float(np.degrees(theta_edges.min())) - 5.0)
        ax.set_thetamax(float(np.degrees(theta_edges.max())) + 5.0)
        title_view = "Polar View"
    else:
        ax = fig.add_subplot(111)
        theta_edges_deg = np.degrees(theta_edges)
        mesh = ax.pcolormesh(theta_edges_deg, r_edges, plot_values, cmap=cmap, norm=norm, shading="auto")
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
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_matrix(frame_path: Path, out_path: Path, point_size: float) -> None:
    matrix = load_matrix_bin(frame_path)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111)

    if matrix.size == 0:
        ax.set_title(f"{frame_path.name} | Empty Matrix")
    elif matrix.ndim == 2 and matrix.shape[1] >= 3:
        sc = ax.scatter(
            matrix[:, 0],
            matrix[:, 1],
            c=matrix[:, 2],
            s=point_size,
            cmap="viridis",
            linewidths=0,
            alpha=0.8,
        )
        fig.colorbar(sc, ax=ax, label="Z")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"{frame_path.name} | XY scatter (color=Z)")
        ax.set_aspect("equal", adjustable="box")
    else:
        mesh = ax.imshow(matrix, cmap="viridis", aspect="auto", interpolation="nearest")
        fig.colorbar(mesh, ax=ax, label="Value")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_title(f"{frame_path.name} | Matrix Heatmap")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def is_traversability_bin(path: Path) -> bool:
    try:
        load_traversability_bin(path)
        return True
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    base_dir = Path(args.base_dir)
    if not base_dir.is_absolute():
        base_dir = project_root / base_dir

    input_dir = base_dir / args.folder_name
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    frame_paths = sorted_frame_paths(input_dir)
    if not frame_paths:
        raise FileNotFoundError(f"No frame_*.bin files found in: {input_dir}")

    selected_format = args.format
    if selected_format == "auto":
        selected_format = "traversability" if is_traversability_bin(frame_paths[0]) else "matrix"

    output_dir = project_root / "traversability_frames" / args.folder_name / selected_format
    if selected_format == "traversability":
        output_dir = output_dir / args.view
    output_dir.mkdir(parents=True, exist_ok=True)

    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_over("black")
    norm = Normalize(vmin=0.0, vmax=1.0, clip=False)

    total = len(frame_paths)
    for idx, frame_path in enumerate(frame_paths, start=1):
        print(f"Processing frame {idx:04d}/{total:04d} ({selected_format})")
        out_path = output_dir / f"{frame_path.stem}.png"
        if selected_format == "traversability":
            render_traversability(frame_path, out_path, args.view, cmap, norm)
        else:
            render_matrix(frame_path, out_path, args.point_size)


if __name__ == "__main__":
    main()
