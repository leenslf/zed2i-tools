#!/usr/bin/env python3
"""Interactive height_map viewer — single NPZ file or folder of frames.

Usage:
    python tools/height_map_viewer.py temp-offline-outs/ROLL-NEG1/
    python tools/height_map_viewer.py temp-offline-outs/ROLL-NEG1/frame_00002.npz

Keyboard navigation (folder mode):
    Left / Right arrow  — previous / next frame
    Home / End          — first / last frame
    Q                   — quit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent))
from travviz_common import draw_polar, load_npz_frame, sorted_frame_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive height_map viewer for NPZ frames.")
    parser.add_argument(
        "path",
        type=Path,
        help="Single frame_*.npz file or folder containing frame_*.npz files.",
    )
    return parser.parse_args()


def _load_height_map(frame_path: Path):
    """Return (height_map, r_edges, theta_edges) for the given frame."""
    _, r_edges, theta_edges, height_map, _ = load_npz_frame(frame_path)
    return height_map, r_edges, theta_edges


def _update(ax, cbar_ax, frame_path: Path, fig: plt.Figure) -> None:
    height_map, r_edges, theta_edges = _load_height_map(frame_path)

    ax.cla()
    mesh = draw_polar(
        ax=ax,
        polar_grid=height_map,
        r_edges=r_edges,
        theta_edges=theta_edges,
        title="Height Map (polar)",
    )
    ax.invert_xaxis()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{-x:.4g}"))

    cbar_ax.cla()
    fig.colorbar(mesh, cax=cbar_ax, label="Height (m)")

    frame_id = int(frame_path.stem.split("_")[1]) if "_" in frame_path.stem else 0
    fig.suptitle(
        f"{frame_path.stem}  |  {frame_path.parent.name}",
        fontsize=12,
        fontweight="bold",
    )
    fig.canvas.draw_idle()


def view_single(frame_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=False)
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])

    _update(ax, cbar_ax, frame_path, fig)
    plt.show()


def view_folder(frames: list[Path]) -> None:
    state = {"idx": 0}

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=False)
    fig.subplots_adjust(right=0.85, bottom=0.12)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])

    hint = fig.text(
        0.5, 0.02,
        "← → navigate  |  Home/End first/last  |  Q quit",
        ha="center", va="bottom", fontsize=8, color="gray",
    )

    def refresh():
        _update(ax, cbar_ax, frames[state["idx"]], fig)
        fig.canvas.set_window_title(
            f"[{state['idx'] + 1}/{len(frames)}] {frames[state['idx']].name}"
        )

    def on_key(event):
        idx = state["idx"]
        if event.key in ("right", "d"):
            idx = min(idx + 1, len(frames) - 1)
        elif event.key in ("left", "a"):
            idx = max(idx - 1, 0)
        elif event.key == "home":
            idx = 0
        elif event.key == "end":
            idx = len(frames) - 1
        elif event.key in ("q", "Q"):
            plt.close(fig)
            return
        if idx != state["idx"]:
            state["idx"] = idx
            refresh()

    fig.canvas.mpl_connect("key_press_event", on_key)
    refresh()
    plt.show()


def main() -> None:
    args = parse_args()
    path = args.path.expanduser().resolve()

    if path.is_file():
        view_single(path)
    elif path.is_dir():
        frames = sorted_frame_paths(path)
        if not frames:
            sys.exit(f"No frame_*.npz files found in {path}")
        view_folder(frames)
    else:
        sys.exit(f"Path not found: {path}")


if __name__ == "__main__":
    main()
