#!/usr/bin/env python3
"""Interactive 3-D point-cloud viewer for tilt_points / input_points in NPZ frames.

Uses Open3D when available (best interactivity), falls back to matplotlib 3-D.

Usage:
    python tools/pointcloud_viewer.py temp-offline-outs/ROLL-NEG1/
    python tools/pointcloud_viewer.py temp-offline-outs/ROLL-NEG1/frame_00002.npz

    # Show both raw and tilt-compensated clouds side-by-side
    python tools/pointcloud_viewer.py temp-offline-outs/ROLL-NEG1/ --both

    # Colour by height instead of cloud source
    python tools/pointcloud_viewer.py temp-offline-outs/ROLL-NEG1/ --color-by height

Keyboard (folder mode, matplotlib backend):
    Left / Right arrow  — previous / next frame
    Home / End          — first / last frame
    T                   — toggle input_points visibility
    Q                   — quit

Keyboard (Open3D backend):
    Left / Right arrow  — previous / next frame
    T                   — toggle input_points
    Q / Escape          — quit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from travviz_common import load_npz_frame, sorted_frame_paths

# ---------------------------------------------------------------------------
# NPZ loading
# ---------------------------------------------------------------------------

def _as_valid_xyz(arr: np.ndarray) -> np.ndarray:
    pts = np.asarray(arr, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return np.zeros((0, 3), dtype=np.float32)
    pts = pts[:, :3]
    return pts[np.isfinite(pts).all(axis=1)]


def load_clouds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (input_points, tilt_points) as Nx3 XYZ float32 arrays."""
    with np.load(path) as data:
        tilt = _as_valid_xyz(data["tilt_points"]) if "tilt_points" in data else np.zeros((0, 3), np.float32)
        inp = _as_valid_xyz(data["input_points"]) if "input_points" in data else np.zeros((0, 3), np.float32)
    return inp, tilt


def _subsample(pts: np.ndarray, max_pts: int, seed: int = 0) -> np.ndarray:
    if max_pts > 0 and pts.shape[0] > max_pts:
        idx = np.random.default_rng(seed).choice(pts.shape[0], size=max_pts, replace=False)
        return pts[idx]
    return pts


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _z_colors(pts: np.ndarray, cmap_name: str = "viridis") -> np.ndarray:
    """Return Nx3 RGB colours mapped from Z height, in [0,1] float64."""
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(cmap_name)
    z = pts[:, 2]
    lo, hi = np.nanpercentile(z, [2.0, 98.0]) if z.size else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1e-3
    norm_z = np.clip((z - lo) / (hi - lo), 0.0, 1.0)
    return cmap(norm_z)[:, :3]  # drop alpha


# ---------------------------------------------------------------------------
# Open3D scene decorations (grid + axis labels)
# ---------------------------------------------------------------------------

def _make_decorations(o3d, step: float = 0.5):
    """Return static scene geometry: coordinate frame, XY grid, and tick labels."""
    geoms = []

    # Coordinate frame at origin
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25))

    # Grid bounds (robot FOV)
    x_lo, x_hi = -0.25, 3.0
    y_lo, y_hi = -1.5, 1.5
    z_lo, z_hi = -1.0, 0.5

    x_vals = np.arange(np.ceil(x_lo / step) * step, x_hi + 1e-6, step)
    y_vals = np.arange(np.ceil(y_lo / step) * step, y_hi + 1e-6, step)
    z_vals = np.arange(np.ceil(z_lo / step) * step, z_hi + 1e-6, step)

    # XY ground grid at Z=0
    pts, lns = [], []
    for xv in x_vals:
        i = len(pts); pts += [[xv, y_lo, 0.0], [xv, y_hi, 0.0]]; lns.append([i, i + 1])
    for yv in y_vals:
        i = len(pts); pts += [[x_lo, yv, 0.0], [x_hi, yv, 0.0]]; lns.append([i, i + 1])
    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(np.array(pts, dtype=np.float64))
    grid.lines  = o3d.utility.Vector2iVector(np.array(lns))
    grid.paint_uniform_color([0.55, 0.55, 0.55])
    geoms.append(grid)

    # Z-axis ruler line
    z_ruler = o3d.geometry.LineSet()
    z_ruler.points = o3d.utility.Vector3dVector([[0.0, 0.0, z_lo], [0.0, 0.0, z_hi]])
    z_ruler.lines  = o3d.utility.Vector2iVector([[0, 1]])
    z_ruler.paint_uniform_color([0.15, 0.65, 0.15])
    geoms.append(z_ruler)

    # Tick-mark stubs along X and Y on the grid
    tick_len = 0.05
    tick_pts, tick_lns = [], []
    for xv in x_vals:
        i = len(tick_pts)
        tick_pts += [[xv, 0.0, 0.0], [xv, -tick_len, 0.0]]
        tick_lns.append([i, i + 1])
    for yv in y_vals:
        i = len(tick_pts)
        tick_pts += [[0.0, yv, 0.0], [-tick_len, yv, 0.0]]
        tick_lns.append([i, i + 1])
    for zv in z_vals:
        i = len(tick_pts)
        tick_pts += [[0.0, 0.0, zv], [-tick_len, 0.0, zv]]
        tick_lns.append([i, i + 1])
    ticks = o3d.geometry.LineSet()
    ticks.points = o3d.utility.Vector3dVector(np.array(tick_pts, dtype=np.float64))
    ticks.lines  = o3d.utility.Vector2iVector(np.array(tick_lns))
    ticks.paint_uniform_color([0.85, 0.85, 0.85])
    geoms.append(ticks)

    # Text labels via o3d.t (open3d >= 0.17); silently skip if unavailable
    txt_scale = 0.04
    try:
        R_upright = o3d.geometry.get_rotation_matrix_from_xyz(
            [np.pi / 2, 0.0, 0.0]
        )  # rotate text from XY plane into XZ plane (standing upright)

        def _label(text: str, pos, color, upright: bool = False):
            m = o3d.t.geometry.TriangleMesh.create_text(text, depth=0.002).to_legacy()
            m.scale(txt_scale, center=[0.0, 0.0, 0.0])
            if upright:
                m.rotate(R_upright, center=[0.0, 0.0, 0.0])
            m.translate(pos)
            m.paint_uniform_color(color)
            return m

        for xv in x_vals:
            geoms.append(_label(f"{xv:.1f}", [xv - 0.04, y_lo - 0.15, 0.0], [0.95, 0.35, 0.35]))
        for yv in y_vals:
            geoms.append(_label(f"{yv:.1f}", [x_lo - 0.20, yv - 0.02, 0.0], [0.35, 0.85, 0.35]))
        for zv in z_vals:
            geoms.append(_label(f"{zv:.1f}", [-0.22, 0.0, zv - 0.02], [0.35, 0.55, 0.95], upright=True))
    except Exception as exc:
        print(f"[pointcloud_viewer] 3-D text labels skipped ({exc})")

    return geoms


# ---------------------------------------------------------------------------
# Open3D backend
# ---------------------------------------------------------------------------

def _try_open3d_viewer(
    frames: list[Path],
    max_pts: int,
    show_input: bool,
    color_by: str,
) -> bool:
    """Attempt to launch Open3D viewer. Returns True on success, False if o3d not available."""
    try:
        import open3d as o3d
    except ImportError:
        return False

    INPUT_COLOR = np.array([0.12, 0.47, 0.71])   # muted blue
    TILT_COLOR  = np.array([1.00, 0.50, 0.05])   # orange

    state = {"idx": 0, "show_input": show_input}

    def _make_pcd(pts: np.ndarray, default_color: np.ndarray) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        if color_by == "height":
            cols = _z_colors(pts)
        else:
            cols = np.tile(default_color, (len(pts), 1))
        pcd.colors = o3d.utility.Vector3dVector(cols)
        return pcd

    def _build_geometry(idx: int) -> list[o3d.geometry.PointCloud]:
        inp, tilt = load_clouds(frames[idx])
        inp = _subsample(inp, max_pts, seed=1)
        tilt = _subsample(tilt, max_pts, seed=0)
        geoms = [_make_pcd(tilt, TILT_COLOR)]
        if state["show_input"] and inp.size:
            geoms.append(_make_pcd(inp, INPUT_COLOR))
        return geoms

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Point-cloud viewer", width=1280, height=720)

    decorations = _make_decorations(o3d)

    def _refresh():
        vis.clear_geometries()
        for d in decorations:
            vis.add_geometry(d)
        for g in _build_geometry(state["idx"]):
            vis.add_geometry(g)
        n = len(frames)
        vis.get_render_option()  # no-op, just ensure vis is alive
        print(
            f"[{state['idx']+1}/{n}] {frames[state['idx']].name}"
            f"  |  input={'on' if state['show_input'] else 'off'}"
        )
        vis.update_renderer()

    def _next(vis):
        if state["idx"] < len(frames) - 1:
            state["idx"] += 1
            _refresh()

    def _prev(vis):
        if state["idx"] > 0:
            state["idx"] -= 1
            _refresh()

    def _toggle_input(vis):
        state["show_input"] = not state["show_input"]
        _refresh()

    # Arrow keys: 262=right, 263=left
    vis.register_key_callback(262, _next)
    vis.register_key_callback(263, _prev)
    vis.register_key_callback(ord("T"), _toggle_input)

    _refresh()
    vis.run()
    vis.destroy_window()
    return True


# ---------------------------------------------------------------------------
# Matplotlib 3-D backend
# ---------------------------------------------------------------------------

def _mpl_viewer(
    frames: list[Path],
    max_pts: int,
    show_input: bool,
    color_by: str,
) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers projection

    INPUT_COLOR = "#1f77b4"
    TILT_COLOR  = "#ff7f0e"

    state = {"idx": 0, "show_input": show_input}
    fig = plt.figure(figsize=(10, 8))
    ax: plt.Axes = fig.add_subplot(111, projection="3d")

    hint = fig.text(
        0.5, 0.01,
        "← → navigate  |  T toggle input  |  Home/End first/last  |  Q quit  |  drag to rotate",
        ha="center", va="bottom", fontsize=8, color="gray",
    )

    def _refresh():
        ax.cla()
        inp, tilt = load_clouds(frames[state["idx"]])
        inp  = _subsample(inp,  max_pts, seed=1)
        tilt = _subsample(tilt, max_pts, seed=0)

        if color_by == "height":
            if tilt.size:
                c_tilt = _z_colors(tilt, "plasma")
                ax.scatter(tilt[:, 0], tilt[:, 1], tilt[:, 2],
                           s=0.5, c=c_tilt, alpha=0.4, linewidths=0, label="tilt-compensated")
            if state["show_input"] and inp.size:
                c_inp = _z_colors(inp, "viridis")
                ax.scatter(inp[:, 0], inp[:, 1], inp[:, 2],
                           s=0.5, c=c_inp, alpha=0.3, linewidths=0, label="input")
        else:
            if tilt.size:
                ax.scatter(tilt[:, 0], tilt[:, 1], tilt[:, 2],
                           s=0.5, c=TILT_COLOR, alpha=0.4, linewidths=0, label="tilt-compensated")
            if state["show_input"] and inp.size:
                ax.scatter(inp[:, 0], inp[:, 1], inp[:, 2],
                           s=0.5, c=INPUT_COLOR, alpha=0.3, linewidths=0, label="input")

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend(loc="upper left", markerscale=6, fontsize=8)

        n = len(frames)
        fig.suptitle(
            f"[{state['idx']+1}/{n}] {frames[state['idx']].name}",
            fontsize=11, fontweight="bold",
        )
        fig.canvas.draw_idle()

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
        elif event.key in ("t", "T"):
            state["show_input"] = not state["show_input"]
            _refresh()
            return
        elif event.key in ("q", "Q"):
            plt.close(fig)
            return
        if idx != state["idx"]:
            state["idx"] = idx
            _refresh()

    fig.canvas.mpl_connect("key_press_event", on_key)
    _refresh()
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive 3-D point-cloud viewer for NPZ traversability frames."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Single frame_*.npz file or folder of frame_*.npz files.",
    )
    parser.add_argument(
        "--frame-id",
        type=int,
        default=None,
        help="Start at this frame id (folder mode).",
    )
    parser.add_argument(
        "--frame-pos",
        type=int,
        default=0,
        help="Start at this frame index (default: 0).",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        default=False,
        help="Show both input_points and tilt_points (default: tilt only).",
    )
    parser.add_argument(
        "--color-by",
        choices=("source", "height"),
        default="source",
        help="Colour points by cloud source (default) or Z height.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=200_000,
        help="Max points per cloud (0 = no limit).",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "open3d", "matplotlib"),
        default="auto",
        help="Rendering backend (default: auto tries open3d first).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = args.path.expanduser().resolve()

    if path.is_file():
        frames = [path]
        start_idx = 0
    elif path.is_dir():
        frames = sorted_frame_paths(path)
        if not frames:
            sys.exit(f"No frame_*.npz files found in {path}")
        if args.frame_id is not None:
            target = f"frame_{args.frame_id:05d}.npz"
            names = [f.name for f in frames]
            if target not in names:
                sys.exit(f"Frame not found: {target}")
            start_idx = names.index(target)
        else:
            start_idx = max(0, min(args.frame_pos, len(frames) - 1))
    else:
        sys.exit(f"Path not found: {path}")

    # Rotate frame list so navigation starts at the chosen frame
    frames = frames[start_idx:] + frames[:start_idx]

    kwargs = dict(
        frames=frames,
        max_pts=args.max_points,
        show_input=args.both,
        color_by=args.color_by,
    )

    if args.backend == "open3d":
        if not _try_open3d_viewer(**kwargs):
            sys.exit("open3d not available. Install it with: pip install open3d")
    elif args.backend == "matplotlib":
        _mpl_viewer(**kwargs)
    else:  # auto
        if not _try_open3d_viewer(**kwargs):
            print("open3d not found, falling back to matplotlib 3-D viewer.")
            _mpl_viewer(**kwargs)


if __name__ == "__main__":
    main()
