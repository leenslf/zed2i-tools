#!/usr/bin/env python3
"""Live Cartesian traversability viewer.

Reads frames from an SVO2 recording or a live ZED camera, runs the full
traversability pipeline in-memory, and displays the result as a Cartesian
green/red/gray grid updated in real time via OpenCV.

RUN (SVO):  python3 src/traversability/python/live_trav_viewer.py --svo <PATH>
RUN (live): python3 src/traversability/python/live_trav_viewer.py

Press 'q' or ESC to quit.
"""

from __future__ import annotations

import argparse
import logging
import queue
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.cm as mcm
import numpy as np
import pyzed.sl as sl

from polarize import polarize
from run_svo_pipeline import DEFAULT_CONFIG, load_config, resolve_svo_path
from tilt_compensate import tilt_compensate
from traversability import compute_traversability
from voxel_filter import voxel_filter

# travviz_common lives in tools/
_TOOLS_DIR = Path(__file__).resolve().parents[3] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))
from travviz_common import rasterize_polar_to_cartesian  # noqa: E402  # type: ignore[import]

LOGGER = logging.getLogger("live_trav_viewer")
WINDOW_NAME = "Traversability"

# BGR equivalents of the matplotlib colours
_BGR_TRAVERSABLE = (80, 175, 76)      # #4CAF50 green
_BGR_NONTRAVERSABLE = (54, 67, 244)   # #F44336 red
_BGR_UNKNOWN = (0x88, 0x88, 0x88)     # gray

# Pixels per cell when scaling up the raster for display
_DISPLAY_SCALE = 10


@dataclass
class PipelineResult:
    trav_grid: np.ndarray
    height_map: np.ndarray
    r_edges: np.ndarray
    theta_edges: np.ndarray
    frame_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live Cartesian traversability viewer.")
    parser.add_argument(
        "--svo",
        type=Path,
        default=None,
        help="Path to SVO/SVO2 file or directory (omit to use live camera).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Pipeline config YAML (default: <project_root>/config/pipeline_config.yaml).",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=None,
        help="Process every Nth frame (overrides config value).",
    )
    parser.add_argument(
        "--nt-height-overlay",
        action="store_true",
        help="Overlay non-traversable cells with height values (magma colormap).",
    )
    parser.add_argument(
        "--grid-res-m",
        type=float,
        default=0.05,
        help="Cartesian raster resolution in metres (default: 0.05).",
    )
    return parser.parse_args()


def open_camera(svo_path: Path | None) -> sl.Camera:
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD # check this 
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    if svo_path is not None:
        init_params.set_from_svo_file(str(svo_path))
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        source = f"SVO file {svo_path}" if svo_path else "live camera"
        raise RuntimeError(f"Failed to open {source}: {err}")
    zed.enable_positional_tracking(sl.PositionalTrackingParameters())
    return zed


def _render_frame(result: PipelineResult, grid_res_m: float, show_nt_height_overlay: bool) -> np.ndarray:
    """Convert a PipelineResult into a display-ready BGR image."""

    def rasterize(values: np.ndarray, mask: np.ndarray):
        return rasterize_polar_to_cartesian(
            values=values,
            source_valid_mask=mask,
            r_edges=result.r_edges,
            theta_edges=result.theta_edges,
            x_min=0.0,
            x_max=2.0,
            y_min=-0.75,
            y_max=0.75,
            grid_res_m=grid_res_m,
        )

    # Rasterize traversability to Cartesian. cart_grid shape: (Ny, Nx)
    cart_grid, _, _ = rasterize(result.trav_grid, ~np.isnan(result.trav_grid))

    traversable = np.isfinite(cart_grid) & np.isclose(cart_grid, 0.0)
    nontraversable = np.isfinite(cart_grid) & np.isclose(cart_grid, 1.0)

    img = np.full((*cart_grid.shape, 3), _BGR_UNKNOWN, dtype=np.uint8)
    img[traversable] = _BGR_TRAVERSABLE
    img[nontraversable] = _BGR_NONTRAVERSABLE

    # Optional magma overlay on non-traversable cells, coloured by height
    if show_nt_height_overlay:
        nt_mask_polar = (
            np.isfinite(result.trav_grid)
            & np.isclose(result.trav_grid, 1.0)
            & np.isfinite(result.height_map)
        )
        height_cart, _, _ = rasterize(result.height_map, nt_mask_polar)
        valid_h = nontraversable & np.isfinite(height_cart)
        if np.any(valid_h):
            h_lo, h_hi = np.percentile(height_cart[valid_h], [5.0, 95.0])
            if not np.isfinite(h_lo) or not np.isfinite(h_hi) or h_hi <= h_lo:
                h_lo = float(np.nanmin(height_cart[valid_h]))
                h_hi = max(float(np.nanmax(height_cart[valid_h])), h_lo + 1e-3)

            norm_h = np.clip((height_cart - h_lo) / (h_hi - h_lo), 0.0, 1.0)
            # magma returns RGBA float [0,1]; convert to uint8 BGR
            magma_rgb = (mcm.magma(norm_h)[..., :3] * 255).astype(np.uint8)
            magma_bgr = magma_rgb[..., ::-1]

            alpha = 0.65
            base = img[valid_h].astype(np.float32)
            overlay = magma_bgr[valid_h].astype(np.float32)
            img[valid_h] = ((1.0 - alpha) * base + alpha * overlay).astype(np.uint8)

    # Orient for display: forward (X) up, far = top of image.
    # cart_grid axes: 0=Y (lateral), 1=X (forward).
    # Transpose → (Nx, Ny, 3): rows=forward, cols=lateral.
    # flipud → row 0 = far end (x_max), last row = near (x_min).
    display = np.ascontiguousarray(np.flipud(img.transpose(1, 0, 2)))

    # Scale up for visibility
    dh, dw = display.shape[:2]
    display = cv2.resize(
        display,
        (dw * _DISPLAY_SCALE, dh * _DISPLAY_SCALE),
        interpolation=cv2.INTER_NEAREST,
    )

    _draw_legend(display, show_nt_height_overlay)
    cv2.putText(
        display,
        f"frame {result.frame_index}",
        (8, display.shape[0] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return display


def _draw_legend(img: np.ndarray, show_nt_height_overlay: bool) -> None:
    items = [
        ("traversable", _BGR_TRAVERSABLE),
        ("non-traversable", _BGR_NONTRAVERSABLE),
        ("unknown", _BGR_UNKNOWN),
    ]
    if show_nt_height_overlay:
        items.append(("nt height (magma)", (180, 50, 180)))

    x0, y0, swatch, pad = 8, 8, 12, 4
    for label, color in items:
        cv2.rectangle(img, (x0, y0), (x0 + swatch, y0 + swatch), color, -1)
        cv2.putText(
            img,
            label,
            (x0 + swatch + pad, y0 + swatch - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        y0 += swatch + pad


def _pipeline_worker(
    zed: sl.Camera,
    result_q: queue.Queue,
    stop_event: threading.Event,
    voxel_cfg: dict,
    polarize_cfg: dict,
    traversability_cfg: dict,
    frame_skip: int,
) -> None:
    """Grab frames, run the pipeline, and post the latest result to result_q.

    Uses a size-1 drop queue: if the display hasn't consumed the previous
    result yet, it is discarded so the display always sees the freshest frame.
    """
    point_cloud = sl.Mat()
    pose = sl.Pose()
    frame_index = 0

    try:
        while not stop_event.is_set():
            err = zed.grab()
            if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                LOGGER.info("End of SVO file reached.")
                break
            if err != sl.ERROR_CODE.SUCCESS:
                LOGGER.warning("grab() returned %s, skipping.", err)
                frame_index += 1
                continue

            frame_index += 1
            if frame_index % frame_skip != 0:
                continue

            try:
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZ, sl.MEM.CPU)
                zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
                q = pose.get_orientation().get()
                quaternion = np.array([q[0], q[1], q[2], q[3]], dtype=np.float64)

                pc_data = point_cloud.get_data()
                points = pc_data[:, :, :3].reshape(-1, 3)
                points = points[np.isfinite(points).all(axis=1)]

                detilted = tilt_compensate(points, quaternion)
                filtered = voxel_filter(detilted, voxel_cfg)
                polarized = polarize(filtered, polarize_cfg)
                trav_grid, r_edges, theta_edges, height_map = compute_traversability(
                    polarized, traversability_cfg
                )

                result = PipelineResult(
                    trav_grid=trav_grid,
                    height_map=height_map,
                    r_edges=r_edges,
                    theta_edges=theta_edges,
                    frame_index=frame_index,
                )
                # Drop stale result if display hasn't caught up yet.
                try:
                    result_q.put_nowait(result)
                except queue.Full:
                    try:
                        result_q.get_nowait()
                    except queue.Empty:
                        pass
                    result_q.put_nowait(result)

            except Exception as exc:
                LOGGER.warning("Frame %d pipeline failed: %s", frame_index, exc)
    finally:
        stop_event.set()


def run(args: argparse.Namespace) -> None:
    project_root = Path(__file__).resolve().parents[3]
    config_path = (
        args.config.expanduser().resolve()
        if args.config is not None
        else project_root / "config" / "pipeline_config.yaml"
    )
    config = load_config(config_path)

    voxel_cfg = config.get("voxel_filter", DEFAULT_CONFIG["voxel_filter"])
    polarize_cfg = config.get("polarize", DEFAULT_CONFIG["polarize"])
    traversability_cfg = config.get("traversability", DEFAULT_CONFIG["traversability"])
    frame_skip = (
        args.frame_skip
        if args.frame_skip is not None
        else int(config.get("svo", {}).get("frame_skip", 10))
    )

    svo_path = resolve_svo_path(args.svo) if args.svo is not None else None
    zed = open_camera(svo_path)
    LOGGER.info(
        "Opened %s, frame_skip=%d",
        f"SVO {svo_path}" if svo_path else "live camera",
        frame_skip,
    )

    result_q: queue.Queue[PipelineResult] = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    worker = threading.Thread(
        target=_pipeline_worker,
        args=(zed, result_q, stop_event, voxel_cfg, polarize_cfg, traversability_cfg, frame_skip),
        daemon=True,
        name="pipeline-worker",
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    worker.start()
    try:
        while not stop_event.is_set():
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                LOGGER.info("Window closed, stopping.")
                stop_event.set()
                break

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or ESC
                stop_event.set()
                break

            try:
                result = result_q.get(timeout=0.05)
            except queue.Empty:
                continue

            frame = _render_frame(result, args.grid_res_m, args.nt_height_overlay)
            cv2.imshow(WINDOW_NAME, frame)
    finally:
        stop_event.set()
        worker.join(timeout=5.0)
        zed.close()
        cv2.destroyAllWindows()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
