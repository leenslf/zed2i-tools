#!/usr/bin/env python3
"""SVO offline pipeline orchestrator.

Current behavior:
- Loads config once from config/pipeline_config.yaml
- Reads frames from an SVO2 recording via pyzed
- Applies tilt compensation
- Applies voxel filtering
- Applies polarization
- Applies traversability
- Optionally writes stage outputs based on config
- RUN: python3 src/traversability/run_svo_pipeline.py <PATH>
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pyzed.sl as sl
import yaml

from traversability.py.tilt_compensate import tilt_compensate
from traversability.py.polarize import polarize
from traversability import compute_traversability
from traversability.py.voxel_filter import voxel_filter

DEFAULT_CONFIG = {
    "tilt_compensate": {
        "write_output": False,
    },
    "voxel_filter": {
        "voxel_size_x": 0.05,
        "voxel_size_y": 0.05,
        "voxel_size_z": 0.05,
        "min_points_per_voxel": 8,
        "write_output": False,
    },
    "polarize": {
        "z_threshold": 1.5,
        "min_range": 0.1,
        "write_output": False,
    },
    "traversability": {
        "danger_threshold": 0.3,
        "scrit_deg": 30.0,
        "rcrit_m": 0.10,
        "hcrit_m": 0.20,
        "polar_grid_size_r_m": 0.10,
        "polar_grid_size_theta_deg": 1.0,
        "write_output": True,
    },
}

LOGGER = logging.getLogger("run_svo_pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SVO offline traversability pipeline.")
    parser.add_argument("svo_path", type=Path, help="Path to SVO2 recording file.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to pipeline config YAML (default: <project_root>/config/pipeline_config.yaml).",
    )
    return parser.parse_args()


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override values into a copy of base."""
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load pipeline config once at startup."""
    if not config_path.exists():
        return DEFAULT_CONFIG

    with config_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping, got: {type(loaded).__name__}")

    return deep_merge(DEFAULT_CONFIG, loaded)


def open_svo(svo_path: Path) -> sl.Camera:
    """Open SVO2 file and enable positional tracking."""
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(svo_path))
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open SVO file {svo_path}: {err}")
    zed.enable_positional_tracking(sl.PositionalTrackingParameters())
    return zed


def process_svo(project_root: Path, svo_path: Path, config: Dict[str, Any]) -> None:
    recording_name = svo_path.stem
    out_root = project_root / "temp-offline-outs" / recording_name

    tilt_cfg = config.get("tilt_compensate", {})
    write_tilt_output = bool(tilt_cfg.get("write_output", False))
    voxel_cfg = config.get("voxel_filter", {})
    write_voxel_output = bool(voxel_cfg.get("write_output", False))
    polarize_cfg = config.get("polarize", {})
    write_polarize_output = bool(polarize_cfg.get("write_output", False))
    traversability_cfg = config.get("traversability", {})
    write_traversability_output = bool(traversability_cfg.get("write_output", True))
    LOGGER.info(
        "Stage output mode: tilt=%s, voxel=%s, polarize=%s, traversability=%s",
        "disk" if write_tilt_output else "in-memory",
        "disk" if write_voxel_output else "in-memory",
        "disk" if write_polarize_output else "in-memory",
        "disk" if write_traversability_output else "in-memory",
    )

    detilt_dir = out_root / "detilted_cloud"
    if write_tilt_output:
        detilt_dir.mkdir(parents=True, exist_ok=True)

    filtered_dir = out_root / "filtered_cloud"
    if write_voxel_output:
        filtered_dir.mkdir(parents=True, exist_ok=True)

    polarized_dir = out_root / "polarized_cloud"
    if write_polarize_output:
        polarized_dir.mkdir(parents=True, exist_ok=True)

    traversability_dir = out_root / "traversability_grid"
    if write_traversability_output:
        traversability_dir.mkdir(parents=True, exist_ok=True)

    zed = open_svo(svo_path)
    point_cloud = sl.Mat()
    pose = sl.Pose()

    processed_frames = 0
    skipped_frames = 0
    frame_index = 0

    try:
        while True:
            err = zed.grab()
            if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            if err != sl.ERROR_CODE.SUCCESS:
                LOGGER.warning("grab() returned %s at frame %d, skipping", err, frame_index)
                skipped_frames += 1
                frame_index += 1
                continue

            frame_name = f"frame_{frame_index:05d}.npz"
            frame_index += 1
            # To process every other frame: `if frame_index % 2 != 0: continue`
            if frame_index % 10 != 0: continue
            try:
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZ, sl.MEM.CPU)
                timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
                zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
                q = pose.get_orientation().get()
                quaternion = np.array([q[0], q[1], q[2], q[3]])

                pc_data = point_cloud.get_data()  # (H, W, 4) float32
                points = pc_data[:, :, :3].reshape(-1, 3)
                valid = np.isfinite(points).all(axis=1)
                points = points[valid]

                detilted_points = tilt_compensate(points, quaternion)

                if write_tilt_output:
                    out_path = detilt_dir / frame_name
                    np.savez(out_path, points=detilted_points, timestamp=timestamp)
                    with np.load(out_path) as detilted_snapshot:
                        voxel_input_points = detilted_snapshot["points"]
                        voxel_timestamp = detilted_snapshot["timestamp"]
                else:
                    voxel_input_points = detilted_points
                    voxel_timestamp = timestamp

                filtered_points = voxel_filter(voxel_input_points, voxel_cfg)

                if write_voxel_output:
                    out_path = filtered_dir / frame_name
                    np.savez(out_path, points=filtered_points, timestamp=voxel_timestamp)
                    with np.load(out_path) as filtered_snapshot:
                        polarize_input_points = filtered_snapshot["points"]
                        polarize_timestamp = filtered_snapshot["timestamp"]
                else:
                    polarize_input_points = filtered_points
                    polarize_timestamp = voxel_timestamp

                polarized_points = polarize(polarize_input_points, polarize_cfg)

                if write_polarize_output:
                    out_path = polarized_dir / frame_name
                    np.savez(out_path, points=polarized_points, timestamp=polarize_timestamp)
                    with np.load(out_path) as polarized_snapshot:
                        traversability_input_points = polarized_snapshot["points"]
                        traversability_timestamp = polarized_snapshot["timestamp"]
                else:
                    traversability_input_points = polarized_points
                    traversability_timestamp = polarize_timestamp

                danger_grid, valid_mask, nontraversable, r_edges, theta_edges = compute_traversability(
                    traversability_input_points,
                    traversability_cfg,
                )

                if write_traversability_output:
                    out_path = traversability_dir / frame_name
                    np.savez(
                        out_path,
                        danger_grid=danger_grid,
                        valid_mask=valid_mask,
                        nontraversable=nontraversable,
                        r_edges=r_edges,
                        theta_edges=theta_edges,
                        timestamp=traversability_timestamp,
                    )

                processed_frames += 1
            except Exception as exc:
                skipped_frames += 1
                LOGGER.info("Skipping %s: %s", frame_name, exc)
    finally:
        zed.close()

    LOGGER.info(
        "Pipeline summary: total_frames=%d, processed=%d, skipped=%d",
        frame_index,
        processed_frames,
        skipped_frames,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    svo_path = args.svo_path.expanduser().resolve()
    if not svo_path.exists():
        raise FileNotFoundError(f"SVO file not found: {svo_path}")

    project_root = Path(__file__).resolve().parents[2]
    config_path = (
        args.config.expanduser().resolve()
        if args.config is not None
        else project_root / "config" / "pipeline_config.yaml"
    )

    config = load_config(config_path)
    LOGGER.info("Starting SVO pipeline for: %s", svo_path)
    LOGGER.info("Using config file: %s", config_path)
    LOGGER.info("Effective config:\n%s", yaml.safe_dump(config, sort_keys=False).strip())
    process_svo(project_root, svo_path, config)


if __name__ == "__main__":
    main()
