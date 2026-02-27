#!/usr/bin/env python3
"""SVO offline pipeline orchestrator.

Current behavior:
- Loads config once from config/pipeline_config.yaml
- Reads frames from an SVO2 recording via pyzed
- Applies tilt compensation
- Applies voxel filtering
- Applies polarization
- Applies traversability
- If one or more stages have write_output enabled, writes a single NPZ file per
  frame under temp-offline-outs/<recording>/ containing only the enabled-stage
  outputs with namespaced keys (e.g. tilt_points, voxel_points, polarize_points,
  traversability_danger_grid, …) plus a shared timestamp field.
- RUN: python3 src/traversability/python/run_svo_pipeline.py <PATH>
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pyzed.sl as sl
import yaml

from tilt_compensate import tilt_compensate
from polarize import polarize
from traversability import compute_traversability
from voxel_filter import voxel_filter

DEFAULT_CONFIG = {
    "svo": {
        "frame_skip": 10,
    },
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
    parser.add_argument(
        "svo_path",
        type=Path,
        help="Path to an SVO/SVO2 file or recording directory containing one.",
    )
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


def resolve_svo_path(path: Path) -> Path:
    """Resolve input into a concrete .svo/.svo2 file path."""
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"SVO path not found: {resolved}")

    if resolved.is_file():
        if resolved.suffix.lower() not in {".svo", ".svo2"}:
            raise ValueError(
                f"Expected .svo or .svo2 file, got: {resolved}"
            )
        return resolved

    if not resolved.is_dir():
        raise ValueError(f"Input path is neither a file nor directory: {resolved}")

    candidates = sorted(resolved.rglob("*.svo")) + sorted(resolved.rglob("*.svo2"))
    if not candidates:
        raise FileNotFoundError(
            f"No .svo/.svo2 file found under directory: {resolved}"
        )
    if len(candidates) > 1:
        pretty = "\n".join(f"- {candidate}" for candidate in candidates)
        raise ValueError(
            "Multiple .svo/.svo2 files found. Pass a specific file path:\n"
            f"{pretty}"
        )
    return candidates[0]


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
        raise RuntimeError(
            f"Failed to open SVO file {svo_path}: {err}. "
            "If this is a recording folder, pass the folder and let the script resolve it, "
            "or pass an explicit .svo/.svo2 file path."
        )
    zed.enable_positional_tracking(sl.PositionalTrackingParameters())
    return zed


def process_svo(project_root: Path, svo_path: Path, config: Dict[str, Any]) -> None:
    recording_name = svo_path.parent.parent.name
    out_root = project_root / "temp-offline-outs" / recording_name

    svo_cfg = config.get("svo", {})
    frame_skip = int(svo_cfg.get("frame_skip", 10))
    tilt_cfg = config.get("tilt_compensate", {})
    write_tilt_output = bool(tilt_cfg.get("write_output", False))
    voxel_cfg = config.get("voxel_filter", {})
    write_voxel_output = bool(voxel_cfg.get("write_output", False))
    polarize_cfg = config.get("polarize", {})
    write_polarize_output = bool(polarize_cfg.get("write_output", False))
    traversability_cfg = config.get("traversability", {})
    write_traversability_output = bool(traversability_cfg.get("write_output", True))
    any_write = write_tilt_output or write_voxel_output or write_polarize_output or write_traversability_output
    LOGGER.info(
        "Stage output flags: tilt=%s, voxel=%s, polarize=%s, traversability=%s — writing %s",
        write_tilt_output,
        write_voxel_output,
        write_polarize_output,
        write_traversability_output,
        f"single NPZ to {out_root}/" if any_write else "nothing (all in-memory)",
    )

    if any_write:
        out_root.mkdir(parents=True, exist_ok=True)

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
            if frame_index % frame_skip != 0: continue
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
                filtered_points = voxel_filter(detilted_points, voxel_cfg)
                polarized_points = polarize(filtered_points, polarize_cfg)
                danger_grid, valid_mask, nontraversable, r_edges, theta_edges = compute_traversability(
                    polarized_points,
                    traversability_cfg,
                )

                if any_write:
                    frame_data: Dict[str, Any] = {"timestamp": timestamp}
                    if write_tilt_output:
                        frame_data["tilt_points"] = detilted_points
                    if write_voxel_output:
                        frame_data["voxel_points"] = filtered_points
                    if write_polarize_output:
                        frame_data["polarize_points"] = polarized_points
                    if write_traversability_output:
                        frame_data["traversability_danger_grid"] = danger_grid
                        frame_data["traversability_valid_mask"] = valid_mask
                        frame_data["traversability_nontraversable"] = nontraversable
                        frame_data["traversability_r_edges"] = r_edges
                        frame_data["traversability_theta_edges"] = theta_edges
                    np.savez(out_root / frame_name, **frame_data)

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

    svo_path = resolve_svo_path(args.svo_path)

    project_root = Path(__file__).resolve().parents[3]
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
