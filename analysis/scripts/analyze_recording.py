#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _load_csv(path: Path, required_cols):
    df = pd.read_csv(path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df


def _timestamp_stats(series: pd.Series):
    ts = series.astype(float).to_numpy()
    ts = ts[~np.isnan(ts)]
    if ts.size == 0:
        return {
            "count": 0,
            "start": np.nan,
            "end": np.nan,
            "duration": np.nan,
            "median_dt": np.nan,
            "gap_count": 0,
            "max_gap": np.nan,
            "negative_dt": 0,
        }
    ts_sorted = np.sort(ts)
    dt = np.diff(ts_sorted)
    median_dt = np.median(dt) if dt.size else np.nan
    negative_dt = int(np.sum(np.diff(ts) < 0))
    gap_threshold = median_dt * 1.5 if np.isfinite(median_dt) else np.nan
    gap_mask = dt > gap_threshold if np.isfinite(gap_threshold) else np.zeros_like(dt, dtype=bool)
    return {
        "count": int(ts.size),
        "start": float(ts_sorted[0]),
        "end": float(ts_sorted[-1]),
        "duration": float(ts_sorted[-1] - ts_sorted[0]),
        "median_dt": float(median_dt) if np.isfinite(median_dt) else np.nan,
        "gap_count": int(np.sum(gap_mask)),
        "max_gap": float(np.max(dt)) if dt.size else np.nan,
        "negative_dt": negative_dt,
    }


def _axis_stats(df: pd.DataFrame, cols):
    stats = {}
    for c in cols:
        v = df[c].astype(float)
        stats[c] = {"mean": float(v.mean()), "std": float(v.std(ddof=0))}
    return stats


def _plot_time_series(df, time_col, cols, title, out_path):
    plt.figure(figsize=(10, 4))
    for c in cols:
        plt.plot(df[time_col], df[c], label=c)
    plt.xlabel("time")
    plt.ylabel("value")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_trajectory(df, x_col, y_col, out_path):
    plt.figure(figsize=(5, 5))
    plt.plot(df[x_col], df[y_col], linewidth=1.0)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Odometry Trajectory")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_velocity(df, time_col, vel_cols, out_path):
    v = np.sqrt(np.square(df[vel_cols[0]]) + np.square(df[vel_cols[1]]) + np.square(df[vel_cols[2]]))
    plt.figure(figsize=(10, 4))
    plt.plot(df[time_col], v)
    plt.xlabel("time")
    plt.ylabel("velocity")
    plt.title("Odometry Velocity")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze ZED IMU/odometry recording.")
    parser.add_argument("recording_dir", type=Path, help="Path to a recording folder")
    args = parser.parse_args()

    recording_dir = args.recording_dir
    imu_path = recording_dir / "imu.csv"
    odom_path = recording_dir / "odometry.csv"

    if not imu_path.exists() or not odom_path.exists():
        raise FileNotFoundError("Expected imu.csv and odometry.csv in recording folder")

    imu_cols = ["timestamp", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
    odom_cols = ["timestamp", "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z"]
    imu = _load_csv(imu_path, imu_cols)
    odom = _load_csv(odom_path, odom_cols)

    results_dir = Path("analysis/results") / recording_dir.name
    results_dir.mkdir(parents=True, exist_ok=True)

    imu_time = _timestamp_stats(imu["timestamp"])
    odom_time = _timestamp_stats(odom["timestamp"])

    imu_accel_stats = _axis_stats(imu, ["accel_x", "accel_y", "accel_z"])
    imu_gyro_stats = _axis_stats(imu, ["gyro_x", "gyro_y", "gyro_z"])

    pos = odom[["pos_x", "pos_y", "pos_z"]].astype(float).to_numpy()
    if pos.shape[0] >= 2:
        diffs = np.diff(pos, axis=0)
        total_distance = float(np.sum(np.linalg.norm(diffs, axis=1)))
    else:
        total_distance = 0.0

    vel = odom[["vel_x", "vel_y", "vel_z"]].astype(float)
    speed = np.sqrt(np.square(vel["vel_x"]) + np.square(vel["vel_y"]) + np.square(vel["vel_z"]))
    avg_speed = float(speed.mean()) if len(speed) else 0.0
    max_speed = float(speed.max()) if len(speed) else 0.0

    stats_path = results_dir / "statistics.txt"
    with stats_path.open("w", encoding="utf-8") as f:
        f.write("IMU Stats\n")
        f.write("Accel mean/std (x,y,z)\n")
        f.write(
            f"  mean: {imu_accel_stats['accel_x']['mean']:.6f}, "
            f"{imu_accel_stats['accel_y']['mean']:.6f}, "
            f"{imu_accel_stats['accel_z']['mean']:.6f}\n"
        )
        f.write(
            f"  std : {imu_accel_stats['accel_x']['std']:.6f}, "
            f"{imu_accel_stats['accel_y']['std']:.6f}, "
            f"{imu_accel_stats['accel_z']['std']:.6f}\n"
        )
        f.write("Gyro mean/std (x,y,z)\n")
        f.write(
            f"  mean: {imu_gyro_stats['gyro_x']['mean']:.6f}, "
            f"{imu_gyro_stats['gyro_y']['mean']:.6f}, "
            f"{imu_gyro_stats['gyro_z']['mean']:.6f}\n"
        )
        f.write(
            f"  std : {imu_gyro_stats['gyro_x']['std']:.6f}, "
            f"{imu_gyro_stats['gyro_y']['std']:.6f}, "
            f"{imu_gyro_stats['gyro_z']['std']:.6f}\n"
        )
        f.write("\nOdometry Stats\n")
        f.write(f"Total distance: {total_distance:.6f}\n")
        f.write(f"Average velocity: {avg_speed:.6f}\n")
        f.write(f"Max velocity: {max_speed:.6f}\n")
        f.write("\nData Quality\n")
        f.write("IMU\n")
        f.write(
            f"  samples: {imu_time['count']}\n"
            f"  timestamp range: {imu_time['start']:.6f} to {imu_time['end']:.6f}\n"
            f"  duration: {imu_time['duration']:.6f}\n"
            f"  median dt: {imu_time['median_dt']:.6f}\n"
            f"  gaps: {imu_time['gap_count']} (max gap {imu_time['max_gap']:.6f})\n"
            f"  negative dt: {imu_time['negative_dt']}\n"
        )
        f.write("Odometry\n")
        f.write(
            f"  samples: {odom_time['count']}\n"
            f"  timestamp range: {odom_time['start']:.6f} to {odom_time['end']:.6f}\n"
            f"  duration: {odom_time['duration']:.6f}\n"
            f"  median dt: {odom_time['median_dt']:.6f}\n"
            f"  gaps: {odom_time['gap_count']} (max gap {odom_time['max_gap']:.6f})\n"
            f"  negative dt: {odom_time['negative_dt']}\n"
        )

    _plot_time_series(
        imu,
        "timestamp",
        ["accel_x", "accel_y", "accel_z"],
        "IMU Acceleration",
        results_dir / "imu_acceleration.png",
    )
    _plot_time_series(
        imu,
        "timestamp",
        ["gyro_x", "gyro_y", "gyro_z"],
        "IMU Gyroscope",
        results_dir / "imu_gyroscope.png",
    )
    _plot_trajectory(odom, "pos_x", "pos_y", results_dir / "odometry_trajectory.png")
    _plot_velocity(odom, "timestamp", ["vel_x", "vel_y", "vel_z"], results_dir / "odometry_velocity.png")


if __name__ == "__main__":
    main()
