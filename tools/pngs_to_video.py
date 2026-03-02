#!/usr/bin/env python3
"""Turn all PNG files in a folder into a video."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a video from PNG files in a folder.")
    parser.add_argument("input_dir", type=Path, help="Folder containing PNG files.")
    parser.add_argument("--out", type=Path, default=None, help="Output video path (default: <folder>.mp4).")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second (default: 24).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")

    pngs = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"])
    if not pngs:
        raise FileNotFoundError(f"No PNG files found in {input_dir}")

    out_path = args.out.expanduser().resolve() if args.out else input_dir.with_suffix(".mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for png in pngs:
            f.write(f"file '{png.as_posix()}'\n")
        list_path = Path(f.name)

    cmd = [
        "ffmpeg",
        "-y",
        "-r",
        str(args.fps),
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True)
    finally:
        list_path.unlink(missing_ok=True)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
