#!/usr/bin/env python3
"""Extract review candidate frames from downloaded egocentric videos."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def find_videos(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*") if path.suffix.lower() in VIDEO_SUFFIXES)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output-dir", default="data/task3_real_egocentric_source_frames")
    parser.add_argument("--fps", default="0.2", help="Frames per second to sample. 0.2 means one frame every 5 seconds.")
    parser.add_argument("--limit-videos", type=int, default=None)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()

    video_dir = Path(args.video_dir).expanduser()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = find_videos(video_dir)
    if args.limit_videos is not None:
        videos = videos[: args.limit_videos]
    if not videos:
        raise FileNotFoundError(f"No video files found under {video_dir}")

    for idx, video in enumerate(videos, start=1):
        safe_stem = "".join(ch if ch.isalnum() else "_" for ch in video.stem).strip("_")
        video_out = output_dir / f"{idx:04d}_{safe_stem}"
        video_out.mkdir(parents=True, exist_ok=True)
        pattern = video_out / "frame_%06d.jpg"
        cmd = [
            args.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            f"fps={args.fps}",
            "-q:v",
            "2",
            str(pattern),
        ]
        print(f"[{idx}/{len(videos)}] extracting {video} -> {video_out}")
        subprocess.run(cmd, check=True)

    print(f"Wrote extracted frames to {output_dir}")


if __name__ == "__main__":
    main()
