"""
MODULE 03 — Video Stabilization (FFmpeg VidStab)

Input  : Deinterlaced video
Output : Stabilized video

Process:
    1) Motion detection → generates transforms.trf
    2) Motion compensation → stabilized output

"""

import argparse
from pathlib import Path
import re
import sys

from utils.ffmpeg_utils import ensure_ffmpeg_available, run_ffmpeg_command

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "output" / "stage_02_deinterlaced"
OUTPUT_DIR = PROJECT_ROOT / "output" / "stage_03_stabilized"
TRANSFORM_DIR = OUTPUT_DIR / "transforms"


def _safe_stem(path: Path) -> str:
    """
    Build a filesystem-safe stem for intermediate vidstab files.
    """
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._-")
    return sanitized or "video"


def _escape_ffmpeg_filter_path(path: Path) -> str:
    """
    Escape a filesystem path for use inside an FFmpeg filter expression.
    """
    normalized = path.resolve().as_posix()
    escaped = normalized.replace(":", r"\:")
    return f"'{escaped}'"


def check_ffmpeg():
    try:
        return ensure_ffmpeg_available()
    except RuntimeError as exc:
        print(str(exc))
        sys.exit(1)


def detect_motion(ffmpeg_bin: str, input_video: Path, transform_file: Path, shakiness=10, accuracy=15):
    transform_file.parent.mkdir(parents=True, exist_ok=True)
    escaped_transform = _escape_ffmpeg_filter_path(transform_file)

    cmd = [
        ffmpeg_bin, "-y",
        "-i", str(input_video),
        "-vf", f"vidstabdetect=shakiness={shakiness}:accuracy={accuracy}:result={escaped_transform}",
        "-f", "null", "-"
    ]

    print("\n Running Motion Analysis:")
    print(" ".join(cmd))

    run_ffmpeg_command(cmd)

    print(f"\n Motion analysis complete → {transform_file}")


def apply_stabilization(ffmpeg_bin: str, input_video: Path, transform_file: Path, output_video: Path, smoothing=30, crf=16):
    output_video.parent.mkdir(parents=True, exist_ok=True)
    escaped_transform = _escape_ffmpeg_filter_path(transform_file)

    cmd = [
        ffmpeg_bin, "-y",
        "-i", str(input_video),
        "-vf", f"vidstabtransform=smoothing={smoothing}:input={escaped_transform},unsharp=5:5:0.8:3:3:0.4",
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        str(output_video)
    ]

    print("\n Applying Stabilization:")
    print(" ".join(cmd))

    run_ffmpeg_command(cmd)

    print(f"\n Module 03 Complete → {output_video}")


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 03 — Video Stabilization (VidStab)")

    parser.add_argument("--input", required=True, type=Path, help="Input deinterlaced video")
    parser.add_argument("--output", required=True, type=Path, help="Output stabilized video")
    parser.add_argument("--transform", required=True, type=Path, help="Transform file (.trf)")
    parser.add_argument("--shakiness", default=10, type=int, help="Detection shakiness (default: 10)")
    parser.add_argument("--accuracy", default=15, type=int, help="Detection accuracy (default: 15)")
    parser.add_argument("--smoothing", default=30, type=int, help="Motion smoothing (default: 30)")
    parser.add_argument("--crf", default=16, type=int, help="Output quality (default: 16)")

    return parser.parse_args()


#def main():
#    args = parse_args()
#
#   if not args.input.exists():
#        print(f" Input file not found: {args.input}")
#        sys.exit(1)
#
#    check_ffmpeg()
#
#    detect_motion(args.input, args.transform, args.shakiness, args.accuracy)
#
#   apply_stabilization(args.input, args.transform, args.output, args.smoothing, args.crf)

def main():

    ffmpeg_bin = check_ffmpeg()

    videos = sorted(path for path in INPUT_DIR.iterdir() if path.is_file())

    if not videos:
        print(f"No input videos found in {INPUT_DIR}")
        sys.exit(1)

    for input_video in videos:
        output_video = OUTPUT_DIR / input_video.name
        transform_file = TRANSFORM_DIR / f"{_safe_stem(input_video)}.trf"

        print(f"\n▶ Stabilizing: {input_video.name}")

        detect_motion(ffmpeg_bin, input_video, transform_file)
        apply_stabilization(ffmpeg_bin, input_video, transform_file, output_video)

    print("\n Stage 03 complete")


if __name__ == "__main__":
    main()
