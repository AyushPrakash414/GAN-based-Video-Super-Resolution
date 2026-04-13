# AyushPrakash414
from pathlib import Path


VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi")


def find_latest_video(directory: Path) -> Path:
    """
    Return the most recently modified video file in a pipeline stage directory.
    """
    candidates = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]

    if not candidates:
        raise FileNotFoundError(f"No video files found in {directory}")

    return max(candidates, key=lambda path: path.stat().st_mtime)
