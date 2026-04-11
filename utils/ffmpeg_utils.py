import os
import shutil
import subprocess
from pathlib import Path


def resolve_ffmpeg_binary():
    env_binary = os.environ.get("FFMPEG_BINARY")
    if env_binary:
        candidate = Path(env_binary).expanduser()
        if candidate.exists():
            return str(candidate)
        return env_binary

    return shutil.which("ffmpeg") or "ffmpeg"


def ensure_ffmpeg_available():
    ffmpeg_bin = resolve_ffmpeg_binary()
    try:
        subprocess.run(
            [ffmpeg_bin, "-version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        raise RuntimeError(
            "FFmpeg was not found. Add `ffmpeg.exe` to PATH or set the "
            "`FFMPEG_BINARY` environment variable to the full executable path."
        ) from exc

    return ffmpeg_bin


def run_ffmpeg_command(cmd):
    """
    Run an FFmpeg command with cleaner Ctrl+C handling on Windows consoles.

    On some Windows terminals, pressing Ctrl+C while Python is waiting can raise
    KeyboardInterrupt even after FFmpeg has already exited successfully. In that
    case we treat the command as successful and avoid surfacing a misleading
    traceback.
    """
    process = subprocess.Popen(cmd)

    try:
        return_code = process.wait()
    except KeyboardInterrupt:
        return_code = process.poll()

        if return_code == 0:
            return

        if return_code is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

        raise SystemExit("\nPipeline interrupted by user.")

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)
