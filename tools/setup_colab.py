#!/usr/bin/env python3
"""
Prepare the repository for Google Colab.

This script:
1. installs FFmpeg through apt,
2. installs Python dependencies from requirements.txt,
3. downloads the missing model weights used by this pipeline.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def install_ffmpeg() -> None:
    run(["apt-get", "update"])
    run(["apt-get", "install", "-y", "ffmpeg"])


def install_python_requirements() -> None:
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=PROJECT_ROOT)


def download_models() -> None:
    run([sys.executable, "tools/download_models.py"], cwd=PROJECT_ROOT)


def main() -> None:
    print("=" * 72)
    print("Preparing GAN_Video_Enhancer for Google Colab")
    print("=" * 72)
    print(f"Project root: {PROJECT_ROOT}")

    install_ffmpeg()
    install_python_requirements()
    download_models()

    print("\nSetup complete.")
    print("Upload a source video into input/raw_videos/ and run:")
    print("python -m scripts.run_pipeline")


if __name__ == "__main__":
    main()
