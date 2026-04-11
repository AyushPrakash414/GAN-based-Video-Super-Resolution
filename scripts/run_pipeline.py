#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import subprocess
import sys

from core.pipeline import VideoEnhancementPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _candidate_venv_pythons():
    return [
        PROJECT_ROOT / "venv" / "Scripts" / "python.exe",
        PROJECT_ROOT / "venv" / "bin" / "python",
    ]


def ensure_project_venv():
    """
    Re-launch the pipeline inside the repository virtual environment when it
    exists but the current interpreter is a different Python installation.
    """
    current_python = Path(sys.executable).resolve()

    venv_python = next((path for path in _candidate_venv_pythons() if path.exists()), None)
    if venv_python is None:
        return

    venv_python = venv_python.resolve()
    if current_python == venv_python:
        return

    env = os.environ.copy()
    env["PIPELINE_VENV_BOOTSTRAPPED"] = "1"

    if env.get("PIPELINE_VENV_BOOTSTRAPPED") == "1" and current_python != venv_python:
        subprocess.run([str(venv_python), "-m", "scripts.run_pipeline", *sys.argv[1:]], check=True, env=env)
        raise SystemExit(0)


def parse_args():
    parser = argparse.ArgumentParser("Video Enhancement Pipeline")

    parser.add_argument(
        "--start-stage",
        type=int,
        default=0,
        help="Resume pipeline from this stage index (default: 0)"
    )

    return parser.parse_args()


def main():
    ensure_project_venv()

    args = parse_args()

    pipeline = VideoEnhancementPipeline()
    try:
        pipeline.run(start_stage=args.start_stage)
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")


if __name__ == "__main__":
    main()
