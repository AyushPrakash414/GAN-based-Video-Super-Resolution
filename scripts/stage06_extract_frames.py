#!/usr/bin/env python3
import os
import cv2
from pathlib import Path
from tqdm import tqdm

from utils.pipeline_utils import find_latest_video

# ================================
# PATH CONFIG
# ================================

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_PATH = Path(ROOT_DIR)
INPUT_DIR = ROOT_PATH / "output" / "stage_05_presharpen"
OUTPUT_DIR = ROOT_PATH / "output" / "stage_06_frames"
FPS_FILE = OUTPUT_DIR / "source_fps.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================
# UTILS
# ================================

def log(msg):
    print(f"[STAGE 06] {msg}")

# ================================
# MAIN
# ================================

def main():
    input_video = find_latest_video(INPUT_DIR)

    log("Starting Frame Extraction")
    log(f"Input  : {input_video}")
    log(f"Output : {OUTPUT_DIR}")

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError("Failed to open input video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log(f"Total frames: {total_frames}")
    FPS_FILE.write_text(f"{fps}\n", encoding="ascii")

    idx = 0

    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        out_path = os.path.join(
            str(OUTPUT_DIR),
            f"frame_{idx:06d}.png"
        )

        cv2.imwrite(out_path, frame)
        idx += 1

    cap.release()

    log(f"Extraction complete: {idx} frames written.")

# ================================
# ENTRY
# ================================

if __name__ == "__main__":
    main()
