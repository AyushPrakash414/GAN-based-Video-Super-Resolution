# AyushPrakash414

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from utils.pipeline_utils import find_latest_video

# ================================
# PATH CONFIG
# ================================

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_PATH = Path(ROOT_DIR)
INPUT_DIR = ROOT_PATH / "output" / "stage_04_deflicker"
OUTPUT_DIR = ROOT_PATH / "output" / "stage_05_presharpen"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================
# PARAMETERS (SAFE)
# ================================

BLUR_SIGMA = 0.8          # very light blur
DETAIL_GAIN = 1.0        # 0.5–1.2 safe range
EDGE_THRESHOLD = 20      # gradient threshold
EDGE_GAIN = 1.2          # edge boost only

# ================================
# UTILS
# ================================

def log(msg):
    print(f"[STAGE 05] {msg}")

# ================================
# MAIN
# ================================

def main():
    input_video = find_latest_video(INPUT_DIR)
    output_video = OUTPUT_DIR / input_video.name

    log("Starting Detail-Preserving Pre-Sharpen")
    log(f"Input  : {input_video}")
    log(f"Output : {output_video}")

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError("Failed to open input video")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    for _ in tqdm(range(frames)):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to float
        f = frame.astype(np.float32)

        # Light blur (for high-pass extraction)
        blur = cv2.GaussianBlur(f, (0,0), BLUR_SIGMA)

        # High-frequency detail
        detail = f - blur

        # Edge detection (Sobel)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)

        # Edge mask
        edge_mask = (grad > EDGE_THRESHOLD).astype(np.float32)
        edge_mask = cv2.GaussianBlur(edge_mask, (0,0), 1.0)

        # Apply detail boost only on edges
        enhanced = f + (detail * DETAIL_GAIN * edge_mask[...,None] * EDGE_GAIN)

        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

        writer.write(enhanced)

    cap.release()
    writer.release()

    log("Detail-preserving pre-sharpen complete.")

if __name__ == "__main__":
    main()
