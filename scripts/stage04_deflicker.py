# AyushPrakash414
#!/usr/bin/env python3
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.signal import savgol_filter

from utils.pipeline_utils import find_latest_video

# ================================
# PATH CONFIG
# ================================

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_PATH = Path(ROOT_DIR)
INPUT_DIR = ROOT_PATH / "output" / "stage_03_stabilized"
OUTPUT_DIR = ROOT_PATH / "output" / "stage_04_deflicker"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================
# PARAMETERS
# ================================

WINDOW_SIZE = 31   # must be odd, larger = smoother brightness curve
POLY_ORDER  = 3    # smoothing polynomial order

# ================================
# UTILS
# ================================

def log(msg):
    print(f"[STAGE 04] {msg}")

# ================================
# MAIN PROCESS
# ================================

def main():
    input_video = find_latest_video(INPUT_DIR)
    output_video = OUTPUT_DIR / input_video.name

    log("Starting Temporal Deflicker")
    log(f"Input  : {input_video}")
    log(f"Output : {output_video}")

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError("Failed to open input video")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    log(f"Resolution : {width}x{height}")
    log(f"FPS        : {fps}")
    log(f"Frames     : {frames}")

    # -------------------------------
    # PASS 1 — LUMINANCE ANALYSIS
    # -------------------------------

    luminance_curve = []

    log("Analyzing luminance curve...")

    for _ in tqdm(range(frames)):
        ret, frame = cap.read()
        if not ret:
            break

        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y = ycrcb[:,:,0]
        luminance_curve.append(np.mean(y))

    cap.release()

    luminance_curve = np.array(luminance_curve)

    # -------------------------------
    # TEMPORAL SMOOTHING
    # -------------------------------

    log("Applying temporal smoothing...")

    if len(luminance_curve) < WINDOW_SIZE:
        log("WARNING: Video too short for Savitzky-Golay. Falling back to moving average.")
        smooth_curve = np.convolve(
            luminance_curve,
            np.ones(7)/7,
            mode='same'
        )
    else:
        smooth_curve = savgol_filter(
            luminance_curve,
            window_length=WINDOW_SIZE,
            polyorder=POLY_ORDER
        )

    gain_curve = smooth_curve / (luminance_curve + 1e-6)

    # Clamp gain for safety
    gain_curve = np.clip(gain_curve, 0.85, 1.15)

    # -------------------------------
    # PASS 2 — APPLY CORRECTION
    # -------------------------------

    log("Applying brightness correction...")

    cap = cv2.VideoCapture(str(input_video))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    for i in tqdm(range(frames)):
        ret, frame = cap.read()
        if not ret:
            break

        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        y = y.astype(np.float32)
        y *= gain_curve[i]
        y = np.clip(y, 0, 255).astype(np.uint8)

        corrected = cv2.merge([y, cr, cb])
        corrected = cv2.cvtColor(corrected, cv2.COLOR_YCrCb2BGR)

        writer.write(corrected)

    cap.release()
    writer.release()

    log("Temporal deflicker complete.")

# ================================
# ENTRY
# ================================

if __name__ == "__main__":
    main()
