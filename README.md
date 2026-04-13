# 🎬 GAN-Based Video Super-Resolution & Enhancement

<div align="center">
  <p>
    <em>A robust 12-stage pipeline for transforming low-quality, noisy, or interlaced videos into high-definition, smooth, and crystal-clear masterpieces using State-of-the-Art Deep Learning models.</em>
  </p>
</div>

---

## 🌟 What is this Project?

This project is an advanced, fully automated video restoration pipeline. It leverages **Generative Adversarial Networks (GANs)** and specialized neural network architectures to tackle severe video degradations. Whether your footage has interlacing artifacts, camera shake, severe low-light noise, low resolution, or blurry faces, this pipeline enhances it frame-by-frame to deliver a modern, high-quality output.

**It integrates industry-leading open-source models:**
- 🌌 **FastDVDnet** for deep spatio-temporal video denoising.
- 🔍 **Real-ESRGAN** for high-fidelity upscaling and super-resolution.
- 👤 **GFPGAN** for blind face restoration.
- 🎞️ **FFmpeg** for video normalization, stabilization, and reconstruction.

---

## ⚙️ How it Works (The 12-Stage Pipeline)

The enhancement process is divided into **12 modular stages**. Intermediate results are securely saved to the `output/` folder, allowing you to pause, inspect, and resume the pipeline at any point without losing progress.

1.  **Normalization:** Standardizes the video format, codecs, and framerate.
2.  **Deinterlacing:** Removes comb-like artifacts from interlaced/older footage.
3.  **Stabilization:** Smooths out camera shake and jitter using FFmpeg physics.
4.  **Deflickering:** Eliminates brightness fluctuations between concurrent frames.
5.  **Presharpening:** Enhances edges slightly to prepare for deep denoising.
6.  **Frame Extraction:** Splits the video into individual image frames for neural network processing.
7.  **FastDVDnet Denoising:** Removes noise utilizing temporal sequences, preserving textures without blurring.
8.  **Real-ESRGAN Super-Resolution:** Upscales frames up to 4x, predicting missing high-frequency details.
9.  **Detail Refinement:** Applies subtle contrasting and sharpening post-upscaling.
10. **GFPGAN Face Enhancement:** Detects and magically restores individual faces in the video.
11. **Temporal Smoothing:** Ensures color and light consistency across enhanced frames to prevent flickering.
12. **Final Reconstruction:** Merges the enhanced frames back into a seamless, high-quality audio-visual `.mp4` stream.

---

## 🚀 Running on Google Colab (Highly Recommended)

⚠️ **WHY COLAB? (HEAVY GPU TASK):** Video processing—especially frame-by-frame GAN inference—is an extremely computationally expensive task. Running this on a local machine without a powerful dedicated NVIDIA GPU (like an RTX 3090/4090) will take days for a short clip. 
**Google Colab** provides **Free High-End GPUs (T4, V100, A100)** to run this pipeline efficiently and effortlessly.

### 🛠️ Google Colab Setup Guide

**Step 1. Open a New Colab Notebook**  
Navigate to [colab.research.google.com](https://colab.research.google.com/) and create a new notebook.

**Step 2. Enable the GPU**  
In the top menu, go to `Runtime` -> `Change runtime type` -> select **Hardware Accelerator**: `GPU` (Select T4 or better).

**Step 3. Clone the Repository & Setup Dependencies**  
Create a code cell, paste the following, and run it. This clones the code and automatically downloads the huge model weights (`.pth` files):
```python
# 1. Clone the repository
!git clone https://github.com/AyushPrakash414/GAN-based-Video-Super-Resolution.git
%cd GAN-based-Video-Super-Resolution

# 2. Setup the environment and download heavy model weights
!python tools/setup_colab.py
```

**Step 4. Upload Your Input Video**  
Upload your footage using Colab's file uploader and move it to the `input` directory:
```python
import os
import shutil
from google.colab import files

# Create input directory
os.makedirs("input/raw_videos", exist_ok=True)

# Upload file
print("Please select your corrupted/low-res video:")
uploaded = files.upload()

# Move the uploaded file to the processing directory
for name in uploaded.keys():
    shutil.move(name, os.path.join("input/raw_videos", name))
    print(f"✅ Successfully prepared {name} for processing.")
```

**Step 5. Start the Magic! (Run Pipeline)**  
Execution will take time depending on video length and the assigned GPU.
```python
!python -m scripts.run_pipeline
```
*💡 Pro Tip: If Colab disconnects, you can resume from a specific stage to save time:*
`!python -m scripts.run_pipeline --start-stage 7` *(e.g., stage 7 is FastDVDnet)*

**Step 6. Download the Final Masterpiece**  
```python
from google.colab import files
files.download("output/stage_12_final/final_restored.mp4")
```

---

## 💻 Local Installation (For Powerful Machines Only)

If you have a high-end NVIDIA GPU (CUDA enabled) and wish to process locally:

**Prerequisites:**
- **Python 3.8+**
- **FFmpeg** installed and added to your system's PATH. (`set FFMPEG_BINARY=path/to/ffmpeg.exe` if needed)

**Installation Steps (in PowerShell/Terminal):**
```powershell
# 1. Install dependencies
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

# 2. Download model weights (~1GB for FastDVDnet, GFPGAN, RealESRGAN)
python tools/download_models.py

# 3. Add your video to input/raw_videos/

# 4. Run the pipeline
python -m scripts.run_pipeline
```

---

## 📁 Repository Structure

```text
GAN-based-Video-Super-Resolution/
│
├── input/raw_videos/     # 📥 Place your source videos here
├── output/               # 📤 Intermediate pipeline stages and final output generated here
├── models/               # 🧠 Neural network weights saved here (.pth files)
├── scripts/              # 📜 Core pipeline processor (run_pipeline.py, stage scripts)
├── tools/                # 🛠️ Helper scripts (setup_colab.py, download_models.py)
├── requirements.txt      # 📦 Python packages list
└── README.md             # 📖 You are here
```

> **Note to Developers / Contributors:** When pushing back to GitHub, ensure you do not upload the huge `.pth` model weights or `output/` folders. This repository relies on `tools/download_models.py` to fetch weights dynamically upon setup to keep the GitHub repo lightweight.
