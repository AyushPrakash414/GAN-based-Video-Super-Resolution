# GAN Video Enhancer

This repository contains a 12-stage video restoration pipeline for:

- normalization
- deinterlacing
- stabilization
- deflickering
- presharpening
- frame extraction
- FastDVDnet denoising
- RealESRGAN super-resolution
- detail refinement
- GFPGAN face enhancement
- temporal smoothing
- final video reconstruction

The project is now prepared for two common workflows:

1. pushing the code cleanly to GitHub
2. cloning and running it in Google Colab with minimal setup

## Repository Layout

- `scripts/run_pipeline.py`: runs the full 12-stage pipeline
- `tools/setup_colab.py`: installs Colab system packages, Python dependencies, and model weights
- `tools/download_models.py`: downloads the model weights required by the pipeline
- `input/raw_videos/`: place your source video here
- `output/`: all generated videos and frames are written here

## Before You Push To GitHub

Do not push local runtime artifacts such as:

- `venv/`
- files inside `output/`
- files inside `input/raw_videos/`
- huge downloaded `.pth` weights unless you intentionally want them in Git

This repo already ignores those paths for normal use.

Important GitHub limit:

- `models/weights/GFPGANv1.4.pth` is about `332 MB`, so it cannot be pushed to normal GitHub history.
- `tools/download_models.py` downloads it automatically after clone, which is the recommended approach.

One important note on your machine: Git reported a safe-directory ownership warning for this folder. Run this once before using Git commands here:

```powershell
git config --global --add safe.directory "C:/Users/praka/OneDrive/Desktop/Machine Learning/Deep-Learning"
```

## Push To GitHub

This folder currently sits inside a larger Git repository on your machine. If you want `GAN_Video_Enhancer-main` to become its own standalone GitHub repository, run the commands below from inside this folder so it gets its own `.git` directory.

If this folder is not already a Git repository connected to GitHub, use:

```powershell
git init
git add .
git commit -m "Prepare GAN video enhancer for GitHub and Colab"
git branch -M main
git remote add origin https://github.com/AyushPrakash414/GAN-based-Video-Super-Resolution.git
git push -u origin main
```

If the remote already exists, use:

```powershell
git add .
git commit -m "Prepare repo for Colab"
git push
```

## Google Colab Usage

Open a new Colab notebook and run the following cells.

### 1. Clone your GitHub repository

```python
!git clone https://github.com/AyushPrakash414/GAN-based-Video-Super-Resolution.git
%cd GAN-based-Video-Super-Resolution
```

### 2. Install everything required by the pipeline

```python
!python tools/setup_colab.py
```

This step installs:

- `ffmpeg`
- Python dependencies from `requirements.txt`
- missing FastDVDnet, GFPGAN, and RealESRGAN model weights

### 3. Upload your input video

```python
from google.colab import files
uploaded = files.upload()
```

Then move the uploaded file into the expected input folder:

```python
import os
import shutil

os.makedirs("input/raw_videos", exist_ok=True)

for name in uploaded.keys():
    shutil.move(name, os.path.join("input/raw_videos", name))
    print("Moved:", name)
```

### 4. Run the full pipeline

```python
!python -m scripts.run_pipeline
```

### 5. Download the final output video

```python
from google.colab import files
files.download("output/stage_12_final/final_restored.mp4")
```

## Resume From A Later Stage

If Colab disconnects, you can resume from a specific stage:

```python
!python -m scripts.run_pipeline --start-stage 7
```

Useful examples:

- `--start-stage 6`: start from frame extraction
- `--start-stage 7`: start from FastDVDnet denoising
- `--start-stage 9`: start from GFPGAN

## Local Installation

If you want to run locally instead of Colab:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python tools/download_models.py
python -m scripts.run_pipeline
```

Make sure `ffmpeg` is available on your PATH, or set `FFMPEG_BINARY` to the full executable path.

## Expected Input And Output

- Put one or more source videos in `input/raw_videos/`
- Final result is written to `output/stage_12_final/final_restored.mp4`

Intermediate stages are stored inside `output/` so you can inspect or resume the pipeline.

## Notes

- `scripts/run_pipeline.py` now works with both Windows-style and Linux-style virtual environments.
- Colab normally already includes CUDA-enabled PyTorch, so GPU stages should use CUDA automatically when available.
- If a model file is missing, rerun `python tools/download_models.py`.
