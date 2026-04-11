#!/usr/bin/env python3

from __future__ import annotations

import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_TARGETS = [
    {
        "name": "GFPGAN v1.4",
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
        "path": PROJECT_ROOT / "models" / "weights" / "GFPGANv1.4.pth",
        "min_size_mb": 300,
    },
    {
        "name": "RealESRGAN x4plus",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "path": PROJECT_ROOT / "models" / "realesrgan" / "RealESRGAN_x4plus.pth",
        "min_size_mb": 50,
    },
    {
        "name": "FastDVDnet",
        "url": "https://github.com/m-tassano/fastdvdnet/raw/master/model.pth",
        "path": PROJECT_ROOT / "models" / "fastdvdnet" / "model.pth",
        "min_size_mb": 5,
    },
]


def progress(block: int, block_size: int, total: int) -> None:
    downloaded = block * block_size
    percent = min(downloaded * 100 / total, 100) if total > 0 else 0
    print(f"\rDownloading: {percent:.1f}%", end="", flush=True)


def download_if_missing(name: str, url: str, path: Path, min_size_mb: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"[skip] {name} already exists at {path} ({size_mb:.2f} MB)")
        return

    print(f"[downloading] {name}")
    print(f"  url : {url}")
    print(f"  path: {path}")
    urllib.request.urlretrieve(url, path, reporthook=progress)
    print()

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"[done] {name} saved ({size_mb:.2f} MB)")

    if size_mb < min_size_mb:
        print(f"[warn] {name} is smaller than expected. Please verify the file.")


def main() -> None:
    print("=" * 72)
    print("Downloading pipeline model weights")
    print("=" * 72)

    for target in MODEL_TARGETS:
        download_if_missing(**target)
        print("-" * 72)

    print("All required model checks completed.")


if __name__ == "__main__":
    main()
