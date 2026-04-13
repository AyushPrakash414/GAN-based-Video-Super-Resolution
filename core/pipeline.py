# AyushPrakash414
from pathlib import Path
from importlib import import_module
import sys
import time

sys.path.append(str(Path(__file__).resolve().parent.parent / "scripts"))

class VideoEnhancementPipeline:
    """
    Professional Pipeline Orchestrator
    Can be called from:
        - CLI
        - FastAPI
        - Flask
        - Django
        - Background workers
    """

    def __init__(self, logger=print):
        self.logger = logger

        self.stages = [
            ("Stage 01  Normalize", "scripts.stage01_normalize", "main"),
            ("Stage 02  Deinterlace", "scripts.stage02_deinterlace", "main"),
            ("Stage 03  Stabilize", "scripts.stage03_stabilize", "main"),
            ("Stage 04  Deflicker", "scripts.stage04_deflicker", "main"),
            ("Stage 05  Presharpen", "scripts.stage05_presharpen", "main"),
            ("Stage 06  Extract Frames", "scripts.stage06_extract_frames", "main"),
            ("Stage 07  FastDVDnet", "scripts.stage07_fastdvdnet", "run"),
            ("Stage 08  Super Res", "scripts.stage08_superres_simple", "main"),
            ("Stage 09  Detail Refine", "scripts.stage09_detail_refine", "main"),
            ("stage 10  GFP_GAN", "scripts.stage10_gfpgan", "main"),
            ("stage 11  Temporal Refine", "scripts.stage11_temporal_refine", "main"),
            ("stage 12  Reconstructing Video", "scripts.stage12_reconstruct_video", "main"),
        ]

    @staticmethod
    def _load_stage(module_name: str, func_name: str):
        module = import_module(module_name)
        return getattr(module, func_name)

    def run(self, start_stage: int = 0):
        """
        Run full pipeline.

        start_stage:
            Allows resume from any stage.
            Example:
                start_stage=1 → skips stage 0
        """

        self.logger("\n================ PIPELINE START ================\n")

        for i, (name, module_name, func_name) in enumerate(self.stages):

            if i < start_stage:
                self.logger(f" Skipping {name}")
                continue

            self.logger(f" Running {name}")
            t0 = time.time()

            func = self._load_stage(module_name, func_name)
            func()

            self.logger(f" {name} completed in {time.time() - t0:.2f}s\n")

        self.logger("================ PIPELINE COMPLETE ================\n")
