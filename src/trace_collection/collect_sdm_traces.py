"""
collect_sdm_traces.py — Collect SDM v1.4 activation traces for Ditto reproduction.

Runs Stable Diffusion v1.4 in PLMS 50-step mode on 20 COCO captions, dumping
the input and output of 6 representative UNet layers at every denoising step.

Output: ~30 GB of .npz files in ~/Ditto/traces/sdm/
        Organized as: image_<i>/step_<t>/layer_<name>.npz

Each .npz contains:
  input:   fp16 tensor, shape [B, C, H, W] or [B, T, D]
  output:  fp16 tensor, same shape
  timestep: int (denoising step index)
  layer:   str (layer name)

Author: njzhx213
Project: Ditto Reproduction (HPCA 2025)
Phase: Week 1, Day 1 evening — kick off trace collection in background

Usage (foreground, watch progress):
    python src/trace_collection/collect_sdm_traces.py

Usage (background):
    nohup python src/trace_collection/collect_sdm_traces.py \\
        > ~/Ditto/traces/sdm/collection.log 2>&1 &
    echo $! > ~/Ditto/traces/sdm/collection.pid
    
    # Monitor:
    tail -f ~/Ditto/traces/sdm/collection.log
    # Check if alive:
    ps -p $(cat ~/Ditto/traces/sdm/collection.pid)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import StableDiffusionPipeline, PNDMScheduler
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# 6 representative UNet layers — covers down/mid/up + attention/conv mix.
# Paper Fig 3a mentions "conv-in" and "up.0.0.skip" explicitly; the others are
# spread across the depth of the network to sample attention at different
# resolutions.
TARGET_LAYERS = [
    "conv_in",                              # initial conv, channels [4→320]
    "down_blocks.1.attentions.0",           # early attention, [4096, 640]
    "down_blocks.2.attentions.0",           # mid-depth attention, [1024, 1280]
    "mid_block.attentions.0",               # bottleneck attention, [256, 1280]
    "up_blocks.1.attentions.0",             # upsampling attention, [1024, 1280]
    "up_blocks.2.attentions.0",             # late attention, [4096, 640]
]

# Trace destination
OUTPUT_ROOT = Path.home() / "Ditto" / "traces" / "sdm"

# Module-level state, set by run_trace_collection before each image, read by custom_callback.
# diffusers' callback_on_step_end signature doesn't allow passing arbitrary kwargs through,
# so we go through module globals instead.
_ACTIVE_COLLECTOR = None
_ACTIVE_IMAGE_IDX = -1

# SDM v1.4 config (paper Table I)
MODEL_ID = "CompVis/stable-diffusion-v1-4"
NUM_INFERENCE_STEPS = 50
NUM_IMAGES = 20
GUIDANCE_SCALE = 7.5
HEIGHT, WIDTH = 512, 512    # native SDM v1.4 resolution
SEED_BASE = 42

# COCO captions — fetched from the 2017 annotations file. The file is ~250MB
# uncompressed but we only need the 100 captions list; we cache it to ~/Ditto.
COCO_CACHE = OUTPUT_ROOT.parent / "coco_captions_100.json"
COCO_DOWNLOAD_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt loading
# ─────────────────────────────────────────────────────────────────────────────

def get_coco_captions(n: int = 100) -> list[str]:
    """Load 100 COCO captions. Cache to disk for re-runs."""
    if COCO_CACHE.exists():
        with open(COCO_CACHE) as f:
            data = json.load(f)
        print(f"[prompts] Loaded {len(data)} captions from cache: {COCO_CACHE}")
        return data[:n]

    print(f"[prompts] Downloading COCO 2017 captions (~250 MB)...")
    import urllib.request
    import zipfile
    import tempfile

    tmp_zip = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    tmp_zip.close()

    urllib.request.urlretrieve(COCO_DOWNLOAD_URL, tmp_zip.name)
    print(f"[prompts] Downloaded to {tmp_zip.name}")

    captions = []
    with zipfile.ZipFile(tmp_zip.name, "r") as z:
        # We want annotations/captions_val2017.json (~250MB) — the smaller val set
        target = "annotations/captions_val2017.json"
        with z.open(target) as f:
            ann_data = json.load(f)

    # ann_data has structure: {"annotations": [{"caption": str, ...}, ...]}
    captions = [a["caption"].strip() for a in ann_data["annotations"]][:n]
    print(f"[prompts] Extracted {len(captions)} captions")

    # Cache
    COCO_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(COCO_CACHE, "w") as f:
        json.dump(captions, f, indent=2)
    print(f"[prompts] Cached to {COCO_CACHE}")

    return captions[:n]


# ─────────────────────────────────────────────────────────────────────────────
# Hook machinery
# ─────────────────────────────────────────────────────────────────────────────

class TraceCollector:
    """
    Collects layer inputs/outputs for one (image, step) pair.

    The hook stores tensors in `self.current_step_buffer`. After each
    denoising step completes, the caller flushes this buffer to disk
    by calling `flush_step(image_idx, step_idx)`.
    """

    def __init__(self, output_root: Path, layer_names: list[str]):
        self.output_root = output_root
        self.layer_names = layer_names
        self.current_step_buffer: dict[str, dict[str, np.ndarray]] = {}
        self.handles: list[Any] = []
        self.bytes_written = 0
        self.files_written = 0

    def attach(self, model: torch.nn.Module) -> None:
        """Register forward hooks on the target layers."""
        name_to_module = dict(model.named_modules())
        missing = []
        for name in self.layer_names:
            if name not in name_to_module:
                missing.append(name)
                continue
            module = name_to_module[name]
            handle = module.register_forward_hook(
                self._make_hook(name)
            )
            self.handles.append(handle)
        if missing:
            print(f"[trace] ⚠ WARNING: {len(missing)} layers not found:")
            for m in missing:
                print(f"          {m}")
            print(f"[trace] Available top-level modules:")
            for k in list(name_to_module.keys())[:30]:
                if k.count(".") <= 1:
                    print(f"          {k}")
            sys.exit(1)
        print(f"[trace] ✓ Attached {len(self.handles)} hooks")

    def detach(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def _make_hook(self, layer_name: str):
        """Create a closure capturing layer_name for the hook."""
        def hook(module, inputs, output):
            # inputs is a tuple; we take inputs[0] as the primary tensor.
            # output may be a tensor or (for attention blocks) a special obj.
            in_tensor = self._extract_primary_tensor(inputs[0] if inputs else None)
            out_tensor = self._extract_primary_tensor(output)

            self.current_step_buffer[layer_name] = {
                "input": in_tensor,
                "output": out_tensor,
            }
        return hook

    @staticmethod
    def _extract_primary_tensor(x) -> np.ndarray | None:
        """Pull the primary tensor out of a (possibly nested) hook payload."""
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().to(torch.float16).numpy()
        # Some attention modules return a tuple (out, attn_weights, ...)
        if isinstance(x, tuple) and len(x) > 0 and isinstance(x[0], torch.Tensor):
            return x[0].detach().cpu().to(torch.float16).numpy()
        # Some return a custom object with .sample attribute (diffusers convention)
        if hasattr(x, "sample") and isinstance(x.sample, torch.Tensor):
            return x.sample.detach().cpu().to(torch.float16).numpy()
        return None  # unknown payload type — silently skip

    def flush_step(self, image_idx: int, step_idx: int) -> int:
        """
        Write current_step_buffer to disk under traces/sdm/image_<i>/step_<t>/.
        Returns the number of bytes written.
        """
        step_dir = self.output_root / f"image_{image_idx:03d}" / f"step_{step_idx:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        bytes_this_step = 0
        for layer_name, payload in self.current_step_buffer.items():
            # Sanitize layer name for filename (replace dots with underscores)
            safe_name = layer_name.replace(".", "_")
            out_path = step_dir / f"{safe_name}.npz"

            np.savez_compressed(
                out_path,
                input=payload["input"] if payload["input"] is not None else np.array([]),
                output=payload["output"] if payload["output"] is not None else np.array([]),
                layer_name=layer_name,
                timestep=step_idx,
                image_idx=image_idx,
            )
            bytes_this_step += out_path.stat().st_size
            self.files_written += 1

        self.bytes_written += bytes_this_step
        self.current_step_buffer = {}
        return bytes_this_step


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def setup_pipeline() -> StableDiffusionPipeline:
    """Load SDM v1.4 in fp16 on GPU, configure PLMS sampler."""
    print(f"[setup] Loading {MODEL_ID}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        safety_checker=None,           # skip — we don't need safety filter
        requires_safety_checker=False,
    )

    # Use PNDM scheduler (PLMS variant, paper Table I "PLMS 50 step")
    pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)  # we use our own tqdm

    # Verify GPU
    print(f"[setup] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[setup] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    return pipe


def custom_callback(pipe, step_idx, timestep, callback_kwargs):
    """
    Called after each denoising step by diffusers pipeline.
    Reads collector + image_idx from module globals (set by run_trace_collection).
    """
    if _ACTIVE_COLLECTOR is not None and _ACTIVE_IMAGE_IDX >= 0:
        _ACTIVE_COLLECTOR.flush_step(_ACTIVE_IMAGE_IDX, step_idx)
    return callback_kwargs


def run_trace_collection(
    pipe: StableDiffusionPipeline,
    prompts: list[str],
    resume_from: int = 0,
) -> None:
    """Main trace collection loop with progress + resume support."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    collector = TraceCollector(OUTPUT_ROOT, TARGET_LAYERS)
    collector.attach(pipe.unet)

    print(f"[trace] Starting collection: {NUM_IMAGES} images × {NUM_INFERENCE_STEPS} steps × {len(TARGET_LAYERS)} layers")
    print(f"[trace] Output: {OUTPUT_ROOT}")
    if resume_from > 0:
        print(f"[trace] Resuming from image {resume_from}")

    start_time = time.time()
    images_completed = resume_from

    # Outer loop: 20 images
    for img_idx in tqdm(range(resume_from, NUM_IMAGES), desc="Images", file=sys.stdout):
        # Skip if already done
        marker = OUTPUT_ROOT / f"image_{img_idx:03d}" / "DONE"
        if marker.exists():
            tqdm.write(f"[trace] image_{img_idx:03d} already complete, skipping")
            continue

        prompt = prompts[img_idx]
        tqdm.write(f"\n[trace] Image {img_idx:03d}: \"{prompt[:80]}{'...' if len(prompt) > 80 else ''}\"")

        # Set module-level state for callback to read.
        global _ACTIVE_COLLECTOR, _ACTIVE_IMAGE_IDX
        _ACTIVE_COLLECTOR = collector
        _ACTIVE_IMAGE_IDX = img_idx

        # Generate with custom callback to flush per-step
        _ = pipe(
            prompt=prompt,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            height=HEIGHT,
            width=WIDTH,
            generator=torch.Generator(device="cuda").manual_seed(SEED_BASE + img_idx),
            callback_on_step_end=custom_callback,
            callback_on_step_end_tensor_inputs=["latents"],
        )

        # Clear after image is done (defensive — not strictly needed)
        _ACTIVE_COLLECTOR = None
        _ACTIVE_IMAGE_IDX = -1

        # Mark image as done
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()

        images_completed += 1
        elapsed = time.time() - start_time
        rate = (images_completed - resume_from) / elapsed if elapsed > 0 else 0
        eta = (NUM_IMAGES - images_completed) / rate if rate > 0 else 0
        tqdm.write(
            f"[trace] Done {images_completed}/{NUM_IMAGES}  "
            f"({collector.bytes_written / 1e9:.2f} GB, "
            f"{collector.files_written} files, "
            f"{rate*60:.1f} img/min, "
            f"ETA {eta/60:.1f} min)"
        )

    collector.detach()

    total_time = time.time() - start_time
    print(f"\n[trace] ✓ Complete.")
    print(f"[trace]   Total: {images_completed} images, {total_time/60:.1f} minutes")
    print(f"[trace]   Storage: {collector.bytes_written / 1e9:.2f} GB across {collector.files_written} files")
    print(f"[trace]   Output: {OUTPUT_ROOT}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global NUM_IMAGES

    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", type=int, default=0,
                    help="Skip first N images (for resume after interruption)")
    ap.add_argument("--num-images", type=int, default=NUM_IMAGES,
                    help=f"Number of images to generate (default {NUM_IMAGES})")
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip pipeline; only verify environment + layer names")
    args = ap.parse_args()

    NUM_IMAGES = args.num_images

    # 1. Load prompts
    prompts = get_coco_captions(NUM_IMAGES)
    if len(prompts) < NUM_IMAGES:
        print(f"[main] ⚠ Only got {len(prompts)} prompts; reducing NUM_IMAGES")
        NUM_IMAGES = len(prompts)

    # 2. Setup pipeline
    pipe = setup_pipeline()

    # 3. Dry run: just check layer names and exit
    if args.dry_run:
        print(f"[dry-run] Checking layer names...")
        collector = TraceCollector(OUTPUT_ROOT, TARGET_LAYERS)
        collector.attach(pipe.unet)
        collector.detach()
        print(f"[dry-run] ✓ All {len(TARGET_LAYERS)} layers found. Exiting.")
        return

    # 4. Run
    run_trace_collection(pipe, prompts, resume_from=args.resume)


if __name__ == "__main__":
    main()
