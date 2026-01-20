import base64
import io
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import requests
import torch
from PIL import Image

def _first_image_from_tensor(t: torch.Tensor) -> Image.Image:
    x = t
    if x is None:
        raise ValueError("image is None")
    if x.ndim == 4:
        x = x[0]
    x = (x.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    if x.shape[-1] == 4:
        x = x[..., :3]
    return Image.fromarray(x, mode="RGB")


def _tensor_to_data_uri_png(image: torch.Tensor) -> str:
    pil = _first_image_from_tensor(image)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    raw = buf.getvalue()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _bytes_to_image_tensor(image_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _download_image_to_tensor(url: str, timeout: int = 300) -> torch.Tensor:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return _bytes_to_image_tensor(r.content)


def _parse_aspect_ratio(text: str) -> Tuple[int, int, float]:
    t = (text or "").strip()
    if not t:
        raise ValueError("aspect_ratio is required (e.g. '16:9')")
    if ":" not in t:
        raise ValueError("aspect_ratio must be in W:H format (e.g. '16:9')")
    left, right = t.split(":", 1)
    try:
        w = int(str(left).strip())
        h = int(str(right).strip())
    except Exception as e:
        raise ValueError("aspect_ratio must be two integers separated by ':' (e.g. '16:9')") from e
    if w <= 0 or h <= 0:
        raise ValueError("aspect_ratio values must be positive")
    ratio = float(w) / float(h)
    if ratio < 0.25 or ratio > 4.0:
        raise ValueError("aspect_ratio must be between 1:4 and 4:1 (inclusive)")
    return w, h, ratio


def _round_to_multiple(x: int, multiple: int) -> int:
    if multiple <= 1:
        return int(x)
    return max(multiple, int(round(x / multiple) * multiple))


def _compute_dimensions(
    ratio: float,
    base_long_side: int,
    rounding_multiple: int,
    min_side: int,
    max_side: int,
    max_megapixels: float,
) -> Tuple[int, int]:
    if ratio >= 1.0:
        width = base_long_side
        height = int(round(width / ratio))
    else:
        height = base_long_side
        width = int(round(height * ratio))

    width = _round_to_multiple(width, rounding_multiple)
    height = _round_to_multiple(height, rounding_multiple)

    width = max(min_side, min(max_side, width))
    height = max(min_side, min(max_side, height))

    # Enforce megapixel limit by scaling down if needed
    mp = (width * height) / 1_000_000.0
    if mp > max_megapixels and mp > 0:
        scale = (max_megapixels / mp) ** 0.5
        width = int(width * scale)
        height = int(height * scale)
        width = _round_to_multiple(width, rounding_multiple)
        height = _round_to_multiple(height, rounding_multiple)
        width = max(min_side, min(max_side, width))
        height = max(min_side, min(max_side, height))

    return width, height


def _safe_int_seed(seed: int) -> int:
    # Keep within JS safe integer range (ComfyUI front-end uses JS numbers)
    max_safe = 9007199254740991
    if seed < 0:
        seed = 0
    if seed > max_safe:
        seed = seed % max_safe
    return int(seed)


def _derive_seed(seed: int, seed_mode: str) -> Tuple[int, int]:
    """Return (seed_to_use, seed_next). Seed_next must update even on failure."""
    seed = _safe_int_seed(int(seed))
    mode = (seed_mode or "").strip().lower()
    if mode == "fixed":
        return seed, seed
    if mode == "increment":
        return seed, _safe_int_seed(seed + 1)
    if mode == "decrement":
        return seed, _safe_int_seed(seed - 1)
    # randomize default
    seed_to_use = _safe_int_seed(int.from_bytes(os.urandom(8), "big"))
    return seed_to_use, seed_to_use


def _get_user_config_dir() -> str:
    # Prefer ComfyUI user directory if available
    try