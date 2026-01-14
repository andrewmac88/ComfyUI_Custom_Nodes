import numpy as np
import torch
import math
from PIL import Image
from typing import Tuple, Optional


def _tensor_to_pil(img_t: torch.Tensor) -> Image.Image:
    # img_t: [B,H,W,C] or [H,W,C]; values 0..1
    t = img_t
    if t.ndim == 4:
        t = t[0]
    t = (t.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    if t.shape[-1] == 4:
        return Image.fromarray(t, mode="RGBA")
    return Image.fromarray(t, mode="RGB")


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    if img is None:
        return
    arr = np.array(img).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    if arr.shape[-1] == 4:
        # drop alpha for ComfyUI IMAGE (expects 3 channels)
        arr = arr[:, :, :3]
    t = torch.from_numpy(arr.astype(np.float32) / 255.0)
    return t.unsqueeze(0)


def _ensure_rgb_and_resize(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    if img.mode == "RGBA":
        # flatten alpha over white background
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    if img.size != size:
        img = img.resize(size, Image.LANCZOS)
    return img


class dynamicGrid:
    CATEGORY = "Utils/Grids"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "make"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE", {}),
                "image2": ("IMAGE", {}),
                "image3": ("IMAGE", {}),
                "image4": ("IMAGE", {}),
                "resize_mode": ( ["max", "min", "first", "fixed"], {"default": "max"} ),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
            },
            "optional": {
                "image5": ("IMAGE", {}),
                "image6": ("IMAGE", {}),
                "image7": ("IMAGE", {}),
                "image8": ("IMAGE", {}),
                "image9": ("IMAGE", {}),
                "image10": ("IMAGE", {})
            }
   
        }

    def make(self, image1, image2, image3, image4,
             image5 = None,   image6 = None,  image7 = None,  image8 = None,  image9 = None,  image10 = None, 
             resize_mode: str = "max", width: int = 1024, height: int = 1024):
        pils = [_tensor_to_pil(x) for x in (image1, image2, image3, image4, image5, image6, image7, image8, image9, image10, ) if x is not None]
        sizes = [im.size for im in pils]
        if resize_mode == "first":
            target = sizes[0]
        elif resize_mode == "min":
            target = (min(s[0] for s in sizes), min(s[1] for s in sizes))
        elif resize_mode == "fixed":
            target = (int(width), int(height))
        else:  # max
            target = (max(s[0] for s in sizes), max(s[1] for s in sizes))

        pils = [_ensure_rgb_and_resize(im, target) for im in pils]
        num_images = len(pils)
         # Determine grid layout
        cols = math.ceil(math.sqrt(num_images))       # number of columns
        rows = math.ceil(num_images / cols)           # number of rows

        
        w, h = target
        canvas = Image.new("RGB", (w * cols, h * rows), (255, 255, 255))
        # canvas.paste(pils[0], (0, 0))
        # canvas.paste(pils[1], (w, 0))
        # canvas.paste(pils[2], (0, h))
        # canvas.paste(pils[3], (w, h))
            # Paste images dynamically
        for idx, im in enumerate(pils):
            row = idx // cols
            col = idx % cols
            canvas.paste(im, (col * w, row * h))

        out_t = _pil_to_tensor(canvas)
        return (out_t,)


NODE_CLASS_MAPPINGS = {
    "dynamicGrid": dynamicGrid,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "dynamicGrid": "dynamicGrid",
}