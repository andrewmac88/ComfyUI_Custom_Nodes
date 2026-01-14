import io
import os
import base64
from typing import Tuple, List

import numpy as np
import torch
from PIL import Image
import requests


def _tensor_to_png_bytes(image: torch.Tensor) -> bytes:
    x = image
    if x.ndim == 4:
        x = x[0]
    arr = (x.clamp(0, 1).cpu().numpy() * 255).astype('uint8')
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='PNG')
    return buf.getvalue()


def _png_bytes_to_tensor(data: bytes) -> Tuple[torch.Tensor]:
    img = Image.open(io.BytesIO(data)).convert('RGB')
    arr = np.array(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).unsqueeze(0)
    return (t,)


class OpenAIImageEdits:
    CATEGORY = "OpenAI/Remote"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "edit"

    @classmethod
    def INPUT_TYPES(cls):
        # Support up to 6 images as separate optional inputs for convenience
        optional_images = {f"image{i}": ("IMAGE", {}) for i in range(1, 7)}
        optional = {
            **optional_images,
            "prompt": ("STRING", {"multiline": True, "default": ""}),
            "model": ("STRING", {"default": "gpt-image-1"}),
            "size": ("STRING", {"default": "1024x1024"}),
            "n": ("INT", {"default": 1, "min": 1, "max": 4}),
            "endpoint": ("STRING", {"default": "https://api.openai.com/v1/images/edits"}),
            "api_key": ("STRING", {"default": os.getenv("OPENAI_API_KEY", ""), "multiline": False}),
        }
        return {"required": {}, "optional": optional}

    def edit(
        self,
        prompt: str = "",
        model: str = "gpt-image-1",
        size: str = "1024x1024",
        n: int = 1,
        endpoint: str = "https://api.openai.com/v1/images/edits",
        api_key: str = "",
        **kwargs,
    ) -> Tuple[torch.Tensor]:
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Provide via input or .env")

        headers = {"Authorization": f"Bearer {api_key}"}
        files = []
        # Collect provided images from image1..image6
        provided = False
        for i in range(1, 7):
            img = kwargs.get(f"image{i}")
            if img is not None:
                provided = True
                files.append(("image[]", (f"image{i}.png", _tensor_to_png_bytes(img), "image/png")))
        if not provided:
            raise RuntimeError("At least one image input (image1..image6) must be connected")

        data = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "n": str(n),
        }

        r = requests.post(endpoint, headers=headers, data=data, files=files, timeout=300)
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            try:
                msg = r.json()
            except Exception:
                msg = r.text
            # Include X-Request-Id if present
            req_id = r.headers.get("x-request-id") or r.headers.get("X-Request-Id")
            if req_id:
                raise type(e)(f"HTTP {r.status_code}: {msg} (x-request-id={req_id})")
            raise type(e)(f"HTTP {r.status_code}: {msg}")

        resp = r.json()
        data_list = resp.get("data") or []
        if not data_list:
            raise RuntimeError(f"Unexpected response: {resp}")

        # For n>1, return only first image to keep signature simple.
        # Users can run multiple times or we can extend to batch outputs later.
        b64 = data_list[0].get("b64_json")
        if not b64:
            raise RuntimeError(f"No b64_json in response: {resp}")
        img_bytes = base64.b64decode(b64)
        return _png_bytes_to_tensor(img_bytes)


NODE_CLASS_MAPPINGS = {
    "OpenAIImageEdits": OpenAIImageEdits,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAIImageEdits": "OpenAI Image Edits (gpt-image-1)",
}
