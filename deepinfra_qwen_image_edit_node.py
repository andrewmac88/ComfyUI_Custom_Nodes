import io
import os
import base64
from typing import Tuple

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


class DeepinfraQwenImageEdit:
    CATEGORY = "DeepInfra/Remote"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "edit"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
            },
            "optional": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "size": ("STRING", {"default": "1024x1024"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 4}),
                "endpoint": ("STRING", {"default": "https://api.deepinfra.com/v1/openai/images/edits"}),
                "model": ("STRING", {"default": "Qwen/Qwen-Image-Edit"}),
                "api_key": ("STRING", {"default": os.getenv("DEEPINFRA_TOKEN", ""), "multiline": False}),
            },
        }

    def edit(
        self,
        image: torch.Tensor,
        prompt: str = "",
        size: str = "1024x1024",
        n: int = 1,
        endpoint: str = "https://api.deepinfra.com/v1/openai/images/edits",
        model: str = "Qwen/Qwen-Image-Edit",
        api_key: str = "",
    ) -> Tuple[torch.Tensor]:
        if not api_key:
            api_key = os.getenv("DEEPINFRA_TOKEN", "")
        if not api_key:
            raise RuntimeError("DEEPINFRA_TOKEN is not set. Provide via input or .env")

        headers = {
            "Authorization": f"Bearer {api_key}",
        }

        files = {
            "image": ("image.png", _tensor_to_png_bytes(image), "image/png"),
        }
        data = {
            "model": model,
            "n": str(n),
            "size": size,
        }
        if prompt:
            data["prompt"] = prompt

        r = requests.post(endpoint, headers=headers, data=data, files=files, timeout=180)
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            try:
                msg = r.json()
            except Exception:
                msg = r.text
            raise type(e)(f"HTTP {r.status_code}: {msg}")

        resp = r.json()
        # Expect OpenAI-compatible response with data[0].b64_json
        if not isinstance(resp, dict) or "data" not in resp or not resp["data"]:
            raise RuntimeError(f"Unexpected response: {resp}")
        b64 = resp["data"][0].get("b64_json")
        if not b64:
            raise RuntimeError(f"No b64_json in response: {resp}")
        img_bytes = base64.b64decode(b64)
        return _png_bytes_to_tensor(img_bytes)


NODE_CLASS_MAPPINGS = {
    "DeepinfraQwenImageEdit": DeepinfraQwenImageEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DeepinfraQwenImageEdit": "DeepInfra Qwen Image Edit",
}
