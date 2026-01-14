import io
import os
import json
import base64
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
import requests


def _to_tensor(images: List[Image.Image]):
    batch = []
    for img in images:
        arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
        batch.append(arr)
    np_batch = np.stack(batch, axis=0)
    return (torch.from_numpy(np_batch),)


def _first_image_from_tensor(t: torch.Tensor) -> Image.Image:
    x = t
    if x.ndim == 4:
        x = x[0]
    x = (x.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    return Image.fromarray(x)


class FireworksImageToImage:
    CATEGORY = "Fireworks/Remote"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "edit"
    OUTPUT_IS_LIST = (False,)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "endpoint": ("STRING", {"default": "https://api.fireworks.ai/inference/v1/images/generations"}),
                "api_key": ("STRING", {"default": os.getenv("FIREWORKS_API_KEY", ""), "multiline": False}),
                "model": ("STRING", {"default": "accounts/fireworks/models/flux-pro-1.1"}),
                "response_format": ("STRING", {"default": "b64_json"}),
                "size": ("STRING", {"default": "1024x1024"}),
                "user": ("STRING", {"default": ""}),
            }
        }

    def edit(
        self,
        image: torch.Tensor,
        prompt: str,
        endpoint: str = "https://api.fireworks.ai/inference/v1/images/generations",
        api_key: str = "",
        model: str = "accounts/fireworks/models/flux-pro-1.1",
        response_format: str = "b64_json",
        size: str = "1024x1024",
        user: str = "",
    ) -> Tuple[torch.Tensor]:
        if not api_key:
            api_key = os.getenv("FIREWORKS_API_KEY", "")
        if not api_key:
            raise RuntimeError("FIREWORKS_API_KEY is not set. Provide via input or .env")

        pil_image = _first_image_from_tensor(image)
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        buf.seek(0)

        headers = {
            "Authorization": f"Bearer {api_key}",
        }

        # Build payload for Fireworks
        data = {
            "prompt": prompt,
            "model": model,
            "size": size,
            "response_format": response_format,
        }
        if user:
            data["user"] = user
        # Generations use JSON; edits use multipart with image
        if endpoint.endswith("/images/generations") or "/images/generations" in endpoint:
            headers_json = {**headers, "Content-Type": "application/json"}
            r = requests.post(endpoint, headers=headers_json, json=data, timeout=180)
        else:
            files = {"image": ("image.png", buf, "image/png")}
            r = requests.post(endpoint, headers=headers, data=data, files=files, timeout=180)
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            raise type(e)(f"HTTP {r.status_code}: {r.text}")

        try:
            resp = r.json()
        except Exception:
            # Try treat as raw image
            try:
                out_img = Image.open(io.BytesIO(r.content)).convert("RGB")
            except Exception:
                raise RuntimeError("Unexpected response from Fireworks (not JSON or image)")
            return _to_tensor([out_img])

        # Parse OpenAI-style image response
        if isinstance(resp, dict) and "data" in resp and resp["data"]:
            item = resp["data"][0]
            if "b64_json" in item:
                img_bytes = base64.b64decode(item["b64_json"])  # type: ignore
                out_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                return _to_tensor([out_img])
            if "url" in item:
                img_r = requests.get(item["url"], timeout=120)
                img_r.raise_for_status()
                out_img = Image.open(io.BytesIO(img_r.content)).convert("RGB")
                return _to_tensor([out_img])

        raise RuntimeError(f"Unrecognized response from Fireworks: {json.dumps(resp)[:800]}")


NODE_CLASS_MAPPINGS = {
    "FireworksImageToImage": FireworksImageToImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FireworksImageToImage": "Fireworks Image to Image",
}
