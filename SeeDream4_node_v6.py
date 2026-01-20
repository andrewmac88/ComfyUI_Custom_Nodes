import base64
import io
import json
import os
from typing import List, Tuple, Any, Optional, Dict

import numpy as np
import requests
import torch
from PIL import Image


def _first_image_from_tensor(t: torch.Tensor) -> torch.Tensor:
    x = t
    if x.ndim == 4:
        x = x[0]
    return x


def _tensor_to_png_bytes(image: torch.Tensor) -> bytes:
    x = _first_image_from_tensor(image)
    arr = (x.clamp(0, 1).cpu().numpy() * 255).astype("uint8")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _to_tensor(images: List[Image.Image]) -> Tuple[torch.Tensor]:
    batch = []
    for img in images:
        arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
        batch.append(arr)
    np_batch = np.stack(batch, axis=0)
    return (torch.from_numpy(np_batch),)


def _load_models() -> List[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(here, "models.json")
    if not os.path.exists(p):
        return ["seedream-4-5-251128", "seedream-4-0-250828"]
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("models"), list):
        return [str(m) for m in data["models"] if str(m).strip()]
    if isinstance(data, list):
        return [str(m) for m in data if str(m).strip()]
    return ["seedream-4-5-251128", "seedream-4-0-250828"]


def _download_images(urls: List[str]) -> List[Image.Image]:
    out: List[Image.Image] = []
    with requests.Session() as s:
        for url in urls:
            r = s.get(url, timeout=300)
            r.raise_for_status()
            out.append(Image.open(io.BytesIO(r.content)))
    return out


class ByteDanceSeedreamNode:
    CATEGORY = "api node/image/ByteDance"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "generate"
    OUTPUT_IS_LIST = (False,)

    @classmethod
    def INPUT_TYPES(cls):
        models = _load_models()
        size_presets = [
            "2048x2048 (1:1)",
            "2304x1728 (4:3)",
            "1728x2304 (3:4)",
            "2560x1440 (16:9)",
            "1440x2560 (9:16)",
            "2496x1664 (3:2)",
            "1664x2496 (2:3)",
            "3024x1296 (21:9)",
            "4096x4096 (1:1)",
            "Custom",
        ]
        return {
            "required": {
                "model": (models, {"default": models[0] if models else "seedream-4-0-250828"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "size_preset": (size_presets, {"default": "2048x2048 (1:1)"}),
                "width": ("INT", {"default": 2048, "min": 1024, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 2048, "min": 1024, "max": 4096, "step": 8}),
                "sequential_image_generation": (["disabled", "auto"], {"default": "disabled"}),
                "max_images": ("INT", {"default": 1, "min": 1, "max": 15, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "seed_mode": (["randomize", "fixed"], {"default": "randomize"}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "prompt_in": ("STRING", {"forceInput": True}),
                "image": ("IMAGE", {}),
            },
        }

    def generate(
        self,
        image: Optional[torch.Tensor] = None,
        prompt: str = "",
        prompt_in: Optional[str] = None,
        model: str = "seedream-4-0-250828",
        size_preset: str = "2048x2048 (1:1)",
        width: int = 2048,
        height: int = 2048,
        sequential_image_generation: str = "disabled",
        max_images: int = 1,
        seed: int = 0,
        seed_mode: str = "randomize",
        api_key: str = "",
    ) -> Tuple[torch.Tensor]:
        api_key = (api_key or os.getenv("ARK_API_KEY") or os.getenv("BYTEPLUS_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("API key missing. Provide api_key input.")

        if prompt_in is not None:
            if isinstance(prompt_in, (list, tuple)):
                prompt_in = prompt_in[0] if prompt_in else ""
            final_prompt = str(prompt_in).strip()
        else:
            if isinstance(prompt, (list, tuple)):
                prompt = prompt[0] if prompt else ""
            final_prompt = str(prompt).strip()

        if not final_prompt:
            raise RuntimeError("Prompt missing. Provide prompt.")

        size_map = {
            "2048x2048 (1:1)": (2048, 2048),
            "2304x1728 (4:3)": (2304, 1728),
            "1728x2304 (3:4)": (1728, 2304),
            "2560x1440 (16:9)": (2560, 1440),
            "1440x2560 (9:16)": (1440, 2560),
            "2496x1664 (3:2)": (2496, 1664),
            "1664x2496 (2:3)": (1664, 2496),
            "3024x1296 (21:9)": (3024, 1296),
            "4096x4096 (1:1)": (4096, 4096),
        }
        if size_preset != "Custom" and size_preset in size_map:
            w, h = size_map[size_preset]
        else:
            w, h = int(width), int(height)

        if seed_mode == "randomize":
            seed_to_use = int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF
        else:
            seed_to_use = int(seed)

        endpoint = os.getenv(
            "BYTEPLUS_IMAGE_ENDPOINT",
            "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations",
        )

        payload: Dict[str, Any] = {
            "model": model,
            "prompt": final_prompt,
            "response_format": "url",
            "size": f"{w}x{h}",
            "seed": seed_to_use,
            "sequential_image_generation": sequential_image_generation,
            "sequential_image_generation_options": {"max_images": int(max_images)},
        }

        if image is not None:
            if image.ndim == 4 and image.shape[0] > 1:
                imgs = [image[i] for i in range(min(int(image.shape[0]), 10))]
            else:
                imgs = [image]
            data_urls = []
            for t in imgs:
                raw = _tensor_to_png_bytes(t)
                b64 = base64.b64encode(raw).decode("ascii")
                data_urls.append(f"data:image/png;base64,{b64}")
            payload["image"] = data_urls

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        r = requests.post(endpoint, headers=headers, json=payload, timeout=300)
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            try:
                msg = r.json()
            except Exception:
                msg = r.text
            raise type(e)(f"HTTP {r.status_code}: {msg}")

        resp = r.json()
        if isinstance(resp, dict) and resp.get("error"):
            err = resp.get("error")
            code = err.get("code") if isinstance(err, dict) else None
            message = err.get("message") if isinstance(err, dict) else str(err)
            raise RuntimeError(f"ByteDance request failed. Code: {code}, message: {message}")

        data = resp.get("data") if isinstance(resp, dict) else None
        if not isinstance(data, list) or not data:
            raise RuntimeError(f"Unexpected response: {resp}")

        urls = [str(item.get("url")) for item in data if isinstance(item, dict) and item.get("url")]
        if not urls:
            raise RuntimeError(f"No image urls in response: {resp}")

        images = _download_images(urls)
        return _to_tensor(images)


NODE_CLASS_MAPPINGS = {
    "ByteDanceSeedreamNode": ByteDanceSeedreamNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ByteDanceSeedreamNode": "ByteDance Seedream 4 (API Key)",
}