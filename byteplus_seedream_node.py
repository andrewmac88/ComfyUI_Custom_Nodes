import os
import base64
import io
from typing import List, Tuple

import numpy as np
from PIL import Image
import httpx
from dotenv import load_dotenv


class BytePlusSeedreamGenerate:
    CATEGORY = "BytePlus/Ark"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate"
    OUTPUT_IS_LIST = (False,)

    @classmethod
    def INPUT_TYPES(cls):
        load_dotenv()
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Interstellar travel, a black hole..."}),
            },
            "optional": {
                "api_key": ("STRING", {"default": os.getenv("BYTEPLUS_API_KEY", ""), "multiline": False}),
                "model": ("STRING", {"default": os.getenv("BYTEPLUS_MODEL", "seedream-4-0-250828")}),
                "size": ("STRING", {"default": os.getenv("BYTEPLUS_SIZE", "2K")}),
                "response_format": ("STRING", {"default": os.getenv("BYTEPLUS_RESPONSE_FORMAT", "url")}),
                "sequential_image_generation": ("STRING", {"default": os.getenv("BYTEPLUS_SEQ", "disabled")}),
                "stream": ("BOOL", {"default": False}),
                "watermark": ("BOOL", {"default": True}),
                "endpoint": ("STRING", {"default": os.getenv("BYTEPLUS_ENDPOINT", "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations")}),
            }
        }

    @staticmethod
    def _to_comfy(images: List[Image.Image]) -> Tuple[np.ndarray]:
        batch = []
        for img in images:
            arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
            batch.append(arr)
        return (np.stack(batch, axis=0),)

    def generate(
        self,
        prompt: str,
        api_key: str = "",
        model: str = "seedream-4-0-250828",
        size: str = "2K",
        response_format: str = "url",
        sequential_image_generation: str = "disabled",
        stream: bool = False,
        watermark: bool = True,
        endpoint: str = "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations",
    ) -> Tuple[np.ndarray]:
        load_dotenv()
        api_key = api_key or os.getenv("BYTEPLUS_API_KEY")
        if not api_key:
            raise RuntimeError("BYTEPLUS_API_KEY not set. Pass api_key or set in environment/.env")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "model": model,
            "prompt": prompt,
            "sequential_image_generation": sequential_image_generation,
            "response_format": response_format,
            "size": size,
            "stream": stream,
            "watermark": watermark,
        }

        # Send request
        with httpx.Client(timeout=120) as client:
            r = client.post(endpoint, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()

        images: List[Image.Image] = []
        # Try multiple common response shapes
        # 1) { data: [ { url: ... }, ... ] }
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            for item in data["data"]:
                if response_format == "url" and isinstance(item, dict) and "url" in item:
                    url = item["url"]
                    with httpx.Client(timeout=120) as client:
                        ir = client.get(url)
                        ir.raise_for_status()
                        images.append(Image.open(io.BytesIO(ir.content)))
                elif response_format in ("b64", "base64") and isinstance(item, dict) and "b64_json" in item:
                    raw = base64.b64decode(item["b64_json"])  # bytes
                    images.append(Image.open(io.BytesIO(raw)))
        # 2) { url: "..." }
        elif isinstance(data, dict) and "url" in data:
            url = data["url"]
            with httpx.Client(timeout=120) as client:
                ir = client.get(url)
                ir.raise_for_status()
                images.append(Image.open(io.BytesIO(ir.content)))
        else:
            # Fallback: try to parse single field 'image' base64
            b64 = data.get("image") if isinstance(data, dict) else None
            if b64:
                images.append(Image.open(io.BytesIO(base64.b64decode(b64))))

        if not images:
            raise RuntimeError(f"BytePlus API returned no images. Response keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")

        return self._to_comfy(images)


NODE_CLASS_MAPPINGS = {
    "BytePlusSeedreamGenerate": BytePlusSeedreamGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BytePlusSeedreamGenerate": "BytePlus Seedream Generate",
}
