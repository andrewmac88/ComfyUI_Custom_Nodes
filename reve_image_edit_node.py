import os
import io
import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import requests
from dotenv import load_dotenv
import torch


class ReveImageEdit:
    CATEGORY = "REVE/API"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "edit"
    OUTPUT_IS_LIST = (False,)

    @classmethod
    def INPUT_TYPES(cls):
        load_dotenv()
        return {
            "required": {
                "edit_instruction": ("STRING", {"multiline": True, "default": "Remove all of the people in the background from this image."}),
            },
            "optional": {
                "image": ("IMAGE", {}),
                "api_key": ("STRING", {"default": os.getenv("REVE_API_KEY", ""), "multiline": False}),
                "endpoint": ("STRING", {"default": os.getenv("REVE_EDIT_ENDPOINT", "https://api.reve.com/v1/image/edit")}),
                "attach_data_url": ("BOOL", {"default": False}),
                "version": ("STRING", {"default": os.getenv("REVE_EDIT_VERSION", "latest")}),
                "save": ("BOOL", {"default": True}),
                "save_dir": ("STRING", {"default": os.getenv("REVE_SAVE_DIR", "output/reve")}),
                "save_prefix": ("STRING", {"default": os.getenv("REVE_SAVE_PREFIX", "reve_edit_")}),
            }
        }

    @staticmethod
    def _to_tensor(images: List[Image.Image]):
        batch = []
        for img in images:
            arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
            batch.append(arr)
        np_batch = np.stack(batch, axis=0)
        return (torch.from_numpy(np_batch),)

    @staticmethod
    def _encode_image_tensor(image_tensor: Optional[object]) -> Optional[bytes]:
        if image_tensor is None:
            return None
        try:
            if isinstance(image_tensor, torch.Tensor):
                t = image_tensor
                if t.ndim == 4:
                    t = t[0]
                t = t.clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()
                pil = Image.fromarray(t)
            elif isinstance(image_tensor, np.ndarray):
                arr = image_tensor[0] if image_tensor.ndim == 4 else image_tensor
                pil = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
            else:
                return None
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            return buf.getvalue()
        except Exception:
            return None

    @staticmethod
    def _save_images(images: List[Image.Image], save_dir: str, save_prefix: str) -> None:
        try:
            out_dir = Path(save_dir)
            if not out_dir.is_absolute():
                out_dir = Path.cwd() / out_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time())
            for idx, img in enumerate(images):
                b = io.BytesIO()
                img.save(b, format="PNG")
                (out_dir / f"{save_prefix}{ts}_{idx:02d}.png").write_bytes(b.getvalue())
        except Exception:
            pass

    def edit(
        self,
        edit_instruction: str,
        image: Optional[object] = None,
        api_key: str = "",
        endpoint: str = "https://api.reve.com/v1/image/edit",
        attach_data_url: bool = False,
        version: str = "latest",
        save: bool = True,
        save_dir: str = "output/reve",
        save_prefix: str = "reve_edit_",
    ) -> Tuple[torch.Tensor]:
        load_dotenv()
        api_key = api_key or os.getenv("REVE_API_KEY")
        if not api_key:
            raise RuntimeError("REVE_API_KEY not set. Pass api_key or set REVE_API_KEY in .env")

        if not (isinstance(endpoint, str) and endpoint.lower().startswith(("http://", "https://"))):
            endpoint = os.getenv('REVE_EDIT_ENDPOINT', 'https://api.reve.com/v1/image/edit')

        raw = self._encode_image_tensor(image)
        img_b64 = None
        if raw is not None:
            import base64
            b64 = base64.b64encode(raw).decode('utf-8')
            img_b64 = f"data:image/png;base64,{b64}" if attach_data_url else b64

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        payload = {
            "edit_instruction": edit_instruction,
            "version": version,
        }
        if img_b64 is not None:
            payload["reference_image"] = img_b64

        r = requests.post(endpoint, headers=headers, json=payload, timeout=300)
        if r.status_code == 401:
            headers_retry = {
                "X-API-Key": api_key,
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
            r = requests.post(endpoint, headers=headers_retry, json=payload, timeout=300)
        try:
            r.raise_for_status()
        except Exception as e:
            msg = f"HTTP {r.status_code}: {r.text[:500]}" if hasattr(r, 'text') else str(e)
            raise type(e)(msg)
        data = r.json()

        images: List[Image.Image] = []
        if isinstance(data, dict):
            if 'image' in data:
                import base64
                images.append(Image.open(io.BytesIO(base64.b64decode(data['image']))))
            elif 'data' in data and isinstance(data['data'], list):
                for item in data['data']:
                    if isinstance(item, dict):
                        if 'b64_json' in item:
                            import base64
                            images.append(Image.open(io.BytesIO(base64.b64decode(item['b64_json']))))
                        elif 'url' in item:
                            ir = requests.get(item['url'], timeout=300)
                            ir.raise_for_status()
                            images.append(Image.open(io.BytesIO(ir.content)))

        if not images:
            raise RuntimeError(f"REVE edit returned no images. Keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")

        if save:
            self._save_images(images, save_dir, save_prefix)

        return self._to_tensor(images)


NODE_CLASS_MAPPINGS = {
    "ReveImageEdit": ReveImageEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReveImageEdit": "REVE Image Edit",
}
