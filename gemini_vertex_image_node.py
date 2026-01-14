import os
import numpy as np
from typing import List, Tuple
from dotenv import load_dotenv

# Vertex AI (provided by google-cloud-aiplatform)
try:
    import vertexai
    from vertexai.preview.vision_models import ImageGenerationModel
except Exception as e:
    raise RuntimeError("vertexai package not available. Ensure google-cloud-aiplatform is installed.") from e


class GeminiVertexImage:
    CATEGORY = "Google/Vertex"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate"
    OUTPUT_IS_LIST = (False,)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A photorealistic banana wearing sunglasses"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "num_images": ("INT", {"default": int(os.getenv("GCP_IMAGE_NUMBER", "1")), "min": 1, "max": 8}),
                "size": ("STRING", {"default": os.getenv("GCP_IMAGE_SIZE", "1024x1024")}),
                "aspect_ratio": ("STRING", {"default": os.getenv("GCP_ASPECT_RATIO", "1:1")}),
                "model_name": ("STRING", {"default": os.getenv("GCP_MODEL", "imagen-3.0-fast")}),
                "project": ("STRING", {"default": os.getenv("GCP_PROJECT_NAME", os.getenv("GCP_PROJECT_ID", ""))}),
                "location": ("STRING", {"default": os.getenv("GCP_LOCATION", "us-central1")}),
            }
        }

    @staticmethod
    def _init_vertex(project: str, location: str):
        # Load environment from .env in project root if available
        try:
            load_dotenv()
        except Exception:
            pass
        if not project:
            # Accept either numeric project number path or project ID
            project = os.getenv("GCP_PROJECT_NAME") or os.getenv("GCP_PROJECT_ID")
        if not location:
            location = os.getenv("GCP_LOCATION", "us-central1")
        if not project:
            raise RuntimeError("Missing GCP project. Set GCP_PROJECT_NAME or GCP_PROJECT_ID in environment or pass via node input.")
        vertexai.init(project=project.replace("projects/", ""), location=location)

    @staticmethod
    def _to_comfy_batch(images: List):
        batch = []
        for img in images:
            # Vertex returns PIL.Image.Image
            arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
            batch.append(arr)
        # Shape: [B, H, W, 3]
        return (np.stack(batch, axis=0),)

    def generate(self, prompt: str, negative_prompt: str = "", num_images: int = 1, size: str = "1024x1024", aspect_ratio: str = "1:1", model_name: str = "imagen-3.0-fast", project: str = "", location: str = "us-central1") -> Tuple[np.ndarray]:
        # Initialize Vertex AI (ADC required: gcloud auth application-default login or service account)
        self._init_vertex(project, location)

        # Choose model
        model_id = model_name or os.getenv("GCP_MODEL", "imagen-3.0-fast")
        model = ImageGenerationModel.from_pretrained(model_id)

        # Compose prompt (basic negative handling)
        full_prompt = prompt if not negative_prompt else f"{prompt}\n\nDo not include: {negative_prompt}"

        # Generate
        images = model.generate_images(
            prompt=full_prompt,
            number_of_images=int(num_images),
            size=size,
            aspect_ratio=aspect_ratio,
        )

        # Convert to ComfyUI batch tensor
        return self._to_comfy_batch(images)


NODE_CLASS_MAPPINGS = {
    "GeminiVertexImage": GeminiVertexImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiVertexImage": "Gemini Vertex Image (Imagen)",
}
