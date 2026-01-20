import base64
import random
import requests
import json
from io import BytesIO

import numpy as np
import torch
from PIL import Image

import folder_paths


class OpenAIImageWithKey:
    """
    Custom OpenAI Image Generation node with user-supplied API key support.
    Supports both text-to-image and image-to-image generation with GPT Image 1 and 1.5 models.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "a beautiful landscape", "multiline": True}),
                "seed": ("INT", {"default": random.randint(0, 2**31 - 1), "min": 0, "max": 2**31 - 1, "step": 1, "control_after_generate": True}),
                "quality": (["low", "medium", "high"], {"default": "low"}),
                "background": (["auto", "opaque", "transparent"], {"default": "auto"}),
                "size": (["auto", "1024x1024", "1024x1536", "1536x1024"], {"default": "auto"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "model": (["gpt-image-1", "gpt-image-1.5"], {"default": "gpt-image-1"}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "image"
    OUTPUT_NODE = False

    def tensor_to_base64(self, image_tensor):
        """Convert tensor to base64 for API with automatic resizing and optimization."""
        try:
            # Convert tensor to PIL Image
            if len(image_tensor.shape) == 4:
                image_tensor = image_tensor.squeeze(0)
            
            # Convert from [H,W,C] to [C,H,W] if needed
            if image_tensor.shape[2] == 3 or image_tensor.shape[2] == 4:
                image_tensor = image_tensor.permute(2, 0, 1)
            
            # Convert to numpy and scale to 0-255
            image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
            
            # Handle different channel counts
            if image_np.shape[0] == 3:
                pil_img = Image.fromarray(image_np.transpose(1, 2, 0), 'RGB')
            elif image_np.shape[0] == 4:
                pil_img = Image.fromarray(image_np.transpose(1, 2, 0), 'RGBA')
            else:
                raise ValueError(f"Unsupported channel count: {image_np.shape[0]}")
            
            # Resize image if it's too large (OpenAI recommends max 2048x2048 for edits)
            max_dim = 2048
            if pil_img.width > max_dim or pil_img.height > max_dim:
                print(f"Resizing image from {pil_img.width}x{pil_img.height} to fit within {max_dim}x{max_dim}")
                pil_img.thumbnail((max_dim, max_dim), Image.LANCZOS)

            # Convert to base64
            buffer = BytesIO()
            pil_img.save(buffer, format='PNG')
            
            # Check file size, if still too large, try reducing quality or further resizing
            if len(buffer.getvalue()) > 4 * 1024 * 1024: # 4MB limit
                print("Image still too large after initial resize, attempting quality reduction.")
                buffer = BytesIO()
                pil_img.save(buffer, format='JPEG', quality=80) # Save as JPEG with reduced quality
                if len(buffer.getvalue()) > 4 * 1024 * 1024:
                    # Further resize if still too large
                    max_dim = 1024
                    pil_img.thumbnail((max_dim, max_dim), Image.LANCZOS)
                    buffer = BytesIO()
                    pil_img.save(buffer, format='JPEG', quality=70)
                    if len(buffer.getvalue()) > 4 * 1024 * 1024:
                        raise ValueError("Image file size remains too large even after compression and resizing. Please use a smaller image.")

            print(f"Final image size: {len(buffer.getvalue()) / 1024 / 1024:.2f} MB")
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Error converting tensor to base64: {e}")
            raise

    def base64_to_tensor(self, base64_str):
        """Convert base64 response to tensor."""
        try:
            image_data = base64.b64decode(base64_str)
            image = Image.open(BytesIO(image_data)).convert('RGBA')
            image_array = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array)
            return image_tensor.unsqueeze(0)
        except Exception as e:
            print(f"Error converting base64 to tensor: {e}")
            raise

    def generate_image(self, api_key, prompt, seed, quality, background, size, n, model, image=None):
        # Validate API key
        if not api_key or api_key.strip() == "":
            raise ValueError("API key is required. Please provide a valid OpenAI API key.")

        if not prompt or prompt.strip() == "":
            raise ValueError("Prompt is required.")

        try:
            # Determine if we're doing image editing or generation
            if image is not None:
                # Image editing mode
                print("Using image editing mode")
                
                # Convert image to base64
                image_base64 = self.tensor_to_base64(image)
                
                # Prepare multipart form data for image editing
                files = {
                    'model': (None, model),
                    'prompt': (None, prompt),
                    'n': (None, str(n)),
                    'size': (None, size),
                    'image': ('image.png', base64.b64decode(image_base64), 'image/png')
                }
                
                # Add model-specific parameters for editing
                # GPT Image models don't use quality parameter in the same way as DALL-E
                # Removing quality parameter to avoid API errors
                if model == "gpt-image-1.5":
                    # Map background to style parameter
                    if background != "auto":
                        files['style'] = (None, background)
                
                headers = {"Authorization": f"Bearer {api_key}"}
                response = requests.post(
                    "https://api.openai.com/v1/images/edits",
                    headers=headers,
                    files=files,
                    timeout=120  # Increased timeout for image uploads
                )
            else:
                # Image generation mode
                print("Using image generation mode")
                
                # Build payload based on model
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "n": n,
                    "size": size,
                }

                # Add model-specific parameters
                # GPT Image models don't use quality parameter in the same way as DALL-E
                # Removing quality parameter to avoid API errors
                if model == "gpt-image-1.5":
                    # Map background to style parameter
                    if background != "auto":
                        payload["style"] = background

                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                print(f"Making request to OpenAI API with model: {model}")
                response = requests.post(
                    "https://api.openai.com/v1/images/generations",
                    headers=headers,
                    json=payload,
                    timeout=60
                )

            # Check response
            if response.status_code == 401:
                raise ValueError("Invalid API key. Please check your OpenAI API key and try again.")
            elif response.status_code == 429:
                raise ValueError("Rate limit exceeded. Please wait a moment and try again.")
            elif response.status_code == 400:
                error_data = response.json() if response.content else {}
                raise ValueError(f"Invalid request: {error_data.get('error', {}).get('message', 'Unknown error')}")
            elif response.status_code != 200:
                raise ValueError(f"API request failed with status {response.status_code}: {response.text}")

            # Parse response
            response_data = response.json()
            
            if "data" not in response_data or len(response_data["data"]) == 0:
                raise ValueError("No images returned from API endpoint")

            # Convert all images to tensors
            image_tensors = []
            for image_data in response_data["data"]:
                # Handle different response formats
                if "b64_json" in image_data:
                    # Base64 response
                    image_tensor = self.base64_to_tensor(image_data["b64_json"])
                elif "url" in image_data:
                    # URL response - download and convert
                    image_url = image_data["url"]
                    print(f"Downloading image from URL: {image_url}")
                    img_response = requests.get(image_url)
                    if img_response.status_code == 200:
                        image = Image.open(BytesIO(img_response.content)).convert('RGBA')
                        image_array = np.array(image).astype(np.float32) / 255.0
                        image_tensor = torch.from_numpy(image_array).unsqueeze(0)
                    else:
                        raise ValueError(f"Failed to download image from URL")
                else:
                    raise ValueError("No image data in response")
                
                image_tensors.append(image_tensor)

            # Stack all images into a single tensor
            if len(image_tensors) > 1:
                final_tensor = torch.cat(image_tensors, dim=0)
            else:
                final_tensor = image_tensors[0]

            print(f"Successfully generated {len(image_tensors)} image(s).")
            
            return (final_tensor,)

        except requests.exceptions.RequestException as e:
            raise ValueError(f"Network error: {str(e)}")
        except Exception as e:
            if "ValueError" in str(type(e)):
                raise
            else:
                raise ValueError(f"Image generation failed: {str(e)}")


# Node mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "OpenAIImageWithKey": OpenAIImageWithKey
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAIImageWithKey": "OpenAI Image With Key"
}