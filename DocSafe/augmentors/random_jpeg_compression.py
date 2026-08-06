import torch
import numpy as np
import io
import torchvision.transforms as transforms
from PIL import Image

class RandomJPEGCompression():
    def __init__(self, 
                 
                 max_compression=30, 
                 
                 ramp_jpeg=20000,
                 device=""):
        """
        Applies JPEG compression with increasing intensity.

        Parameters:
        - max_compression (int): Max compression level (lowest JPEG quality).
        - ramp_jpeg (int): Number of steps before reaching max compression.
        """
        self.max_compression = max_compression
        self.ramp_jpeg = ramp_jpeg
        self.device = device

    def __call__(self, img_tensor, steps):
        """
        Applies JPEG compression to a batch of images.

        Parameters:
        - img_tensor (Tensor): (B, 3, H, W) tensor in [0, 1] or [-1, 1] range.
        - steps (int): Current training step.

        Returns:
        - Tensor: JPEG compressed batch (B, 3, H, W).
        """
        batch_size = img_tensor.shape[0]
        compressed_images = []

        # Compute JPEG quality
        ramp_fn = lambda steps: int(100 - np.min([steps / self.ramp_jpeg * (100 - self.max_compression), 100 - self.max_compression]))
        quality = ramp_fn(steps)

        for i in range(batch_size):
            img_pil = self.tensor_to_pil(img_tensor[i])  # Convert to PIL
            img_pil = self.apply_jpeg_compression(img_pil, quality)  # Apply JPEG compression
            img_tensor_compressed = self.pil_to_tensor(img_pil)  # Convert back to tensor
            compressed_images.append(img_tensor_compressed)

        return torch.stack(compressed_images).to(self.device)  # Stack batch back together

    @staticmethod
    def tensor_to_pil(img_tensor):
        """Convert tensor image (C, H, W) to PIL Image."""
        img_tensor = img_tensor.clamp(0, 1)  # Ensure valid range
        return transforms.ToPILImage()(img_tensor)

    @staticmethod
    def pil_to_tensor(img_pil):
        """Convert PIL image back to tensor (C, H, W)."""
        return transforms.ToTensor()(img_pil)

    @staticmethod
    def apply_jpeg_compression(img, quality):
        """Compresses the image using JPEG format."""
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer)
