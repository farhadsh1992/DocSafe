


import torch

# Example: Assume img_tensor is a 3D or 4D PyTorch tensor (C, H, W) or (B, C, H, W)
# def rgb_to_grayscale(img_tensor):
#     if img_tensor.dim() == 3:  # (C, H, W)
#         r, g, b = img_tensor[0], img_tensor[1], img_tensor[2]
#     elif img_tensor.dim() == 4:  # (B, C, H, W)
#         r, g, b = img_tensor[:, 0, :, :], img_tensor[:, 1, :, :], img_tensor[:, 2, :, :]
#     else:
#         raise ValueError("Expected input tensor shape (C, H, W) or (B, C, H, W)")

#     # Apply grayscale conversion formula
#     gray_tensor = 0.2989 * r + 0.5870 * g + 0.1140 * b

#     # Add a channel dimension back to maintain (1, H, W) or (B, 1, H, W)
#     return gray_tensor.unsqueeze(0) if img_tensor.dim() == 3 else gray_tensor.unsqueeze(1)


def rgb_to_grayscale(img_tensor):
    """ Convert RGB Tensor to Grayscale using the standard formula. """
    r, g, b = img_tensor[:, 0, :, :], img_tensor[:, 1, :, :], img_tensor[:, 2, :, :]
    gray_tensor = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray_tensor.unsqueeze(1)  # Add channel dimension back
