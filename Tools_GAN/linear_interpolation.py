



"""
@--05.02.2025--@
Author: github/farhadsh1992
INFO:
    -ref: 
        
    
LAST_UPDATE:
"""


import torch

def blend_images(input_images, backgrounds, precent):
    """
    Implements blending similar to `tfa.image.blend` in PyTorch.
    Args:
        input_images (torch.Tensor): Tensor of input images (B, C, H, W).
        backgrounds (torch.Tensor): Tensor of background images (B, C, H, W).
        ramp_fn (torch.Tensor): Alpha blending factor (B, 1, 1, 1) or (B, C, H, W).

    Returns:
        torch.Tensor: Blended image tensor.
    """
    # Ensure ramp_fn is broadcastable
    # ramp_fn = ramp_fn.expand_as(input_images)

    # Perform blending
    output_img = (input_images * precent + backgrounds * (1 - precent))
    return output_img
