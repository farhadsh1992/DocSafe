
"""
@--03.02.2025--@
Author: github/farhadsh1992
INFO:
	- StampOne v89 torch format
    - REF: 
		

    
LAST_UPDATE:
"""
#############################################################################################################
#####                                               #####
############################################################################################################

import torch
from torch import nn
import torch.nn.functional as F
from math import exp
import numpy as np

#############################################################################################################
#####                                               #####
############################################################################################################

class YUVLoss(nn.Module):
    def __init__(self, input_size, yuv_scales_pl, device):
        super(YUVLoss, self).__init__()

        self.device = device
        self.size = (input_size, input_size) 
       
        self.yuv_scales = yuv_scales_pl.to(self.device)

    def forward(self, encoded_image, image_input, l2_edge_gain):
        encoded_image = F.interpolate(encoded_image, 
                    size=self.size, 
                    mode="nearest")
        image_input = F.interpolate(image_input, 
                    size=self.size, 
                    mode="nearest")
        ##########################################################################
        falloff_speed = 4 # Cos dropoff that reaches 0 at distance 1/x into image
        falloff_im = np.ones(self.size)
        for i in range(int(falloff_im.shape[0]/falloff_speed)):
            falloff_im[-i,:] *= (np.cos(4*np.pi*i/self.size[0]+np.pi)+1)/2
            falloff_im[i,:] *= (np.cos(4*np.pi*i/self.size[0]+np.pi)+1)/2
        for j in range(int(falloff_im.shape[1]/falloff_speed)):
            falloff_im[:,-j] *= (np.cos(4*np.pi*j/self.size[0]+np.pi)+1)/2
            falloff_im[:,j] *= (np.cos(4*np.pi*j/self.size[0]+np.pi)+1)/2
        falloff_im = 1-falloff_im
        falloff_im = torch.from_numpy(falloff_im.astype(np.float32)).to(self.device)
        falloff_im *= l2_edge_gain
        ##########################################################################

        encoded_image_yuv = rgb_to_yuv(encoded_image).to(self.device)
        image_input_yuv = rgb_to_yuv(image_input).to(self.device)
        im_diff = (encoded_image_yuv-image_input_yuv).to(self.device)
        im_diff += im_diff * torch.unsqueeze(falloff_im, dim=0).to(self.device)
        yuv_loss_op = torch.mean(torch.square(im_diff), dim=[0,2,3]).to(self.device)
  
        image_loss_op = torch.tensordot(yuv_loss_op, self.yuv_scales, dims=1).to(self.device)
        return image_loss_op





#############################################################################################################
#####                                               #####
############################################################################################################
def rgb_to_yuv(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YUV.

    .. image:: _static/img/rgb_to_yuv.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: RGB Image to be converted to YUV with shape :math:`(*, 3, H, W)`.

    Returns:
        YUV version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_yuv(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    u: torch.Tensor = -0.147 * r - 0.289 * g + 0.436 * b
    v: torch.Tensor = 0.615 * r - 0.515 * g - 0.100 * b

    out: torch.Tensor = torch.stack([y, u, v], -3)

    return out


#############################################################################################################
#####                                               #####
############################################################################################################