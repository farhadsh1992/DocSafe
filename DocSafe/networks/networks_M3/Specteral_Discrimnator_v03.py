
"""
@--12.04.2025--@
Author: github/farhadsh1992
INFO:
	- Rimman Loss
    - REF: 
    
		

    
LAST_UPDATE:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import math
from FarhadCV.Tools import tcolors, bcolors

class SpectralDiscriminator(nn.Module):
    """
    INFO:
        - https://arxiv.org/pdf/2012.05535.pdf 
        - We adopt the Spectral discriminator in https://github.com/cyq373/SSD-GAN
        - To guide the generator to generate more realistic images by reducing the gap 
        between the clean and denoised image in the frequency domain. 
    """
    def __init__(self, batch_size, height, device=None):
        super(SpectralDiscriminator, self).__init__()

        self.device = device
        self.thresh = int(height / (2 * math.sqrt(2)))
        self.linear = nn.Linear(self.thresh, 1)

    def high_pass_filtered(self, x):
        out = torch.zeros(x.shape, dtype=x.dtype)
        B, C, H, W = x.shape  # assuming input shape (bchw)
        out[:, :, :int(H/2), :int(W/2)] = x[:, :, int(H/2):, int(W/2):]
        out[:, :, :int(H/2), int(W/2):] = x[:, :, int(H/2):, :int(W/2)]
        out[:, :, int(H/2):, :int(W/2)] = x[:, :, :int(H/2), int(W/2):]
        out[:, :, int(H/2):, int(W/2):] = x[:, :, :int(H/2), :int(W/2)]
        return out

    # def azimuthal_average(self, image, center=None):
    #     H, W = image.shape[-2], image.shape[-1]
    #     y, x = np.indices((H, W))
    #     radius = np.sqrt((x - H/2)**2 + (y - W/2)**2)
    #     radius = radius.astype(np.int32).ravel()
    #     nr = np.bincount(radius)
    #     tbin = np.bincount(radius, image.ravel())
    #     radial_prof = tbin / (nr + 1e-10)
    #     return radial_prof[1:-2]
    def azimuthal_average(self, image, center=None):
        H, W = image.shape[-2], image.shape[-1]
        y, x = torch.from_numpy(np.indices((H, W)))
        radius = torch.sqrt((x - H/2)**2 + (y - W/2)**2)
        radius = radius.type(torch.int32).ravel().to(self.device)
        nr = torch.bincount(radius)
        tbin = torch.bincount(radius, image.ravel())
        radial_prof = tbin / (nr + 1e-10)
        return radial_prof[1:-2]

    def get_fft_feature(self, xrgb):
        epsilon = 1e-8
        x_gray = torchvision.transforms.functional.rgb_to_grayscale(xrgb)#.type('torch.float32')  # convert to grayscale 
        fft = calculate_2dft(x_gray)
        fft += epsilon
        magnitude_spectrum = torch.log(torch.sqrt(fft.real**2 + fft.imag**2 + 1e-10) + 1e-10)
        
        magnitude_spectrum = self.high_pass_filtered(magnitude_spectrum).to(self.device)

        out = []
        for i in range(magnitude_spectrum.shape[0]):
            out.append(torch.unsqueeze(torch.tensor(self.azimuthal_average(magnitude_spectrum[i]), 
                                        dtype=torch.float32), dim=0))
        out = torch.cat(out, dim=0)

        out = (out - torch.min(out, dim=1, keepdim=True)[0]) / (torch.max(out, dim=1, keepdim=True)[0] - torch.min(out, dim=1, keepdim=True)[0])
        return out

    def forward(self, inputs):
        az_fft_feature = self.get_fft_feature(inputs)
        return self.linear(az_fft_feature[:, -self.thresh:])
###########################################################################
class SpectralDiscriminatorRouter1(nn.Module):
    
    def __init__(self, batch_size, height, device=None):
        super(SpectralDiscriminatorRouter1, self).__init__()
        self.disS = SpectralDiscriminator(batch_size, height, device=device)
        self.MSE_Loss = nn.MSELoss()  # LSGAN loss

    def forward(self, original_image, encoded_image):
        origin_spectral_logit = self.disS(original_image)
        encoded_spectral_logit = self.disS(encoded_image)
        l1 = self.MSE_Loss(encoded_spectral_logit, torch.zeros_like(encoded_spectral_logit))
        l2 = self.MSE_Loss(origin_spectral_logit, torch.ones_like(origin_spectral_logit))
        disc_ad_loss_spectral = l2 + l1
        return disc_ad_loss_spectral
###########################################################################
class SpectralDiscriminatorRouter2(nn.Module):
    def __init__(self, batch_size, height, device=None):
        super(SpectralDiscriminatorRouter2, self).__init__()

        self.device = device
        self.disS = SpectralDiscriminator(batch_size, height, device=device).to(device)
        self.MSE_Loss = nn.MSELoss()  # LSGAN loss

    def forward(self, original_image, encoded_image):
        
        origin_spectral_logit = self.disS(original_image)
        encoded_spectral_logit = self.disS(encoded_image)
        disc_ad_loss_spectral = self.MSE_Loss(origin_spectral_logit, encoded_spectral_logit)
        return disc_ad_loss_spectral.to(self.device)
###########################################################################3
def calculate_2dft(inputs):
    inputs = torch.fft.ifftshift(inputs, dim=(-2, -1))
    inputs = torch.fft.fft2(inputs, dim=(-2, -1))
    inputs = torch.fft.fftshift(inputs, dim=(-2, -1))
    return inputs



def rgb2gray(rgb):
    r, g, b = rgb[:, 0, :,:], rgb[:,1,:, :], rgb[:,2:,:]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray