

"""
@--12.04.2023--@
Author: github/farhadsh1992
INFO:
	- Rimman Loss
    - REF: 
    
		

    
LAST_UPDATE:
"""

import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
import numpy as np
import math
########################################################################
class SpectralDiscriminatorRouter2(nn.Module):
    def __init__(self, batch_size: int, height: int, name: str = "Spectral_Disc", device=None):
        super(SpectralDiscriminatorRouter2, self).__init__()

        self.device     = device
        self.batch_size = batch_size
        self.height     = height
        self.net        = SpectralDiscriminator(batch_size, height)
        self.mse_loss   = nn.MSELoss()  # LSGAN loss

    def forward(self, original_image: torch.Tensor, 
                       encoded_image: torch.Tensor) -> torch.Tensor:
        origin_spectral_logit  = self.net(original_image)
        encoded_spectral_logit = self.net(encoded_image)
        disc_ad_loss_spectral  = self.mse_loss(origin_spectral_logit, encoded_spectral_logit)
        return disc_ad_loss_spectral
########################################################################
class SpectralDiscriminator(nn.Module):
    """
    INFO:
        - https://arxiv.org/pdf/2012.05535.pdf
        - We adopt the Spectral discriminator in https://github.com/cyq373/SSD-GAN
        - Guides the generator to generate more realistic images by reducing the gap 
          between clean and denoised images in the frequency domain.
    """

    def __init__(self, batch_size: int, height: int, name: str = "Spectral_Discriminator", device=None):
        super(SpectralDiscriminator, self).__init__()

        self.device = device
        self.batch_size = batch_size
        self.height = height
        self.thresh = int(height / (2 * math.sqrt(2)))
        self.linear = nn.Linear(self.thresh, 1)
        self.s2dft_layer = Calculate2DFT(batch_size, image_size=height)
        self.eps = 1e-10

        self.variables2 = torch.zeros((batch_size, height, height, 1), dtype=torch.float32)
        self.variables_out = torch.zeros((batch_size, 9), dtype=torch.float32)

    def high_pass_filtered(self, x):
        self.variables2.zero_()
        H, W = self.height, self.height
        self.variables2[:, :H//2, :W//2, :] = x[:, H//2:, W//2:, :]
        self.variables2[:, :H//2, W//2:, :] = x[:, H//2:, :W//2, :]
        self.variables2[:, H//2:, :W//2, :] = x[:, :H//2, W//2:, :]
        self.variables2[:, H//2:, W//2:, :] = x[:, :H//2, :W//2, :]
        return self.variables2

    def azimuthal_average(self, image):
        H, W = image.shape[:2]
        y, x = np.indices([H, W])
        radius = np.sqrt((x - H / 2) ** 2 + (y - W / 2) ** 2 + self.eps).astype(np.float32).ravel()
        nr = np.bincount(radius.astype(int))
        tbin = gpu_bincount(radius, image)
        nr = nr.astype(np.float32)
        radial_prof = tbin / (nr + 1e-10)
        return radial_prof[1:-2]

    def get_fft_feature(self, x):
        epsilon = 1e-8
        x_gray = torch.mean(x, dim=-1, keepdim=True)  # Convert RGB to grayscale
        fft = self.s2dft_layer(x_gray) + epsilon
        magnitude_spectrum = torch.log(torch.sqrt(fft.real ** 2 + fft.imag ** 2 + 1e-10) + 1e-10)
        magnitude_spectrum = self.high_pass_filtered(magnitude_spectrum)

        out = []
        for i in range(magnitude_spectrum.shape[0]):
            out.append(self.azimuthal_average(magnitude_spectrum[i]))
        out = torch.tensor(out, dtype=torch.float32)

        min_vals = out.min(dim=1, keepdim=True)[0]
        max_vals = out.max(dim=1, keepdim=True)[0]
        out = (out - min_vals) / (max_vals - min_vals + 1e-10)

        return out

    def forward(self, inputs):
        az_fft_feature = self.get_fft_feature(inputs)
        return self.linear(az_fft_feature[:, -self.thresh:])

class Calculate2DFT(nn.Module):
    def __init__(self, batch_size: int, image_size: int = 16):
        super(Calculate2DFT, self).__init__()
        self.batch_size = batch_size
        self.image_size = image_size
        self.variables = torch.zeros((batch_size, image_size, image_size, 1), dtype=torch.float32)

    def forward(self, inputs):
        self.variables = inputs.clone()
        ft = torch.fft.fftshift(self.variables)
        inp_complex = torch.complex(ft, torch.zeros_like(ft))
        ft = torch.fft.fft2(inp_complex)
        ft = torch.fft.fftshift(ft)
        return ft

def gpu_bincount(radius, image):
    radius = radius.astype(int)
    image = image.ravel()
    return np.bincount(radius, weights=image)
