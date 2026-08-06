
################
"""
@--27.09.2023--@
Author: 
INFO:
	> Pytorch version of StampOne_v89
"""
################



import torch
import torchvision
from torch import nn

import pywt
import pywt.data
from FarhadCV.Tools import tcolors

import warnings
warnings.filterwarnings('ignore')

def wavelet_decomposition(tensor):
    """
    Perform 2D Discrete Wavelet Transform (DWT) on a PyTorch tensor.

    Args:
        tensor (torch.Tensor): Input tensor of shape (B, C, H, W).
    
    Returns:
        LL, (LH, HL, HH): Wavelet decomposition components.
    """
    batch_size, channels, height, width = tensor.shape
    LL_list, LH_list, HL_list, HH_list = [], [], [], []

    for i in range(batch_size):
        for j in range(channels):
            # Convert tensor to NumPy for PyWavelets processing
            np_img = tensor[i, j].cpu().numpy()
            
            # Perform 2D Wavelet Decomposition
            LL, (LH, HL, HH) = pywt.dwt2(np_img, 'haar')  # 'haar' is a common wavelet
            
            # Convert back to PyTorch tensors
            LL_list.append(torch.tensor(LL, dtype=tensor.dtype, device=tensor.device))
            LH_list.append(torch.tensor(LH, dtype=tensor.dtype, device=tensor.device))
            HL_list.append(torch.tensor(HL, dtype=tensor.dtype, device=tensor.device))
            HH_list.append(torch.tensor(HH, dtype=tensor.dtype, device=tensor.device))

    # Stack wavelet components into tensors
    LL = torch.stack(LL_list).view(batch_size, channels, LL.shape[0], LL.shape[1])
    LH = torch.stack(LH_list).view(batch_size, channels, LH.shape[0], LH.shape[1])
    HL = torch.stack(HL_list).view(batch_size, channels, HL.shape[0], HL.shape[1])
    HH = torch.stack(HH_list).view(batch_size, channels, HH.shape[0], HH.shape[1])
    out = torch.cat((tensor, LL, HL, HL, HH), dim=1)  
    return out





class WaveletDecompositionLayer(nn.Module):
    def __init__(self, wavelet='haar', device=None):
        """
        Custom PyTorch layer for 2D Discrete Wavelet Transform (DWT).

        Args:
            wavelet (str): The wavelet type (default: 'haar').
        """
        super(WaveletDecompositionLayer, self).__init__()
        self.wavelet = wavelet
        self.up_layer = torch.nn.Upsample(scale_factor=(2,2)).to(device)
        self.device = device

    def forward(self, inputs, sobel):
        """
        Forward pass for Wavelet Decomposition.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Concatenated tensor (inputs, LL, LH, HL, HH)
        """
        # x = inputs
        
        x = self.up_layer(inputs)
        batch_size, channels, height, width = x.shape
        LL_list, LH_list, HL_list, HH_list = [], [], [], []

        for i in range(batch_size):
            for j in range(channels):
                # Convert PyTorch tensor to NumPy
                np_img = x[i, j].detach().cpu().numpy()
                
                # Perform 2D DWT using PyWavelets
                LL, (LH, HL, HH) = pywt.dwt2(np_img, self.wavelet)

                # Convert back to PyTorch tensors
                LL_list.append(torch.tensor(LL, dtype=x.dtype, device=x.device))
                LH_list.append(torch.tensor(LH, dtype=x.dtype, device=x.device))
                HL_list.append(torch.tensor(HL, dtype=x.dtype, device=x.device))
                HH_list.append(torch.tensor(HH, dtype=x.dtype, device=x.device))

        # Stack components into tensors
        LL = torch.stack(LL_list).view(batch_size, channels, LL.shape[0], LL.shape[1])
        LH = torch.stack(LH_list).view(batch_size, channels, LH.shape[0], LH.shape[1])
        HL = torch.stack(HL_list).view(batch_size, channels, HL.shape[0], HL.shape[1])
        HH = torch.stack(HH_list).view(batch_size, channels, HH.shape[0], HH.shape[1])

        # gray = RGBToGrayLayer()(inputs)
        # LLgray = RGBToGrayLayer()(LL)


        # Concatenate along channel dimension
        # output = torch.cat((gray, gray, gray, LLgray,LLgray,LLgray, LH, HL, HH), dim=1)
        output = torch.cat((inputs, LL, sobel, HL, HH), dim=1)

        output = output.to(self.device)
        return output
    
######################################################
class RGBToGrayLayer(nn.Module):
    def __init__(self):
        super(RGBToGrayLayer, self).__init__()

    def forward(self, x):
        # Ensure the input has 3 channels (RGB)
        if x.shape[1] != 3:
            print(tcolors.RED, "Input Shape: ", x.shape, tcolors.ENDC)
            raise ValueError("Input must have 3 channels (RGB)")

        # Apply the standard luminosity method for RGB to Grayscale conversion
        # Weights are based on the human perception of colors: 0.2989 * R + 0.5870 * G + 0.1140 * B
        r, g, b = x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:3, :, :]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray