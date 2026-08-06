"""
@--30.09.2023--@
Author: github/farhadsh1992
INFO:
	- StampOne v89 torch format
    - REF: 
		

    
LAST_UPDATE:
"""

import math
import torch
from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_mimicry.nets.basemodel import basemodel
# from torch_mimicry.modules import losses
from . import Tools_Specteral_Discrimnator_v03_Losses as losses
import numpy as np


from FarhadCV.Tools import tcolors


import warnings
warnings.filterwarnings('ignore')



##################################################################################################
#####              SNLinear                               #####
##################################################################################################
class SpectralNorm(object):
    r"""
    Spectral Normalization for GANs (Miyato 2018).

    Inheritable class for performing spectral normalization of weights,
    as approximated using power iteration.

    Details: See Algorithm 1 of Appendix A (Miyato 2018).

    Attributes:
        n_dim (int): Number of dimensions.
        num_iters (int): Number of iterations for power iter.
        eps (float): Epsilon for zero division tolerance when normalizing.
    """
    def __init__(self, n_dim, num_iters=1, eps=1e-12):
        self.num_iters = num_iters
        self.eps = eps

        # Register a singular vector for each sigma
        self.register_buffer('sn_u', torch.randn(1, n_dim))
        self.register_buffer('sn_sigma', torch.ones(1))

    @property
    def u(self):
        return getattr(self, 'sn_u')

    @property
    def sigma(self):
        return getattr(self, 'sn_sigma')

    def _power_iteration(self, W, u, num_iters, eps=1e-12):
        with torch.no_grad():
            for _ in range(num_iters):
                v = F.normalize(torch.matmul(u, W), eps=eps)
                u = F.normalize(torch.matmul(v, W.t()), eps=eps)

        # Note: must have gradients, otherwise weights do not get updated!
        sigma = torch.mm(u, torch.mm(W, v.t()))

        return sigma, u, v

    def sn_weights(self):
        r"""
        Spectrally normalize current weights of the layer.
        """
        W = self.weight.view(self.weight.shape[0], -1)

        # Power iteration
        sigma, u, v = self._power_iteration(W=W,
                                            u=self.u,
                                            num_iters=self.num_iters,
                                            eps=self.eps)

        # Update only during training
        if self.training:
            with torch.no_grad():
                self.sigma[:] = sigma
                self.u[:] = u

        return self.weight / sigma
###################################################################################
class SNLinear_spectral(nn.Linear, SpectralNorm):
    r"""
    Spectrally normalized layer for Linear.

    Attributes:
        in_features (int): Input feature dimensions.
        out_features (int): Output feature dimensions.
    """
    def __init__(self, in_features, out_features, *args, **kwargs):
        nn.Linear.__init__(self, in_features, out_features, *args, **kwargs)

        SpectralNorm.__init__(self,
                              n_dim=out_features,
                              num_iters=kwargs.get('num_iters', 1))

    def forward(self, x):
        return F.linear(input=x, weight=self.sn_weights(), bias=self.bias)
###################################################################################
def SNLinear(*args, default=True, **kwargs):
    r"""
    Wrapper for applying spectral norm on linear layer.
    """
    if default:
        return nn.utils.spectral_norm(nn.Linear(*args, **kwargs))

    else:
        return SNLinear_spectral(*args, **kwargs)
##################################################################################################
#####              DBlockOptimized                               #####
##################################################################################################

class DBlockOptimized(nn.Module):
    """
    Optimized residual block for discriminator. This is used as the first residual block,
    where there is a definite downsampling involved. Follows the official SNGAN reference implementation
    in chainer.

    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        spectral_norm (bool): If True, uses spectral norm for convolutional layers.        
    """
    def __init__(self, in_channels, out_channels, spectral_norm=True, device:str=None):
        super().__init__()

        self.cuda = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spectral_norm = spectral_norm

        # Build the layers
        if self.spectral_norm:
            self.c1 = SNConv2d(self.in_channels, self.out_channels, 3, 1, 1).to(self.cuda)
            self.c2 = SNConv2d(self.out_channels, self.out_channels, 3, 1, 1).to(self.cuda)
            self.c_sc = SNConv2d(self.in_channels, self.out_channels, 1, 1, 0).to(self.cuda)
        else:
            self.c1 = nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1).to(self.cuda)
            self.c2 = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1).to(self.cuda)
            self.c_sc = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0).to(self.cuda)

        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.c1.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.c2.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.c_sc.weight.data, 1.0)

    def _residual(self, x):
        """
        Helper function for feedforwarding through main layers.
        """
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = F.avg_pool2d(h, 2)

        return h

    def _shortcut(self, x):
        """
        Helper function for feedforwarding through shortcut layers.
        """
        return self.c_sc(F.avg_pool2d(x, 2))

    def forward(self, x):
        """
        Residual block feedforward function.
        """
        return self._residual(x) + self._shortcut(x)
##################################################################################################
def SNConv2d(*args, default=True, **kwargs):
    r"""
    Wrapper for applying spectral norm on conv2d layer.
    """
    if default:
        return nn.utils.spectral_norm(nn.Conv2d(*args, **kwargs))

    else:
        return SNConv2d_spectral(*args, **kwargs)
##################################################################################################
class SNConv2d_spectral(nn.Conv2d, SpectralNorm):
    r"""
    Spectrally normalized layer for Conv2d.

    Attributes:
        in_channels (int): Input channel dimension.
        out_channels (int): Output channel dimensions.
    """
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        nn.Conv2d.__init__(self, in_channels, out_channels, *args, **kwargs)

        SpectralNorm.__init__(self,
                              n_dim=out_channels,
                              num_iters=kwargs.get('num_iters', 1))

    def forward(self, x):
        return F.conv2d(input=x,
                        weight=self.sn_weights(),
                        bias=self.bias,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=self.groups)

##################################################################################################
#####              DBlock                               #####
##################################################################################################
class DBlock(nn.Module):
    """
    Residual block for discriminator.

    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        hidden_channels (int): The channel size of intermediate feature maps.
        downsample (bool): If True, downsamples the input feature map.
        spectral_norm (bool): If True, uses spectral norm for convolutional layers.        
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=None,
                 downsample=False,
                 spectral_norm=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels if hidden_channels is not None else in_channels
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        self.spectral_norm = spectral_norm

        # Build the layers
        if self.spectral_norm:
            self.c1 = SNConv2d(self.in_channels, self.hidden_channels, 3, 1, 1)
            self.c2 = SNConv2d(self.hidden_channels, self.out_channels, 3, 1,
                               1)
        else:
            self.c1 = nn.Conv2d(self.in_channels, self.hidden_channels, 3, 1,
                                1)
            self.c2 = nn.Conv2d(self.hidden_channels, self.out_channels, 3, 1,
                                1)

        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.c1.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.c2.weight.data, math.sqrt(2.0))

        # Shortcut layer
        if self.learnable_sc:
            if self.spectral_norm:
                self.c_sc = SNConv2d(in_channels, out_channels, 1, 1, 0)
            else:
                self.c_sc = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

            nn.init.xavier_uniform_(self.c_sc.weight.data, 1.0)

    def _residual(self, x):
        """
        Helper function for feedforwarding through main layers.
        """
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = F.avg_pool2d(h, 2)

        return h

    def _shortcut(self, x):
        """
        Helper function for feedforwarding through shortcut layers.
        """
        if self.learnable_sc:
            x = self.c_sc(x)
            return F.avg_pool2d(x, 2) if self.downsample else x

        else:
            return x

    def forward(self, x):
        """
        Residual block feedforward function.
        """
        return self._residual(x) + self._shortcut(x)
##################################################################################################
##################################################################################################
