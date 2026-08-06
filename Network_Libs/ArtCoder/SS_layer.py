# library
# standard library
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
from .utils_code import get_3DGauss


class SSlayer(nn.Module):
    def __init__(self, model_size, requires_grad=False):
        super(SSlayer, self).__init__()

        cov_module = nn.Conv2d(in_channels=3, 
                               out_channels=3, 
                               kernel_size=model_size, 
                               stride=model_size, 
                               padding=0, 
                               bias=False)

        weight = get_3DGauss(model_size)  # [16,16]
        weight = weight.unsqueeze(0).unsqueeze(0)  # [1,1,16,16]
        weight = torch.cat([weight, weight, weight], dim=1)  # [1,3,16,16]
        cov_module.weight = nn.Parameter(weight)
        self.conv_module = nn.Sequential(
            cov_module
        )

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False # each kernel is fixed to gauss weight

    def forward(self, x):
        x = x.repeat(1, 1, 1, 1)
        x = self.conv_module(x)
        return x  # return x for visualization



