################
"""
@--28.07.2024--@
Author: 
INFO:
	> Pytorch version of StampOne_v89
"""
################

"""
    INT: Image > (batch, height, width, channel)
    -------------------------------------------------------
    OUT:Image > (batch, height, width, channel)
    -------------------------------------------------------
    INFO:
        - Special transformer Network
        - https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
        
        - stn_router = STN(name="stn", filters=32)
          out = stn_router(image)
"""

from FarhadCV.Tools import tcolors, bcolors
import torch
import torch.nn as nn
import torch.nn.functional as F



class STNMASK(nn.Module):
    """
    INT: Image > (batch, height, width, channel)
    -------------------------------------------------------
    OUT: Image > (batch, height, width, channel)
    -------------------------------------------------------
    INFO:
        - https://pyimagesearch.com/2022/05/23/spatial-transformer-networks-using-tensorflow/
    """
    def __init__(self, filters, batch, channel_mask):
        super(STNMASK, self).__init__()
        self.B = batch
        self.H = None
        self.W = None
        self.C = channel_mask

        self.filter = filters

        # Localization network
        self.localizationNet = nn.Sequential(
            nn.Conv2d(channel_mask, filters // 4, kernel_size=3, stride=1, padding=1,bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(filters // 4, filters // 2, kernel_size=3, stride=1, padding=1,bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(filters // 2, filters, kernel_size=3, stride=1, padding=1,bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Regressor network
        self.regressorNet = nn.Sequential(
            nn.Linear(filters, filters, bias=True),
            nn.ReLU(),
            nn.Linear(filters, filters // 2, bias=True),
            nn.ReLU(),
            nn.Linear(filters // 2, 6, bias=True),
        )
        
        self.regressorNet[4].weight.data.zero_()
        self.regressorNet[4].bias.data.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def forward(self, x, mask):
        B, C, H, W = x.size()
        localFeatureMap = self.localizationNet(mask)
        localFeatureMap = localFeatureMap.view(B, -1)
        theta = self.regressorNet(localFeatureMap)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x
    
#########################################################################
def get_pixel_value(B, H, W, featureMap, x, y):
    b = torch.arange(0, B).view(B, 1, 1).repeat(1, H, W)
    indices = torch.stack([b, y, x], dim=3)
    gatheredPixelValue = featureMap[indices]
    return gatheredPixelValue
#########################################################################
def affine_grid_generator(B, H, W, theta):
    x = torch.linspace(-1.0, 1.0, H)
    y = torch.linspace(-1.0, 1.0, W)
    xT, yT = torch.meshgrid(x, y)
    xTFlat = xT.view(-1)
    yTFlat = yT.view(-1)
    ones = torch.ones_like(xTFlat)
    samplingGrid = torch.stack([xTFlat, yTFlat, ones])
    samplingGrid = samplingGrid.expand(B, 3, H * W)
    theta = theta.float()
    samplingGrid = samplingGrid.float()
    batchGrids = torch.bmm(theta, samplingGrid)
    batchGrids = batchGrids.view(B, 2, H, W)
    return batchGrids
#########################################################################
def bilinear_sampler(B, H, W, featureMap, x, y):
    maxY = H - 1
    maxX = W - 1
    zero = 0
    x = x.float()
    y = y.float()
    x = 0.5 * ((x + 1.0) * (maxX - 1))
    y = 0.5 * ((y + 1.0) * (maxY - 1))
    x0 = torch.floor(x).int()
    x1 = x0 + 1
    y0 = torch.floor(y).int()
    y1 = y0 + 1
    x0 = torch.clamp(x0, zero, maxX)
    x1 = torch.clamp(x1, zero, maxX)
    y0 = torch.clamp(y0, zero, maxY)
    y1 = torch.clamp(y1, zero, maxY)
    Ia = get_pixel_value(B, H, W, featureMap, x0, y0)
    Ib = get_pixel_value(B, H, W, featureMap, x0, y1)
    Ic = get_pixel_value(B, H, W, featureMap, x1, y0)
    Id = get_pixel_value(B, H, W, featureMap, x1, y1)
    x0 = x0.float()
    x1 = x1.float()
    y0 = y0.float()
    y1 = y1.float()
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)
    wa = wa.unsqueeze(3)
    wb = wb.unsqueeze(3)
    wc = wc.unsqueeze(3)
    wd = wd.unsqueeze(3)
    transformedFeatureMap = wa * Ia + wb * Ib + wc * Ic + wd * Id
    return transformedFeatureMap

