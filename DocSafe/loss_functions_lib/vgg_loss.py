"""
@--28.09.2023--@
Author: github/farhadsh1992
INFO:
	- StampOne v89 torch format
    - REF: 
		

    
LAST_UPDATE:
"""


from FarhadCV.Tools import tcolors
import torchvision.models as models
import warnings
warnings.filterwarnings('ignore')


import torch
from torch import nn
import torch.nn.functional as F
from torcheval.metrics import MeanSquaredError
from torchvision.models import vgg16, VGG16_Weights, VGG16_BN_Weights
from torchvision.models import vgg19, VGG19_Weights, VGG19_BN_Weights

class VGGLoss(nn.Module):
    """
    INPUTS: images(224×224×3), encoded-images(224×224×3)
    -----------------------------------------------------------------------
    OUPUTS:
        - Outputs of VGG-features extractor: 7 x 7 x 512
    -----------------------------------------------------------------------
    Part of pre-trained VGG16. This is used in case we want perceptual loss instead of Mean Square Error loss.
    See for instance https://arxiv.org/abs/1603.08155
    
    https://pytorch.org/torcheval/main/generated/torcheval.metrics.MeanSquaredError.html
    https://pytorch.org/vision/stable/models.html
    """
    def __init__(self, vgg_version: str="vgg16", device='cpu'):
        super(VGGLoss, self).__init__()

        self.device  = device
        if vgg_version=="vgg16":
            # self.VGGModel = models.vgg16(pretrained=True)
            # self.VGGModel.eval()
            self.VGGModel = vgg16(weights=VGG16_Weights).to(device)
            self.VGGModel.eval()
           
        elif vgg_version=="vgg19":
            # self.VGGModel = models.vgg16(pretrained=True)
            # self.VGGModel.eval()
            self.VGGModel = vgg19(weights=VGG19_Weights).to(device)
            self.VGGModel.eval()
        
        self.mse_loss = MeanSquaredError()

    def forward(self, cover_images, encoded_images):
        vgg_on_cov = self.VGGModel(cover_images)
        vgg_on_enc = self.VGGModel(encoded_images)
        self.mse_loss.update(vgg_on_cov, vgg_on_enc)
        g_loss_enc = self.mse_loss.compute()
        return g_loss_enc




