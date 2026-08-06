





from FarhadCV.Tools import tcolors
import torch
from torch import nn


################################################################################################
##                    give  weight wavelet                                                    ##
################################################################################################

def weight_wavelet(in_channels:int=15, out_channels:int=32, name:str=""):
    
    # inputs = tf.keras.layers.Input((shape[0], shape[1], 15))
    # inputs = tf.keras.layers.Input((256, 256, 32))
    modules = []
    sc2d = SeparableConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, 
        stride = (1,1), padding='same', use_bias = False)
    modules.append(sc2d)
    modules.append(torch.nn.Sigmoid())
    
    model = torch.nn.Sequential(*modules)
    return model



class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                               groups=in_channels, bias=use_bias, padding=0)
        self.depthwise.weight = torch.nn.init.uniform_(self.depthwise.weight, a=0.0, b=0.2)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=1, stride=stride, bias=use_bias)
        self.pointwise .weight = torch.nn.init.uniform_(self.pointwise .weight, a=0.0, b=0.2)

    def forward(self, x):
     
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    
################################################################################################