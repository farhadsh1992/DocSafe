#################
"""
@--27.09.2023--@
Author: 
INFO:
	> Pytorch version of StampOne_v89
"""
################
import os
from FarhadCV.Tools import tcolors
import torch
from torch import nn

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 





################################################################################################
##                    give  weight wavelet                                                    ##
################################################################################################

def weight_wavelet(in_channels:int=15, out_channels:int=32, name:str="",  device:str=None):
    
    # inputs = tf.keras.layers.Input((shape[0], shape[1], 15))
    # inputs = tf.keras.layers.Input((256, 256, 32))
    modules = []
    sc2d = SeparableConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, 
        stride = (1,1), padding='same', use_bias = False)
    modules.append(sc2d)
    modules.append(torch.nn.Sigmoid())
    modules.append(torch.nn.BatchNorm2d(num_features=out_channels))
    model = torch.nn.Sequential(*modules).to(device)
    return model


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bias=False, device:str=None):
        super(SeparableConv2d, self).__init__()

        self.cuda = device
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                               groups=in_channels, bias=use_bias, padding=0)
        self.depthwise.weight = torch.nn.init.uniform_(self.depthwise.weight, a=0.0, b=0.2)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=1, stride=stride, bias=use_bias)
        self.pointwise.weight = torch.nn.init.uniform_(self.pointwise .weight, a=0.0, b=0.2)
        self.last_layer_bn = torch.nn.BatchNorm2d(num_features=out_channels).to(self.cuda)
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        # out =  self.last_layer_bn(out)
        return out
################################################################################################
##                    down and up block for AttentionVnet                                     ##
################################################################################################

# class encoder_block(nn.Module):
#     def __init__(self, in_channels:int, out_channels:int, kernel_size:int=4, apply_batchnorm= True,  device:str=None):
#         super(encoder_block, self).__init__()

#         self.cuda = device
#         self.apply_batchnorm = apply_batchnorm
#         self.conv2D = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, kernel_size), stride=(2,2), padding=1, bias=False).to(self.cuda)
#         self.conv2D.weight = torch.nn.init.uniform_(self.conv2D.weight, a=0.0, b=0.02)
    
#         if apply_batchnorm:
#             self.BatchNorm2d = torch.nn.BatchNorm2d(num_features=out_channels).to(self.cuda)
#         self.LeakyReLU = torch.nn.LeakyReLU(inplace=True).to(self.cuda)

#     def forward(self, inputs):
#         x = self.conv2D(inputs)
        
#         if self.apply_batchnorm:
#             self.BatchNorm2d(x)
#         out = self.LeakyReLU(x)
#         return out

def encoder_block(in_channels:int, out_channels:int, kernel_size:int=4, apply_batchnorm= True,  device:str=None):

    modules = []

    conv2D = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                             kernel_size=(kernel_size, kernel_size), stride=(2,2), padding=1, bias=False)
    # conv2D.weight = torch.nn.init.uniform_(conv2D.weight, a=0.0, b=0.02)
    conv2D.weight=  nn.init.xavier_uniform_(conv2D.weight, gain=nn.init.calculate_gain('relu'))
    modules.append(conv2D)
    if apply_batchnorm:
        modules.append(torch.nn.BatchNorm2d(num_features=out_channels))
    modules.append(torch.nn.LeakyReLU(inplace=True))
    model = torch.nn.Sequential(*modules).to(device)

    return model
def encoder_block2(in_channels:int, out_channels:int, kernel_size:int=4, apply_batchnorm= True,  device:str=None):

    modules = []

    conv2D = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                             kernel_size=(kernel_size, kernel_size), stride=(2,2), padding=1, bias=False)
    # conv2D.weight = torch.nn.init.uniform_(conv2D.weight, a=0.0, b=0.02)
    conv2D.weight=  nn.init.xavier_uniform_(conv2D.weight, gain=nn.init.calculate_gain('relu'))
    modules.append(conv2D)
    if apply_batchnorm:
        modules.append(torch.nn.BatchNorm2d(num_features=out_channels))
    modules.append(torch.nn.LeakyReLU(inplace=True))
    model = torch.nn.Sequential(*modules).to(device)

    return model

def decoder_block(in_channels:int, out_channels:int, kernel_size:int=4, stride:int=2, apply_dropout=False,  device:str=None):
    modules = []

    convtrans_01 = torch.nn.ConvTranspose2d(in_channels=in_channels, 
                                            out_channels=out_channels, 
                                            kernel_size=kernel_size, 
                                            stride=stride, 
                                            padding=0,  
                                            bias=False, dilation=1, 
                                            padding_mode='zeros', 
                                            device=None, dtype=None)
    # convtrans_01.weight = torch.nn.init.uniform_(convtrans_01.weight, a=0.0, b=0.02)
    convtrans_01.weight=  nn.init.xavier_uniform_(convtrans_01.weight, gain=nn.init.calculate_gain('relu'))
    modules.append(convtrans_01)
    modules.append(torch.nn.BatchNorm2d(num_features=out_channels))
    if apply_dropout:
        modules.append(torch.nn.Dropout(p=0.5))
    modules.append(torch.nn.ReLU(inplace=True))

    model = torch.nn.Sequential(*modules).to(device)
    return model
#############################################################################################################################
def decoder_block2(in_channels:int, out_channels:int, kernel_size:int=4, stride:int=2, apply_dropout=False,  device:str=None):
    # initialzer = tf.random_normal_initializer(0., 0.4) #??
    modules = []
    convtrans_01 = torch.nn.ConvTranspose2d(in_channels=in_channels, 
                                            out_channels=out_channels, 
                                            kernel_size=kernel_size, 
                                            stride=stride, 
                                            padding=0,  
                                            bias=False, dilation=1, 
                                            padding_mode='zeros', 
                                            device=None, dtype=None)
    # convtrans_01.weight = torch.nn.init.uniform_(convtrans_01.weight, a=0.0, b=0.02)
    convtrans_01.weight=  nn.init.xavier_uniform_(convtrans_01.weight, gain=nn.init.calculate_gain('relu'))
    modules.append(convtrans_01)
    modules.append(torch.nn.LeakyReLU())
    modules.append(torch.nn.BatchNorm2d(num_features=out_channels))
    
    if apply_dropout:
        modules.append(torch.nn.Dropout(p=0.5))

    model = torch.nn.Sequential(*modules).to(device)
    return model

################################################################################################
##           signal gate and attention block for attentionVNet                                ##
################################################################################################


def gating_signal(in_channels, out_channels, batch_norm=False,  device:str=None):
    """
    resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :return: the gating feature map with the same dimension of the up layer feature map
    """
    
    

    modules = []

    modules.append(torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=1, bias=True))
    if batch_norm:
        modules.append(torch.nn.BatchNorm2d(num_features=out_channels))
    modules.append(torch.nn.ReLU())
    model = torch.nn.Sequential(*modules).to(device)
    return model

################################################################################################
class attention_block(nn.Module):
    def __init__(self, in_channels_x ,in_channels_gating , out_channels, number=0,  device:str=None):
        super(attention_block, self).__init__()
        
        self.cuda = device
        self.in_channels_x = in_channels_x
        self.in_channels_gating = in_channels_gating
        self.num = number
        self.shape_x  = [32, 64, 128, 256]
        self.shape_x3 = [512, 256, 128, 64]
        self.shape_g  = [16, 32,  64, 128]
        self.shape_theta_x = [16, 32, 64, 128,]
        self.shape_sigmoid = [16, 32, 64, 128,]
        
        self.inter_shape = out_channels
        # Getting the x signal to the same shape as the gating signal
        self.theta_x_layer = nn.Conv2d(in_channels=in_channels_x, out_channels=out_channels, kernel_size=(2, 2), stride=(2, 2), bias=True).to(self.cuda)
        # shape_theta_x = K.int_shape(theta_x)
        # self.theta_x_layer_bn = torch.nn.BatchNorm2d(num_features=out_channels).to(self.cuda)
        # Getting the gating signal to the same number of filters as the inter_shape
        self.phi_g_layer = nn.Conv2d(in_channels=in_channels_gating, out_channels=out_channels, kernel_size=(1, 1),stride=(1, 1), bias=True).to(self.cuda)
        # self.phi_g_bn = torch.nn.BatchNorm2d(num_features=out_channels).to(self.cuda)
        
        self.convTrans2D_layer = torch.nn.ConvTranspose2d(in_channels=self.inter_shape, 
                                                         out_channels=self.inter_shape, 
                                                         kernel_size=(1, 1), 
                                                         stride=(self.shape_theta_x[self.num] // self.shape_g[self.num], self.shape_theta_x[self.num] // self.shape_g[self.num]), 
                                                         padding=0,  
                                                         bias=True, 
                                                         padding_mode='zeros', 
                                                         device=None).to(self.cuda)
        self.relu_01 = torch.nn.ReLU().to(self.cuda)
        # self.convTrans2D_layer_bn = torch.nn.BatchNorm2d(num_features=self.inter_shape).to(self.cuda)

        self.psi_layer = nn.Conv2d(in_channels=self.in_channels_x, out_channels=1, kernel_size=(1, 1), stride=(1, 1)).to(self.cuda)
        self.sigmoid_01 = torch.nn.Sigmoid().to(self.cuda)
        # self.sigmoid_01_bn = torch.nn.BatchNorm2d(num_features=1).to(self.cuda)

        self.upsample_psi_layer = torch.nn.Upsample(scale_factor=(self.shape_x[self.num] // self.shape_sigmoid[self.num],
                                                        self.shape_x[self.num] // self.shape_sigmoid[self.num])).to(self.cuda)

        self.last_layer = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1), bias=True).to(self.cuda)
        self.last_layer_bn = torch.nn.BatchNorm2d(num_features=out_channels).to(self.cuda)

        self.repeat_elem = LambdaLayer(lambda x: x.repeat(1, int(self.shape_x3[self.num]),1,1)).to(self.cuda)
        self.multiply_layer = Multiply(device=self.cuda).to(self.cuda)
        
    def forward(self, x, gating):
        shape_xi = x.shape
        # shape_g = gating.shape
        # Getting the x signal to the same shape as the gating signal
        theta_x = self.theta_x_layer(x)  # 16
        # shape_theta_x = theta_x.shape
        # theta_x = self.theta_x_layer_bn(theta_x)

    

        # Getting the gating signal to the same number of filters as the inter_shape

        phi_g = self.phi_g_layer(gating)
        # phi_g = self.phi_g_bn(phi_g)
        upsample_g = self.convTrans2D_layer(phi_g)
        # upsample_g = self.convTrans2D_layer_bn(upsample_g)
        
        concat_xg  = upsample_g + theta_x
    
        act_xg = self.relu_01(concat_xg)
        psi =  self.psi_layer(act_xg)
        sigmoid_xg  = self.sigmoid_01(psi)    
        # sigmoid_xg  = self.sigmoid_01_bn(sigmoid_xg)

        upsample_psi = self.upsample_psi_layer(sigmoid_xg) # 32
        upsample_psi = self.repeat_elem(upsample_psi)
       
        # print("x:", x.device)
        # print("upsample_psi:", upsample_psi.device)
        y = self.multiply_layer([upsample_psi.to(self.cuda), x.to(self.cuda)])
    

        result =  self.last_layer(y)
        result_bn = self.last_layer_bn(result)
        return result_bn
    
################################################################################################
class Multiply(nn.Module):
  def __init__(self,  device:str=None):
    super(Multiply, self).__init__()
    self.cuda = device
  def forward(self, tensors):
    result = torch.ones(tensors[0].size()).to(self.cuda)
    for t in tensors:
      result *= t
    return t.to(self.cuda)

################################################################################################
class LambdaLayer(nn.Module):
    def __init__(self, lambd,  device:str=None):
        super(LambdaLayer, self).__init__()
        self.cuda = device
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)
################################################################################################
def repeat_elem(tensor, rep,  device:str=None):
    # lambda function to repeat Repeats the elements of a tensor along an axis
    #by a factor of rep.
    # If tensor has shape (None, 256,256,3), lambda will return a tensor of shape 
    #(None, 256,256,6), if specified axis=3 and rep=2.
    
    return LambdaLayer(lambda x: x.repeat(1, rep,1,1))(tensor)
################################################################################################
####                                     ####
################################################################################################

class Reshape_layer(nn.Module):
    def __init__(self, shape,  device:str=None):
        super(Reshape_layer, self).__init__()
        self.shape = shape
        self.cuda = device
    def forward(self, x):
        # return x.view(self.shape)
        return torch.reshape(x, self.shape)
#################################################################################
class conv1d_layer_de1(nn.Module):
    def __init__(self, batch,  device:str=None):
        super(conv1d_layer_de1, self).__init__()
        ACTIVE = "relu" #"gelu" #Snake(beta=0.5, trainable=True)

        self.cuda = device
        self.flatten = torch.nn.Flatten()
        self.Conv1D = torch.nn.Conv1d(in_channels=1, out_channels=2, kernel_size=1, stride=1, bias=True)
        # self.Conv1D.weight = torch.nn.init.uniform_(self.Conv1D.weight, a=0.0, b=0.02)
        self.Conv1D.weight=  nn.init.xavier_uniform_(self.Conv1D.weight, gain=nn.init.calculate_gain('relu'))
        self.Conv1D.bias = torch.nn.init.zeros_(self.Conv1D.bias)
        self.act_fn = torch.nn.ReLU()
        
        self.reshape = Reshape_layer((batch, 64, 256, 256))
        self.last_layer_bn = torch.nn.BatchNorm2d(num_features=64).to(self.cuda)
    def forward(self, inputs):
      
        x = self.flatten(inputs)
        x = torch.unsqueeze(x, dim=1)

        x = self.Conv1D(x)
        x = self.act_fn(x)
        shape_x = x.shape[0]
        x =  Reshape_layer((shape_x, 64, 256, 256))(x)
        # x = self.reshape(x)
        x = self.last_layer_bn(x)
        return x
#################################################################################
class conv1d_layer_de2(nn.Module):
    def __init__(self, out_channels, batch,  device:str=None):
        super(conv1d_layer_de2, self).__init__()
        # inputs = tf.keras.layers.Input(shape=[10, 10, 64])
        ACTIVE = "gelu" #"gelu" #Snake(beta=0.5, trainable=True)

        self.cuda = device
        self.flatten = torch.nn.Flatten()
        self.Conv1D  = torch.nn.Conv1d(in_channels=1, out_channels=4, kernel_size=1, stride=1, bias=True)
        # self.Conv1D.weight = torch.nn.init.uniform_(self.Conv1D.weight, a=0, b=0.02)
        self.Conv1D.weight=  nn.init.xavier_uniform_(self.Conv1D.weight, gain=nn.init.calculate_gain('relu'))
        self.Conv1D.bias = torch.nn.init.zeros_(self.Conv1D.bias)
        self.act_fn  = torch.nn.GELU()
        self.reshape = Reshape_layer((batch, out_channels, 16, 16))
        self.out_channels = out_channels
        self.last_layer_bn = torch.nn.BatchNorm2d(num_features=out_channels).to(self.cuda)
    def forward(self, inputs):
        
        x = self.flatten(inputs)
        x = torch.unsqueeze(x, dim=1)
        x = self.Conv1D(x)
        x = self.act_fn(x)
        shape_x = x.shape[0]
        x = Reshape_layer((shape_x, self.out_channels, 16, 16))(x)
        # x = self.reshape(x)
        x = self.last_layer_bn(x)
        return x

