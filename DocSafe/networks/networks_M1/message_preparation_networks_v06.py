

#################
"""
@--04.02.2024--@
Author: 
INFO:
	> Pytorch version of StampOne_v89
"""
################


from FarhadCV.Tools import tcolors, bcolors
import torch
import torch.nn as nn
import torch.nn.functional as F

# from .Tools_StampOne_Net2 import AttentionBlock
# from .Tools_StampOne_Net2 import GatingSignal

from .Snake_activation_function import Snake
from .Tools_StampOne_Net2 import DecoderBlock
from .Tools_StampOne_Net2 import DecoderBlock2
##################################################################
from .Wavelet_transfer3 import WaveletDecompositionLayer as Wavelet_Layer
##################################################################
from .Sobel_Egdes import sobel_egdes
from .weight_wavelet import weight_wavelet
from .Tools_StampOne_Net2 import WeightWavelet
from .Wavelet_transfer3_message import WaveletDecompositionLayer as Wavelet_Layer_message
from .Special_Transform_Network_vt02 import STNMASK
######################################################
#####                                    #####
######################################################
## Message Preparation Networks (MPNs)
class MessagePreparationNetworks(nn.Module):
    def __init__(self, batch_size, 
                 input_shape, 
                 channelin,
                 batch_norm=True, 
                 device=None, name="MPN"):
        super(MessagePreparationNetworks, self).__init__()

        ######################################################
        #####                                    #####
        ######################################################
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.device     = device
        self.ACTIVATION = Snake(beta=0.5, trainable=True)
        ######################################################
        #####                                    #####
        ######################################################
        ## Transtion into special domain
        # self.sobel_edge_message_router = sobel_egdes(input_shape = self.input_shape, 
        #                                            name = 'Sobel_Egdes_imageY').to(self.device)
        # self.wavelet_layer_message = Wavelet_Layer()
        # self.message_ww_layer = WeightWavelet(shape=self.input_shape, 
        #                                       name="encoded").to(self.device)
        ######################################################
        #####                                    #####
        ######################################################

        ## Flatten Layer
        self.flatten = nn.Flatten()

        ## Conv1D Equivalent (Conv2D with kernel_size=1 for PyTorch)
        self.dense_1 = nn.Linear(input_shape[0]*input_shape[1]*3, 
                                 256*256*3).to(device)  # Equivalent to Dense(64)
        # self.msg_embeddings = torch.nn.Embedding(num_embeddings=2, 
        #                                          embedding_dim=128).to(device)
        # self.msg_embeddings = torch.nn.Embedding(num_embeddings=2, 
        #                                          c
        # self.conv1d = nn.Conv1d(1, 32, kernel_size=1, stride=1, padding=0).to(device)
        ## Activation
        self.act_1 = Snake(beta=0.5, trainable=True).to(device)
        
        self.reshape_1 = lambda x: x.view(self.batch_size, 3, 256, 256).to(device)  
        # self.padding_1 = torch.nn.ZeroPad2d(96)
       
        self.mask_layer = nn.Conv2d(in_channels  = 3, 
                                    out_channels = 3, 
                                    kernel_size  = 1, 
                                    stride       = 1,
                                    padding      = 0, 
                                    bias         = True).to(device)
        self.stn = STNMASK(filters=32, batch=batch_size, channel_mask=3)

        #########################################################################
         ## message transtion into special domain
        self.sobel_edge_message_router = sobel_egdes(input_shape = 256, 
                                                     channel_in  = 3,
                                                     name        = 'Sobel_Egdes_messageY').to(self.device)
        self.wavelet_message_layer = Wavelet_Layer_message(device=device).to(self.device)
        

        self.message_ww_layer = WeightWavelet(shape       = (256, 256), 
                                              channel_in  = 15, 
                                              channel_out = 32, 
                                              name        = "encoded").to(self.device)
        #########################################################################

        self.last_layer = nn.Conv2d(in_channels  = 32, 
                                    out_channels = 256, 
                                    kernel_size  = 1, 
                                    stride       = 1,
                                    padding      = 0, 
                                    bias         = True).to(device)


    def forward(self, inputs, mask):
        """
        Forward pass for the Embeddding4 module.
        """
        ######################################################
        #####                                    #####
        ######################################################
        # dev_mess = self.sobel_edge_message_router(inputs)
        # wavelet_mess = self.wavelet_layer_message(dev_mess)
        # ww_mess = self.message_ww_layer(wavelet_mess)
        ######################################################
        #####                                    #####
        ######################################################
        x = self.flatten(inputs)  # Flatten input
        x = x.unsqueeze(1).to(self.device)
        x = self.dense_1(x)
        # x = self.act_1(x)  # Apply activation
       
        # x = x.to(torch.int64)  # Convert to long (int64)
        # x = torch.clamp(x, min=0, max=1)  # Clamp values to valid range
        # # x = x.unsqueeze(1).to(self.device)  # Expand dimensions equivalent to `expand_dims`
        # x = self.msg_embeddings(x.to(torch.int64)) # .long()
        x = self.act_1(x)  
        # x = x.unsqueeze(1).to(self.device)
        # x = self.conv1d(x)
        # x = x.unsqueeze(2)  # Expand dimensions equivalent to `expand_dims
        # x = self.conv1d(x)  # Apply Conv1D equivalent
        
        x = self.reshape_1(x)  # Reshape
        
        # x = self.padding_1(x)
        # mask64 = self.mask_layer(mask)
        # x = torch.multiply(x, mask64)
        dev_x      = self.sobel_edge_message_router(x)
        wavelet_x  = self.wavelet_message_layer(dev_x)
        xmessage   = self.message_ww_layer(wavelet_x)
        xmessage   = self.stn(xmessage, mask)
        # x = self.reshape_1(x)  # Reshape
        # out = self.padding(out)
        out = self.last_layer(xmessage)
        ######################################################
        #####                                    #####
        ######################################################
        ## Reduce from 128 t0 3 RGB channel output.
        # out = self.last_layer(x)
        return out
#####################################################
#####                              #####
#####################################################
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
#####################################################
#####                              #####
#####################################################
class GatingSignal(nn.Module):
    """
    Resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :return: the gating feature map with the same dimension of the up layer feature map
    """
    def __init__(self, in_channels, out_size, batch_norm=False):
        super(GatingSignal, self).__init__()
        self.conv2d_layer = nn.Conv2d(in_channels  = in_channels, 
                                      out_channels = out_size, 
                                      kernel_size  = (1, 1), 
                                      padding      = 'same', 
                                      bias         = not batch_norm)
        self.batch_norm = batch_norm
        if batch_norm:
            self.batch_layer = nn.BatchNorm2d(out_size)
        self.act_layer = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = self.conv2d_layer(inputs)
        if self.batch_norm:
            x = self.batch_layer(x)
        x = self.act_layer(x)
        return x
    

#####################################################
#####                              #####
#####################################################
class AttentionBlock(nn.Module):
    def __init__(self, inter_shape, out_shape, number):
        super(AttentionBlock, self).__init__()
        self.inter_shape = inter_shape
        self.number = number
        shape_x  = [32, 64, 128, 256]
        shape_x3 = [512, 256, 128, 64, 32]
        shape_g  = [16, 32,  64, 128]
        shape_theta_x = [16, 32, 64, 128]
        shape_sigmoid = [16, 32, 64, 128]

        # Getting the x signal to the same shape as the gating signal
        self.theta_x_layer = nn.Conv2d(inter_shape, inter_shape, 
                                       kernel_size=(2, 2), stride=(2, 2), padding=0, bias=True)

        # Getting the gating signal to the same number of filters as the inter_shape
        self.phi_g_layer = nn.Conv2d(inter_shape, inter_shape, kernel_size=(1, 1), padding=0)
        self.upsample_g_layer = nn.ConvTranspose2d(inter_shape, inter_shape, kernel_size=(3, 3),
                                                   stride=(shape_theta_x[number] // shape_g[number],
                                                           shape_theta_x[number] // shape_g[number]),
                                                   padding=1, bias=True)
        self.concat_xg_layer = nn.Sequential()
        self.act_xg_layer = nn.ReLU()
        self.psi_layer = nn.Conv2d(inter_shape, 1, kernel_size=(1, 1), padding=0, bias=True)
        self.sigmoid_xg_layer = nn.Sigmoid()
        self.upsample_psi_layer1 = nn.Upsample(scale_factor=(shape_x[number] // shape_sigmoid[number],
                                                             shape_x[number] // shape_sigmoid[number]),
                                               mode='bilinear', align_corners=True)

        self.upsample_psi_layer2 = lambda x: x.repeat_interleave(shape_x3[number], dim=1)
        self.multiply_layer = lambda x, y: x * y

        self.result_layer = nn.Conv2d(inter_shape, 
                                      out_shape, kernel_size=(1, 1), padding=0, bias=True)
        self.result_bn_layer = nn.BatchNorm2d(out_shape)

    def forward(self, x, gating):
       
        # Getting the x signal to the same shape as the gating signal
        theta_x = self.theta_x_layer(x)
        # Getting the gating signal to the same number of filters as the inter_shape
        phi_g = self.phi_g_layer(gating)
        upsample_g = self.upsample_g_layer(phi_g)

        
        concat_xg = torch.add(upsample_g, theta_x)
        # concat_xg = torch.mul(upsample_g, theta_x)
        act_xg = self.act_xg_layer(concat_xg)
      
        psi = self.psi_layer(act_xg)
 
        sigmoid_xg = self.sigmoid_xg_layer(psi)

        upsample_psi = self.upsample_psi_layer1(sigmoid_xg)
        upsample_psi = self.upsample_psi_layer2(upsample_psi)


        


        y = self.multiply_layer(upsample_psi, x)

        result = self.result_layer(y)
        result_bn = self.result_bn_layer(result)
        return result_bn    