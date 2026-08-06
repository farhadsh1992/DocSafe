

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
######################################################
#####                                    #####
######################################################
## Message Preparation Networks (MPNs)
class MessagePreparationNetworks(nn.Module):
    def __init__(self, batch_size, input_shape, batch_norm=True, device=None, name="MPN"):
        super(MessagePreparationNetworks, self).__init__()

        ######################################################
        #####                                    #####
        ######################################################
        self.batch_size  = batch_size
        self.input_shape = input_shape
        self.device      = device
        self.ACTIVATION  = Snake(beta=0.5, trainable=True)
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
        ## Reshape Layer # Manually reshape in PyTorch
        

        ######################################################
        #####                                    #####
        ######################################################
        ## Flatten Layer
        self.flatten_1 = nn.Flatten()
        ## Dense (Fully Connected) Layers
        
        self.dense_1 = nn.Linear(input_shape[0]*input_shape[1]*32, 
                                 64*64*32).to(device)  # Equivalent to Dense(64)
        self.act_1 = Snake(beta=0.5, trainable=True).to(device)
        self.reshape_1 = lambda x: x.view(self.batch_size, 32, 64, 64).to(device)  
        self.padding = torch.nn.ZeroPad2d(96)

        # Decoder Blocks (Assuming decoder_block2 is a U-Net-like block)
        self.conv2d_layer= nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0).to(device)
        self.act_2 = Snake(beta=0.5, trainable=True).to(device)
        self.up1_2 = DecoderBlock2(32, 32, 1, activation=self.ACTIVATION, name=f"up1_2{name}").to(device)
        self.up2_2 = DecoderBlock2(32, 32, 1, activation=self.ACTIVATION, name=f"up2_2{name}").to(device)
        # self.padding2 = torch.nn.ZeroPad2d(16)
        # self.conv2d_layer= nn.Conv2d(3, 32, kernel_size=1, stride=1, padding=0)
        # self.up3_2 = DecoderBlock2(64, 64, 1, activation=self.ACTIVATION, name=f"up3_2{name}").to(device)
        # self.up4_2 = DecoderBlock2(64, 64, 1, activation=self.ACTIVATION, name=f"up3_2{name}").to(device)

        ## Gating Signal for Attention Mechanism
        self.signal_3 = GatingSignal(in_channels = 32,
                                     out_size    = 32, 
                                     batch_norm  = batch_norm).to(device)
        
        #################################################################
        ####                                   ####
        #################################################################

        ## Attention Mechanism
        self.attention_3 = AttentionBlock(
                                input_shape_x = (256, 256, 32), 
                                input_shape_g = (64, 64, 32),
                                out_shape   = 32, 
                                # number      =  3,
                        # 0->512, 1->256, 2->128, 3->64, 4->32, 5->3
                                shape_theta_x = 128, 
                                shape_g = 64,
                                number = 2
                                          ).to(device)
        
        #################################################################
        ####                                   ####
        #################################################################
        # self.last_layer = nn.Conv2d(in_channels  = 3, 
        #                             out_channels = 3, 
        #                             kernel_size  = 1, 
        #                             stride       = 1,
        #                             padding      = 0, 
        #                             bias         = True).to(device)


    def forward(self, inputs):
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
       

        ######################################################
        #####                                    #####
        ######################################################
        # Dense layer transformations
        x1 = self.flatten(inputs)  # Flatten input
        x1 = self.dense_1(x1)
        x1 = self.act_1(x1)  
        x1 = self.reshape_1(x1)
        x1 = self.padding(x1)
        ######################################################
        #####                                    #####
        ######################################################
        # Decoder blocks
        x2 = self.conv2d_layer(inputs)
        x2 = self.act_2(x2)
        x2 = self.up1_2(x2)
        x2 = self.up2_2(x2)
        # self.padding2()
        # x1 = self.up3_2(x1)
        # x1 = self.up4_2(x1)

        ######################################################
        #####                                    #####
        ######################################################
        # Attention mechanism
        gating_64 = self.signal_3(x2)
        att_64 = self.attention_3(x1, gating_64)
        # Concatenation (PyTorch uses `torch.cat`)
        out = torch.cat([x1, att_64], dim=1)  # Concatenate along channel axis
        ######################################################
        #####                                    #####
        ######################################################
        ## Reduce from 32 t0 3 RGB channel output.
        # out = self.last_layer(att_64)
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
    def __init__(self, input_shape_x, input_shape_g, out_shape, number, 
                 shape_theta_x, shape_g
                 ):
        super(AttentionBlock, self).__init__()
        self.h_x,self.w_x,self.c_x = input_shape_x
        self.h_g,self.w_g,self.c_g = input_shape_g

        inter_shape = self.c_g
        self.number = number
        shape_x  = self.c_x #[32, 64, 128, 256]
        shape_x3 = self.c_x #[512, 256, 128, 64, 32, 3]
        # shape_g  = [16, 32,  64, 128]
        # shape_theta_x = [16, 32, 64, 128]
        shape_sigmoid = [16, 32, 64, 128]

        # Getting the x signal to the same shape as the gating signal
        self.theta_x_layer = nn.Conv2d(self.c_x, self.c_g, 
                                       kernel_size=(2, 2), stride=(2, 2), padding=0, bias=True)

        # Getting the gating signal to the same number of filters as the inter_shape
        self.phi_g_layer = nn.Conv2d(self.c_g, self.c_g, kernel_size=(1, 1), padding=0)
        self.upsample_g_layer = nn.ConvTranspose2d(self.c_g, self.c_g, kernel_size=(2, 2),
                                                   stride=(number,number),
                                                   padding=0, 
                                                   bias=True)
        self.concat_xg_layer = nn.Sequential()
        self.act_xg_layer = nn.ReLU()
        self.psi_layer = nn.Conv2d(inter_shape, 1, kernel_size=(1, 1), padding=0, bias=True)
        self.sigmoid_xg_layer = nn.Sigmoid()
        self.upsample_psi_layer1 = nn.Upsample(scale_factor=(number,number),
                                               mode='bilinear', align_corners=True)

        self.upsample_psi_layer2 = lambda x: x.repeat_interleave(self.c_g, dim=1)

        self.weight_rouer = WeightIncreses(in_channel=3, out_channel=32)
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

        # print(tcolors.RED,"theta_x: ", theta_x.shape,tcolors.ENDC)
        # print(tcolors.RED,"upsample_g: ", upsample_g.shape,tcolors.ENDC)
        


        # concat_xg = torch.mul(upsample_g, theta_x)
        concat_xg = torch.add(upsample_g, theta_x)
        act_xg = self.act_xg_layer(concat_xg)
      
        psi = self.psi_layer(act_xg)
 
        sigmoid_xg = self.sigmoid_xg_layer(psi)

        upsample_psi = self.upsample_psi_layer1(sigmoid_xg)
        upsample_psi = self.upsample_psi_layer2(upsample_psi)


        # print(tcolors.RED,
        #       "upsample_psi: ",
        #       upsample_psi.shape, tcolors.ENDC)
        # print(tcolors.RED,
        #       "x: ",
        #       x.shape, tcolors.ENDC)
        xi = self.weight_rouer(x)
        y = self.multiply_layer(upsample_psi, xi)

        result = self.result_layer(y)
        result_bn = self.result_bn_layer(result)
        return result_bn    
    

###############################################################
#####                                                #####
###############################################################
class WeightIncreses(nn.Module):
    def __init__(self, in_channel, out_channel, name=""):
        super(WeightIncreses, self).__init__()
        self.name = name
        self.separable_conv = nn.Sequential(
            nn.Conv2d(in_channels  = in_channel, 
                      out_channels = in_channel, 
                      kernel_size  = 1, 
                      stride   = 1, 
                      padding  ='same', 
                      groups   = in_channel, 
                      bias     = False),
            nn.Conv2d(in_channels  = in_channel, 
                      out_channels = out_channel, 
                      kernel_size  = 1, 
                      stride       = 1, 
                      padding      = 'same', 
                      bias         = False)
        )

    def forward(self, x):
        return self.separable_conv(x)
    

###############################################################
#####                                                #####
###############################################################