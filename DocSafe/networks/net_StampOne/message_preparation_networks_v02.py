

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

##################################################################
# from ..DeformableDetr.deformable_transformer import DeformableTransformerEncoderLayer as DeformEncoder
# from ..DeformableDetr.deformable_transformer import DeformableTransformerEncoder 

# from ..DeformableDetr.deformable_transformer import DeformableTransformerDecoderLayer as DeformDecoder
# from ..DeformableDetr.deformable_transformer import DeformableTransformerDecoder 
##################################################################

from ..Deformable_Conv2D import DeformConv


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
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.device     = device
        self.ACTIVATION = Snake(beta=0.5, trainable=True)
        self.defdec_layers = 6
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
        # self.conv1d = nn.Conv2d(768, 768, kernel_size=1, stride=1, padding=0).to(device)
        self.conv1d = nn.Conv1d(1, 128, kernel_size=1, stride=1, padding=0).to(device)


        ## Activation
        self.act_1 = Snake(beta=0.5, trainable=True).to(device)

        ## Reshape Layer # Manually reshape in PyTorch
        self.reshape_1 = lambda x: x.view(self.batch_size, 64, 128, 128).to(device)  

        ######################################################
        #####                                    #####
        ######################################################
        ## Dense (Fully Connected) Layers
        # self.dense_2 = nn.Linear(2048, 64).to(device)  # Equivalent to Dense(64)
        self.dense_2 = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0).to(device)
        self.act_2 = Snake(beta=0.5, trainable=True).to(device)


        ## Decoder Blocks (Assuming decoder_block2 is a U-Net-like block)
        self.up1_2 = DecoderBlock2(64, 64, 1, activation=self.ACTIVATION, name=f"up1_2{name}").to(device)
        self.up2_2 = DecoderBlock2(64, 64, 1, activation=self.ACTIVATION, name=f"up2_2{name}").to(device)
        self.up3_2 = DecoderBlock2(64, 64, 1, activation=self.ACTIVATION, name=f"up3_2{name}").to(device)
        self.up4_2 = DecoderBlock2(64, 64, 1, activation=self.ACTIVATION, name=f"up3_2{name}").to(device)

        ## Gating Signal for Attention Mechanism
        self.signal_3 = GatingSignal(in_channels = 64 ,
                                     out_size    = 64, 
                                     batch_norm  = batch_norm).to(device)


        ## Attention Mechanism
        self.attention_3 = AttentionBlock(inter_shape = 64, 
                                          out_shape   = 64, 
                                          number      =  3 # 0->512, 1->256, 2->128, 3->64
                                          ).to(device)
        
        ########################################################################
        ########################################################################

        # self.defEncoder = DeformEncoder(d_model=256,
        #                         dim_feedforward=102,
        #                         dropout=0.1,
        #                         activation="relu",
        #                         num_feature_levels=4,
        #                         nhead=8,
        #                         enc_n_points=4,)
        
        ########################################################################
        # self.defDecoder = DeformDecoder(d_model=256,
        #                             dim_feedforward=102,
        #                             dropout=0.1,
        #                             activation="relu",
        #                             num_feature_levels=4,
        #                             nhead=8,
        #                             dec_n_points=4)
        # self.deform_decoder_trans = DeformableTransformerDecoder(
        #                                     decoder_layer = self.defDecoder, 
        #                                     num_decoder_layers = self.defdec_layers, 
        #                                     return_intermediate_dec=True)

        self.deform_conv = DeformConv(in_channels=128, 
                                    out_channels=128,
                                    kernel_size=1, 
                                    stride=1, 
                                    padding=0)
        
        ########################################################################
        self.last_layer = nn.Conv2d(in_channels  = 128, 
                                    out_channels = 128, 
                                    kernel_size  = 1, 
                                    stride       = 1,
                                    padding      = 0, 
                                    bias         = True).to(device)


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
        x = self.flatten(inputs)  # Flatten input
        x = x.unsqueeze(1)  # Expand dimensions equivalent to `expand_dims`
        # x = x.unsqueeze(2)  # Expand dimensions equivalent to `expand_dims
        x = self.conv1d(x)  # Apply Conv1D equivalent
        x = self.act_1(x)  # Apply activation
        x = x.reshape(self.batch_size, 64, 128, 128)  # Reshape
        # x = self.reshape_1(x)  # Reshape

        ######################################################
        #####                                    #####
        ######################################################
        # Dense layer transformations
        x1 = self.dense_2(inputs)
        x1 = self.act_2(x1)  
        ######################################################
        #####                                    #####
        ######################################################
        # Decoder blocks
        x1 = self.up1_2(x1)
        x1 = self.up2_2(x1)
        x1 = self.up3_2(x1)
        x1 = self.up4_2(x1)

        ######################################################
        #####                                    #####
        ######################################################
        # Attention mechanism
        gating_64 = self.signal_3(x)
        att_64 = self.attention_3(x1, gating_64)
        # Concatenation (PyTorch uses `torch.cat`)
        x1 = torch.cat([x1, att_64], dim=1)  # Concatenate along channel axis
        x1 = self.deform_conv(x1)
        ######################################################
        #####                                    #####
        ######################################################
        ## Reduce from 128 t0 3 RGB channel output.
        out = self.last_layer(x1)
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

        


        concat_xg = torch.mul(upsample_g, theta_x)
        act_xg = self.act_xg_layer(concat_xg)
      
        psi = self.psi_layer(act_xg)
 
        sigmoid_xg = self.sigmoid_xg_layer(psi)

        upsample_psi = self.upsample_psi_layer1(sigmoid_xg)
        upsample_psi = self.upsample_psi_layer2(upsample_psi)


        

        y = self.multiply_layer(upsample_psi, x)

        result = self.result_layer(y)
        result_bn = self.result_bn_layer(result)
        return result_bn    