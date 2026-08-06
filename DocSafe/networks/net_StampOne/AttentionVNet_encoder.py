

#################
"""
@--28.07.2024--@
Author: 
INFO:
	> Pytorch version of StampOne_v89
"""
################



from FarhadCV.Tools import tcolors, bcolors
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Snake_activation_function import Snake
from .Tools_StampOne_Net2 import DecoderBlock
from .Tools_StampOne_Net2 import EncoderBlock
from .Tools_StampOne_Net2 import AttentionBlock
from .Tools_StampOne_Net2 import GatingSignal
from .Tools_StampOne_Net2 import WeightWavelet
from .Wavelet_transfer3 import WaveletDecompositionLayer as Wavelet_Layer
from .Wavelet_transfer3_message import WaveletDecompositionLayer as Wavelet_Layer_message
##################################################################
from .Sobel_Egdes import sobel_egdes
from .weight_wavelet import weight_wavelet
##################################################################
# from .message_preparation_networks_v01 import MessagePreparationNetworks
# from .message_preparation_networks_v03 import MessagePreparationNetworks
# from .message_preparation_networks_v04 import MessagePreparationNetworks
# from .message_preparation_networks_v05 import MessagePreparationNetworks
from .message_preparation_networks_v06 import MessagePreparationNetworks
##################################################################
# from ..Deformable_Conv2D import DeformConv
from ..AffineTransform.Affine_Layer_en_v01 import Affine_Layer_En


##################################################################
#####                                               #####
##################################################################
class AttentionVnetEncoder(nn.Module):
    def __init__(self, 
                       detr_load = None,
                       batch_size:int=None, 
                       image_shape:int=(256,256),
                       message_shape:int=(16,16),
                       croper_size = 64, 
                       batch_norm=True, 
                       device=None):
        super(AttentionVnetEncoder, self).__init__()
        
        self.device        = device
        self.image_shape   = image_shape
        self.message_shape = message_shape
        self.batch_size    = batch_size
        self.batch_norm    = batch_norm

        self.detr_load    = detr_load
        self.using_affine = False
        OUTPUT_CHANNELS = 3
        self.croper_size = croper_size
        #########################################################################
        ####                                                          ####
        #########################################################################
         ## cover image transtion into special domain
        self.sobel_edge_image_router = sobel_egdes(input_shape = self.image_shape[0], 
                                                   channel_in  = 3,
                                                   name = 'Sobel_Egdes_imageY').to(self.device)
        self.wavelet_image_layer = Wavelet_Layer(device=device).to(self.device)
        

        self.image_ww_layer = WeightWavelet(shape = self.image_shape, 
                                            name  = "encoded").to(self.device)
        #########################################################################
         ## message transtion into special domain
        self.sobel_edge_message_router = sobel_egdes(input_shape = self.message_shape[0], 
                                                     channel_in  = 3,
                                                     name        = 'Sobel_Egdes_messageY').to(self.device)
        self.wavelet_message_layer = Wavelet_Layer_message(device=device).to(self.device)
        

        self.message_ww_layer = WeightWavelet(shape      = self.message_shape, 
                                              channel_in = 15, 
                                              channel_out = 32, 
                                              name       = "encoded").to(self.device)
        #########################################################################
        ####                                                          ####
        #########################################################################
        # self.embedding_layer = MessagePreparationNetworks(
        #                             batch_size  = self.batch_size,  
        #                             input_shape = self.message_shape,
        #                             channelin   = 32, 
        #                             device      = self.device)
        self.embedding_layer = MessagePreparationNetworks(
                                    batch_size  = self.batch_size,  
                                    input_shape = self.message_shape,
                                    channelin   = 3, 
                                    device      = self.device)
        # self.deform_conv = DeformConv(in_channels=160, 
        #                             out_channels=160,
        #                             kernel_size=1, 
        #                             stride=1, 
        #                             padding=0)
        #########################################################
        ####                                       ####
        #########################################################
        # if self.detr_load != None:
        #     self.using_affine = True
        #     self.affine_router = Affine_Layer_En(
        #                             Deformable_DetrNet = self.detr_load,
        #                             input_size  = self.image_shape[0],
        #                             channel_in  = 160,
        #                             channel_out = 160,
        #                             batch_size  = batch_size, 
        #                             trainable   = True, 
        #                             image_size  = 256,
        #                             mask_size   = 256,
        #                             device = self.device)
        #     # self.uper_1 = DecoderBlock(32, 32, 4)
        #     # self.uper_2 = DecoderBlock(32, 32, 4)

        #########################################################
        
        #########################################################################
        ####                                                          ####
        #########################################################################
        self.down_stack = nn.ModuleList([
            EncoderBlock(288, 128, 4).to(self.device),  # (bs, 64, 64, 128)
            EncoderBlock(128, 256, 4).to(self.device),  # (bs, 32, 32, 256)
            EncoderBlock(256, 512, 4).to(self.device),  # (bs, 16, 16, 512)
            EncoderBlock(512, 512, 4).to(self.device),  # (bs, 8, 8, 512)
            EncoderBlock(512, 512, 4).to(self.device),  # (bs, 4, 4, 512)
            EncoderBlock(512, 512, 4).to(self.device),  # (bs, 2, 2, 512)
            EncoderBlock(512, 512, 4).to(self.device)   # (bs, 1, 1, 512)
        ])

        #########################################################################
        ####                                                          ####
        #########################################################################
        self.up_stack = nn.ModuleList([
            DecoderBlock(512, 512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
            DecoderBlock(544, 512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
            DecoderBlock(544, 512, 4),  # (bs, 16, 16, 1024)
            DecoderBlock(544, 256, 4),  # (bs, 32, 32, 512)
            DecoderBlock(288, 128, 4),  # (bs, 64, 64, 256)
            DecoderBlock(192, 64, 4)    # (bs, 128, 128, 128)
        ])

        self.signal_stack = nn.ModuleList([
            GatingSignal(in_channels=512 ,out_size=512, batch_norm=batch_norm),
            GatingSignal(in_channels=544 ,out_size=512, batch_norm=batch_norm),
            GatingSignal(in_channels=544 ,out_size=512, batch_norm=batch_norm),
            GatingSignal(in_channels=544 ,out_size=512, batch_norm=batch_norm),
            GatingSignal(in_channels=288 ,out_size=256, batch_norm=batch_norm),
            GatingSignal(in_channels=192 ,out_size=128, batch_norm=batch_norm),
        ])

        self.att_stack = nn.ModuleList([
            AttentionBlock(inter_shape=512, out_shape=32 , number=0),
            AttentionBlock(inter_shape=512, out_shape=32 , number=0),
            AttentionBlock(inter_shape=512, out_shape=32 , number=0),
            AttentionBlock(inter_shape=512, out_shape=32 , number=0),
            AttentionBlock(inter_shape=256, out_shape=64 , number=1),
            AttentionBlock(inter_shape=128, out_shape=128 , number=2),
        ])

        #########################################################################
        ####                                                          ####
        #########################################################################
        self.up_final = DecoderBlock(192, 64, 4)

        #########################################################################
        ####                      160 224                                    ####
        #########################################################################
        self.conv9  = nn.Conv2d(352, 64, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.last_layer = nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0)
        self.snake = Snake(beta=0.5, trainable=True)
        # self.snake = nn.Tanh()
        # self.snake = nn.ReLU()

    def forward(self, images, mask,  secrets):
   
        #########################################################
        ####                                       ####
        #########################################################
        dev_image     = self.sobel_edge_image_router(images)
        wavelet_image = self.wavelet_image_layer(dev_image)
        ximage        = self.image_ww_layer(wavelet_image)

        #########################################################
        ####                                       ####
        #########################################################
        
        
        # dev_message     = self.sobel_edge_message_router(secrets)
        # wavelet_message = self.wavelet_message_layer(dev_message)
        # xmessage        = self.message_ww_layer(wavelet_message)
        embedding       = self.embedding_layer(secrets, mask)
        # embedding = self.embedding_layer(xmessage)
        # print(tcolors.RED,
        #       "x: ",
        #       x.shape, tcolors.ENDC)
        # xmessage = self.deform_conv(xmessage)
        #########################################################
        ####                                       ####
        #########################################################
        # if self.using_affine:
        #     xmessage, boxes = self.affine_router(embedding, mask)
        #     effine_en = xmessage
        #     # xmessage = self.uper_1(xmessage)
        #     # xmessage = self.uper_2(xmessage)
        # else:
        #     effine_en = xmessage
        #     boxes = [0,0,0,0]
        #########################################################
        ####                                       ####
        #########################################################
        x = x1 = torch.cat([embedding, ximage], dim=1)
        # print(tcolors.RED,
        #       "x: ",
        #       x.shape, tcolors.ENDC)
        # print(tcolors.RED, "x:", x.shape, tcolors.ENDC)
        skips = []
        for down in self.down_stack:
            x = down(x)
            skips.append(x)
        
        skips = reversed(skips[:-1])

        #########################################################
        ####                                       ####
        #########################################################
        i = 0
        for up, skip in zip(self.up_stack, skips):
            gating = self.signal_stack[i](x)
            att = self.att_stack[i](skip, gating)
            x = up(x)
            x = torch.cat([x, att], dim=1)
            i += 1
            
        #########################################################
        ####                                       ####
        #########################################################
        x = self.up_final(x)
        x = torch.cat([x, x1], dim=1)
        # print(tcolors.RED, "x:", x.shape, tcolors.ENDC)
        x = self.conv9(x)
        x = self.conv10(x)
        residual = self.last_layer(x)
        residual = self.snake(residual)
        return residual, embedding

    def get_config(self):
        config = {
            # "reshape_size_1": self.reshape_size_1,
            # "reshape_size_2": self.reshape_size_2,
            # "batch_size": self.batch,
            "batch_norm": self.batch_norm,
        }
        return config
