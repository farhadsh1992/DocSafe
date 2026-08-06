#################
"""
@--04.02.2024--@
Author: 
INFO:
	> Pytorch version of StampOne_v89
"""
################


##################################################################
from FarhadCV.Tools import tcolors, bcolors
import torch
import torch.nn as nn
# import torch.nn.functional as F
##################################################################
from .Snake_activation_function import Snake
from .Tools_StampOne_Net2 import DecoderBlock
from .Tools_StampOne_Net2 import DecoderBlock2
from .Tools_StampOne_Net2 import EncoderBlock
from .Tools_StampOne_Net2 import AttentionBlock
from .Tools_StampOne_Net2 import GatingSignal
from .Tools_StampOne_Net2 import WeightWavelet
from .Tools_StampOne_Net2 import Conv1DLayer1
from .Tools_StampOne_Net2 import Conv1DLayer2
##################################################################
# from .Special_Transform_Network_vt01 import STN
# from .Probabilistic_Special_Transform_Network_v01 import PSTN
##################################################################
# # from .Wavelet_transfer import Wavelet_Layer
# from .Wavelet_transfer2 import Wavelet_Layer
# from .Wavelet_transfer_keras3 import Wavelet_Layer_Keras3
from .Wavelet_transfer3 import WaveletDecompositionLayer as Wavelet_Layer
##################################################################
from .Sobel_Egdes import sobel_egdes
from .weight_wavelet import weight_wavelet
##################################################################
# from Network_Libs.DeformableDetr.deformable_transformer import DeformableTransformerEncoderLayer as DeformEncoder
# from ..Deformable_Conv2D import DeformConv
from .Special_Transform_Network_vt02 import STNMASK
##################################################################
# from ..AffineTransform.Affine_Layer_de_v01 import Affine_Layer_De
# from ..PSTNet.Probabilistic_Special_Transform_Network_v02 import ProbabilisticSTN
##################################################################
#####                                               #####
##################################################################
class AttentionVnetDecoder(nn.Module):
    def __init__(self, 
                 detr_load        = None,
                 batch_size:int   = 1, 
                 image_shape      = (256, 256), 
                 croper_size:int  = 32,
                 batch_norm:bool  = True, 
                 device:str       = None):
        super(AttentionVnetDecoder, self).__init__()

        self.device      = device
        self.image_shape = image_shape

        self.detr_load    = detr_load
        self.using_affine = False
        self.croper_size = croper_size
        # self.up_in_1 = DecoderBlock2(32, 1, nn.ReLU())  # none, 64, 64, ??

        #########################################################
        ####                                       ####
        #########################################################
        self.sobel_edge_encoded_router = sobel_egdes(input_shape = self.image_shape, 
                                            name = 'Sobel_Egdes_imageY').to(self.device)
        self.wavelet_encoded_layer = Wavelet_Layer(device=device)
        self.encoded_ww_layer = WeightWavelet(shape=self.image_shape, name="encoded")
        #########################################################
        ####                                       ####
        #########################################################
        # self.first_layer = nn.Conv2d(in_channels  = 3, 
        #                             out_channels = 32, 
        #                             kernel_size  = 1, 
        #                             stride       = 1,
        #                             padding      = 0, 
        #                             bias         = True)
        # self.stn_router = STN(filters=32+3, batch=batch_size, channel=32+3)
        self.stn_router = STNMASK(filters=32+3, batch=batch_size, channel_mask=3)

        # self.stn_router = ProbabilisticSTN(
        #                     input_size = 256, 
        #                     batch_size = batch_size,  
        #                     channel_in = 3, 
        #                     filters    = 32+3)
        #########################################################
        ####                                       ####
        #########################################################
        # if self.detr_load != None:
        #     self.using_affine  = True
        #     self.affine_router = Affine_Layer_De(
        #                             Deformable_DetrNet = self.detr_load,
        #                             input_size  = self.image_shape[0],
        #                             channel_in  = 32,
        #                             channel_out = 32,
        #                             batch_size  = batch_size, 
        #                             trainable   = True, 
        #                             image_size_out = self.croper_size, 
        #                             image_size  = 256,
        #                             mask_size   = 256,
        #                             device = self.device)
        #     grow_num_block = int((256/self.croper_size)/4)+1
        #     self.upper_1 = nn.ModuleList([
        #         DecoderBlock(32, 32, 4) for i in range(grow_num_block)
        #         ])
        #    # self.uper_1 = DecoderBlock(32, 32, 4)
        #    # self.uper_2 = DecoderBlock(32, 32, 4)
        #     # self.uper_3 = DecoderBlock(32, 32, 4)

        #########################################################
        ####                                       ####
        #########################################################
        self.conv1d_r1 = Conv1DLayer1()
        #########################################################
        ####                                       ####
        #########################################################

        self.down_stack = nn.ModuleList([
            EncoderBlock(64, 64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
            EncoderBlock(64, 128, 4),  # (bs, 64, 64, 128)
            EncoderBlock(128, 256, 4),  # (bs, 32, 32, 256)
            EncoderBlock(256, 512, 4),  # (bs, 16, 16, 512)
            EncoderBlock(512, 512, 4),  # (bs, 8, 8, 512)
            EncoderBlock(512, 512, 4),  # (bs, 4, 4, 512)
            EncoderBlock(512, 512, 4),  # (bs, 2, 2, 512)
            EncoderBlock(512, 512, 4)   # (bs, 1, 1, 512)
        ])
        # self.stn_stack = nn.ModuleList([
        #     STN(filters=64, batch=batch_size, channel=64),
        #     STN(filters=128, batch=batch_size, channel=128),
        #     STN(filters=256, batch=batch_size, channel=256),
        #     STN(filters=512, batch=batch_size, channel=512),
        #     STN(filters=512, batch=batch_size, channel=512),
        #     # STN(filters=512, batch=batch_size, channel=512),
        #     # STN(filters=512, batch=batch_size, channel=512),
        #     # STN(filters=512, batch=batch_size, channel=512),
        # ])
        #########################################################
        ####                                       ####
        #########################################################
        self.up_stack = nn.ModuleList([
            DecoderBlock(512, 512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
            DecoderBlock(544, 512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
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
            GatingSignal(in_channels=544 ,out_size=256, batch_norm=batch_norm),
            GatingSignal(in_channels=288 ,out_size=128, batch_norm=batch_norm),
            GatingSignal(in_channels=192 ,out_size=64, batch_norm=batch_norm)
        ])

        self.att_stack = nn.ModuleList([
            AttentionBlock(inter_shape=512, out_shape=32, number=0),
            AttentionBlock(inter_shape=512, out_shape=32, number=0),
            AttentionBlock(inter_shape=512, out_shape=32, number=0),
            AttentionBlock(inter_shape=512, out_shape=32, number=0),
            AttentionBlock(inter_shape=256, out_shape=32, number=1),
            AttentionBlock(inter_shape=128, out_shape=64, number=2),
            AttentionBlock(inter_shape=64, out_shape=128, number=3)
        ])
        #########################################################
        ####                                       ####
        #########################################################
        self.up_layer = DecoderBlock(192, 192, 4)
        #########################################################
        ####                                       ####
        #########################################################
        self.conv9  = EncoderBlock(224, 96, 4)  # 128
        self.conv10 = EncoderBlock(96, 64, 4)  # 64
        self.conv11 = EncoderBlock(64, 64, 4)  # 32
        self.conv12 = EncoderBlock(64, 64, 4)  # 16


        
        # DeformEncoder()
        #########################################################
        ####                                       ####
        #########################################################
        self.conv1d_router2 = Conv1DLayer2()

        #########################################################
        ####                                       ####
        #########################################################
        self.last_layer = nn.Conv2d(in_channels  = 128, 
                                    out_channels = 3, 
                                    kernel_size  = 1, 
                                    stride       = 1,
                                    padding      = 0, 
                                    bias         = True)
        #########################################################    
        # self.snake = Snake(beta=0.5, trainable=True)
        # self.snake = nn.Tanh()
        # self.snake = nn.ReLU()
        self.snake = nn.Sigmoid()

    def forward(self, inputs, mask):
        
        #########################################################
        ####                                       ####
        #########################################################
        dev_encimg = self.sobel_edge_encoded_router (inputs)
        wavelet_encimg = self.wavelet_encoded_layer(inputs=inputs, sobel= dev_encimg)
        xencoded = self.encoded_ww_layer(wavelet_encimg)
        # x1 = self.up_in_1(x1)
        # x1 = self.first_layer(inputs)
        #########################################################
        ####                                       ####
        #########################################################
        xMix = torch.cat([mask, xencoded], dim=1)
        xMix = self.stn_router(xMix, mask)
        x = x1 = xMix[:,3:,:,:]
        xmask =  xMix[:,:3,:,:]
        #########################################################
        ####                                       ####
        #########################################################
        # if self.using_affine:
        #     x, boxes, cropped_mask = self.affine_router(x, xmask)
        #     croped = x
        #     for up1 in self.upper_1:
        #         x = up1(x)
        #     # x = self.uper_1(x)
        #     # x = self.uper_2(x)
        #     # x = self.uper_3(x)
        # else:
        #     croped = x
        #     cropped_mask = x
        #     boxes = [0,0,0,0]
        x  = self.conv1d_r1(x1)
        # print(tcolors.RED,"(AttentionVNet_decoder) x: ", 
        #       x.shape,tcolors.ENDC)
        #########################################################
        ####                                       ####
        #########################################################
        skips = []
        i = 0
        for down in self.down_stack:
            x = down(x)
            # if i < len(self.stn_stack):
            #     # print(x.shape)
            #     x = self.stn_stack[i](x)
            skips.append(x)
            i += 1

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
        x = self.up_layer(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv1d_router2(x)
        recovered = self.last_layer(x)
        recovered = self.snake(recovered)
        return recovered #, croped, boxes,cropped_mask
