


#################
"""
@--08.02.2025--@
Author: 
INFO:
	> Pytorch version of StampOne_v89
"""
################




#############################################################
from FarhadCV.Tools import tcolors, bcolors
import torch
import torch.nn as nn
import torch.nn.functional as F
#############################################################
# from .DeformableDetr import Deformable_DetrNet
# from Network_Libs.DETR_NET.detr_net import detr_resnet50
#############################################################
from FarhadCV.Tools import tcolors, bcolors


#############################################################
#####                                       #####
#############################################################
class Affine_Layer_En(nn.Module):
    """
    A custom affine transformation layer.
    """
    def __init__(self, 
                 Deformable_DetrNet,
                 input_size:int,
                 channel_in:int,
                 channel_out:int,
                 batch_size: int = 1, 
                 trainable=True, 
                 image_size_out = 32, 
                 image_size = 256,
                 mask_size = 256,
                 croper_size = 64, 
                 device= "cpu", 
                 **kwargs):
        super(Affine_Layer_En, self).__init__(**kwargs)
        #############################################################
        ### Create the transform factor as a trainable weight    ####
        #############################################################
        self.supports_masking = True
        self.batch_size = batch_size
        self.trainable  = trainable
        self.image_size = image_size
        self.input_size  = input_size
        self.channel_in  = channel_in
        self.channel_out = channel_out
        self.mask_size = mask_size

        self.image_size_out = int(image_size_out/2.0)
        self.device = device

        self.croper_size = int(croper_size)
        self.croper_half = int(croper_size/2)
        #############################################################
        ###     ####
        #############################################################
        self.detr_decoder = Deformable_DetrNet


        #############################################################
        ### Create the transform factor as a trainable weight     ####
        #############################################################
        self.transform_factor = nn.Parameter(torch.tensor([[1, 0, -1, 0, 1, -1, 0, 0] 
                                                           for _ in range(batch_size)], dtype=torch.float32), 
                                                           requires_grad=trainable)
        # self.boxes = nn.Parameter(torch.tensor([[0, 0, 0, 0] 
        #                         for _ in range(batch_size)], dtype=torch.float32), 
        #                         requires_grad=False)
        
        # self.img2 = nn.Parameter(torch.zeros((batch_size, channel_in, 
        #                          image_size,image_size), 
        #                          requires_grad=False)).to(self.device)

    def forward(self, inputs, mask):
        # print(tcolors.RED,"(Affine_Layer_de_v01) mask: ",mask.shape,tcolors.ENDC)
        downsampled = self.detr_decoder(mask)
        transform_factor = downsampled['pred_boxes']
        # transform_factor = torch.sigmoid(transform_factor)
        # print(tcolors.RED, "(Affine_Layer_en_v01) transform_factor: ", transform_factor.shape, tcolors.ENDC)
        #############################################################
        # bbox_x1 = transform_factor[:, 0, 0]
        # bbox_x2 = transform_factor[:, 0, 1] + 32

        # middle_face_y =  ((transform_factor[:,0, 2] - transform_factor[:,0, 0])/2-0.4)*128
        # middle_face_x =  ((transform_factor[:,0, 3] - transform_factor[:,0, 1])/2-0.4)*128

        # print(tcolors.RED,"middle_face_x: ",  middle_face_x.shape, tcolors.ENDC)
        # self.transform_factor[:, 2] = -1 * middle_face_y[0]
        # self.transform_factor[:, 5] = -1 * middle_face_x[0]

        # # Affine transformation
        # grid = F.affine_grid(self.transform_factor[:, :6].view(-1, 2, 3), inputs.size(), align_corners=False)
        # img2 = F.grid_sample(inputs, grid, align_corners=False)
        #############################################################
        # with torch.no_grad():
        self.img2 = torch.zeros(inputs.shape).to(self.device)
        ################################
        ## [ymin, xmin, ymax, xmax]
        # y_min = int(transform_factor[:,0, 0]*256)
        # x_min = int(transform_factor[:,0, 1]*256)
        # y_max = int(transform_factor[:,0, 2]*256)
        # x_max = int(transform_factor[:,0, 3]*256)
        # middle_x = int((y_max - y_min) / 2)
        # middle_y =  int((x_max - x_min) / 2)
        ################################
        ## y_max, y_min, x_max, x_min
        # box = (transform_factor[:,0, 2][0], transform_factor[:,0, 0][0], 
        #        transform_factor[:,0, 3][0], transform_factor[:,0, 1][0])
        # print("boxes: ", self.boxes.shape)
        ################################
        ## [ymin, xmin, ymax, xmax]
        self.boxes = torch.tensor([[0.0, 0.0, 0, 0] 
                                for _ in range(self.batch_size)], dtype=torch.float32)
        self.boxes[:,0] = transform_factor[:,0, 0]
        self.boxes[:,1] = transform_factor[:,0, 1]
        self.boxes[:,2] = transform_factor[:,0, 2]
        self.boxes[:,3] = transform_factor[:,0, 3]

        ################################
        # print("middle_y: ", middle_y,middle_y+self.croper_size)
        for i in range(self.batch_size):
            y_min = int(transform_factor[:,0, 0][i]*256)
            x_min = int(transform_factor[:,0, 1][i]*256)
            y_max = int(transform_factor[:,0, 2][i]*256)
            x_max = int(transform_factor[:,0, 3][i]*256)
            # self.img2[i,:,
            #         y_min:y_min+self.croper_size,
            #         x_min:x_min+self.croper_size] = inputs[i,:, 
            #                     (128-self.croper_half):(128+self.croper_half), 
            #                     (128-self.croper_half):(128+self.croper_half)].to(self.device)
            
            middle_y = int((((y_max-y_min)/2.0)+y_min))-16
            middle_x = int((((x_max-x_min)/2.0)+x_min))-16
            

            self.img2[i,:,
                    middle_y:middle_y+self.croper_size,
                    middle_x:middle_x+self.croper_size] = inputs[i,:, 
                                (128-self.croper_half):(128+self.croper_half), 
                                (128-self.croper_half):(128+self.croper_half)].to(self.device)
        # img2 = img2.to(self.device)

        del(transform_factor)
        del(y_max)
        del(y_min)
        del(x_max)
        del(x_min)

        return self.img2, self.boxes
