










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






class Affine_Layer_De(nn.Module):
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
                 device= "cpu", 
                 **kwargs):
        super(Affine_Layer_De, self).__init__(**kwargs)
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
        self.device = device

        self.croper_size = int(image_size_out)
        #############################################################
        ###     ####
        #############################################################
        self.detr_decoder = Deformable_DetrNet


        #############################################################
        ### Create the transform factor as a trainable weight     ####
        #############################################################
        self.transform_factor = nn.Parameter(
                                    torch.tensor([[1, 0, -1, 0, 1, -1, 0, 0] 
                                    for _ in range(batch_size)], dtype=torch.float32), 
                                        requires_grad=trainable)
        

        size = image_size_out / 2 / 256
        # self.boxes = nn.Parameter(torch.tensor([[(128-size)/256, (128-size)/256, 
        #                                          (128+size)/256, (128+size)/256] 
        #                         for _ in range(batch_size)], dtype=torch.float32), 
        #                         requires_grad=False)


    def forward(self, inputs, mask):
        # print(tcolors.RED,"(Affine_Layer_de_v01) mask: ",mask.shape,tcolors.ENDC)
        downsampled = self.detr_decoder(mask.clone())
        transform_factor = downsampled['pred_boxes']
        # transform_factor = torch.sigmoid(transform_factor)
        # print(tcolors.RED, "(Affine_Layer_de_v01) transform_factor: ", transform_factor.shape, tcolors.ENDC)
        ##########################################################
        # bbox_x1 = transform_factor[:, 0, 0] - 2
        # bbox_x2 = transform_factor[:, 0, 1] - 22

        # middle_face_y =  ((transform_factor[:,0, 2] - transform_factor[:,0, 0])/2-0.4)*128
        # middle_face_x =  ((transform_factor[:,0, 3] - transform_factor[:,0, 1])/2-0.4)*128

       

        # self.transform_factor[:, 2] = - middle_face_y #bbox_x1
        # self.transform_factor[:, 5] = -middle_face_x #bbox_x2

        # # Affine transformation
        # grid = F.affine_grid(self.transform_factor[:, :6].view(-1, 2, 3), inputs.size(), align_corners=False)
        # img2 = F.grid_sample(inputs, grid, align_corners=False)

        # cropped_images = self.crop_and_resize_layer(img_in=img2, boxes=self.boxes)
        ##########################################################
        ## [ymin, xmin, ymax, xmax]
        # y_min = int(transform_factor[:,0, 0]*256)
        # x_min = int(transform_factor[:,0, 1]*256)
        # y_max = int(transform_factor[:,0, 2]*256)
        # x_max = int(transform_factor[:,0, 3]*256)
        
        

        ## y_max, y_min, x_max, x_min
        # box = (transform_factor[:,0, 2][0], transform_factor[:,0, 0][0], 
        #        transform_factor[:,0, 3][0], transform_factor[:,0, 1][0])

        self.boxes = torch.tensor([[0.0, 0.0, 0, 0] 
                        for _ in range(self.batch_size)], dtype=torch.float32)
        self.boxes[:,0] = transform_factor[:,0, 0]
        self.boxes[:,1] = transform_factor[:,0, 1]
        self.boxes[:,2] = transform_factor[:,0, 2]
        self.boxes[:,3] = transform_factor[:,0, 3]

        


        # middle_x = int((y_max - y_min) / 2)
        # middle_y = int((x_max - x_min) / 2)

        # cropped_images = inputs[:,:, 
        #                         middle_y:middle_y+self.croper_size,
        #                         middle_x:middle_x+self.croper_size].clone().to(self.device)
        
        self.cropped_images = torch.zeros((self.batch_size, self.channel_in, 
                                           self.croper_size, self.croper_size)).to(self.device)
        self.cropped_mask = torch.zeros((self.batch_size, 3, 
                                         self.croper_size, self.croper_size)).to(self.device)

        for i in range(self.batch_size):
            y_min = int(transform_factor[:,0, 0][i]*256)
            x_min = int(transform_factor[:,0, 1][i]*256)
            y_max = int(transform_factor[:,0, 2][i]*256)
            x_max = int(transform_factor[:,0, 3][i]*256)
            # self.cropped_images[i] = inputs[i,:, 
            #                         y_min:y_min+self.croper_size,
            #                         x_min:x_min+self.croper_size].clone().to(self.device)
            # self.cropped_mask[i] = mask[i,:, 
            #                         y_min:y_min+self.croper_size,
            #                         x_min:x_min+self.croper_size].clone().to(self.device)
            
            middle_y = int((((y_max-y_min)/2.0)+y_min))-int(16)
            middle_x = int((((x_max-x_min)/2.0)+x_min))-int(16)
            self.cropped_images[i] = inputs[i,:, 
                                    middle_y:middle_y+self.croper_size,
                                    middle_x:middle_x+self.croper_size].clone().to(self.device)
            self.cropped_mask[i] = mask[i,:, 
                                    middle_y:middle_y+self.croper_size,
                                    middle_x:middle_x+self.croper_size].clone().to(self.device)
        # cropped_images = cropped_images.to(self.device)

        del(transform_factor)
        del(y_max)
        del(y_min)
        del(x_max)
        del(x_min)

        return self.cropped_images, self.boxes, self.cropped_mask









class CropAndResizeLayer(nn.Module):
    def __init__(self, batch_size, image_size_out):
        super(CropAndResizeLayer, self).__init__()
        self.batch_size = batch_size
        self.image_size_out = image_size_out

    def forward(self, img_in, boxes):
        box_indices = torch.arange(self.batch_size, device=img_in.device)
        cropped_images = F.grid_sample(img_in, F.affine_grid(boxes[:, :6].view(-1, 2, 3), 
                                                             [self.batch_size, img_in.size(1), 
                                                              self.image_size_out, self.image_size_out],
                                                                align_corners=False), align_corners=False)
        return cropped_images
