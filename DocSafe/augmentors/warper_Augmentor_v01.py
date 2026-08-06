"""
@--05.02.2025--@
Author: github/farhadsh1992
INFO:
    -ref: 
        
    
LAST_UPDATE:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry
import numpy as np
from .utils import get_rand_transform_matrix
from FarhadCV.Tools import tcolors, bcolors

class Warper_Transformer(nn.Module):
    def __init__(self, 
                 args, 
                 noise_args, 
                 batch_size:int, 
                 image_size:int, 
                 device = None):
        super(Warper_Transformer, self).__init__()

        self.args = args
        self.noise_args = noise_args

        self.batch_size = batch_size
        self.image_size = image_size
        self.device = device

        self.borders = noise_args.borders

        print(bcolors.BLACK+tcolors.WHITE+tcolors.BOLD,
            f"Adding Warping Version 2 - RAMP:{noise_args.rnd_rotation_ramp}"+
            f", max_rotate:{noise_args.max_rotation}"
            + tcolors.ENDC)

    def generate_random_matrix(self, epoch):
        speed_rotation    = self.noise_args.speed_rotation
        max_rotation      = self.noise_args.max_rotation # StegaStamp is 0.1
        rnd_rotation_ramp = self.noise_args.rnd_rotation_ramp
        
        rnd_tran = min(speed_rotation * epoch / rnd_rotation_ramp, max_rotation)
        rnd_tran = np.random.uniform() * rnd_tran
        self.warper_matrix = get_rand_transform_matrix(self.image_size, 
                np.floor(self.image_size * rnd_tran), self.batch_size)
        self.warper_matrix = self.warper_matrix.to(self.device)
        
        # return self.warper_matrix
    def warper_inputs(self, image_input):


        input_warped = torchgeometry.warp_perspective(image_input, 
                                                    self.warper_matrix[:, 1, :, :], 
                                                    dsize=(self.image_size, self.image_size), 
                                                    flags='bilinear')
        mask_warped = torchgeometry.warp_perspective(torch.ones_like(input_warped), 
                                                     self.warper_matrix[:, 1, :, :], 
                                                     dsize=(self.image_size, self.image_size),
                                                     flags='bilinear')
        
        input_warped += (1 - mask_warped) * image_input
         
        return input_warped

    def warper_encoded_outputs(self, 
                               image, 
                               input_warped, 
                               residual_warped):
        


        
        
        encoded_warped = residual_warped + input_warped
        residual = torchgeometry.warp_perspective(residual_warped, 
                                                  self.warper_matrix[:, 0, :, :], 
                                                  dsize=(self.image_size, self.image_size), 
                                                  flags='bilinear')
        


        if self.borders == 'no_edge':
            encoded_image = residual + image #encoded_warped 
            #encoded_image = kornia.enhance.add_weighted(residual, 0.5, image_input,0.5,1.0)
            encoded_image = torch.clamp(encoded_image,0,1)
            
        elif self.borders == 'black':
            encoded_image = residual_warped + input_warped
            encoded_image = torchgeometry.warp_perspective(encoded_image, 
                                                           self.warper_matrix[:, 0, :, :], 
                                                           dsize=(self.image_size, self.image_size), 
                                                           flags='bilinear')
            input_unwarped = torchgeometry.warp_perspective(input_warped, 
                                                            self.warper_matrix[:, 0, :, :], 
                                                            dsize=(self.image_size, self.image_size), 
                                                            flags='bilinear')
        elif self.borders.startswith('random'):
            mask = torchgeometry.warp_perspective(torch.ones_like(residual), 
                                                  self.warper_matrix[:, 0, :, :], 
                                                  dsize=(self.image_size, self.image_size),
                                                flags='bilinear')
            input_unwarped = torchgeometry.warp_perspective(input_warped, 
                                                            self.warper_matrix[:, 0, :, :], 
                                                            dsize=(self.image_size, self.image_size),
                                                            flags='bilinear')
            warped_encoded = residual_warped + input_unwarped
            warped_encoded= torchgeometry.warp_perspective(warped_encoded, 
                                                           self.warper_matrix[:, 0, :, :],
                                                           dsize=(self.image_size, self.image_size), 
                                                          flags='bilinear')
            ch = 3 if self.borders.endswith('rgb') else 1
            warped_encoded += (1 - mask) * torch.ones_like(residual) * torch.rand([ch])
        elif self.borders == 'white':
            mask = torchgeometry.warp_perspective(torch.ones_like(residual), 
                                                  self.warper_matrix[:, 0, :, :], 
                                                  dsize=(self.image_size, self.image_size),
                                                  flags='nearest')
        
        
            encoded_image = residual_warped + input_warped
            encoded_image = torchgeometry.warp_perspective(encoded_image, 
                                                           self.warper_matrix[:, 0, :, :], 
                                                           dsize=(self.image_size, self.image_size), 
                                                           flags='nearest')
            input_unwarped = torchgeometry.warp_perspective(input_warped, 
                                                            self.warper_matrix[:, 0, :, :], 
                                                            dsize=(self.image_size, self.image_size), 
                                                            flags='nearest')
            warped_encoded += (1 - mask) * torch.ones_like(residual)

        elif self.borders == 'image':
            mask = torchgeometry.warp_perspective(torch.ones_like(residual), 
                                                  self.warper_matrix[:, 0, :, :], 
                                                  dsize=(self.image_size, self.image_size),
                                                  flags='bilinear')
            
            encoded_image = residual_warped + input_warped
            encoded_image = torchgeometry.warp_perspective(encoded_image, 
                                                           self.warper_matrix[:, 0, :, :], 
                                                           dsize=(self.image_size, self.image_size), 
                                                           flags='bilinear')
            encoded_image += (1 - mask) * torch.roll(image, 1, 0)

        encoded_image2 = residual + image
        # encoded_image has noises 
        return  encoded_image,  encoded_warped, residual, encoded_image2
    