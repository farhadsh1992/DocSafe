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

import numpy as np
import kornia

from .utils import random_blur_kernel
from .utils import get_rnd_brightness_torch
# from .utils import jpeg_compress_decompress
from .utils_jpeg_torch import jpeg_compress_decompress
from .utils import round_only_at_0
from FarhadCV.Tools import tcolors, bcolors



class Augmentation_Transformer(nn.Module):
    def __init__(self, 
                 args, 
                 batch_size:int, 
                 image_size:int, 
                 device = None):
        super(Augmentation_Transformer, self).__init__()

        self.args = args
        self.batch_size = batch_size
        self.image_size = image_size
        self.device = device
        ramp_fn = (lambda ramp, steps: torch.minimum(torch.tensor(steps, 
                                                          dtype=torch.float32) / ramp, torch.tensor(1.0)))
        ramp_max_fn = (lambda ramp, max_val, steps: torch.minimum(torch.tensor(steps, dtype=torch.float32) / 
                                                          ramp, torch.tensor(max_val)))


        
        self.jpeq_router  = jpeg_compress_decompress(rounding = round_only_at_0,
                                                     height   = image_size, 
                                                     width    = image_size,
                                                     device   = self.device)

    def forward(self, encoded_image:torch.Tensor, steps:int):
        
        ramp_fn = lambda ramp: np.min([steps / ramp, 1.])
        
        
        ######################################################
        ####  blur   ####
        ######################################################
        #if global_step > 5000:
        encoded_image = kornia.augmentation.RandomPosterize(bits = 7,p = 0.5)(encoded_image)
        
        N_blur = 7
        f = random_blur_kernel(probs  = [.25, .25], 
                               N_blur = N_blur, 
                               sigrange_gauss = [1., 3.], 
                               sigrange_line  = [.25, 1.],
                               wmin_line = 3)
        # if self.args.cuda:
        f = f.to(self.device)
        encoded_image = F.conv2d(encoded_image, f, padding="same")    
        ######################################################
        #### Rnd noise         ####
        ######################################################
        rnd_noise = torch.rand(1)[0] * ramp_fn(self.args.rnd_noise_ramp) * self.args.rnd_noise
        noise = torch.normal(mean  = 0, 
                             std   = rnd_noise, 
                             size  = encoded_image.size(), 
                             dtype = torch.float32)
        # if self.args.cuda:
        noise = noise.to(self.device)
        encoded_image = encoded_image + noise
        encoded_image = torch.clamp(encoded_image, 0, 1)
        
        ######################################################
        ####  poisson noise         ####
        ######################################################
        ## vals = len(torch.unique(encoded_image))
        ## vals = 2 ** np.ceil(np.log2(vals))
        ## encoded_image = torch.poisson(encoded_image * vals)/float(vals)
        ## encoded_image = torch.clamp(encoded_image, 0, 1)
        ## np.random.poisson(image * vals) / float(vals)

        ## b,ch,row,col = encoded_image.shape
        ## gauss = torch.rand(b,ch,row,col).cuda()
        ## encoded_image = encoded_image + encoded_image * gauss
        ## encoded_image = torch.clamp(encoded_image, 0, 1)

        ######################################################
        #### contrast & brightness ####
        ######################################################
        rnd_bri = ramp_fn(self.args.rnd_bri_ramp) * self.args.rnd_bri
        rnd_hue = ramp_fn(self.args.rnd_hue_ramp) * self.args.rnd_hue

        rnd_brightness = get_rnd_brightness_torch(rnd_bri, rnd_hue, self.batch_size)


        contrast_low = 1. - (1. - self.args.contrast_low) * ramp_fn(self.args.contrast_ramp)
        contrast_high = 1. + (self.args.contrast_high - 1.) * ramp_fn(self.args.contrast_ramp)
        contrast_params = [contrast_low, contrast_high]

        contrast_scale = torch.Tensor(encoded_image.size()[0]).uniform_(contrast_params[0], 
                                                                        contrast_params[1])
        contrast_scale = contrast_scale.reshape(encoded_image.size()[0], 1, 1, 1)
        ## encoded_image = kornia.augmentation.RandomPlasmaBrightness(
        ##     roughness=(contrast_params[0], contrast_params[1]), p=1.)(encoded_image)
        
        # if args.cuda:
        contrast_scale = contrast_scale.to(self.device)
        rnd_brightness = rnd_brightness.to(self.device)

        encoded_image = encoded_image * contrast_scale
        encoded_image = encoded_image + rnd_brightness
        encoded_image = torch.clamp(encoded_image, 0, 1)

        ######################################################
        ####  saturation ####
        ######################################################
        rnd_sat = torch.rand(1)[0] * ramp_fn(self.args.rnd_sat_ramp) * self.args.rnd_sat
        sat_weight = torch.FloatTensor([.3, .6, .1]).reshape(1, 3, 1, 1)
      
        sat_weight = sat_weight.to(self.device)
        encoded_image_lum = torch.sum(encoded_image * sat_weight, dim=1).unsqueeze_(1)
        encoded_image = ((1 - rnd_sat) * encoded_image + rnd_sat * encoded_image_lum)
        
        ######################################################
        ##jpeg_quality = torch.tensor([int(jpeg_quality)]*8).cuda() 
        ##print(jpeg_quality)
        ##encoded_image = diff_jpeg_coding(image_rgb=encoded_image, jpeg_quality=jpeg_quality)
        ##(encoded_image.size())
        ##encoded_image = kornia.augmentation.RandomPosterize(bits=7)(encoded_image) 
        ## #kornia.enhance.posterize(encoded_image, bits=7)

        ##encoded_image = torch.clamp(encoded_image, 0, 1)
        ##encoded_image = dithering(encoded_image)
        ######################################################
        #### jpeg                       ####
        ######################################################
        ## Problem With JPEG
        jpeg_quality = (100. - torch.rand(1)[0] * 
                        ramp_fn(self.args.jpeg_quality_ramp) * (100. - self.args.jpeg_quality))
        #encoded_image = encoded_image.reshape([-1, 3, 256, 256]).contiguous()
        # encoded_image2 = jpeg_compress_decompress(encoded_image, 
        #                                          rounding = round_only_at_0,
        #                                          quality  = jpeg_quality,
        #                                          device=self.device).to(self.device)
        # encoded_image =  self.jpeq_router.call(encoded_image, factor=jpeg_quality)
        # encoded_image = encoded_image.to(self.device)
        # print(tcolors.RED,"encoded_image", encoded_image.shape,tcolors.ENDC)
        # encoded_image = encoded_image.reshape([-1,3,256,256]).contiguous()

        ######################################################
        ## encoded_image = kornia.filters.motion_blur(encoded_image, 5, 
        ##                                            torch.tensor([90., 180,]), 
        ##                                            torch.tensor([1., -1.])).cuda()
    
        #print(encoded_image.size())
        return encoded_image