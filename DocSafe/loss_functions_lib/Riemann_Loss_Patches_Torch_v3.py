


"""
@--23.10.2023--@
Author: github/farhadsh1992
INFO:
	- Rimman Loss
    - REF: 
    https://medium.com/@fernandopalominocobo/demystifying-visual-transformers-with-pytorch-understanding-patch-embeddings-part-1-3-ba380f2aa37f
		

    
LAST_UPDATE:
"""


import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from FarhadCV.Tools import tcolors, bcolors
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms

import numpy as np
################################################
from Network_Libs.Vision_Transformer.Patch_Embedding import  PatchEmbedding


#############################################
import warnings

# tf.get_logger().setLevel("DEBUG")
warnings.filterwarnings('ignore')
# warnings.filterwarnings("ignore", message=".*tf_half_pixel_for_nn.*")
# # import tesnorflow as tf
# tf.autograph.set_verbosity(3)
#################################################################################
#######                                #######
#################################################################################


class Riemann_Patcher_Loss3(nn.Module):
    def __init__(self, 
                batch_size:int=1, 
                image_size:int=16,
                # embed_dim:int=16, 
                patch_size:int=16, 
                # num_patches:int=16, 
                dropout:int=0.001, 
                in_channels:int=3,
                device=""):
        super(Riemann_Patcher_Loss3, self).__init__()

        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.grayscale_to_rgb = transforms.Lambda(lambda x: x.repeat(3, 1, 1) )
        
        self.convv_fun = LambdaLayer(lambda med_x: 
                                     torch.matmul(med_x.transpose(1,2), med_x)/(torch.shape(med_x)[0]*1.0))
        self.dis_fun2 = dis_fun2
        

        self.shape_img_layer       = Reshape_layer((self.batch_size, 3, self.image_size*self.image_size))
        self.shape_generated_layer = Reshape_layer((self.batch_size, 3, self.image_size*self.image_size))
        ##########################################################################
        #####                                              #####
        ##########################################################################
        self.NUM_PATCHES = (image_size // patch_size) ** 2 # 49
        self.EMBED_DIM = (patch_size ** 2) * in_channels # 16
        # DROPOUT = 0.001

        self.embeddings_block = PatchEmbedding(
               embed_dim = self.EMBED_DIM, 
               patch_size=patch_size, 
               num_patches = self.NUM_PATCHES, 
               dropout=dropout, 
               in_channels=in_channels,
               device=device)
        ##########################################################################
        #####                                              #####
        ##########################################################################
        
    def tf_cov(self, x:torch.Tensor)->torch.Tensor:
        """
        Computes covariance matrix of the given tensor.
        Equivalent to the TensorFlow version `tf_cov`.
        """
        mean_x = torch.mean(x)
        med_x = x-mean_x
     
        x2 = med_x.transpose(1, 2)
        # x2 = x2.transpose(1, 2)
        # cov_xx = self.convv_fun(med_x)
        #####################################################
        ## This line printer an array
        # x2 = tf.map_fn(lambda med_x: 
        #                tf.matmul(tf.transpose(med_x), med_x)/tf.cast(tf.shape(med_x)[0], 
        #                 tf.float32), x2.detach().cpu().numpy())
        x2 = torch.stack([
            (med_x.T @ med_x) / med_x.shape[0]  
            for med_x in x2
            ])
        #####################################################
        # x2 = torch.from_numpy(x2.numpy())
        cov_xx = x2.transpose(2, 1)
        # print(tcolors.RED,"x2: ", x2.shape,tcolors.ENDC)
       
        # cov_xx = x2
        return cov_xx
    
    def torch_cov(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes covariance matrix of the given tensor.
        Equivalent to the TensorFlow version `tf_cov`.
        """
        mean_x = torch.mean(x, dim=0, keepdim=True)
        med_x = x - mean_x
        cov_xx = torch.matmul(med_x.T, med_x) / x.shape[0]
        return cov_xx
    
    def distance_riemann(self, AB)->torch.Tensor:
        A, B = AB
        # A = normalize_fixed(A, current_range = [-1,1], normed_range=[0,1])
        # B = normalize_fixed(B, current_range = [-1,1], normed_range=[0,1])
        A = self.tf_cov(A*255)
        B = self.tf_cov(B*255) 
        # A = self.torch_cov(A*255)
        # B = self.torch_cov(B*255) 

        B, A = torch.abs(B +1e-12), torch.abs(A +1e-12)
        
        c = A*torch.reciprocal(B) 
 
        # dist = torch.real(torch.trace((torch.log(c[0])**2))) 
        dist = self.dis_fun2(c)
        return dist


    def forward(self, 
                original_image:  torch.Tensor, 
                generated_image: torch.Tensor, 
                fake_output_disc:torch.Tensor = None
                )->torch.Tensor:
        # if original_image.shape[3] == 1:
        if False:
            ## Gray image to RGB image (real and fake)
            img = grayscale_to_rgb(original_image)
            generated = grayscale_to_rgb(generated_image)
        else:
            img = original_image
            generated = generated_image
            
        img = self.embeddings_block(img)
        generated = self.embeddings_block(generated)
        B,patch_num, w = img.shape 
        loss_op = 0
        for i in range(self.batch_size):
            # print(tcolors.RED,"img",img.shape,tcolors.ENDC)
            # print(tcolors.RED,"generated", generated.shape,tcolors.ENDC)
            
            ## reshape real and fake from (B, w, h, 3) into (B, w*h, 3)
            ## img =  self.reshape_fn(img)
            # img = img.view(self.batch_size*patch_num, 1, w)#(self.patch_size ** 2)
            # ##img = self.shape_img_layer(img)
            ## generated = self.reshape_fn(generated)
            # generated = generated.view(self.batch_size*patch_num, 1, w)

            # generated = self.shape_generated_layer(generated)

            img1 = img[0].view(patch_num, 1, w)#(self.patch_size ** 2)
            generated1 = generated[0].view(patch_num, 1, w)
            ## compute distance in riemann space
            loss = self.distance_riemann((img1, generated1))
        

            ## Normalize the loss value
            loss_op += loss/((patch_num)*1.0)
            # loss_op = torch.mean(loss_op)
        loss_op2 = loss_op/((self.batch_size)*1.0)

        ## (????)
        # loss_c = torch.nn.functional.binary_cross_entropy_with_logits(
        #                                     tf.ones_like(fake_output_disc), 
        #                                     fake_output_disc)
        out = torch.mean(loss_op2) #/5000 + loss_c ## (????)
        
        return out 










#################################################################################
#######                                #######
#################################################################################

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)
########################################################
class Reshape_layer(nn.Module):
    def __init__(self, shape):
        super(Reshape_layer, self).__init__()
        self.shape = shape

    def forward(self, x):
        # return x.view(self.shape)
        return torch.reshape(x, self.shape)
########################################################
def dis_fun2(ints_batch):
    batch = ints_batch.shape[0]
    out_batch = []
    for i in range(batch):
        x = torch.real(torch.trace((torch.log(ints_batch[i])**2)))
        out_batch.append(x)
    ##########################################
    # out_batch = np.array(out_batch)
    # out_batch = torch.from_numpy(out_batch)
    out_batch = torch.stack(out_batch)
    ##########################################
    return out_batch

###################################################################################################
def normalize_fixed(x, current_range, normed_range):
    # current_min, current_max = tf.expand_dims(current_range[:, 0], 1), tf.expand_dims(current_range[:, 1], 1)
    current_min, current_max = current_range[0], current_range[1]

    # normed_min, normed_max = tf.expand_dims(normed_range[:, 0], 1), tf.expand_dims(normed_range[:, 1], 1)
    normed_min, normed_max = normed_range[0], normed_range[1]

    x_normed = (x - current_min) / (current_max - current_min)
    x_normed = x_normed * (normed_max - normed_min) + normed_min
    return x_normed
###################################################################################################  