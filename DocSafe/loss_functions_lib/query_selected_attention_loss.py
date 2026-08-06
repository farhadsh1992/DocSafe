"""
@--28.09.2023--@
Author: github/farhadsh1992
INFO:
	- StampOne v89 torch format
    - REF: https://github.com/sapphire497/query-selected-attention/blob/main/models/qs_model.py

    
LAST_UPDATE:
"""


# from DocSafe.loss_functions_lib.query_selected_attention.base_model import BaseModel

from DocSafe.loss_functions_lib.query_selected_attention_tools import networks_global 
from DocSafe.loss_functions_lib.query_selected_attention_tools import networks_local
from DocSafe.loss_functions_lib.query_selected_attention_tools import networks_local_global

# from DocSafe.loss_functions_lib.query_selected_attention_tools.patchnce import PatchNCELoss
# import loss_functions_lib.query_selected_attention_tools.util.util as util


from DocSafe.loss_functions_lib.query_selected_attention_tools.qs_tools import PatchNCELoss
from DocSafe.loss_functions_lib.query_selected_attention_tools.qs_tools import Normalize
from DocSafe.loss_functions_lib.query_selected_attention_tools.qs_tools import  PatchSampleF

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from FarhadCV.Tools import tcolors



import warnings
warnings.filterwarnings('ignore')


######################################################################################################################
#######                                                                      #######
######################################################################################################################
class QSModel(nn.Module):
    """
    REF:
        - https://github.com/sapphire497/query-selected-attention/blob/main/models/qs_model.py
    """
    

    def __init__(self,  args, net="", n_layers="", kind_net=""):
        super(QSModel, self).__init__()

        self.batch_size = args.batch_size
        ##############################################
        ##               PARAMTERS                  ##
        ##############################################
        self.nce_layers = '0,4,8,12,16'
        # QS_mode:str = "" # choices='(global, local, local_global)'
        # input_nc:int  = 
        # output_nc:int =
        # ngf = 
        # netG = 
        # normG = 
        # no_dropout:bool = 
        # init_type:str = 
        # init_gain:int= 0.2
        no_antialias:no_antialias = False
        no_antialias_up:no_antialias = False
        
        # self.gpu_ids = 
        self.device = "cpu"
        self.opt ={
            "QS_mode": "local_global", # choices='(global, local, local_global)'
            "input_nc":3,
            "output_nc":3,
            "ngf":64,
            "netG":"resnet_6blocks", # choices=['resnet_9blocks', 'resnet_6blocks', 'unet_256', 'unet_128', 'stylegan2', 'smallstylegan2', 'resnet_cat']
            "normG":"none", # choices=['instance', 'batch', 'none']
            "no_dropout":True,
            "init_type":'xavier', # choices=['normal', 'xavier', 'kaiming', 'orthogonal']
            "init_gain":0.02,
            "no_antialias":True,
            "no_antialias_up":True,
            "gpu_ids":[],
            "batch_size":self.batch_size,
            "nce_T":0.07,  
            "flip_equivariance":False, 
            "num_patches":64,
            "lambda_NCE":1.0
        }
        self.flipped_for_equivariance = False
        self.nce_layers = [int(i) for i in self.nce_layers.split(',')]
        ##############################################
        ##           SELECT THE NETWORK             ##
        ##############################################
        # choose the network:
        if self.opt['QS_mode'] == 'global':
            from DocSafe.loss_functions_lib.query_selected_attention_tools import networks_global 
            networks = networks_global
        elif self.opt['QS_mode'] == 'local':
            from DocSafe.loss_functions_lib.query_selected_attention_tools import networks_local
            networks = networks_local
        else:
            from DocSafe.loss_functions_lib.query_selected_attention_tools import networks_local_global
            networks = networks_local_global
        ########################################################    
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(self.opt["input_nc"], 
                         self.opt["output_nc"], 
                         self.opt["ngf"], 
                         self.opt["netG"], 
                         self.opt["normG"], 
                         not self.opt["no_dropout"], 
                         self.opt["init_type"], 
                         self.opt["init_gain"], 
                         self.opt["no_antialias"], 
                         self.opt["no_antialias_up"], 
                         self.opt["gpu_ids"], self.opt)
        self.netF = PatchSampleF()
        ############################
        self.criterionNCE = []

        for nce_layer in self.nce_layers:
            self.criterionNCE.append(PatchNCELoss(self.opt).to(self.device))
        # PatchNCELoss(self.opt)
        

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)
        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        
        if self.opt["flip_equivariance"] and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        
        
        feat_k_pool, sample_ids, attn_mats = self.netF(feat_k, self.opt['num_patches'], None, None)
        feat_q_pool, _, _ = self.netF(feat_q, self.opt['num_patches'], sample_ids, attn_mats)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt['lambda_NCE']
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers




######################################################################################################################
#######                                                                      #######
######################################################################################################################


class PatchNCELoss(nn.Module):
    """
    # https://github.com/sapphire497/query-selected-attention/blob/main/models/patchnce.py
    """
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        #self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        self.mask_dtype = torch.bool
        
    def forward(self, feat_q, feat_k):
        
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]  # channel
        feat_k = feat_k.detach()

        # pos logit    
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        # neg logit -- current batch
        # reshape features to batch size
        feat_q = feat_q.view(self.opt["batch_size"], -1, dim)
        feat_k = feat_k.view(self.opt["batch_size"], -1, dim)
        npatches = feat_q.size(1)
        
       
        
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1)) # b*np*np
        
   
        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        
        
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
      
      
        l_neg = l_neg_curbatch.view(-1, npatches)
      
        out = torch.cat((l_pos, l_neg), dim=1) / self.opt["nce_T"]

   


        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))
        return loss

######################################################################################################################
#######                                                                      #######
######################################################################################################################
class Normalize(torch.nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

######################################################################################################################
#######                                                                      #######
######################################################################################################################

class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None, attn_mats=None):
        return_ids = []
        return_feats = []
        return_mats = []
        k_s = 7 # kernel size in unfold
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, C, H, W = feat.shape[0], feat.shape[1], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2) # B*HW*C
            if num_patches > 0:
                if feat_id < 3:
                    if patch_ids is not None:
                        patch_id = patch_ids[feat_id]
                    else:
                        patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)  # random id in [0, HW]
                        patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                    x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
                    attn_qs = torch.zeros(1).to(feat.device)
                else:
                    if attn_mats is not None:
                        attn_qs = attn_mats[feat_id]
                    else:
                        feat_local = F.unfold(feat, kernel_size=k_s, stride=1, padding=3)  # (B, ks*ks*C, L)
                        L = feat_local.shape[2]
                        feat_k_local = feat_local.permute(0, 2, 1).reshape(B, L, k_s*k_s, C).flatten(0, 1) # (B*L, ks*ks, C)
                        feat_q_local = feat_reshape.reshape(B*L, C, 1)
                        dots_local = torch.bmm(feat_k_local, feat_q_local)  # (B*L, ks*ks, 1)
                        attn_local = dots_local.softmax(dim=1)
                        attn_local = attn_local.reshape(B, L, -1)  # (B, L, ks*ks)
                        prob = -torch.log(attn_local)
                        prob = torch.where(torch.isinf(prob), torch.full_like(prob, 0), prob)
                        entropy = torch.sum(torch.mul(attn_local, prob), dim=2)
                        _, index = torch.sort(entropy)
                        patch_id = index[:, :num_patches]
                        feat_q_global = feat_reshape
                        feat_k_global = feat_reshape.permute(0, 2, 1)
                        dots_global = torch.bmm(feat_q_global, feat_k_global)  # (B, HW, HW)
                        attn_global = dots_global.softmax(dim=2)
                        attn_qs = attn_global[torch.arange(B)[:, None], patch_id, :]
                    feat_reshape = torch.bmm(attn_qs, feat_reshape)  # (B, n_p, C)
                    x_sample = feat_reshape.flatten(0, 1)
                    patch_id = []
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            return_mats.append(attn_qs)
            
            
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids, return_mats

######################################################################################################################
#######                                                                      #######
######################################################################################################################





