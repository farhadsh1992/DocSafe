



"""
@--04.02.2025--@
Author: github/farhadsh1992
INFO:
    -ref: 
        
    
LAST_UPDATE:
"""



import torch
import torch.nn as nn
import torch.nn.functional as F

class AffineLoss(nn.Module):
    def __init__(self, device=None):
        super(AffineLoss, self).__init__()

        self.device = device
        self.affine_mae_touter = nn.L1Loss(reduction="mean")  # equivalent to "sum_over_batch_size"

    def forward(self, x_true, x_pred):
        return self.affine_mae_touter(x_true, x_pred)