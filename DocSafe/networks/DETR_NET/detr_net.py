
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch

from .backbone import Backbone, Joiner
from .detr import DETR
from .detr import PostProcess
from .position_encoding import PositionEmbeddingSine
from .segmentation import DETRsegm
# from .segmentation import PostProcessPanoptic
from .transformer import Transformer


######################################################################################
######################################################################################
def _make_detr(backbone_name: str, 
               dilation:bool   = False, 
               num_classes:int = 2, 
               num_queries:int = 1, 
               mask:bool       = False,
               device:str      = None):
    hidden_dim = 256
    backbone = Backbone(backbone_name, train_backbone=True, return_interm_layers=mask, dilation=dilation)
    pos_enc = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
    backbone_with_pos_enc = Joiner(backbone, pos_enc)
    backbone_with_pos_enc.num_channels = backbone.num_channels
    transformer = Transformer(d_model=hidden_dim, return_intermediate_dec=True)
    detr = DETR(backbone_with_pos_enc, transformer, num_classes=num_classes, num_queries=num_queries)
    if mask:
        return DETRsegm(detr)
    return detr

######################################################################################
######################################################################################
def detr_resnet50(pretrained  = False, 
                  num_classes = 91, 
                  num_queries = 1,
                  return_postprocessor = False,
                  device=None):
    """
    DETR R50 with 6 encoder and 6 decoder layers.

    Achieves 42/62.4 AP/AP50 on COCO val5k.
    """
    model = _make_detr(backbone_name = "resnet50",
                       dilation      = False, 
                       num_classes   = num_classes, 
                       num_queries   = num_queries,
                       device = device)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth", 
            map_location="cpu", 
            check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model