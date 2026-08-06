import torch
import torch.nn as nn

from DocSafe.loss_functions_lib.lpips.networks import get_network, LinLayers
from DocSafe.loss_functions_lib.lpips.utils import get_state_dict
from FarhadCV.Tools import tcolors, bcolors, estimator, read_files
import torch.nn.functional as F
###############################################################################
####    LPIPS                    ####
###############################################################################

class LPIPS(nn.Module):
    r"""Creates a criterion that measures
    Learned Perceptual Image Patch Similarity (LPIPS).
    Arguments:
        net_type (str): the network type to compare the features:
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    def __init__(self, input_size, net_type: str = 'alex', version: str = '0.1', device=""):

        assert version in ['0.1'], 'v0.1 is only supported now'

        super(LPIPS, self).__init__()

        self.device = device
        self.size = (input_size, input_size)
        # pretrained network
        self.net = get_network(net_type).to(device)

        # linear layers
        self.lin = LinLayers(self.net.n_channels_list).to(device)
        self.lin.load_state_dict(get_state_dict(net_type, version))

    def forward(self, x: torch.Tensor, y: torch.Tensor):

        # print(tcolors.RED,"(LPIPS) x", x.shape,tcolors.ENDC)
        # print(tcolors.RED,"(LPIPS) y", y.shape,tcolors.ENDC)
        x = F.interpolate(x, 
                    size=self.size, 
                    mode="nearest")
        y = F.interpolate(y, 
                    size=self.size, 
                    mode="nearest")

        feat_x, feat_y = self.net(x), self.net(y)

        diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
        res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]

        return torch.sum(torch.cat(res, 0)) / x.shape[0]
