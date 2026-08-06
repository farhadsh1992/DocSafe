
#################
"""
@--27.09.2023--@
Author: 
INFO:
	> Pytorch version of StampOne_v89
"""
################





from FarhadCV.Tools import tcolors

import  torch
from torch import nn
# from snake.activations import Snake
from .Snake_activation_function import Snake
import torchvision

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

# print(torch.__version__)

################################################################################################
class Stega_Discriminator(nn.Module):
    def __init__(self, args, device:str=None):
        super(Stega_Discriminator, self).__init__()


        self.cuda = device
        modules = [
             torch.nn.Conv2d(in_channels=3,  out_channels=8,   kernel_size=(3, 3), stride=2, bias =  True),
             torch.nn.ReLU(),
             torch.nn.Conv2d(in_channels=8,  out_channels=16,  kernel_size=(3, 3), stride=2, bias =  True),
             torch.nn.ReLU(),
             torch.nn.Conv2d(in_channels=16, out_channels=32,  kernel_size=(3, 3), stride=2, bias =  True),
             torch.nn.ReLU(),
             torch.nn.Conv2d(in_channels=32, out_channels=64,  kernel_size=(3, 3), stride=2, bias =  True),
             torch.nn.ReLU(),
             torch.nn.Conv2d(in_channels=64, out_channels=1,   kernel_size=(3, 3), stride=2, bias =  True)

        ]
        
        self.model = torch.nn.Sequential(*modules).to(self.cuda)
  
    def forward(self, image):
            #x = image - .5
            #image = tf.keras.layers.Input(shape=(400,400, 3))
            x = self.model(image)
            output = torch.mean(x).to(self.cuda)
            #model = tf.keras.models.Model(inputs=[output, x], outputs=[output], name="DiscriminatorEnc")
            return output, x













