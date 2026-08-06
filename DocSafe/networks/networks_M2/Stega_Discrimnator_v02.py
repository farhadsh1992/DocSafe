







"""
@--12.04.2025--@
Author: github/farhadsh1992
INFO:
	- Rimman Loss
    - REF: 
    
		

    
LAST_UPDATE:
"""



import torch
import torch.nn as nn
import torch.nn.functional as F



class StegaDiscriminator(nn.Module):
    def __init__(self, device):
        super(StegaDiscriminator, self).__init__()
        # initializer = tf.random_normal_initializer(0.0, 0.02)
    
        # inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
        # tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

        self.device = device
    
        self.down1  = encoder_block(in_channels=3, out_channels=64, 
                                    kernel_size=4, 
                                    apply_batchnorm=False, 
                                    device=device).to(device) # (bs, 64, 128, 128 )
        self.down2  = encoder_block(in_channels=64, out_channels=128, 
                                    kernel_size=4,device=device).to(device) # (bs, 128, 64, 64)
        self.down3  = encoder_block(in_channels=128, out_channels=256, 
                                    kernel_size=4, device=device).to(device) # (bs, 256, 32, 32)

        
        self.zero_pad1 = torch.nn.ZeroPad2d(padding=1).to(device) # (bs, 512, 34, 34) #???
    
        self.conv =  torch.nn.Conv2d(in_channels=256, 
                                     out_channels=512, 
                                     kernel_size=4, 
                                     stride=1, 
                                     bias =  False).to(device) # (bs, 512, 31, 31)
        # self.conv.weight = torch.nn.init.uniform_(self.conv.weight, a=0.0, b=0.02)
        self.batchnorm1  = torch.nn.BatchNorm2d(num_features=512).to(device)
        self.leaky_relu  = torch.nn.LeakyReLU().to(device)
        
        self.zero_pad2 = self.zero_pad1 = torch.nn.ZeroPad2d(padding=1).to(device)
    
  
        self.last_layer =  torch.nn.Conv2d(in_channels=512,
                                            out_channels=1, 
                                            kernel_size=4, 
                                            stride=1, bias = False).to(device) # (bs, 512, 31, 31)
        # self.last_layer.weight = torch.nn.init.uniform_(self.last_layer.weight, a=0.0, b=0.02)
        
    def forward(self,inputs):
        # x = torch.cat((inputs, target), dim=1)  # (B, channels*2, 256, 256)
        
        x = self.down1(inputs)
        x = self.down2(x)
        x = self.down3(x)
        x = self.zero_pad1(x)
        x = self.conv(x)
        x = self.batchnorm1(x)
        x = self.leaky_relu(x)
        x = self.zero_pad2(x)
        
        out = self.last_layer(x)
        return out, x
    


def encoder_block(in_channels:int, out_channels:int, kernel_size:int=4, apply_batchnorm= True, device=None):

    modules = []

    conv2D = torch.nn.Conv2d(in_channels=in_channels, 
                             out_channels=out_channels, 
                             kernel_size=(kernel_size, kernel_size), 
                             stride=(2,2), padding=1, bias=False).to(device)
    conv2D.weight = torch.nn.init.uniform_(conv2D.weight, a=0.0, b=0.02).to(device)
    modules.append(conv2D)
    if apply_batchnorm:
        modules.append(torch.nn.BatchNorm2d(num_features=out_channels).to(device))
    modules.append(torch.nn.LeakyReLU().to(device))
    model = torch.nn.Sequential(*modules).to(device)

    return model