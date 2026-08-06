



from FarhadCV.Tools import tcolors, bcolors
import torch
import torch.nn as nn





#####################################################
#####                              #####
#####################################################
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, filters, size, apply_batchnorm=True):
        super(EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=filters, 
                      kernel_size=size, 
                      stride=2, 
                      padding=1, bias=False)
        ]
        
        if apply_batchnorm:
            layers.append(nn.BatchNorm2d(filters))
        
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
#####################################################
#####                              #####
#####################################################
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, filters, size, apply_dropout=False):
        super(DecoderBlock, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, 
                               filters, 
                               kernel_size=size, 
                               stride=2, 
                               padding=1, 
                               output_padding=0, bias=False),
           
            nn.BatchNorm2d(filters),
            # nn.ReLU(inplace=True),
            nn.SiLU(inplace=True),
            # nn.LeakyReLU(inplace=True)

        ]
        if apply_dropout:
            layers.insert(-1, nn.Dropout(0.5))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    

#####################################################
#####                              #####
#####################################################
class WeightWavelet(nn.Module):
    def __init__(self, shape, channel_in=15, channel_out=32, name=""):
        super(WeightWavelet, self).__init__()
        self.name = name
        self.separable_conv = nn.Sequential(
            nn.Conv2d(in_channels=channel_in, 
                      out_channels=channel_in, 
                      kernel_size=1, stride=1, padding='same', 
                      groups=channel_in, bias=False),
            nn.Conv2d(in_channels = channel_in,
                      out_channels = channel_out, kernel_size=1, stride=1, padding='same', bias=False)
        )

    def forward(self, x):
        return self.separable_conv(x)

#####################################################
#####                              #####
#####################################################
class GatingSignal(nn.Module):
    """
    Resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :return: the gating feature map with the same dimension of the up layer feature map
    """
    def __init__(self, in_channels, out_size, batch_norm=False):
        super(GatingSignal, self).__init__()
        self.conv2d_layer = nn.Conv2d(in_channels  = in_channels, 
                                      out_channels = out_size, 
                                      kernel_size  = (1, 1), 
                                      padding      = 'same', 
                                      bias         = not batch_norm)
        self.batch_norm = batch_norm
        if batch_norm:
            self.batch_layer = nn.BatchNorm2d(out_size)
        # self.act_layer = nn.ReLU(inplace=True)
        self.act_layer = nn.SiLU(inplace=True)

    def forward(self, inputs):
        x = self.conv2d_layer(inputs)
        if self.batch_norm:
            x = self.batch_layer(x)
        x = self.act_layer(x)
        return x
    
#####################################################
#####                              #####
#####################################################
class AttentionBlock(nn.Module):
    def __init__(self, inter_shape, out_shape, number):
        super(AttentionBlock, self).__init__()
        self.inter_shape = inter_shape
        self.number = number
        shape_x  = [32, 64, 128, 256]
        shape_x3 = [512, 256, 128, 64, 32, 3]
        shape_g  = [16, 32,  64, 128]
        shape_theta_x = [16, 32, 64, 128]
        shape_sigmoid = [16, 32, 64, 128]

        # Getting the x signal to the same shape as the gating signal
        self.theta_x_layer = nn.Conv2d(inter_shape, inter_shape, 
                                       kernel_size=(2, 2), stride=(2, 2), padding=0, bias=True)

        # Getting the gating signal to the same number of filters as the inter_shape
        self.phi_g_layer = nn.Conv2d(inter_shape, inter_shape, kernel_size=(1, 1), padding=0)
        self.upsample_g_layer = nn.ConvTranspose2d(inter_shape, inter_shape, kernel_size=(3, 3),
                                                   stride=(shape_theta_x[number] // shape_g[number],
                                                           shape_theta_x[number] // shape_g[number]),
                                                   padding=1, bias=True)
        self.concat_xg_layer = nn.Sequential()
        self.act_xg_layer = nn.ReLU()
        self.psi_layer = nn.Conv2d(inter_shape, 1, kernel_size=(1, 1), padding=0, bias=True)
        self.sigmoid_xg_layer = nn.Sigmoid()
        self.upsample_psi_layer1 = nn.Upsample(scale_factor=(shape_x[number] // shape_sigmoid[number],
                                                             shape_x[number] // shape_sigmoid[number]),
                                               mode='bilinear', align_corners=True)

        self.upsample_psi_layer2 = lambda x: x.repeat_interleave(shape_x3[number], dim=1)
        self.multiply_layer = lambda x, y: x * y

        self.result_layer = nn.Conv2d(inter_shape, 
                                      out_shape, kernel_size=(1, 1), padding=0, bias=True)
        self.result_bn_layer = nn.BatchNorm2d(out_shape)

    def forward(self, x, gating):

        # Getting the x signal to the same shape as the gating signal
        theta_x = self.theta_x_layer(x)
        # Getting the gating signal to the same number of filters as the inter_shape
        phi_g = self.phi_g_layer(gating)
        upsample_g = self.upsample_g_layer(phi_g)

        concat_xg = torch.add(upsample_g, theta_x)
        # concat_xg = torch.mul(upsample_g, theta_x)
        act_xg = self.act_xg_layer(concat_xg)
      
        psi = self.psi_layer(act_xg)
 
        sigmoid_xg = self.sigmoid_xg_layer(psi)

        upsample_psi = self.upsample_psi_layer1(sigmoid_xg)
        upsample_psi = self.upsample_psi_layer2(upsample_psi)

       
        y = self.multiply_layer(upsample_psi, x)

        result = self.result_layer(y)
        result_bn = self.result_bn_layer(result)
        return result_bn    
#####################################################
#####                              #####
#####################################################
class Conv1DLayer1(nn.Module):
    def __init__(self):
        super(Conv1DLayer1, self).__init__()
        self.flatten    = nn.Flatten()
        self.conv1d     = nn.Conv1d(1, 2, kernel_size = 1, stride = 1, padding = 'same', bias = True)
        # self.activation = nn.ReLU()
        # self.activation = nn.GELU()
        self.activation = nn.SiLU(inplace=True)
        self.reshape    = nn.Unflatten(1, (64, 256, 256))

    def forward(self, x):
        x = self.flatten(x)
        x = torch.unsqueeze(x, dim=1)
        x = self.conv1d(x)
        x = self.activation(x)
        x = x.view(-1, 64, 256, 256)
        return x  

#####################################################
#####                              #####
#####################################################
class Conv1DLayer2(nn.Module):
    def __init__(self):
        super(Conv1DLayer2, self).__init__()
        self.flatten    = nn.Flatten()
        self.conv1d     = nn.Conv1d(1, 2, kernel_size = 1, stride = 1, padding = 'same', bias = True)
        # self.activation = nn.GELU()
        self.activation = nn.SiLU(inplace=True)
        self.reshape    = nn.Unflatten(1, (128, 16, 16))

    def forward(self, x):
        x = self.flatten(x)
        x = torch.unsqueeze(x, dim=1)
        x = self.conv1d(x)
        x = self.activation(x)
        x = x.view(-1, 128, 16, 16)
        return x


#####################################################
#####                              #####
#####################################################
class DecoderBlock2(nn.Module):
    def __init__(self, filters_in,filters_out,  size, activation, apply_dropout=False, name=""):
        super(DecoderBlock2, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels    = filters_in, 
                               out_channels   = filters_out, 
                               kernel_size    = size, 
                               stride         = 2, 
                               padding        = 0, 
                               output_padding = 1,  
                               bias           = False),
            activation,
            nn.BatchNorm2d(filters_out),
            
        ]
        if apply_dropout:
            layers.append(nn.Dropout(0.5))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

#####################################################
#####                              #####
#####################################################