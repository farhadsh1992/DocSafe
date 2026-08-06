











import torch
import torch.nn as nn
import torch.nn.functional as F



class StegaDiscriminator(nn.Module):
    """
  
    """
    def __init__(self, device=None):
        super(StegaDiscriminator, self).__init__()
        
        self.device = device
        self.model = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=3, stride=2),
                nn.LeakyReLU(),
                nn.Conv2d(8, 16, kernel_size=3, stride=2),
                nn.LeakyReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2),
                nn.LeakyReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2),
                nn.LeakyReLU(),
                nn.Conv2d(64, 1, kernel_size=3),  # No activation needed if None
                nn.Sigmoid(),
            ).to(device)
        
        # self.logits = nn.LazyLinear(1).to(device)
    def forward(self, image):
        #image = torch.clamp(image, 0, 1)
        # x = image.clone() - 0.5  # Avoid in-place 
        # x = image.clone()
        x = self.model(image)
      

        output = torch.mean(x).to(self.device)
        

        # D_loss = torch.add(D_output_real, -1*D_output_fake)
        
        # output = nn.Flatten()(x).to(self.device)
        # output = self.logits(output).to(self.device)
        # output = output.squeeze().to(self.device)
        
        return output, x