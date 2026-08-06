



import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class sobel_egdes(nn.Module):
    """
        Sobel Edge Detection Layer for RGB images.
        - Input: (B, 3, H, W)
        - Output: (B, 15, H, W) where each channel undergoes Sobel edge detection
    """
    def __init__(self, 
                 input_shape = 256, 
                 channel_in = 3, 
                 name=""):
        super(sobel_egdes, self).__init__()
        self.input_shape = input_shape

        # Define Sobel filters
        self.sobel_x = nn.Conv2d(in_channels=channel_in, out_channels=channel_in, 
                                 kernel_size=3, stride=1, padding=1, bias=False, groups=channel_in)
        self.sobel_y = nn.Conv2d(in_channels=channel_in, out_channels=channel_in, 
                                 kernel_size=3, stride=1, padding=1, bias=False, groups=channel_in)

        repeat = int(channel_in)
        # Sobel kernels (one per channel)
        sobel_kernel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], 
                                      dtype=torch.float32).repeat(repeat, 1, 1, 1)
        sobel_kernel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], 
                                      dtype=torch.float32).repeat(repeat, 1, 1, 1)

        # Define Sobel filters
        self.sobel_x.weight = nn.Parameter(sobel_kernel_x, requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_kernel_y, requires_grad=False)

    def forward(self, x):
        """
        Forward pass to compute Sobel edges.
        Args:
            x: (B, 3, H, W) input RGB image
        Returns:
            output: (B, 15, H, W) concatenated features
        """
        grad_x = self.sobel_x(x)  # (B, 3, H, W)
        grad_y = self.sobel_y(x)  # (B, 3, H, W)

        # Combine edges
        sobel_edges = grad_x + grad_y  # (B, 3, H, W)

        return sobel_edges

# Example usage:
# model = SobelEdges(input_size=256)
# input_tensor = torch.randn(1, 3, 256, 256)  # Example input (batch of 1, 3-channel image, 256x256)
# output_tensor = model(input_tensor)

# print(output_tensor.shape)  # Should be (1, 15, 256, 256)
