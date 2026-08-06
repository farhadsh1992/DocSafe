"""
@--29.07.2024--@
AUTHOR: github/farhadsh1992
INFO:
LAST_UPDATE:
    - https://pypi.org/project/torch-snake/
    - pip install torch-snake
    - https://github.com/EdwardDixon/snake
    - https://github.com/EdwardDixon/snake/blob/master/snake/activations.py
"""




from FarhadCV.Tools import tcolors, bcolors
import torch
import torch.nn as nn
import torch.nn.functional as F



# def snake_(X, beta):
#     return X + (1 / beta) * torch.square(torch.sin(beta * X))

# class Snake(nn.Module):
#     """
#     Snake activation function: X + (1/b) * sin^2(b * X). Proposed to learn periodic targets.

#     Y = Snake(beta=0.5, trainable=False)(X)

#     ----------
#     Ziyin, L., Hartwig, T. and Ueda, M., 2020. Neural networks fail to learn periodic functions
#     and how to fix it. arXiv preprint arXiv:2006.08195.
#     """
#     def __init__(self, beta=0.25, trainable=True):
#         super(Snake, self).__init__()
#         self.beta = beta
#         self.trainable = trainable
        
#         if self.trainable:
#             self.beta_factor = nn.Parameter(torch.tensor(beta))
#         else:
#             self.register_buffer('beta_factor', torch.tensor(beta))

#     def forward(self, inputs):
#         # Convert from BCHW to BHWC
#         # inputs = inputs.permute(0, 2, 3, 1)
#          # Apply the Snake activation function
#         outputs = snake_(inputs, self.beta_factor)
        
#         # Convert back from BHWC to BCHW
#         # outputs = outputs.permute(0, 3, 1, 2)
#         return outputs
    
#     def extra_repr(self):
#         return f'beta={self.beta_factor.item()}, trainable={self.trainable}'
    





# class SnakeLinear(nn.Module):
#     """
#     Snake activation function: X + (1/b) * sin^2(b * X). Proposed to learn periodic targets.

#     Y = Snake(beta=0.5, trainable=False)(X)

#     ----------
#     Ziyin, L., Hartwig, T. and Ueda, M., 2020. Neural networks fail to learn periodic functions
#     and how to fix it. arXiv preprint arXiv:2006.08195.
#     """
#     def __init__(self, beta=0.25, trainable=True):
#         super(SnakeLinear, self).__init__()
#         self.beta = beta
#         self.trainable = trainable
        
#         if self.trainable:
#             self.beta_factor = nn.Parameter(torch.tensor(beta))
#         else:
#             self.register_buffer('beta_factor', torch.tensor(beta))

#     def forward(self, inputs):
#         # Convert from BCHW to BHWC
#         # inputs = inputs.permute(0, 2, 1)
#          # Apply the Snake activation function
#         outputs = snake_(inputs, self.beta_factor)
        
#         # Convert back from BHWC to BCHW
#         # outputs = outputs.permute(0, 2, 1)
#         return outputs
    
#     def extra_repr(self):
#         return f'beta={self.beta_factor.item()}, trainable={self.trainable}'
    

class SnakeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, a):
        ctx.save_for_backward(input, a)
        return input + (1.0 / a) * torch.sin(a * input) ** 2

    @staticmethod
    def backward(ctx, grad_output):
        input, a = ctx.saved_tensors
        sin = torch.sin(a * input)
        cos = torch.cos(a * input)
        grad_input = grad_output * (1 + (2 / a) * sin * cos)
        grad_a = grad_output * (- (1 / (a * a)) * sin * sin * input + (2 / a) * sin * cos * input)
        return grad_input, grad_a.sum()

class Snake(nn.Module):
    def __init__(self, beta=0.5, trainable=True):
        super(Snake, self).__init__()
        self.a = nn.Parameter(torch.tensor(beta))

    def forward(self, input):
        return SnakeFunction.apply(input, self.a)
