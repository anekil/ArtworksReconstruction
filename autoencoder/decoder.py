import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from residual import ResBlock


class Decoder(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi 
    maps back to the original space z -> x.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim=8, out_dim=3, h_dim=128, res_h_dim=32):
        super(Decoder, self).__init__()

        blocks = [
            nn.Conv2d(10, 240, 3, padding=1),

            nn.ReLU(inplace=True),
            nn.Conv2d(240, 240, 1),

            ResBlock(240, 240, 120),

            nn.ReLU(inplace=True),
            nn.Conv2d(240, 240, 1),

            ResBlock(240, 240, 120),
            ResBlock(240, 240, 120),

            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(240, 120, kernel_size=4, stride=2, padding=1),
            ResBlock(120, 120, 60),
            nn.ReLU(inplace=True),
            nn.Conv2d(120, 120, 1),

            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(120, 120, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(120, 120, 1),
            ResBlock(120, 120, 60),
            nn.ReLU(inplace=True),
            nn.Conv2d(120, 120, 3, padding=2, dilation=2),

            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(120, 90, kernel_size=4, stride=2, padding=1),
            ResBlock(90, 90, 60),
            nn.ReLU(inplace=True),
            nn.Conv2d(90, 60, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(60, 60, 3, padding=1),

            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(60, 60, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(60, 30, 1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(30, out_dim, 3, padding=1),
        ]
        
        self.inverse_conv_stack = nn.Sequential(*blocks)


    def forward(self, x):
        return self.inverse_conv_stack(x)


if __name__ == "__main__":
    from torchinfo import summary
    summary(Decoder(), input_size=(4, 10, 16, 16))