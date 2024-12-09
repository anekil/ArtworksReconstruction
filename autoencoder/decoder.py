
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
            nn.Conv2d(in_dim, h_dim, 3, padding=1),

            ResBlock(h_dim, h_dim, res_h_dim),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(h_dim, h_dim // 2, 4, stride=2, padding=1),

            ResBlock(h_dim // 2, h_dim // 2, res_h_dim),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(h_dim // 2, h_dim // 4, 4, stride=2, padding=1),

            ResBlock(h_dim // 4, h_dim // 4, res_h_dim // 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(h_dim // 4, h_dim // 8, 4, stride=2, padding=1),

            ResBlock(h_dim // 8, h_dim // 8, res_h_dim // 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(h_dim // 8, h_dim // 16, 4, stride=2, padding=1),

            ResBlock(h_dim // 16, h_dim // 16, res_h_dim // 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(h_dim // 16, out_dim, 4, stride=2, padding=1),
        ]
        
        self.inverse_conv_stack = nn.Sequential(*blocks)


    def forward(self, x):
        return self.inverse_conv_stack(x)


if __name__ == "__main__":
    from torchinfo import summary
    summary(Decoder(), input_size=(4, 8, 8, 8))
