import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from residual import ResBlock


class ChannelReduction(nn.Module):
    def __init__(self, in_channels=128, out_channels=8, h_dim=32):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, h_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(h_dim, h_dim, kernel_size=1),
            
            ResBlock(h_dim, h_dim, h_dim//2, stride=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(h_dim, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.reduce(x)
    

class Encoder(nn.Module):
    def __init__(self, in_dim=3, h_dim=128, res_h_dim=32, pre_quantizer_dim=32):
        super(Encoder, self).__init__()
        kernel = 4
        stride = 2
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 60, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(60, 60, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(60, 120, kernel_size=kernel, stride=stride, padding=1),

            ResBlock(120, 120, 60),
            nn.ReLU(inplace=True),

            nn.Conv2d(120, 240, kernel_size=kernel, stride=stride, padding=1),

            ResBlock(240, 240, 120),
            ResBlock(240, 240, 120),

            nn.ReLU(inplace=True),
            nn.Conv2d(240, 240, kernel_size=1),
            
            ResBlock(240, 240, 120),

            nn.ReLU(inplace=True),
            nn.Conv2d(240, 10, kernel_size=1),
        )

    def forward(self, x):
        return self.conv_stack(x)


if __name__ == "__main__":
    from torchinfo import summary

    summary(Encoder(), input_size=(4, 3, 256, 256))