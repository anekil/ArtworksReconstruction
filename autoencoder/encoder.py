import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from residual import ResBlock


class Encoder(nn.Module):
    def __init__(self, in_dim=3, h_dim=128, res_h_dim=32):
        super(Encoder, self).__init__()
        kernel = 4
        stride = 2
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(h_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1),

            ResBlock(h_dim, h_dim, res_h_dim, 1),

            ResBlock(h_dim, h_dim, res_h_dim, 2),

            ResBlock(h_dim, h_dim, res_h_dim, 1),

            ResBlock(h_dim, h_dim, res_h_dim, 2),

            ResBlock(h_dim, h_dim, res_h_dim, 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_stack(x)


if __name__ == "__main__":
    from torchinfo import summary

    summary(Encoder(), input_size=(4, 3, 256, 256))
