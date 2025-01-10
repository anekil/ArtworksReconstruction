import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from residual import ResBlock
from torchvision import models
    

class Encoder(nn.Module):
    def __init__(self, in_dim=3, h_dim=128, res_h_dim=32, pre_quantizer_dim=32):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 196, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(196, 2, kernel_size=1),
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    from torchinfo import summary

    summary(Encoder(), input_size=(4, 3, 256, 256))