import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from residual import ResBlock


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 192, kernel_size=3, stride=1, padding=1),
            ResBlock(192, 128),
            ResBlock(192, 128),

            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(192, 192, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(192, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)



if __name__ == "__main__":
    from torchinfo import summary
    summary(Decoder(), input_size=(4, 2, 32, 32))