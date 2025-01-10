import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )


    def forward(self, input):
        out = self.conv(input)
        out += input

        return out



if __name__ == "__main__":
    pass