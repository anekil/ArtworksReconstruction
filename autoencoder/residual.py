
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResBlock(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim, stride=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, res_h_dim, 3, stride=stride, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(res_h_dim, h_dim, 1, bias=False),
        )
        if stride != 1 or in_dim != h_dim:
            self.shortcut = nn.Conv2d(in_dim, h_dim, 1, stride=stride, bias=False)
        else:
            self.shortcut = None


    def forward(self, input):
        out = self.conv(input)

        if self.shortcut is not None:
          shortcut = self.shortcut(input)
        else:
          shortcut = input
        out += shortcut

        return out


if __name__ == "__main__":
    pass
