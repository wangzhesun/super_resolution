import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=1, bias=False):
        super(ResBlock, self).__init__()
        self.conv_ = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=bias)
        self.act_ = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x
        out = self.conv_(x)
        out = self.act_(out)
        out = self.conv_(out)
        out *= 0.1

        out = torch.add(out, shortcut)
        
        return out
