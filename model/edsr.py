import torch
from torch import nn


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range=255, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0),
                 sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ResBlock(nn.Module):
    def __init__(self, in_channel=255, out_channel=266, kernel_size=3, stride=1, padding=1,
                 bias=False):
        super(ResBlock, self).__init__()
        self.conv_ = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=bias)
        self.act_ = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv_(x)
        out = self.act_(out)
        out = self.conv_(out)
        out *= 0.1

        out = torch.add(out, residual)

        return out


class EDSR(nn.Module):
    def __init__(self, scale):
        super(EDSR, self).__init__()
        self.scale_ = scale
        self.sub_mean_ = MeanShift(sign=-1)
        self.add_mean_ = MeanShift(sign=1)
        self.input_conv_ = nn.Conv2d(3, 255, kernel_size=3, stride=1, padding=1, bias=False)

        self.res_block_ = [ResBlock() for i in range(32)]
        self.res_block_ = nn.Sequential(*self.res_block_)

        self.up_sampler_ = []
        if self.scale_ == 2 or self.scale_ == 4:
            self.up_sampler_.append(nn.Conv2d(255, 255 * 4, kernel_size=3, stride=1, padding=1,
                                              bias=False))
            self.up_sampler_.append(nn.PixelShuffle(2))
            if self.scale_ == 4:
                self.up_sampler_.append(nn.Conv2d(255, 255 * 4, kernel_size=3, stride=1, padding=1,
                                                  bias=False))
                self.up_sampler_.append(nn.PixelShuffle(2))
        elif self.scale_ == 3:
            self.up_sampler_.append(nn.Conv2d(255, 255 * 9, kernel_size=3, stride=1, padding=1,
                                              bias=False))
            self.up_sampler_.append(nn.PixelShuffle(3))
        else:
            raise NotImplementedError
        self.up_sampler_.append(nn.Conv2d(255, 3, kernel_size=3, stride=1, padding=1, bias=False))
        self.up_sampler_ = nn.Sequential(*self.up_sampler_)

    def forward(self, x):
        out = self.sub_mean_(x)
        out = self.input_conv_(out)
        residual = out
        out = self.res_block_(out)
        out = torch.add(out, residual)
        out = self.up_sampler_(out)
        out = self.add_mean_(out)
