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
    def __init__(self, in_channel=256, out_channel=256, kernel_size=3, stride=1, padding=1,
                 bias=True):
        super(ResBlock, self).__init__()
        layers = [nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                            padding=padding, bias=bias),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                            padding=padding, bias=bias)]

        self.body = nn.Sequential(*layers)
        self.res_scale = 0.1

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Upsampler(nn.Sequential):
    def __init__(self, scale):
        layers = []
        if scale == 2 or scale == 4:
            layers.append(nn.Conv2d(256, 256 * 4, kernel_size=3, stride=1, padding=1, bias=True))
            layers.append(nn.PixelShuffle(2))
            if scale == 4:
                layers.append(nn.Conv2d(256, 256 * 4, kernel_size=3, stride=1, padding=1,
                                        bias=True))
                layers.append(nn.PixelShuffle(2))
        elif scale == 3:
            layers.append(nn.Conv2d(256, 256 * 9, kernel_size=3, stride=1, padding=1, bias=True))
            layers.append(nn.PixelShuffle(3))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*layers)


class EDSR(nn.Module):
    def __init__(self, scale):
        super(EDSR, self).__init__()

        head_layers = [nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=1, bias=True)]

        body_layers = []
        for _ in range(32):
            body_layers.append(ResBlock())
        body_layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True))

        tail_layers = [Upsampler(scale=scale),
                       nn.Conv2d(256, 3, kernel_size=3, stride=1, padding=1, bias=True)]

        self.sub_mean = MeanShift(sign=-1)
        self.add_mean = MeanShift(sign=1)
        self.head = nn.Sequential(*head_layers)
        self.body = nn.Sequential(*body_layers)
        self.tail = nn.Sequential(*tail_layers)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x
