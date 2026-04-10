import torch
import torch.nn as nn
import torch.nn.functional as F

class WSConv2d(nn.Module):
    def __init__(self, in_c, out_c, k, s, p):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p)
        self.scale = (2 / (in_c * k * k)) ** 0.5

    def forward(self, x):
        return self.conv(x * self.scale)


class Discriminator(nn.Module):
    def __init__(self, base_channels=256):
        super().__init__()

        self.channels = [
            base_channels,
            base_channels,
            base_channels,
            base_channels // 2,
            base_channels // 4,
            base_channels // 8,
            base_channels // 16,
        ]

        self.progression = nn.ModuleList()
        self.from_rgb = nn.ModuleList()

        for ch in self.channels:
            self.from_rgb.append(WSConv2d(3, ch, 1, 1, 0))

        for i in range(len(self.channels) - 1):
            self.progression.append(
                nn.Sequential(
                    WSConv2d(self.channels[i+1], self.channels[i], 3, 1, 1),
                    nn.LeakyReLU(0.2),
                    nn.AvgPool2d(2),
                )
            )

        self.final = nn.Sequential(
            WSConv2d(base_channels, base_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(base_channels * 4 * 4, 1),
        )

    def forward(self, x, step, alpha):
        out = self.from_rgb[step](x)

        if step == 0:
            return self.final(out).view(-1)

        downscaled = F.avg_pool2d(x, 2)
        prev = self.from_rgb[step-1](downscaled)

        out = self.progression[step-1](out)
        out = alpha * out + (1 - alpha) * prev

        for i in reversed(range(step-1)):
            out = self.progression[i](out)

        return self.final(out).view(-1)