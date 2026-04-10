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


class PixelNorm(nn.Module):
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)


class Generator(nn.Module):
    def __init__(self, z_dim, base_channels=256):
        super().__init__()

        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim, base_channels, 4, 1, 0),
            nn.LeakyReLU(0.2),
            WSConv2d(base_channels, base_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            PixelNorm(),
        )

        # Channel progression (SAFE)
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
        self.to_rgb = nn.ModuleList()

        for i in range(len(self.channels) - 1):
            self.progression.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    WSConv2d(self.channels[i], self.channels[i+1], 3, 1, 1),
                    nn.LeakyReLU(0.2),
                    WSConv2d(self.channels[i+1], self.channels[i+1], 3, 1, 1),
                    nn.LeakyReLU(0.2),
                    PixelNorm(),
                )
            )

        for ch in self.channels:
            self.to_rgb.append(WSConv2d(ch, 3, 1, 1, 0))

    def forward(self, x, step, alpha):
        out = self.initial(x)

        if step == 0:
            return self.to_rgb[0](out)

        for i in range(step):
            prev = out
            out = self.progression[i](out)

        final = self.to_rgb[step](out)
        prev = self.to_rgb[step-1](F.interpolate(prev, scale_factor=2))

        return alpha * final + (1 - alpha) * prev