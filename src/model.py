"""Compact RGB-guided Residual U-Net for edge-preserving depth refinement."""
import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        ) if in_c != out_c else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))

class DepthRefineNet(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.e1 = ResidualBlock(5, width)
        self.p1 = nn.MaxPool2d(2)
        self.e2 = ResidualBlock(width, width * 2)
        self.p2 = nn.MaxPool2d(2)
        self.mid = ResidualBlock(width * 2, width * 4)
        
        self.u2 = nn.ConvTranspose2d(width * 4, width * 2, 2, stride=2)
        self.d2 = ResidualBlock(width * 4, width * 2)
        self.u1 = nn.ConvTranspose2d(width * 2, width, 2, stride=2)
        self.d1 = ResidualBlock(width * 2, width)
        
        self.out = nn.Conv2d(width, 1, 3, padding=1)
        # Initialize output layer to zero for pure residual learning from input depth
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        # x: [B, 5, H, W] -> RGB (0:3), Coarse Depth (3:4), Mask (4:5)
        coarse_depth = x[:, 3:4]
        a = self.e1(x)
        b = self.e2(self.p1(a))
        m = self.mid(self.p2(b))
        z = self.d2(torch.cat([self.u2(m), b], dim=1))
        z = self.d1(torch.cat([self.u1(z), a], dim=1))
        residual = self.out(z)
        # Residual correction bounded to valid normalized depth [0, 1]
        return torch.clamp(coarse_depth + residual, 0.0, 1.0)

