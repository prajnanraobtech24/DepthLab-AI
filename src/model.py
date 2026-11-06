"""Compact RGB-guided U-Net for depth refinement."""
import torch
from torch import nn

class Block(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(a,b,3,padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                                 nn.Conv2d(b,b,3,padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True))
    def forward(self,x): return self.net(x)

class DepthRefineNet(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.e1=Block(5,width); self.p1=nn.MaxPool2d(2)
        self.e2=Block(width,width*2); self.p2=nn.MaxPool2d(2)
        self.mid=Block(width*2,width*4)
        self.u2=nn.ConvTranspose2d(width*4,width*2,2,2); self.d2=Block(width*4,width*2)
        self.u1=nn.ConvTranspose2d(width*2,width,2,2); self.d1=Block(width*2,width)
        self.out=nn.Conv2d(width,1,1); self.sig=nn.Sigmoid()
    def forward(self,x):
        a=self.e1(x); b=self.e2(self.p1(a)); m=self.mid(self.p2(b))
        z=self.d2(torch.cat([self.u2(m),b],1)); z=self.d1(torch.cat([self.u1(z),a],1))
        return self.sig(self.out(z))
