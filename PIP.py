import torch.nn as nn
import torch

class PIP(nn.Module):
    def __init__(self, in_channels,n_feat=32):
        super(PIP, self).__init__()
        self.a=nn.Parameter(torch.tensor([0.5]))
        self.b = nn.Parameter(torch.tensor([0.5]))

        self.conv_fus1 = nn.Conv2d(in_channels, n_feat, kernel_size=3, padding=(3 - 1) // 2,bias=False)
        self.conv_fus2 = nn.Conv2d(n_feat, in_channels, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.lrelu=nn.LeakyReLU()

    def forward(self, x,y):
        res=self.conv_fus2(self.lrelu(self.conv_fus1(y-x)))
        return self.a*res+self.b*x
