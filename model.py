from sSR import HSCNN_plus as sSRH
from PIP import PIP as PIP
import torch.nn as nn
import torch
class sSRHelp(nn.Module):
    def __init__(self, channel_msi,channel_pan):
        super(sSRHelp, self).__init__()
        self.sD = sSRH(in_channel=channel_msi, out_channel=channel_pan)
        self.sSR = sSRH(in_channel=channel_pan, out_channel=channel_msi)
        self.PanFN = PIP(in_channels=channel_pan)
        self.MSFN = PIP(in_channels=channel_msi)

    def forward(self, pan,IniFus):
        Fused_sD=self.sD(IniFus)
        Panfused=self.PanFN(Fused_sD,pan)
        Fused_sSR=self.sSR(Panfused)
        output = self.MSFN(IniFus, Fused_sSR)
        Spa_down=nn.functional.interpolate((output+IniFus)/2.,scale_factor=0.25,mode='bicubic')
        return (output+IniFus)/2.,Spa_down