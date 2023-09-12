import torch
import torch.nn as nn
from math import sqrt
from math import pi

import torch
import numpy as np
import torch.nn.functional as F

##    CanNet
class CanNet(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(CanNet, self).__init__()
        self.conv1 = self.convlayer(in_channel, 128, 5)
        self.conv_residual = self.convlayer(in_channel, out_channel, 7)
        self.conv2 = self.convlayer(128, 32, 1)
        self.resB1=Resblock(32)
        self.resB2=Resblock(32)
        self.conv5 = self.convlayer(64, 128, 1)
        self.conv6 = self.convlayer(128, out_channel, 5)


    def convlayer(self, in_channels, out_channels, kernel_size):
        pader = nn.ReplicationPad2d(int((kernel_size - 1) / 2))
        conver = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)
        layers = filter(lambda x: x is not None, [pader, conver])
        return nn.Sequential(*layers)

    def forward(self, x):
        out1=self.conv1(x)
        out2=self.conv2(out1)
        out3=self.resB1(out2)
        out4=self.resB2(out3)
        out4=torch.cat([out4,out2],1)
        out5=self.conv5(out4)
        out6=self.conv6(out5)
        residual=self.conv_residual(x)
        output=torch.add(out6,residual)
        return output

class Resblock(nn.Module):
    def __init__(self, in_channels):
        super(Resblock, self).__init__()
        self.conv1 = self.convlayer(in_channels, 32, 3)
        self.conv2 = self.convlayer(in_channels, 32, 3)
        self.prelu=nn.PReLU()

    def convlayer(self, in_channels, out_channels, kernel_size):
        pader = nn.ReplicationPad2d(int((kernel_size - 1) / 2))
        conver = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)
        layers = filter(lambda x: x is not None, [pader, conver])
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.prelu(self.conv1(x))
        residual=self.conv2(out1)
        out2=torch.add(x,residual)
        output=self.prelu(out2)
        return output


##    HSRnet
class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes, 1, bias=False)
        self.gelu1 =nn.LeakyReLU()
        self.fc2 = nn.Conv2d(in_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.gelu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.gelu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class HSRnet(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(HSRnet, self).__init__()
        self.conv_B = self.convlayer(in_channel, out_channel, 3)
        self.conv_G = self.convlayer(in_channel, out_channel, 3)
        self.conv_R = self.convlayer(in_channel, out_channel, 3)
        self.convinputb = self.convlayer(out_channel, out_channel, 1)
        self.convinputg = self.convlayer(out_channel, out_channel, 1)
        self.convinputr = self.convlayer(out_channel, out_channel, 1)
        self.fusion=self.convlayer(3*out_channel, out_channel, 1)
        self.conv1 = self.convlayer(out_channel, out_channel, 3)
        self.e1 = ChannelAttention(out_channel)
        self.HSI1 = HSI_block(out_channel, out_channel)
        self.en1 = ChannelAttention(out_channel)
        self.conv2 = self.convlayer(out_channel, out_channel, 3)
        self.e2 = ChannelAttention(out_channel)
        self.HSI2 = HSI_block(out_channel, out_channel)
        self.en2 = ChannelAttention(out_channel)
        self.conv3 = self.convlayer(out_channel, out_channel, 3)
        self.e3 = ChannelAttention(out_channel)
        self.HSI3 = HSI_block(out_channel, out_channel)
        self.en3 = ChannelAttention(out_channel)
        self.conv4 = self.convlayer(out_channel, out_channel, 3)
        self.e4 = ChannelAttention(out_channel)
        self.HSI4 = HSI_block(out_channel, out_channel)
        self.en4 = ChannelAttention(out_channel)
        self.conv5 = self.convlayer(out_channel, out_channel, 3)
        self.e5 = ChannelAttention(out_channel)
        self.HSI5 = HSI_block(out_channel, out_channel)
        self.en5 = ChannelAttention(out_channel)
        self.conv6 = self.convlayer(out_channel, out_channel, 3)
        self.e6 = ChannelAttention(out_channel)
        self.HSI6 = HSI_block(out_channel, out_channel)
        self.en6 = ChannelAttention(out_channel)
        self.conv7 = self.convlayer(out_channel, out_channel, 3)
        self.e7 = ChannelAttention(out_channel)
        self.HSI7 = HSI_block(out_channel, out_channel)
        self.en7 = ChannelAttention(out_channel)
        self.conv8 = self.convlayer(out_channel, out_channel, 3)
        self.e8 = ChannelAttention(out_channel)
        self.HSI8 = HSI_block(out_channel, out_channel)
        self.en8 = ChannelAttention(out_channel)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)

    def convlayer(self, in_channels, out_channels, kernel_size):
        pader = nn.ReplicationPad2d(int((kernel_size - 1) / 2))
        conver = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)
        gelu = nn.LeakyReLU()
        layers = filter(lambda x: x is not None, [pader, conver, gelu])
        return nn.Sequential(*layers)

    def forward(self, x):
        B = self.conv_B(x)
        G = self.conv_G(x)
        R = self.conv_R(x)
        B = self.convinputb(B)
        G = self.convinputg(G)
        R = self.convinputr(R)
        f0 = torch.cat([B, G], 1)
        f0 = torch.cat([f0, R], 1)
        f0 = self.fusion(f0)
        Sf0 = self.HSI1(f0)
        e1 = self.e1(f0)
        en1 = self.en1(f0)
        f0_e = f0 * e1
        Sf0_en = Sf0 * en1
        f0_conv = self.conv1(f0)
        f1 = torch.add(f0_conv, f0_e)
        f1 = torch.add(f1, Sf0_en)
        Sf1 = self.HSI2(f1)
        e2 = self.e2(f1)
        en2 = self.en2(f1)
        f1_e = f0 * e2
        Sf1_en = Sf1 * en2
        f1_conv = self.conv2(f1)
        f2 = torch.add(f1_conv, f1_e)
        f2 = torch.add(f2, Sf1_en)
        Sf2 = self.HSI3(f2)
        e3 = self.e3(f2)
        en3 = self.en3(f2)
        f2_e = f0 * e3
        Sf2_en = Sf2 * en3
        f2_conv = self.conv3(f2)
        f3 = torch.add(f2_conv, f2_e)
        f3 = torch.add(f3, Sf2_en)
        Sf3 = self.HSI4(f3)
        e4 = self.e4(f3)
        en4 = self.en4(f3)
        f3_e = f0 * e4
        Sf3_en = Sf3 * en4
        f3_conv = self.conv4(f3)
        f4 = torch.add(f3_conv, f3_e)
        f4 = torch.add(f4, Sf3_en)
        Sf4 = self.HSI5(f4)
        e5 = self.e5(f4)
        en5 = self.en5(f4)
        f4_e = f0 * e5
        Sf4_en = Sf4 * en5
        f4_conv = self.conv5(f4)
        f5 = torch.add(f4_conv, f4_e)
        f5 = torch.add(f5, Sf4_en)
        Sf5 = self.HSI6(f5)
        e6 = self.e6(f5)
        en6 = self.en6(f5)
        f5_e = f0 * e6
        Sf5_en = Sf5 * en6
        f5_conv = self.conv6(f5)
        f6 = torch.add(f5_conv, f5_e)
        f6 = torch.add(f6, Sf5_en)
        Sf6 = self.HSI7(f6)
        e7 = self.e7(f6)
        en7 = self.en7(f6)
        f6_e = f0 * e7
        Sf6_en = Sf6 * en7
        f6_conv = self.conv7(f6)
        f7 = torch.add(f6_conv, f6_e)
        f7 = torch.add(f7, Sf6_en)
        Sf7 = self.HSI8(f7)
        e8 = self.e8(f7)
        en8 = self.en8(f7)
        f7_e = f0 * e8
        Sf7_en = Sf7 * en8
        f7_conv = self.conv8(f7)
        f8 = torch.add(f7_conv, f7_e)
        f8 = torch.add(f8, Sf7_en)

        return f8

class HSI_block(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(HSI_block, self).__init__()
        self.conv1 = self.convlayer(in_channels,128,3)
        self.conv2 = nn.Conv2d(128, out_channels, 3, stride=1,padding=1, bias=True)
        self.conv3 = self.convlayer(out_channels,out_channels,1)
        self.gelu = nn.LeakyReLU()

    def convlayer(self, in_channels, out_channels, kernel_size):
        pader = nn.ReplicationPad2d(int((kernel_size - 1) / 2))
        conver = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)
        gelu = nn.LeakyReLU(0.2, inplace=True)
        layers = filter(lambda x: x is not None, [pader, conver, gelu])
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        residuals=torch.add(x,out2)
        residuals=self.gelu(residuals)
        output=self.conv3(residuals)
        return output


##    HSCNN_plus
class Exblock(nn.Module):
    def __init__(self, in_channels):
        super(Exblock, self).__init__()
        self.conv1 = self.convlayer(in_channels, 64, 1,1)
        self.conv3_1 = self.convlayer(64, 16, 3,1)
        self.conv3_2 = self.convlayer(64, 16, 3,1)
        self.conv1_1 = self.convlayer(16, 8, 1,1)
        self.conv1_2 = self.convlayer(16, 8, 1,1)
        self.conv1_final = self.convlayer(48, 16, 1,1)

    def convlayer(self, in_channels, out_channels, kernel_size,stride):
        pader = nn.ReplicationPad2d(int((kernel_size - 1) / 2))
        conver = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=True)
        relu = nn.ReLU()
        layers = filter(lambda x: x is not None, [pader, conver, relu])
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.conv1(x)
        f3_1=self.conv3_1(out1)
        f3_2 = self.conv3_2(out1)
        f1_1 = self.conv1_1(f3_1)
        f1_2 = self.conv1_2(f3_2)
        f=torch.cat([f3_1,f3_2,f1_1,f1_2],1)
        out=self.conv1_final(f)
        output=torch.cat([x,out],1)
        return output

class HSCNN_plus(nn.Module):
    def __init__(self, in_channel,out_channel,n=4):
        super(HSCNN_plus, self).__init__()
        self.conv3_1 = self.convlayer(in_channel, 16, 3, 1)
        self.conv3_2 = self.convlayer(in_channel, 16, 3, 1)
        self.conv1_1 = self.convlayer(16, 16, 1, 1)
        self.conv1_2 = self.convlayer(16, 16, 1, 1)
        self.conv = self.Exlayer(n,64)
        self.conv1_final = self.convlayer(64+n*16, out_channel, 1, 1)

    def convlayer(self, in_channels, out_channels, kernel_size, stride):
        pader = nn.ReplicationPad2d(int((kernel_size - 1) / 2))
        conver = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=True)
        relu = nn.ReLU()
        layers = filter(lambda x: x is not None, [pader, conver, relu])
        return nn.Sequential(*layers)

    def Exlayer(self, n,in_channels):
        main = nn.Sequential()
        for i in range(n):
            name='Ex'+str(i)
            conv=Exblock(in_channels+16*i)
            main.add_module(name,conv)
        return main

    def forward(self, x):
        f3_1=self.conv3_1(x)
        f3_2 =self.conv3_2(x)
        f1_1=self.conv1_1(f3_1)
        f1_2 = self.conv1_1(f3_2)
        f_in=torch.cat([f3_1,f3_2,f1_1,f1_2],1)
        f_out = self.conv(f_in)
        output=self.conv1_final(f_out)
        return output


##    GDNet
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class My_Bn_1(nn.Module):
    def __init__(self):
        super(My_Bn_1, self).__init__()

    def forward(self, x):
        # print(x.shape)
        # _,C,_,_ = x.shape
        # x1,x2 = torch.split(x,C//2,dim=1)
        # x1 = x1 - nn.AdaptiveAvgPool2d(1)(x1)
        # x1 = torch.cat([x1,x2],dim=1)
        # x =
        return x - torch.mean(x, dim=1, keepdim=True)

class My_Bn_2(nn.Module):
    def __init__(self):
        super(My_Bn_2, self).__init__()

    def forward(self, x):
        # print(x.shape)
        # _,C,_,_ = x.shape
        # x1,x2 = torch.split(x,C//2,dim=1)
        # x1 = x1 - nn.AdaptiveAvgPool2d(1)(x1)
        # x1 = torch.cat([x1,x2],dim=1)
        # x =
        return x - nn.AdaptiveAvgPool2d(1)(x)

class BCR(nn.Module):
    def __init__(self, kernel, cin, cout, group=1, stride=1, RELU=True, padding=0, BN=False, spatial_norm=False,
                 bias=False):
        super(BCR, self).__init__()
        if stride > 0:
            self.conv = nn.Conv2d(in_channels=cin, out_channels=cout, kernel_size=kernel, groups=group, stride=stride,
                                  padding=padding, bias=bias)
        else:
            self.conv = nn.ConvTranspose2d(in_channels=cin, out_channels=cout, kernel_size=kernel, groups=group,
                                           stride=int(abs(stride)), padding=padding, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        self.Swish = MemoryEfficientSwish()

        if RELU:
            if BN:
                if spatial_norm:
                    self.Bn = My_Bn_2()
                    self.Module = nn.Sequential(
                        self.Bn,
                        self.conv,
                        self.relu,
                    )
                else:
                    self.Bn = My_Bn_1()
                    self.Module = nn.Sequential(
                        self.Bn,
                        self.conv,
                        self.relu,
                    )
            else:
                self.Module = nn.Sequential(
                    self.conv,
                    self.relu
                )
        else:
            if BN:
                if spatial_norm:
                    self.Bn = My_Bn_2()
                    self.Module = nn.Sequential(
                        self.Bn,
                        self.conv,
                    )
                else:
                    self.Bn = My_Bn_1()
                    self.Module = nn.Sequential(
                        self.Bn,
                        self.conv,
                        self.relu,
                    )
            else:
                self.Module = nn.Sequential(
                    self.conv,
                )

    def forward(self, x):
        output = self.Module(x)
        return output

class denselayer(nn.Module):
    def __init__(self, cin, cout=31, RELU=True, BN=True, bias=True):
        super(denselayer, self).__init__()
        self.compressLayer = BCR(kernel=1, cin=cin, cout=cout, RELU=RELU, BN=BN, spatial_norm=False, bias=bias)
        self.actlayer = BCR(kernel=3, cin=cout, cout=cout, group=cout, RELU=RELU, padding=1, BN=BN, spatial_norm=False,
                            bias=bias)

    def forward(self, x):
        output = self.compressLayer(x)
        output = self.actlayer(output)

        return output

class stage(nn.Module):
    def __init__(self, cin, cout, final=False, BN=False, linear=True, bias=False):
        super(stage, self).__init__()
        self.Upconv = BCR(kernel=3, cin=cin, cout=cout, stride=1, padding=1, RELU=False, BN=False)
        if final == True:
            f_cout = cout + 1
        else:
            f_cout = cout
        mid = cout * 3
        self.denselayers = nn.ModuleList([
            denselayer(cin=1 * cout, cout=cout * 2, BN=BN, bias=bias),
            denselayer(cin=3 * cout, cout=cout * 2, BN=BN, bias=bias),
            denselayer(cin=5 * cout, cout=cout * 2, BN=BN, bias=bias),
            denselayer(cin=7 * cout, cout=cout * 2, BN=BN, bias=bias),
            denselayer(cin=9 * cout, cout=f_cout, RELU=False, BN=False, bias=bias)])
        self.bn = My_Bn_1()
        self.linear = linear

    def forward(self, MSI):
        MSI = self.Upconv(MSI)
        # print('-----------ERROR------------')
        # print(MSI.shape)
        x = [MSI]
        # print(MSI.shape)
        for layer in self.denselayers:
            x_ = layer(torch.cat(x, 1))
            x.append(x_)
        if self.linear == True:
            return x[-1] + MSI
        else:
            return x[-1]

class GDNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GDNet, self).__init__()

        self.stages = nn.ModuleList([
            stage(cin=in_channel, cout=out_channel, BN=False, linear=False, bias=True),
            stage(cin=in_channel, cout=out_channel, BN=True, linear=True, bias=False),
            stage(cin=in_channel, cout=out_channel, BN=True, linear=True, bias=False),
            stage(cin=in_channel, cout=out_channel, BN=True, linear=True, bias=False),
            stage(cin=in_channel, cout=out_channel, BN=True, linear=True, bias=False),
            stage(cin=in_channel, cout=out_channel, BN=True, linear=True, bias=False),
            stage(cin=in_channel, cout=out_channel, BN=True, linear=True, bias=False),
            stage(cin=in_channel, cout=out_channel, BN=True, linear=True, bias=False),
            # stage(cin=3,cout=31,extra = extra[1],BN = True,linear=True,bias=False),
            # stage(cin=3,cout=31,extra = extra[1],BN = True,linear=True,bias=False),
            # stage(cin=3,cout=31,extra = extra[1],BN = True,linear=True,bias=False),
            # stage(cin=3,cout=31,extra = extra[1],BN = True,linear=True,bias=False),
            # stage(cin=3,cout=31,extra = extra[1],BN = True,linear=True,bias=False),
            # stage(cin=3,cout=31,extra = extra[1],BN = True,linear=True,bias=False),
            stage(cin=in_channel, cout=out_channel,  BN=True, linear=True, bias=False)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.degradation = BCR(kernel=3, cin=out_channel, cout=in_channel, group=1, stride=1, RELU=False, BN=False, padding=1, bias=False)

    def forward(self, MSI, extra_data=None):

        ref = [np.array(range(8)) * 4, np.array(range(16)) * 2]
        ref[0][-1] = 30
        ref[1][-1] = 30
        recon_out = None
        MSI = [MSI]
        for index, stage in enumerate(self.stages):
            recon = stage(MSI[-1])
            if recon_out is None:
                recon_out = recon
            else:
                recon_out = recon_out + recon
            msi_ = MSI[0] - self.degradation(recon_out)
            MSI.append(msi_)

        return recon_out


##    SSDCN
class SSDCN(nn.Module):
    def __init__(self,in_channel,out_channel,NNN = 64):
        super(SSDCN, self).__init__()
        # to extract intial feature
        self.Spe1_conv = nn.Conv2d(in_channel, NNN, 1)
        # ------------------residual spectral-spatial blocks 1------------------
        # 1. 1x1 spectral branch
        self.Spe2_conv = nn.Conv2d(NNN, NNN, 1)
        self.Spe3_conv = nn.Conv2d(NNN, NNN, 1)
        self.Spe3_sea = SEA(NNN)
        # ------------------residual spectral-spatial blocks 2------------------
        # 2. 1x1 spectral branch
        self.Spe4_conv = nn.Conv2d(NNN, NNN, 1)
        self.Spe5_conv = nn.Conv2d(NNN, NNN, 1)
        self.Spe5_sea = SEA(NNN)
        # ------------------residual spectral-spatial blocks 3------------------
        # 3. 3x3 spatial branch
        self.Spe6_conv = nn.Conv2d(NNN, NNN, 3)
        self.Spe7_conv = nn.Conv2d(NNN, NNN, 3)
        self.padding = nn.ReflectionPad2d(1)
        self.Spe7_sea = SEA(NNN)
        # ------------------residual spectral-spatial blocks 4------------------
        # 4. 3x3 spatial branch
        self.Spe8_conv = nn.Conv2d(NNN, NNN, 1)
        self.Spe9_conv = nn.Conv2d(NNN, NNN, 1)
        self.Spe9_sea = SEA(NNN)
        # Out
        self.out = nn.Conv2d(NNN, out_channel, 1)

        # ----------------------------------------Dual Network--------------------------------------------
        self.down = nn.Conv2d(out_channel, in_channel, 1)
        self.relu=nn.ReLU()

    def forward(self, x):
        Spe1=self.Spe1_conv(x)
        # ------------------residual spectral-spatial blocks 1------------------
        # 1. 1x1 spectral branch
        Spe2 = self.Spe2_conv(Spe1)
        Spe3 = self.Spe3_conv(Spe2)
        Spe3 = self.Spe3_sea(Spe3)
        Spe3_residual =self.relu(Spe3+Spe1)

        # ------------------residual spectral-spatial blocks 2------------------
        # 2. 1x1 spectral branch
        Spe4 = self.Spe4_conv(Spe3_residual)
        Spe5 = self.Spe5_conv(Spe4)
        Spe5 = self.Spe5_sea(Spe5)
        Spe5_residual = self.relu(Spe5+Spe3_residual)

        # ------------------residual spectral-spatial blocks 3------------------
        # 3. 3x3 spatial branch
        Spe6 = self.Spe6_conv(self.padding(Spe5_residual))
        Spe7 = self.Spe7_conv(self.padding(Spe6))
        Spe7 = self.Spe7_sea(Spe7)
        Spe7_residual = self.relu(Spe7 + Spe5_residual)

        # ------------------residual spectral-spatial blocks 4------------------
        # 4. 3x3 spatial branch
        Spe8 = self.Spe8_conv(Spe7_residual)
        Spe9 = self.Spe9_conv(Spe8)
        Spe9 = self.Spe9_sea(Spe9)
        Spa9_residual = self.relu(Spe9 + Spe7_residual)

        # Out
        Output_HSI = self.out(Spa9_residual)

        # ----------------------------------------Dual Network--------------------------------------------
        Output_MSI = self.down(Output_HSI)
        return Output_HSI

class SEA(nn.Module):
    def __init__(self, in_planes):
        super(SEA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, 16, 1, bias=False)
        self.relu1 =nn.ReLU()
        self.fc2 = nn.Conv2d(16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        out = self.sigmoid(avg_out)*x
        return out



