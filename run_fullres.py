import logging
import scipy.io as io
import os
import numpy as np
import time
import math
import torch.nn as nn
import torch.optim as optim
import torch
from torch.autograd import Variable

from MSDCNN import MSDCNN
from model import sSRHelp
from ass import SSIM,Loss_SAM,initialize_logger
from store2tiff import writeTiff as wtiff
from analysis import analysis_accu
import argparse

parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument("--U_ite", type=int, default=500, help="opt.U_ite")
parser.add_argument("--D_ite", type=int, default=500, help="opt.D_ite")
parser.add_argument("--U_lr", type=float, default=2e-3, help="Upscale learning rate")
parser.add_argument("--D_lr", type=float, default=1e-3, help="Degra learning rate")
parser.add_argument("--gpus", type=str, default='0', help='gpu name')
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
outfile = 'output/fullresolution/QB/'
CSoutfile=outfile+'MSDCNN2.tiff'
data=io.loadmat('data/QB_fullresolution.mat')
msi = data['msi']
msi = Variable(torch.from_numpy(msi).float()).view(1, -1, msi.shape[1], msi.shape[2]).cuda()
msi=msi.permute(0,3,1,2)
pan = data['pan']
pan = Variable(torch.from_numpy(pan).float()).view(1, -1, pan.shape[0], pan.shape[1]).cuda()
msi_down = data['msi_lr']
msi_down = Variable(torch.from_numpy(msi_down).float()).view(1, -1, msi_down.shape[1], msi_down.shape[2]).cuda()
msi_down=msi_down.permute(0,3,1,2)
pan_down = data['pan_lr']
pan_down = Variable(torch.from_numpy(pan_down).float()).view(1, -1, pan_down.shape[0], pan_down.shape[1]).cuda()

def cut(msi,pan,msi_down,pan_down,width_msi):
    width_pan=width_msi*4
    width_msi_down = width_msi // 4
    width_pan_down = width_msi_down * 4
    msi = msi[:, :, 0:width_msi, 0:width_msi]
    pan = pan[:, :, 0:width_pan, 0:width_pan]
    msi_down = msi_down[:, :, 0:width_msi_down, 0:width_msi_down]
    pan_down = pan_down[:, :, 0:width_pan_down, 0:width_pan_down]
    return msi,pan,msi_down,pan_down

width_msi=320
msi,pan,msi_down,pan_down=cut(msi,pan,msi_down,pan_down,width_msi)

SSIMloss=SSIM()
SAMloss=Loss_SAM()
Upscale_criterion = nn.L1Loss(reduction='mean')
Degra_criterion = nn.L1Loss(reduction='mean')

def PSNR(img_base, img_out):
    mse = torch.mean((img_base- img_out) ** 2,2)
    mse = torch.mean(mse, 2)
    rmse = mse**0.5
    temp=torch.log(1 / rmse)/math.log(10)
    PSNR = 20 * temp
    return PSNR.mean()

def IniFused(msi,pan, model):

    starttime = time.time()
    output = model(msi, pan)
    endtime = time.time()

    print("\r\nRuningTime: {:.6f}s".format(endtime - starttime))
    return output

def Upscale_selftraining(msi,msi_down,pan_down, model,criterion, iterationmax):
    print("===> Upscale Self-learning")
    optimizer = optim.Adam(model.parameters(), lr=opt.U_lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=6)
    model = model.cuda()

    for iteration in range(iterationmax):
        starttime = time.time()
        output = model(msi_down,pan_down)
        loss = criterion(output, msi)
        ssim = SSIMloss(msi,output)
        psnr = PSNR(msi,output)
        sam = SAMloss(msi,output)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),0.1)
        optimizer.step()
        optimizer.zero_grad()

        temp=loss.data
        temp=temp.cpu()
        endtime = time.time()
        scheduler.step(temp.numpy())

        print("\r ===> Interation[{}]: Loss:{:.10f} SSIM:{:.4f} PSNR:{:.4f} SAM:{:.4f} Time:{:.6f} lr={}".format(iteration, loss.data,ssim.data,psnr,sam.data,endtime - starttime,optimizer.param_groups[0]["lr"]),end='')

    return output

def Degra_selftraining(msi,pan,IniFus, model,criterion,iterationmax):
    print("===> Degradation Self-learning")
    optimizer = optim.Adam(model.parameters(), lr=opt.D_lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=6)

    model=model.cuda()

    for iteration in range(iterationmax):
        starttime = time.time()
        output,Spa_down=model(pan,IniFus)
        loss=criterion(Spa_down,msi)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        temp=loss.data
        temp=temp.cpu()
        endtime = time.time()
        scheduler.step(temp.numpy())
        print("\r ===> Interation[{}]: Loss:{:.10f} Time:{:.6f} lr={}".format(iteration, loss.data,endtime - starttime,optimizer.param_groups[0]["lr"]),end='')
    return output

def torch2tiff(torch,filename):
    out_temp = torch.cpu()
    output_temp = out_temp.data[0].numpy().astype(np.float32)
    output_temp = np.transpose(output_temp, [1, 2, 0])
    output = output_temp
    wtiff(output, output.shape[2], output.shape[0], output.shape[1], filename)

channel_msi=msi.shape[1]
channel_pan=pan.shape[1]
ratio=pan.shape[2]/msi.shape[2]

outfile=outfile+'Uite'+str(opt.U_ite)+'_lr'+str(opt.U_lr)+'_Dite'+str(opt.D_ite)+'_lr'+str(opt.D_lr)+'/'
if not os.path.exists(outfile):
    os.makedirs(outfile)


IFN=MSDCNN(channel_msi+channel_pan,channel_msi,ratio=ratio)
# sSRHN=sSRHelp(channel_msi, channel_pan)

star_time=time.time()
Upscale_selftraining(msi,msi_down,pan_down,IFN,Upscale_criterion,opt.U_ite)
with torch.no_grad():
    IniFus=IniFused(msi,pan, IFN)
print("Time for IFN_UpscaleST is {:6f}".format(time.time()-star_time))
torch2tiff(IniFus,CSoutfile)
star_time=time.time()
FinalFus=Degra_selftraining(msi,pan,IniFus, sSRHN,Degra_criterion,opt.D_ite)
print("\nTime for sSRHN_DegraST is {:6f}".format(time.time()-star_time))

torch2tiff((FinalFus+IniFus)/2.,outfile +  'FinalFus_Ensem.tiff')
