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
from model import sSRHelp
from ass import SSIM,Loss_SAM,initialize_logger
from store2tiff import readTiff,writeTiff
from analysis import analysis_accu
import argparse

parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument("--D_ite", type=int, default=50, help="opt.D_ite")
parser.add_argument("--D_lr", type=float, default=1e-3, help="Degra learning rate")
parser.add_argument("--gpus", type=str, default='0', help='gpu name')
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus

outfile = 'output/GF2/'
data=io.loadmat('data/GF2.mat')
GT = data['msi']
GT = Variable(torch.from_numpy(GT).float()).view(1, -1, GT.shape[1], GT.shape[2]).cuda()
GT=GT.permute(0,3,1,2)
msi = data['msi_lr']
msi = Variable(torch.from_numpy(msi).float()).view(1, -1, msi.shape[1], msi.shape[2]).cuda()
msi=msi.permute(0,3,1,2)
pan = data['pan_lr']
pan = Variable(torch.from_numpy(pan).float()).view(1, -1, pan.shape[0], pan.shape[1]).cuda()
msi_down = data['msi_llr']
msi_down = Variable(torch.from_numpy(msi_down).float()).view(1, -1, msi_down.shape[1], msi_down.shape[2]).cuda()
msi_down=msi_down.permute(0,3,1,2)
pan_down = data['pan_llr']
pan_down = Variable(torch.from_numpy(pan_down).float()).view(1, -1, pan_down.shape[0], pan_down.shape[1]).cuda()

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

def analysis(GT,output,ratio):
    index=analysis_accu(GT[0,:,:,:],output[0,:,:,:], ratio)
    print("\r===>Test:    CC    RMSE    PSNR   SSIM    SAM  ERGAS")
    print("           {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(index[0,0].data,index[1,0].data,index[2,0].data,index[3,0].data,index[4,0].data,index[5,0].data))

def Degra_selftraining(msi,pan,IniFus, model,criterion,iterationmax,logger):
    print("===> Degradation Self-learning")
    optimizer = optim.Adam(model.parameters(), lr=opt.D_lr, betas=(0.9, 0.999), weight_decay=1e-4)
    # optimizer = optim.SGD(model.parameters(), lr=opt.D_lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40,eta_min=0.0005)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=0)

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
        # scheduler.step(temp.numpy())
        # scheduler.step()
        print("\r ===> Interation[{}]: Loss:{:.10f} Time:{:.6f} lr={}".format(iteration, loss.data,endtime - starttime,optimizer.param_groups[0]["lr"]),end='')
        with torch.no_grad():
            results_enm=(IniFus+output)/2.
            index = analysis_accu(GT[0, :, :, :], results_enm[0, :, :, :], ratio)
        logger.info(" Iter[%06d], learning rate : %.9f, Train Loss: %.6f, Test: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f " % (
                    iteration, optimizer.param_groups[0]['lr'], temp.numpy(), index[0,0].data,index[1,0].data,index[2,0].data,index[3,0].data,index[4,0].data,index[5,0].data))
        # scheduler.step(index[2,0].data)
    return output

def torch2tiff(torch,filename):
    out_temp = torch.cpu()
    output_temp = out_temp.data[0].numpy().astype(np.float32)
    output_temp = np.transpose(output_temp, [1, 2, 0])
    output = output_temp
    writeTiff(output, output.shape[2], output.shape[0], output.shape[1], filename)

channel_msi=msi.shape[1]
channel_pan=pan.shape[1]
ratio=pan.shape[2]/msi.shape[2]

outfile=outfile+'_Dite'+str(opt.D_ite)+'_lr'+str(opt.D_lr)+'/'
if not os.path.exists(outfile):
    os.makedirs(outfile)
log_dir_U = os.path.join(outfile, 'train_UpscaleST.log')
logger_U = initialize_logger(log_dir_U,'logU')
log_dir_D = os.path.join(outfile, 'train_DegraST.log')
logger_D = initialize_logger(log_dir_D,'logD')

sSRHN=sSRHelp(channel_msi, channel_pan)

star_time=time.time()
FinalFus=Degra_selftraining(msi,pan,IniFus, sSRHN,Degra_criterion,opt.D_ite,logger_D)
print("\nTime for sSRHN_DegraST is {:6f}".format(time.time()-star_time))

analysis(GT,(FinalFus+IniFus)/2.,ratio)

# torch2tiff((FinalFus+IniFus)/2.,outfile +  'FinalFus_Ensem.tiff')
