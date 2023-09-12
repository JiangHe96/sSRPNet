import math
import torch
import torch.nn.functional as F


def analysis_accu(img_base,img_out,ratio):
    chanel = img_out.shape[0]
    h = img_out.shape[1]
    w = img_out.shape[2]

#计算CC
    C1=torch.sum(torch.sum(img_base*img_out,1),1)-h*w*(torch.mean(torch.mean(img_base,1),1)*torch.mean(torch.mean(img_out,1),1))
    C2=torch.sum(torch.sum(img_out**2,1),1)-h*w*(torch.mean(torch.mean(img_out,1),1)**2)
    C3 = torch.sum(torch.sum(img_base**2,1),1)-h*w*(torch.mean(torch.mean(img_base,1),1)**2)
    CC=C1/((C2*C3)**0.5)

#计算SAM
    sum1 = torch.sum(img_base* img_out,0)
    sum2 = torch.sum(img_base* img_base,0)
    sum3 = torch.sum(img_out* img_out,0)
    t=(sum2*sum3)**0.5
    numlocal=torch.gt(t, 0)
    num=torch.sum(numlocal)
    t=sum1 / t
    angle = torch.acos(t)
    sumangle= torch.where(torch.isnan(angle), torch.full_like(angle, 0), angle).sum()
    if num==0:
        averangle=sumangle
    else:
        averangle=sumangle/num
    SAM=averangle*180/3.14159256

#计算ERGAS
    summ=0
    for i in range(chanel):
        a1 = torch.mean((img_base[i,:, :] - img_out[i,:, :])**2)
        m1=torch.mean(img_base[i,:, :])
        a2=m1*m1
        summ=summ+a1/a2
    ERGAS=100*(1/ratio)*((summ/chanel)**0.5)

#计算RMSE,PSNR
    mse = torch.mean((img_base- img_out) ** 2,1)
    mse = torch.mean(mse, 1)
    rmse = mse**0.5
    temp=torch.log(1 / rmse)/math.log(10)
    PSNR = 20 * temp

# 计算SSIM
    img_base = img_base.unsqueeze(0)
    img_out = img_out.unsqueeze(0)
    SSIM=_ssim(img_base,img_out)

    index=torch.zeros((6,chanel+1))

    index[0, 1:chanel+1] =CC
    index[1, 1:chanel + 1]=rmse
    index[2, 1:chanel+1] =PSNR
    index[3, 1:chanel+1] =SSIM
    index[0, 0] = torch.mean(CC)
    index[1, 0] = torch.mean(rmse)
    index[2, 0] = torch.mean(PSNR)
    index[3, 0] = torch.mean(SSIM)
    index[4, 0] =SAM
    index[5, 0] =ERGAS
    return index

def _ssim(img1, img2):
    channel = img1.shape[1]
    max_val = 1
    _, c, w, h = img1.size()
    window_size = min(w, h, 11)
    sigma = 1.5 * window_size / 11
    window = create_window(window_size, sigma, channel).cuda()
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)


    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2
    C1 = (0.01*max_val)**2
    C2 = (0.03*max_val)**2
    V1 = 2.0 * sigma12 + C2
    V2 = sigma1_sq + sigma2_sq + C2
    ssim_map = ((2*mu1_mu2 + C1)*V1)/((mu1_sq + mu2_sq + C1)*V2)
    t=ssim_map.shape
    return  ssim_map.mean(2).mean(2)

from torch.autograd import Variable

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, sigma, channel):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

#full resolution evaluation

