from skimage.measure import compare_ssim
#from skimage.metrics import structural_similarity as compare_ssim

from skimage.color import rgb2gray
import numpy as np
from PIL import Image
import scipy.misc
from scipy.misc import imread
from scipy.misc import imresize
import skimage
import lpips
import torch


def test_quality():
    real_path = " "
    fake_path = " "
    ssim_scores = []
    psnr_scores = []
    l1_losses = []
    lpips_scores=[]

    for i in range(2000):

        s = '%04d' % (i+1) # 
     
        fake = fake_path + "places2_{:s}.jpg".format(s)
        real  = real_path + "places2_{:s}.jpg".format(s)
        imageA = rgb2gray(imread(fake)/255)#
        imageB = rgb2gray(imread(real)/255)
     
        

        score =compare_ssim(imageA, imageB, win_size=51, data_range = 1)
        ssim_scores.append(score)
        psnr_scores.append(psnr1(imageA, imageB))
        l1_losses.append(L1loss(imageA, imageB))

        imageA_tensor=torch.from_numpy(imageA).to(torch.float32)
        imageB_tensor=torch.from_numpy(imageB).to(torch.float32)
        loss_fn_alex=lpips.LPIPS(net='alex')
        lpips_scores.append(loss_fn_alex(imageA_tensor, imageB_tensor))
    
    print("SSIM score is:", sum(ssim_scores)/len(ssim_scores))
    print("PSNR score is:", sum(psnr_scores)/len(psnr_scores))
    print("L1 Losses score is:", sum(l1_losses)/len(l1_losses))
    print("lpips score is:", sum(lpips_scores)/len(lpips_scores))
    
    
def psnr1(img1, img2):
   return skimage.measure.compare_psnr(img1, img2, data_range = 1)

def L1loss(img1, img2):
    return np.sum(np.abs(img1 - img2))/np.sum(img1 + img2)

test_quality()