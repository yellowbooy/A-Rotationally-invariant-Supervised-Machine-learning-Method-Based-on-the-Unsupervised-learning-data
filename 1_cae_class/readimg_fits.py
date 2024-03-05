import PIL.Image as I
import PIL.ImageFilter as filters
import numpy as np
import os,glob
from tqdm import tqdm
from astropy.io import fits
import pickle as pkl
import imageio


def mkdirs(name):
    if not os.path.exists(name):
        os.makedirs(name) 
    return name 


def ioreadfit(path,i,savepath):
    filelists = os.listdir(path+i)
    #print(filelists)
    for j in tqdm(filelists):
        with open(path+i+'/'+j,'rb') as f:
            img = pkl.load(f)
        img = np.array(img)
        paths = mkdirs(savepath+i+'/')
        imageio.imsave(savepath+i+'/'+j.split('.txt')[0]+'.png', img )

path = r'result_final_50/'
pathfiles = glob.glob(path+'/*')
savepath = r'result_img_50/'
savepath1 = r'D:\宋杰\cutimage\0002.png'
k = 0
for i in tqdm(pathfiles):
    try:
        img1 = ioreadfit(path,i.split('/')[-1]+'/',savepath)
    except:
        print('%s有问题'%i)

