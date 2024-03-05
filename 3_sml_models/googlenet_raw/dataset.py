""" train and test dataset

author baiyu
"""
import os
import sys
import pickle
import glob
#from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image as I
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from conf import settings

class CIFAR100Train(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):      #文件地址格式
        #self.transform=transform
        #if transform is given, we transoform data using
        path_train=path+"/"+"train"+"/*"
        # if os.path.exists(path_train)
        train_files=glob.glob(path_train)
        #print(train_files,"输出的图片地址")
        x=[]
        y=[]
        #0107train文件pickle输入修改
        for im in train_files:
           #with open(im,"rb") as f:
   
            img = I.open(im).convert("L") 
            #img = img.resize((28,28))
          
            #img = pickle.load(f)
            img=np.array(img)
            #img=self.zuida_zuixiao_guiyihua(img)
            # img=self.junzhi_fangcha_guiyihua(img)
            #img = self.junzhi_fangcha_guiyihua(img)
            img = img.transpose()
            #img = np.resize(128,128)            
            #exit()
            #print(img.shape,'img.shape')
            x.append(img)
            
            label=im.split("/")[-1].split("_")[0]
            
            y.append(int(label))
        x=np.array(x)
        #print(x.shape,'x.shape')
        self.label = np.array(y)
        
        self.data = np.vstack(x).reshape(-1, 1,settings.IMG_SIZE[0], settings.IMG_SIZE[1])
        self.image = self.data
        print(self.image.shape,'data_train处理后的形状')
        #exit()
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        label = self.label[index]
        image = self.image[index]
        #image=torch.from_numpy(image)
        return image, label

    def zuida_zuixiao_guiyihua(self,x):
        x_,y_=x.shape
        x_zhong=x.reshape([x_*y_])
        #print(x_zhong)
        Max=max(x_zhong)
        Min=min(x_zhong)
        x = (x_zhong - Min) / (Max - Min)
        x=x.reshape([x_,y_])
        return x
        
    def junzhi_fangcha_guiyihua(self,x):
        junzhi=np.mean(x)
        max_num=np.max(x)
        x = (x - junzhi) / max_num
        return x
class CIFAR100Test(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """
    #0107test文件pickle输入修改
    def __init__(self, path, name,transform=None):
        #self.transform=transform
        #path_test=path+"/"+"test"+"/*"
        # print(name,'==name==')
        path_test=path+"/"+name+"/*"
        #print(path_test,'==path_test==')
        # if os.path.exists(path_train):
        test_files=glob.glob(path_test)
        #print(test_files,'==test_files==')
        x=[]
        y=[]
        for im in test_files:
          #with open(im,"rb") as f:

            img = I.open(im).convert("L")
            #img = img.resize((28,28))

            #img = pickle.load(f)
            img=np.array(img)
            img = img.transpose()
            #img = np.resize(28,175)
            #img=self.junzhi_fangcha_guiyihua(img)
            x.append(img)
            label=im.split("/")[-1].split("_")[0]

            y.append(int(label))

        x=np.array(x)
        self.label_test = np.array(y)
        try:
            self.data_test = np.vstack(x).reshape(-1, 1,settings.IMG_SIZE[0], settings.IMG_SIZE[1])  #0107尺寸修改
            print(self.data_test.shape,"data_test处理后的形状")
            self.image_test = self.data_test
        except:
            print('%s该数据没有源'%name)
        #exit()
    def __len__(self):
        return self.data_test.shape[0]

    def __getitem__(self, index):
        label = self.label_test[index]
        image = self.image_test[index]
        image=torch.from_numpy(image)
        return image, label

    def zuida_zuixiao_guiyihua(self,x):
        x_,y_=x.shape
        x_zhong=x.reshape([x_*y_])
        #print(x_zhong)
        Max=max(x_zhong)
        Min=min(x_zhong)
        x = (x_zhong - Min) / (Max - Min)
        x=x.reshape([x_,y_])
        return x
        
    def junzhi_fangcha_guiyihua(self,x):
        junzhi=np.mean(x)
        fangcha=np.var(x)
        x = (x - junzhi) / (np.sqrt(fangcha))
        #one = torch.ones_like(x)
        #one_300=one*300
        #one_fu300=one*(-300)
        #x = torch.where(-300>x, one_fu300, a)
        #x = torch.where(x>3000, one_300, a)
        return x

class xr_Train(Dataset):

    def __init__(self, path, transform=None):
        #if transform is given, we transoform data using
        pwd = os.path.join(path,"train_data")
        files = glob.glob(pwd+"/*")
        #print(files[0], len(files))
        imgs = []
        labels = []
        t=transforms.Compose([transforms.Resize((32,32)),])
        for i, png in enumerate(files):
            tag = png.split("/")[-1].split("_")[0]
            if tag == "positive":
                labels.append(1)
            else:
                labels.append(0)
            image = Image.open(png).convert("RGB")
            image = t(image)
            #print(np.shape(image))
            #print(labels)
            #exit()
            imgs.append(image)
        
        self.transform = transform
        data = np.vstack(imgs).reshape(-1, 3, 32, 32)
        self.image = data.transpose((0, 2, 3, 1))
        self.label = labels

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        label = self.label[index]
        #print(type(label))
        image = self.image[index]
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image, label

class xr_Test(Dataset):

    def __init__(self, path, transform=None):
        #if transform is given, we transoform data using
        pwd = os.path.join(path,"test_data")
        files = glob.glob(pwd+"/*")
        #print(files[0], len(files))
        imgs = []
        labels = []
        t=transforms.Compose([transforms.Resize((32,32)),])
        for i, png in enumerate(files):
            tag = png.split("/")[-1].split("_")[0]
            
            if tag == "positive":
                labels.append(1)
            else:
                labels.append(0)
            image = Image.open(png).convert("RGB")
            image = t(image)
            #print(np.shape(image))
            #print(labels)
            #exit()
            imgs.append(image)

        self.transform = transform
        data = np.vstack(imgs).reshape(-1, 3, 32, 32)
        self.image = data.transpose((0, 2, 3, 1))
        self.label = labels

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        label = self.label[index]
        image = self.image[index]
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image, label

#train = xr_Train("/home/tuhailong/pytorch-cifar100/data/xr_wrist")
#train = CIFAR100Train("/home/tuhailong/pytorch-cifar100/data/cifar-100-python")
#print(train.__len__())
#print(train.__getitem__(0))
