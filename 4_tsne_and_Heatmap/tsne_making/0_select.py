import numpy as np
import PIL.Image as I
from tqdm import tqdm
# import config as C
import random
import timm
from torch.autograd import Variable
import torch
import os,glob ,shutil
from torchvision import transforms# Load ViT
from pytorch_pretrained_vit import ViT
import torchvision.models as models
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F


class SimCLRStage1(nn.Module):
    def __init__(self, feature_dim=800):
        super(SimCLRStage1, self).__init__()

        # self.f = []
        # for name, module in resnet50().named_children():
        #     if name == 'conv1':
        #         module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        #     if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
        #         self.f.append(module)
        # # encoder
        # self.f = nn.Sequential(*self.f)
        # projection head
        # self.g = nn.Sequential(nn.Linear(2048, 512, bias=False),
        #                        nn.BatchNorm1d(512),
        #                        nn.ReLU(inplace=True),
        #                        nn.Linear(512, feature_dim, bias=True))
        self.g = nn.Sequential(nn.Linear(2048,1024, bias=False),
                               nn.BatchNorm1d(1024),
#                                nn.ReLU(inplace=True),
                               nn.Sigmoid(),
                               nn.Linear(1024, feature_dim, bias=True))
    def forward(self, x):
        # x = self.f(x)
        # feature = torch.flatten(x, start_dim=1)
        # out = self.g(feature)
        out = self.g(x)
        # return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
        return  F.normalize(out, dim=-1)



def makedata(data):
    new = []
    for i in data:
        #print(i)
        #exit()
        new.append(i)
    return new

def get_image_data(image_path):
    # ret=I.open(image_path).convert("L")
    ret=I.open(image_path).convert("RGB")
    # print(ret.size)
    ret = ret.resize((128,128))
    # print(np.array(ret).shape)
    ret=np.reshape(np.array(ret),[128*128*3])
    return list(ret)


def vit_encoding(image_path):
    img1 = I.open(image_path).convert('RGB')
    img1 = img1.rotate(90)
    img1 = ys(img1)
    img1 = img1.unsqueeze(0)
    #print(img1.shape,'==')
    with torch.no_grad():
        img1 = img1.to(device)
        outputs = model(img1)        
        outputs = outputs.cpu()        
        #outputs = outputs
        np.set_printoptions(suppress=True)        
        outputs = np.array(outputs,dtype = 'float32')
        # imgs.append([outputs,label])
        datas = makedata(outputs[0])
    return datas


def convnext_encoding(image_path):
    img1 = I.open(image_path).convert('RGB')
    img1 = ys(img1)
    xR = Variable(torch.unsqueeze(img1, dim=0).float(), requires_grad=False)
    #print(xR.shape)
    xR = xR.to(device)
    yR= resnet50_feature_extractor(xR)
    #print(yR.shape)
    yR = yR.cpu()
    yR = yR.data.numpy()
    #yR = yR.squeeze(0)
    outputs = np.array(yR,dtype = 'float32')
    datas = makedata(outputs[0])

    return datas



def compare_encoding():
    pre_model=('../block1/2_compare/pth_60_11/conv+vit1280_sdata800_426_631000.pth')
    model_=SimCLRStage1().to(device)
    model_.load_state_dict(torch.load(pre_model, map_location='cpu'), strict=False)
    model_.eval()
    # ../block1/2_compare/pth_60_11/
    arrR = np.loadtxt(open("../block1/X1_60.csv","rb"),delimiter=",",skiprows=0)
    data=np.array(arrR)
    data = torch.tensor(data, dtype=torch.float)
    data_CLR = []
    # nums = 
    # data = torch.tensor(data, dtype=torch.float)
    # data.shape
    # data_CLR=[]
    for i,da in enumerate(data):
        dac=da.unsqueeze(0)
        dac = dac.to(device)
        #print(i,dac.shape)
        with torch.no_grad():
            outputs=model_(dac)
            outputs = outputs.cpu()
            outputs = np.array(outputs)
            #print(i,j,k)# (1, 1000)
        data_CLR.append(outputs)
        

    return data_CLR



def main():
    #打标签后的数据地址
    # label_list=glob.glob(r"E:\文件\项目列表\垃圾分类\VIT-Sklearn\无监督聚类\result_final_10_2\\*")
    label_list=glob.glob("data_raw/*")
    # print(label_list)
    # label_list = random.sample(label_list,9)
    #print('strat convnext encoding===============')
    

    #with open("img_data_conv","w",encoding="utf-8")as f:
    #    for label in label_list:
    #        print(label)
    #        file_list=glob.glob(label+"/*")
    #        label_=label.split("/")[-1]
    #        for file_ in file_list:
    #            img=convnext_encoding(file_)
    #            f.write("%s\t%s\n"%(file_.split("/")[-1],"<=>".join([str(x)for x in img])))
    #print('strat ViT_large encoding===============')
    
    #with open("img_data_ViT","w",encoding="utf-8")as f:
    #    for label in label_list:
    #        print(label)
    #        file_list=glob.glob(label+"/*")
    #        label_=label.split("/")[-1]
    #        for file_ in file_list:
    #            img=vit_encoding(file_)
    #            f.write("%s\t%s\n"%(file_.split("/")[-1],"<=>".join([str(x)for x in img])))
    #print('strat compareing encoding===============')
    #name = []
    #with open("img_data_compare","w",encoding="utf-8")as f:
    #    for label in label_list:
            # name.append([])
    #        print(label)
    #        file_list=glob.glob(label+"/*")
    #        label_=label.split("/")[-1]
    #        for file_ in file_list:
    #            name.append(file_)
    #        data = compare_encoding()
    #        #print(len(name))
    #        for i in range(len(name)):
            # for i,da in enumerate(data):              
     #           f.write("%s\t%s\n"%(name[i].split("/")[-1],"<=>".join([str(x)for x in data[i][0]])))
                #print(data[i][0])
                #exit()
    print('strat img encoding===============')
    num = 0
    with open("img_data_img","w",encoding="utf-8")as f:
        for label in label_list:
            print(label)
            file_list=glob.glob(label+"/*")
            label_=label.split("/")[-1]
            for file_ in file_list:
                img=get_image_data(file_)
                f.write("%s\t%s\n"%(file_.split("/")[-1],"<=>".join([str(x)for x in img])))








if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
    # model=ViT('B_16_imagenet1k',pretrained=True)
    model = timm.create_model('vit_large_r50_s32_224_in21k', pretrained=True, num_classes=0)
    model = model.to(device)
    model.eval()# Load image# NOTE: Assumes an image `img.jpg` exists in the current directory
    img=transforms.Compose([transforms.Resize((384,384)),transforms.ToTensor(),transforms.Normalize(0.5,0.5),])


    ys=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])

    resnet50_feature_extractor = timm.create_model('convnext_xlarge_in22k', pretrained=True, num_classes=0)
    resnet50_feature_extractor = resnet50_feature_extractor.to(device)
    resnet50_feature_extractor.eval()

    main()
