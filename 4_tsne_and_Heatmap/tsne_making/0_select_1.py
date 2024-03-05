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


def main():
    #打标签后的数据地址
    # label_list=glob.glob(r"E:\文件\项目列表\垃圾分类\VIT-Sklearn\无监督聚类\result_final_10_2\\*")
    label_list=glob.glob("data_raw/*")
    # print(label_list)
    # label_list = random.sample(label_list,9)
    print('strat convnext encoding===============')
    with open("img_data_conv","w",encoding="utf-8")as f:
        for label in label_list:
            print(label)
            file_list=glob.glob(label+"/*")
            label_=label.split("/")[-1]
            for file_ in file_list:
                img=convnext_encoding(file_)
                f.write("%s\t%s\n"%(file_.split("/")[-1],"<=>".join([str(x)for x in img])))
    print('strat ViT_large encoding===============')
    with open("img_data_ViT","w",encoding="utf-8")as f:
        for label in label_list:
            print(label)
            file_list=glob.glob(label+"/*")
            label_=label.split("/")[-1]
            for file_ in file_list:
                img=vit_encoding(file_)
                f.write("%s\t%s\n"%(file_.split("/")[-1],"<=>".join([str(x)for x in img])))
    print('strat img encoding===============')
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
