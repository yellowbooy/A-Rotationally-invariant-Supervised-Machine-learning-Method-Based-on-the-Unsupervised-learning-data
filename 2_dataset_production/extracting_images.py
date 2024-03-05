import os, random, shutil
import PIL.Image as I
from tqdm import tqdm

#构造验证集
def make_testingset(fileDir,tarDir):
        pathDir = os.listdir(fileDir)    #取图片的原始路径
        filenumber=2900            #确定验证集的数量
        picknumber=int(filenumber) 
        sample = random.sample(pathDir, picknumber)  #随机选取picknumber数量的样本图片
        n = 0
        namelist=[]
        print('开始划分验证集')
        for name in tqdm(sample):
                img = I.open(fileDir+name).convert('L')
                img = img.save(tarDir+name)
                namelist.append(name)
                n = n + 1
                #print(n,'====验证集图像抽取数，总数为%s====')
        return namelist

#构造训练和验证集
def make_dataset(fileDir,tarDir,tarDir1):
        if not os.path.exists(tarDir):
            os.makedirs(tarDir)
        if not os.path.exists(tarDir1):
            os.makedirs(tarDir1)
        pathDir = os.listdir(fileDir)    #取图片的原始路径
        namelist = make_testingset(fileDir,tarDir)
        m = 0
        print('开始划分训练集')
        for name in tqdm(pathDir):
            if name not in namelist:
                img = I.open(fileDir+name).convert('L')
                img = img.save(tarDir1+name)
                m = m + 1
                #print(m,'====训练集图像抽取数，总数为%s====')
        return

if __name__ == '__main__':
    fileDir = "./1_cae_class/2_cae_class/dataset/"    #原始图像文件夹路径
    tarDir = 'dataset/test/'    #验证集地址
    tarDir1 = 'dataset/train/'    #训练集地址
    make_dataset(fileDir,tarDir,tarDir1)
