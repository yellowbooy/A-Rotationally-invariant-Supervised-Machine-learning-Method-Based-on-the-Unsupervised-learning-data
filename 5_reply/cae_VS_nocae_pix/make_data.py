import PIL.Image as I
import numpy as np
import glob,os,random

#随机挑选十张影像
imgpath=r'E:\python\wangting\tiantidataset\dataset0305\dataset0326\28_28_Mk3_noCAE\train\\'
imgname = os.listdir(imgpath)
imgnames = random.sample(imgname,10)

#将这十张影像对应的降噪和未降噪版本存入相应文件夹中
nocaepath=r'E:\python\wangting\tiantidataset\dataset0305\dataset0326\28_28_Mk3_noCAE\train\\'
nocaelist = os.listdir(nocaepath)

caepath = r'E:\python\wangting\tiantidataset\dataset0305\dataset0326\28_28_Mk3\train\\'
caelist = os.listdir(caepath)

for filename in imgnames:
    if filename in nocaelist:
        img = I.open(nocaepath+filename)
        img.save(r'E:\文件\天体分类\汇报总结\补充图\rawimg\nocae\\'+filename)
    if filename in caelist:
        img = I.open(caepath+filename)
        img.save(r'E:\文件\天体分类\汇报总结\补充图\rawimg\cae\\'+filename)