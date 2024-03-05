import PIL.Image as I
import numpy as np
import glob,os
import matplotlib.pyplot as plt



def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path




def make_log(name,rotates):
    listname = name.split('\\')[-1].split('_')[-1]
    typename = name.split('\\')[1].split('_')[0]+'_'

    savename = typename+listname
    # print(savename)
    # exit()
    path_1 = makedir('acc\\'+savename.split('_')[0]+'\\%s\\'%(rotates))                   
    if not os.path.exists(path_1+rotates+'_'+savename): 
        with open(name,'r',encoding = 'utf-8') as f:
            for name in f:
     
                line = name.strip().split()

                if len(line) == 18 :
                    step = line[17].split(':')[-1]
                    rotate = line[16].split(':')[-1].split(',')[0]
                    acc = line[12].split(',')[0]
                    if rotate == rotates:                  
                        path = makedir('acc\\'+savename.split('_')[0]+'\\%s\\'%(rotate))
                        with open (path+rotates+'_'+savename,'a',encoding='utf-8') as f1:
                            f1.write('%s\t%s\n'%(step,acc))
                elif len(line) == 20 :

                    step = line[19].split(':')[-1]
                    rotate = line[16].split(':')[-1].split(',')[0]
                    acc = line[12].split(',')[0]
                    if rotate == rotates:
                        path = makedir('acc\\'+savename.split('_')[0]+'\\%s\\'%(rotate))
                        with open (path+rotates+'_'+savename,'a',encoding='utf-8') as fii:
                            fii.write('%s\t%s\n'%(step,acc))



for log in ['log\\raw2_log','log\\raw5_log','log\\raw8_log','log\\raw_log','log\\polar_log','log\\raw20_log','log\\rawnocae_log','log\\rawgauss_log']:
    names_all=glob.glob(log+'\\*')


    for name in names_all:
        
        for rotates in ['rotation_0','rotation_80','rotation_180','rotation_270','rotation_90']:
            make_log(name,rotates)
        print(name)

  