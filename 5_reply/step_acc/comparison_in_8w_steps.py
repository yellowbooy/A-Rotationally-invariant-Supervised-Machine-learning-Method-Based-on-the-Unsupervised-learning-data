import matplotlib.pyplot as plt
import numpy as np
import os,glob


listname = ['acc\\raw2','acc\\raw5','acc\\raw8','acc\\polar','acc\\raw','acc\\raw20']#'acc\\rawnocae','acc\\raw20']
# listname = ['acc\\raw','acc\\rawnocae','acc\\rawgauss']
#载入各个数据的各个旋转角度的每100步准确率数据
def load_data():
    t={}
    max_=-1
    for dirname in listname:
        t[dirname]={}
        file_names = glob.glob(dirname+'\\*')
        for file in file_names:
            files = glob.glob(file+'\\*')
            t[dirname][file]={'1':[],'2':[],'3':[],'error_raw':[],'mean_raw':[],'error':[],'mean':[]}
            for file_ in files:
                the_number = file_.split('\\')[-1].split('_')[-1].split('.')[0]
                t[dirname][file][the_number]={"step":[],'acc':[]}
                for ll,line in enumerate(open(file_,"r",encoding="utf-8")):
                    line = line.strip().split()
                    step = int(line[0])
                    acc = float(line[1])
                    if step <= 80000:
                        t[dirname][file][the_number]['acc'].append(acc)
                        t[dirname][file][the_number]['step'].append(step)
    return t

#计算每种数据在各个旋转角度验证集每100步准确率的均值和标准差
def make_error(t):
    mean_err = {}
    for dirname in listname:
        file_names = glob.glob(dirname+'\\*')
        for file in file_names:
            files = glob.glob(file+'\\*')
            for file_ in files:
                the_number = file_.split('\\')[-1].split('_')[-1].split('.')[0]
                t[dirname][file]['mean_raw'].append(t[dirname][file][the_number]['acc'])
            t[dirname][file]['mean_raw']=np.array(t[dirname][file]['mean_raw'])
            t[dirname][file]['mean_raw']=t[dirname][file]['mean_raw'].transpose()
            t[dirname][file]['mean_raw']=t[dirname][file]['mean_raw'].transpose()
            t[dirname][file]['mean_raw']=t[dirname][file]['mean_raw'].transpose()
            
            for i in t[dirname][file]['mean_raw']:
                means = round(np.mean(i),4)
                stds = round(np.std(i),4)
                t[dirname][file]['mean'].append(means)
                t[dirname][file]['error'].append(stds)
    return t


def change_title(f):
    if f == 'raw2':
        f='raw image *2'
    elif f == 'raw5':
        f='raw image *5'
    elif f == 'raw8':
        f='raw image *8'
    elif f == 'raw20':
        f='raw image *20'        
    elif f == 'raw':
        f='Without APCT'        
    elif f == 'polar':
        f='With APCT'
    return f
#根据提取的数据画图    
def plot(t):
    lab = []
    for dirname in listname:
        labs = change_title(dirname.split('\\')[-1])
        lab.append(labs)
    # roattes = ['rotation_0','rotation_80','rotation_90','rotation_180','rotation_270']
    roattes = ['rotation_0']
    for file_ in roattes:
        plt.figure(figsize=(20,10))
        for dirname in listname:
            file = dirname+'\\'+file_
            x_1 = []
            y_1= []
            std_1=[]
            for i in range(0,800,10):
                x_1.append(t[dirname][file]['1']['step'][i])
                y_1.append(t[dirname][file]['mean'][i])
                std_1.append(t[dirname][file]['error'][i])
            x = x_1
            y = y_1
            std = std_1
            plt.plot(x,y,marker='o',markersize = '4')
            #误差棒绘制
            #plt.errorbar(x,y,marker='o',yerr=std)
            title = file.split('\\')[-1].split('_')[-1]
            plt.title('The overall accuraices versus steps at different situations.The images in test set are rotated by %s degrees.'%title)
        plt.legend(lab)
        plt.xlabel('Steps',fontdict=None,labelpad=None)
        plt.ylabel('Overall accuracies',fontdict=None,labelpad=None)
        plt.savefig(title+'.jpg')
        # plt.show()
        # exit()



if __name__=="__main__":
    t=load_data()
    make_error(t)
    plot(t)