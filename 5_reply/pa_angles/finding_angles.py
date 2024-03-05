import matplotlib.pyplot as plt
import glob
import sys,os
import numpy as np
import matplotlib
import PIL.Image as I
from tqdm import tqdm

# print(glob.glob(r'E:/python/wangting/refactor_cpu_Mk3/data/val_corrupted/paris_eval_corrupted/*.png'))
# exit()
def writting_padata():
    namelist=[] 
    with open("train_name_lst","r",encoding="utf-8")as f:
        for line in f:
            name = line.strip().split()
            name = str(name[0]).split('_')[1]+'_'+str(name[0]).split('_')[2]
            # print(name)
            # exit()
            namelist.append(str(name))
            if str(name) == ' ':
                break

    with open("name2label2angle.txt","r",encoding="utf-8")as f:
        for line in f:
            name = line.strip().split()
            if str(name) == ' ':
                break        
            elif name[0] in namelist:
                with open('1661data.txt','a',encoding="utf-8")as g:
                    g.write('%s\t%s\t%s\n'%(name[0],name[2],name[3]))


def cal_an(an):
    if an >360:
        an=an -360
    if an >180:
        an=360-an
    if an <-180:
        an=an +360
    if an <-90:
        an=180+an
    if an >90:
        an=180-an

    
    return an


def making_dataimags():
    palist = []
    panum = []
    with open('1661data.txt','r',encoding="utf-8")as f:
        for line in f:
            if str(line)==' ':
                break
            else:            
                name = line.strip().split()
                palist.append(str(name[2]))
                panum.append(cal_an(int(round(float(name[2]),3))))
    return palist,panum

def pa_24900():
    names = []
    path = r'all_raw_img\\'
    path_list=os.listdir(path)
    for name in path_list:
        name = name.split('_')[1]+'_'+name.split('_')[2]
        name = name.split('.')[0]
        names.append(name)
    with open("name2label2angle.txt","r",encoding="utf-8")as f:
        for line in f:
            Name = line.strip().split()
            if str(Name) == ' ':
                break        
            elif Name[0] in names:
                with open('24900data.txt','a',encoding="utf-8")as g:
                    g.write('%s\t%s\t%s\n'%(Name[0],Name[2],Name[3]))
    pa_24900data = []
    with open("24900data.txt","r",encoding="utf-8")as e:
        for line in e:
            Name_1 = line.strip().split()
            pa_24900data.append(int(round(float(Name_1[2]),1)))
    return pa_24900data
    
    
def count_24900():
    pa_24900data = []
    with open("24900data.txt","r",encoding="utf-8")as e:
        for line in e:
            Name_1 = line.strip().split()
            pa_24900data.append(int(round(float(Name_1[2]),1)))
    return pa_24900data
    
def count_2900():
    pa_2900data = []
    with open("2900data.txt","r",encoding="utf-8")as e:
        for line in e:
            Name_1 = line.strip().split()
            pa_2900data.append(int(round(float(Name_1[2]),1)))
    return pa_2900data

def count_22000():
    pa_22000data = []
    with open("22000data.txt","r",encoding="utf-8")as e:
        for line in e:
            Name_1 = line.strip().split()
            pa_22000data.append(int(round(float(Name_1[2]),1)))
            
    return pa_22000data    

def pa_test(path):
    filename = []
    # path = r'E:\python\wangting\tiantidataset\dataset0305\dataset0326\28_28_Mk3\test\\'
    filelist = os.listdir(path)
    for i in filelist:
        filename.append(i.split('_')[1]+'_'+i.split('_')[2].split('.')[0])
        # print(i.split('_')[1]+'_'+i.split('_')[2].split('.')[0])
        # exit()
    kk =0 
    with open('24900data.txt','r',encoding='utf-8') as f:
        for ii in f:
            line = ii.strip().split()
            print(line,kk+1)
            if line[0] in filename:
                with open('2900data.txt','a',encoding='utf-8') as f_:
                    f_.write('%s\t%s\t%s\n'%(line[0],line[1],line[2]))


def pa_train(path):
    filename = []
    # path = r'E:\python\wangting\tiantidataset\dataset0305\dataset0326\28_28_Mk3\test\\'
    filelist = os.listdir(path)
    for i in filelist:
        filename.append(i.split('_')[1]+'_'+i.split('_')[2].split('.')[0])
        # print(i.split('_')[1]+'_'+i.split('_')[2].split('.')[0])
        # exit()
    kk =0 
    with open('24900data.txt','r',encoding='utf-8') as f:
        for ii in f:
            line = ii.strip().split()
            kk+=1
            print(line,kk)
            if line == ' ':
                pass
             
            elif line[0] in filename:
                with open('22000data.txt','a',encoding='utf-8') as f_:
                    f_.write('%s\t%s\t%s\n'%(line[0],line[1],line[2]))


    
def probability_distribution(data,num, bins_interval=4, margin=3):
    print('number of imgas :',num)
    num = str(num)
    bins = range(min(data), max(data) + bins_interval - 1, bins_interval)
    # print(len(bins))
    print( 'mix angle :', min(data),'max angle :',max(data) )
    # for i in range(0, len(bins)):
        # print(bins[i],'====%s==='%i)
    #plt.xlim(min(data) - margin, max(data) + margin)
    plt.xlim(-90 , 90)
    # print(-90 - margin, 90 + margin)
    # exit()
    plt.title("The distribution of positional angles in the UML-dataset")
    plt.xlabel('Angles')
    plt.ylabel('Numbers')
    bins = range(-90, 90, 2)
    # 频率分布normed=True，频次分布normed=False
    prob,left,rectangle = plt.hist(x=data, bins=bins, density=False, histtype='bar', color='steelblue',edgecolor='black')
    #for x, y in zip(left, prob):
        # 字体上边文字
        # 频率分布数据 normed=True
        #plt.text(x + bins_interval / 2, y + 0.003, '%.2f' % y, ha='center', va='top')
        # 频次分布数据 normed=False
        #plt.text(x + bins_interval / 2, y + 0.25, '%.2f' % y, ha='center', va='top')
    plt.savefig('The_positionalangles_in_%s.jpg'%num)
    # plt.show()



def probability_distribution_data(datas, bins_interval=4, margin=3):
    lab=['train','test','1661']
    lens = len(datas)
    print(lens)
    # fig, axs = plt.subplots(1,3, figsize = (15,4) )
    # fig, axs = plt.subplots(1,1, figsize = (10,4) )
    # fig.subplots_adjust(left = 0.08, bottom = 0.08, right = 0.93, top = 0.95, wspace = 0.25, hspace = 0.2)
    # for data ind datas:    


    for i in range(0,lens):
    # for i in [0]:
        num = str(len(datas[i]))
        print(type(datas[i]))
        bins = range(-90, 90, 15)
        # axs[i].hist(datas[i], bins, histtype='bar', rwidth=2,color='steelblue',edgecolor='black')
    # axs[0,i].legend('cae')
        # axs[0,i].hist(y_nc, group,histtype='bar', rwidth=2,color='red',edgecolor='black')
    # axs[0,i].legend('nocae')
        # axs[i].set_title("The distribution of positional angles in the UML-dataset")
        # axs[i].set_xlabel('Positional Angles')
        # axs[i].set_ylabel('Numbers')


        plt.hist(datas[i], bins, histtype='bar', rwidth=2,color='lightblue',edgecolor='red')
        plt.title("The distribution of positional angles in the UML-dataset")
        plt.xlabel('Positional Angles')
        plt.ylabel('Numbers')
        
        plt.savefig('Distribution_angles_%s.png'%lab[i])
        plt.close()
    # plt.show()    
        # print('number of imgas :',num)
        # num = str(num)
        # bins = range(min(data), max(data) + bins_interval - 1, bins_interval)
        # print(len(bins))
        # print( 'mix angle :', min(data),'max angle :',max(data) )
        # for i in range(0, len(bins)):
            # print(bins[i],'====%s==='%i)
        #plt.xlim(min(data) - margin, max(data) + margin)
        # plt.xlim(-90 , 90)
        # print(-90 - margin, 90 + margin)
        # exit()
        # plt.title("The distribution of positional angles in %s images"%num)
        # plt.xlabel('Positional Angles')
        # plt.ylabel('Numbers')
        # bins = range(-90, 90, 2)
        # 频率分布normed=True，频次分布normed=False
        # prob,left,rectangle = plt.hist(x=data, bins=bins, density=False, histtype='bar', color='steelblue',edgecolor='black')
        #for x, y in zip(left, prob):
            # 字体上边文字
            # 频率分布数据 normed=True
            #plt.text(x + bins_interval / 2, y + 0.003, '%.2f' % y, ha='center', va='top')
            # 频次分布数据 normed=False
            #plt.text(x + bins_interval / 2, y + 0.25, '%.2f' % y, ha='center', va='top')
        # plt.savefig('The_positionalangles_in_%s.jpg'%num)
        # plt.show()


def making_piltimages(palist):
    pa_list = np.array(palist)
    plt.figure(figsize=(10,5))
    #plt.rcParams['font.family'] = 'SimHei' # 显示中文
    fre_tuple = plt.hist(pa_list, bins=20, color='steelblue') # 返回值元组
    plt.title("The distribution of positional angles in the UML-dataset", fontsize=15)
    # plt.show()


def snr_make(path):
    snrs = []
    for i in tqdm(glob.glob(path+'*')):
        img = I.open(i).convert('L')
        # img_ = I.fromarray(img)
        img_ = np.array(img)
    
        mean = np.mean(img_)
        var = np.var(img_)
        snr = np.abs(10*np.log10((mean/var)))
        snrs.append(snr)
    return snrs
        
    
   
if __name__=="__main__":
    
    snsr_path = 'all_raw_img\\'
    snrs = snr_make(snsr_path)
    mins = np.min(np.array(snrs))
    maxs = np.max(np.array(snrs))
    bins = range(int(mins),int(maxs),1)
    plt.title('The distribution of the SNR in the UML-dataset')
    # plt.hist(snrs, bins=bins, color='steelblue',edgecolor='black')
    plt.hist(snrs, bins=bins, color='lightblue',edgecolor='red')
    plt.savefig('Distribution_SNR.png')
    plt.close()
    # plt.show()
    # exit()
    
    # path = r'E:\python\wangting\tiantidataset\dataset0305\dataset0326\28_28_Mk3\train\\'
    # pa_test(path)
    # pa_train(path)
    panum = count_22000()
    panum1 = count_2900()
    palist,panum2 = making_dataimags()
    # panum = count_2900()
    # probability_distribution(panum,len(panum))
    datas = [panum,panum1,panum2]
    probability_distribution_data(datas)
    exit()
    palist,panum = making_dataimags()
    
    probability_distribution(panum,len(panum))
    
    panum = count_24900()
    # print(len(panum))
    # exit()
    probability_distribution(panum,len(panum))
    
    # pa_24900data = []
    # with open("24900data.txt","r",encoding="utf-8")as e:
        # for line in e:
            # Name_1 = line.strip().split()
            # result = cal_an(int(round(float(Name_1[2]),1)))
            # pa_24900data.append(result)
            
    # probability_distribution(panum)
    # making_piltimages(palist)
