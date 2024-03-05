import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle
import matplotlib
import random,os,glob
import numpy as np
# import config as C
import sys
from matplotlib import rcParams
matplotlib.use('Agg')

config = {
            #"font.family":'Times New Roman',  
            #"font.size": 80,
            "mathtext.fontset":'stix',
            }
rcParams.update(config)


#读取tsne数据
def random_choose(lst,N):
    index=[i for i in range(len(lst))]
    choice_index=np.random.choice(index,N,replace=False)
    ret=[]
    for ind in choice_index:
        ret.append(lst[ind])
    return ret

def load_name2label():
    name2label={}
    names = []
    for line in open("namelabel","r",encoding="utf-8"):
        # print(line)
        try:
            name,raw =line.strip().split("\t")
            # print(line)
            name2label[name]={"raw":raw}#,"raw1":raw1,"raw2":raw2,"raw3":raw3,"raw4":raw4,"raw5":raw5,"raw6":raw6,"raw7":raw7,"raw8":raw8}
            names.append(name)
        except:
            pass
    return name2label,names


def load_data(N,filename):
    name2label,names=load_name2label()
    # print(names)
    all_tsne=[]
    all_raw=[]
    # for line in open("img_data_tsne",'r',encoding="utf-8"):
    for line in open("%s_tsne"%filename,'r',encoding="utf-8"):
        name,tsne = line.strip().split("\t")
        tsne=[float(x)for x in tsne.split("<=>")]
        #name=name.split(".")[0]
        # polar=name2label[name]["polar"]
        if name in names:
            # print(name,raw)
            raw=name2label[name]["raw"]
            # raw_1 = name2label[name]["raw1"]
            # raw_2 = name2label[name]["raw2"]
            # raw_3 = name2label[name]["raw3"]
            # raw_4 = name2label[name]["raw4"]
            # raw_5 = name2label[name]["raw5"]
            # raw_6 = name2label[name]["raw6"]
            # raw_7 = name2label[name]["raw7"]
            # raw_8 = name2label[name]["raw8"]
            # hc15=name2label[name]["hc15"]
            # if hc15=="NO":continue
            all_tsne.append(tsne)
            # all_polar.append(polar)
            all_raw.append(raw)
            # all_hc15.append(hc15)
    index=[i for i in range(len(all_tsne))]
    print(len(index))
    choice_index=np.random.choice(index,N,replace=False)
    
    all_tsne_final=[]
    # all_polar_final=[]
    all_raw_final=[]
    # all_hc15_final=[]
    for index in choice_index:
        all_tsne_final.append(all_tsne[index])
        # all_polar_final.append(all_polar[index])
        all_raw_final.append(all_raw[index])
        # all_hc15_final.append(all_hc15[index])

    return all_tsne_final,all_raw_final

#排序函数，使得图例的顺序按SPH，ETD，LTD，IRR，UNC顺序排列    
def chooselist(all_final,data,n=150):
    change = {}
    changedata = {}
    for index in range(n):
        #print(all_final[index])
          
        if int(all_final[index]) == 4 and all_final[0] != 4:
            change[index] = all_final[0]
            all_final[0] = all_final[index]
            all_final[index] = change[index]

            changedata[index] = data[0]
            data[0] = data[index]
            data[index] = changedata[index]

        if int(all_final[index]) == 3 and all_final[1] != 3:
            change[index] = all_final[1]
            all_final[1] = all_final[index]
            all_final[index] = change[index]

            changedata[index] = data[1]
            data[1] = data[index]
            data[index] = changedata[index]


        if int(all_final[index]) == 2 and all_final[2] != 2:
            change[index] = all_final[2]
            all_final[2] = all_final[index]
            all_final[index] = change[index]

            changedata[index] = data[2]
            data[2] = data[index]
            data[index] = changedata[index]

        if int(all_final[index]) == 1 and all_final[3] != 1:
            change[index] = all_final[3]
            all_final[3] = all_final[index]
            all_final[index] = change[index]

            changedata[index] = data[3]
            data[3] = data[index]
            data[index] = changedata[index]
        if int(all_final[index]) == 0 and all_final[4] != 0:
            change[index] = all_final[4]
            all_final[4] = all_final[index]
            all_final[index] = change[index]

            changedata[index] = data[4]
            data[4] = data[index]
            data[index] = changedata[index]

    return all_final,data


def plot_embedding(data, label,color_lst,labs,label_name,index=None):
    fig = plt.figure()
    #label,data = chooselist(label,data)
    # print(label)
    
    max_=max(label)
    min_=min(label)
    
    label_num=max_-min_+1
    # print((label))
    label1 = list(label)
    label1.sort()
    l1 = np.array(label1)
    l2 = set(l1)
    # print((l2))
    # print(l1)
    # l1 = (list(set(label.sort())))
    # print(set(l2))
    l2 = list(set(l2))
    l2.sort()
    label_num = len(l2)
    print((l2),'===')
    print(label_num,labs)
    color_list=['red','blue','green','cyan','green','blue','grey','red','purple','pink','orange','darkviolet','yellow','gold','lightgray','gray']
    maker_list=['<','.','^','1','*','+','3','4','p','h','d','>','o']
    color_name = []
    maker_name = []
    for i in range(0,labs):
        color_name.append([names[i],color_list[i]])
        maker_name.append([names[i],maker_list[i]])
    # print(color_name,'===',maker_name)
    # exit()
    # label_num = 9
    if label_num==labs or label_num==labs+1:
        # color_dict={0:'blue',1:'green',2:'red'}
        color_dict = {}
        maker_dict = {}
        for i in color_name:
            color_dict.update({int(i[0]):i[1]})
        for i in maker_name:
            maker_dict.update({int(i[0]):i[1]})
        # print(color_dict,'===',maker_dict)
        # print()
        # exit()
        # color_dict={0:'blue',1:'red',2:'green',3:'pink',4:'orange',5:'darkviolet',6:'yellow',7:'gold',8:'lightgray',9:'purple'}#,10:'gray'}
        # maker_dict={0:'+',1:'*',2:'h',3:'3',4:'.',5:'1',6:'<',7:'p',8:'d',9:'>'}#,10:'o'}
        flag=[0]*100
        print(flag)
    print(flag)
    for data,index in zip(data,label):
        #print(flag)
        #print(data)
        #print(index)
        #print(flag[index])
        #exit(#)
        if flag[index]==0:
            plt.scatter(data[0],data[1],color=color_dict[index],marker=maker_dict[index],label=label2label[index],s=11)
            flag[index]+=1
        else:
            plt.scatter(data[0],data[1],color=color_dict[index],marker=maker_dict[index],s=11)
    
    plt.xticks([])
    plt.yticks([])
    #plend = ['train','val','test']#,'IRR','UNC']
    #plt.title('t-SNE %s performance'%(label_name))
    plt.legend(loc='upper left', frameon=False,ncol=2)
    #plt.legend(loc='upper left', frameon=False,ncol=2)
    
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.9, left=0.1, hspace=0.9, wspace=0.1)
    # plt.legend(loc='lower right', frameon=False,ncol=2)
    # plt.legend().set_location('upper left')
    # plt.legend(['Line'], loc='upper right')
    # plt.gca().get_legend().set_frame_on(True)
    # plt.gca().get_legend().set_bbox_to_anchor((5, 5))
    # print(plt.gca().get_legend().set_frame_on())
    # plt.gca().get_legend().set_fontsize(20)
    os.makedirs('plot_result/',exist_ok=True)
    plt.savefig('plot_result/t_SNE_%s.png'%(label_name),dpi=800)
    plt.close()

if __name__=="__main__":
    plt.rcParams['font.sans-serif']=['stix']
    # label2label={2:'real_dog',1:'real_cat',0:'generate_cat'}
    path = 'data_raw'
    alls = 0
    label2label = {}
    names = os.listdir(path)
    for i in glob.glob(path+'/*'):
        alls+=len(glob.glob(i+'/*'))
    # print(names)
    for i in names:
        label2label.update({int(i):str(i)})
    # print(label2label)
    # exit()
    # label2label={0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9'}#,10:'rest'}
    #随机读取的文件数
    # N=int(sys.argv[1])
    N = int(input('please check nums (all files has %d imgs)'%alls))
    labs= int(input('please check labels:'))
        
    #for j in [['img_data_cae','cae']]:
    for j in [['img_data_img','img']]:#,["img_data",'ViT']]:
    #for j in [['conv_tsne','conv']["conv_compare_tsne",'conv_compare']:
        filename = j[0]
        i = j[1]
        raw_tsne_data,all_raw_final = load_data(N,filename)
        color_lst = list(matplotlib.colors.cnames.keys()) #获取matplotlib颜色列表
        
        raw_tsne_data=np.array(raw_tsne_data)
        max_,min_ = np.max(raw_tsne_data,0),np.min(raw_tsne_data,0)
        data = (raw_tsne_data - min_)/(max_ - min_)
        
        # label=np.array([int(x)for x in all_polar_final])
        # plot_embedding(data, label, color_lst, 'the_result_polar_%s'%N)
        label=np.array([int(x)for x in all_raw_final])
        plot_embedding(data, label, color_lst, labs,'the_result_raw_%s_%s'%(N,i))
        # label=np.array([int(x)for x in all_hc15_final])
        # plot_embedding(data, label, color_lst, 'the_result_hc15_%s'%N)
