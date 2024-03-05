import PIL.Image as I
import numpy as np
import glob,os
import matplotlib.pyplot as plt



#读取八万步的验证集准确率
y = {}
for log in ['acc\\raw2','acc\\raw5','acc\\raw8','acc\\polar','acc\\raw','acc\\raw20']:
    names_all=glob.glob(log+'\\*')
    y[log]={'0':{},'80':{},'90':{},'180':{},'270':{}}
    for name in names_all:
        num = name.split('\\')[-1].split('_')[-1]
        filenames = glob.glob(name+'\\*')
        y[log][num] = {'acc':[],'steps':[]}
        for i in filenames:
            with open(i,'r') as f:
                for names in f:
                    line = names.strip().split()
                    
                    if int(line[0]) == 80000:
                        y[log][num]['steps'].append(int(line[0]))
                        y[log][num]['acc'].append(float(line[1]))
                        
                        
#读取原图转两次，五次，八次数据的准确率数据                        
acc = []
for name1,name2 in y.items():
    for name3,name4 in name2.items():
        print(name1,'===',name3,'===',name4)
        # print(y[name1][name3][name4]['acc'])
        acc.append(y[name1][name3]['acc'])
        
# print(len(acc))
# exit()        
#读取原图转两次的准确率，并根据三次数据计算均值和标准差 
acc_1=[]
acc_1_mean = []
acc_1_std = []
print(len(acc))        
for i in range(0,len(acc)-25):        
    acc_1.append(acc[i])
acc_1 = np.array(acc_1)
acc_1 = acc_1.transpose()
acc_1 = acc_1.transpose()

for its in acc_1:
    acc_1_mean.append(round(np.mean(its),4))
    acc_1_std.append(round(np.std(its),4))
print(acc_1_mean)
print(acc_1_std)    

#读取原图转五次的准确率，并根据三次数据计算均值和标准差
acc_2=[]
acc_2_mean = []
acc_2_std = []
print(len(acc))        
for i in range(5,len(acc)-20):        
    acc_2.append(acc[i])
acc_2 = np.array(acc_2)
acc_2 = acc_2.transpose()
acc_2 = acc_2.transpose()
for its in acc_2:
    acc_2_mean.append(round(np.mean(its),4))
    acc_2_std.append(round(np.std(its),4))
print(acc_2_mean)
print(acc_2_std) 

#读取原图转八次的准确率，并根据两次数据计算均值和标准差
acc_3=[]
acc_3_mean = []
acc_3_std = []
print(len(acc))        
for i in range(10,len(acc)-15):        
    acc_3.append(acc[i])
acc_3 = np.array(acc_3)
acc_3 = acc_3.transpose()
acc_3 = acc_3.transpose()
for its in acc_3:
    acc_3_mean.append(round(np.mean(its),4))
    acc_3_std.append(round(np.std(its),4))
print(acc_3_mean)
print(acc_3_std) 

#读取极坐标的准确率，并根据三次数据计算均值和标准差
acc_4=[]
acc_4_mean = []
acc_4_std = []
print(len(acc))        
for i in range(15,len(acc)-10):        
    acc_4.append(acc[i])
acc_4 = np.array(acc_4)
acc_4 = acc_4.transpose()
acc_4 = acc_4.transpose()
for its in acc_4:
    acc_4_mean.append(round(np.mean(its),4))
    acc_4_std.append(round(np.std(its),4))
print(acc_4_mean)
print(acc_4_std)



#读取原图的准确率，并根据三次数据计算均值和标准差
acc_5=[]
acc_5_mean = []
acc_5_std = []
print(len(acc))        
for i in range(20,len(acc)-5):        
    acc_5.append(acc[i])
acc_5 = np.array(acc_5)
acc_5 = acc_5.transpose()
acc_5 = acc_5.transpose()
for its in acc_5:
    acc_5_mean.append(round(np.mean(its),4))
    acc_5_std.append(round(np.std(its),4))
print(acc_5_mean)
print(acc_5_std)
    
#读取原图转20次的准确率，并根据三次数据计算均值和标准差
acc_6=[]
acc_6_mean = []
acc_6_std = []
print(len(acc))        
for i in range(25,len(acc)):        
    acc_6.append(acc[i])
acc_6 = np.array(acc_6)
acc_6 = acc_6.transpose()
acc_6 = acc_6.transpose()
for its in acc_6:
    acc_6_mean.append(round(np.mean(its),4))
    acc_6_std.append(round(np.std(its),4))
print(acc_6_mean)
print(acc_6_std)

#记录画图的图例、横坐标、纵坐标、标准差值
legend=[]
Y={'1':{},'2':{},'3':{},'4':{},'5':{},'6':{}}
for datalist in [acc_3_mean,acc_2_mean,acc_1_mean,acc_4_mean,acc_5_mean,acc_6_mean]:
    if datalist == acc_3_mean:
        #这里设定横坐标是为了改善图片显示，但还没找到显示指定横坐标的方法，所以画出来效果不太好
        x3 = [0,50,100,150,200]
        y3 = acc_3_mean
        std3 = acc_3_std
        Y['3']={'x':x3,'y':y3,'std':std3,'lab':'raw_8'}
    elif datalist == acc_2_mean:
        x2 = [0,50,100,150,200]
        y2 = acc_2_mean
        std2 = acc_2_std
        Y['2']={'x':x2,'y':y2,'std':std2,'lab':'raw_5'}        
    elif datalist == acc_1_mean:
        x1 = [0,50,100,150,200]
        y1 = acc_1_mean
        std1 = acc_1_std
        Y['1']={'x':x1,'y':y1,'std':std1,'lab':'raw_2'}
    elif datalist == acc_4_mean:
        x4 = [0,50,100,150,200]
        y4 = acc_4_mean
        std4 = acc_4_std
        Y['4']={'x':x4,'y':y4,'std':std4,'lab':'polar'}
    elif datalist == acc_5_mean:
        x5 = [0,50,100,150,200]
        y5 = acc_5_mean
        std5 = acc_5_std
        Y['5']={'x':x5,'y':y5,'std':std5,'lab':'raw'}
    elif datalist == acc_6_mean:
        x6 = [0,50,100,150,200]
        y6 = acc_6_mean
        std6 = acc_6_std
        Y['6']={'x':x6,'y':y6,'std':std6,'lab':'raw20'}
for i in range(1,7):
    print(i)
    i = str(i)
    legend.append(Y[i]['lab'])

#画图
plt.figure(figsize=(17,10))
for i in range(1,7):
    i = str(i)
    plt.errorbar(Y[i]['x'],Y[i]['y'],marker='o',yerr=Y[i]['std'])
    # plt.plot
    plt.title('Results of polar,raw,raw_2,raw_5,raw_8')
plt.legend(legend)
plt.savefig('erroebar_acc_8w.jpg')
# plt.show()