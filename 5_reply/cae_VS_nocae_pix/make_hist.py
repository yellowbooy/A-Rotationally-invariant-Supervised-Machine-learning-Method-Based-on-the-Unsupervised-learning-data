# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
#font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)  
#f = open("name2errors.txt","r",encoding="utf-8")
#f = open("pickleerrors.txt","r",encoding="utf-8")the_raw_images.txt
f = open("the_polar_images_cae.txt","r",encoding="utf-8")
# f = open("test.txt","r",encoding="utf-8")
name = f.readline()
k = 0
diffname = []
filename = []
plend = []
pix_cae = []
while name:
    pix_cae = []
    name1 = name.strip().split()
    # print(len(name1),name1[1])
    # exit()
    for i in range(1,len(name1)):
        pix_cae.append(int(name1[i].split('[')[-1].split(']')[0].split(',')[0]))
    # name1[2] = int(name[2])
    # name1[4] = float(name1[4])
    # if name1[4] < 0:
        # name1[4] = -name1[4]
    diffname.append(pix_cae)
    filename.append(name1[0])
    # if name1[4] < 0.1:
        # img=I.open('alltest\\'+name1[0]).convert("L")
        # img=img.save('min\\'+name1[0])
    k = k + 1
    print(k,'===k====')
    name = f.readline()
f.close()
print(k,'========')  
#print(diffname)
#exit()
salary = diffname
plend.append('With CAE')

f = open("the_polar_images_nocae.txt","r",encoding="utf-8")
# f = open("test.txt","r",encoding="utf-8")
name = f.readline()
k = 0
diffname1 = []
filename1 = []
pix_cae1 = []
while name:
    pix_cae1 = []
    name1 = name.strip().split()
    #name1[2] = int(name[2])
    for i in range(1,len(name1)):
        pix_cae1.append(int(name1[i].split('[')[-1].split(']')[0].split(',')[0]))
    # name1[4] = float(name1[4])
    # if name1[4] < 0:
        # name1[4] = -name1[4]
    diffname1.append(pix_cae1)
    filename1.append(name1[0])
    # if name1[4] < 0.1:
        # img=I.open('alltest\\'+name1[0]).convert("L")
        # img=img.save('min\\'+name1[0])
    k = k + 1
    print(k,'===k====')
    name = f.readline()
f.close()
print(k,'========')  
#print(diffname)
#exit()
salary1 = diffname1
plend.append('Without CAE')    



# group = range(0,255,50)
# group1 = [0,20,40,60,80,100,120,140,160]
# plt.hist(group1, group, histtype='bar', rwidth=2,color='steelblue',edgecolor='black')
# plt.hist(group1, group, histtype='bar', rwidth=2,color='red',edgecolor='black')
# plt.legend(plend)

# plt.xlabel('The mean difference of pixel values between images with the polar expansion in different rotation angles')
# plt.xlabel('The mean difference of pixel values ')#between images with different rotation angles in validation set
# plt.ylabel('The number of the images')

# plt.title(u'Distribution of mean pixel differences between images rotated in different angles with and without APCT')
# plt.title(u'The histogram about the processed images in validation set')#, FontProperties=font)
# plt.xlim([0,150])
# plt.show()  

# plt.hist(salary1, 2, histtype='bar', rwidth=2,color='grey',edgecolor='black')
# plt.legend(plend)
# plt.show()

# exit()

group = range(0,255,5)
fig, axs = plt.subplots(2,5, figsize = (20,10) )
# plt.legend('noCAE')
# plt.legend('CAE')
fig.subplots_adjust(left = 0.08, bottom = 0.08, right = 0.93, top = 0.95, wspace = 0.25, hspace = 0.2)
print(len(salary1))

# for i in range(0,len(salary1)):
    # y_c = salary[i]
    # y_nc = salary1[i]
    # group = range(0,255,10)
    # group1 = range(0,255,10)
    # plt.hist(y_c, group1, histtype='bar', rwidth=2,color='steelblue',edgecolor='black')
    # plt.show()
    # plt.hist(y_nc, group, histtype='bar', rwidth=2,color='red',edgecolor='black')
    # plt.legend(plend)
    # plt.show()
    # exit()

for i in range(0,len(salary1)):
    y_c = salary[i]
    y_nc = salary1[i]
    group = range(0,255,5)
    # axs[0,0].legend('c')
    # axs[0,1].legend('nc')
    if i <= 4:
        # axs[0,0].legend('c')
        # axs[0,0].legend('nc')
        if i == 3:
            axs[0,i].hist(y_c, group, histtype='bar', rwidth=2,color='red',edgecolor='black')
            # axs[0,i].legend('cae')
            axs[0,i].hist(y_nc, group,histtype='bar', alpha=0.6,rwidth=2,color='steelblue',edgecolor='black')
            # axs[0,i].legend('nocae')
            axs[0,i].set_title( filename1[i] )
            axs[0,i].set_xlabel('The pixels of the image')
            axs[0,i].set_ylabel('The number of pixels')
        else:
            axs[0,i].hist(y_c, group, histtype='bar', rwidth=2,color='red',edgecolor='black')
        # axs[0,i].legend('cae')
            axs[0,i].hist(y_nc, group,histtype='bar', alpha=0.6,rwidth=2,color='steelblue',edgecolor='black')
            axs[0,i].set_title( filename1[i] )
            axs[0,i].set_xlabel('The pixels of the image')
            axs[0,i].set_ylabel('The number of pixels')
    else:
        # print(y_c)
        if i ==7:
            axs[1,i-5].hist(y_c, group,histtype='bar', rwidth=2,color='red',edgecolor='black')
            
        # axs[1,i-5].legend('cae')
            axs[1,i-5].hist(y_nc, group,histtype='bar', alpha=0.6,rwidth=2,color='steelblue',edgecolor='black')
            axs[1,i-5].set_title( filename1[i] )
            axs[1,i-5].set_xlabel('The pixels of the image')
            axs[1,i-5].set_ylabel('The number of pixels')
        # axs[1,i-5].legend('nocae')
        else:
            axs[1,i-5].hist(y_c, group,histtype='bar', rwidth=2,color='red',edgecolor='black')
        # axs[1,i-5].legend('cae')
            axs[1,i-5].hist(y_nc, group,histtype='bar', alpha=0.6,rwidth=2,color='steelblue',edgecolor='black')
            axs[1,i-5].set_title( filename1[i] )
            axs[1,i-5].set_xlabel('The pixels of the image')
            axs[1,i-5].set_ylabel('The number of pixels')
    
    axs[0,4].legend(['Denoised by CAE','Raw'])
    # axs[0,4].legend('nocae')
    
    # axs[i] = plt.
    # plt.savefig('his_%s.png'%(i),dpi=600)

# plt.legend('noCAE')
plt.show()

exit()
# group = [-1,0,1,2,3,4,5,6,7,8,9,10]#,11,12,13,14,15,16,17,18,19,20]
# group = [90,95,100,105,110,115,120,125,130,135]
group = range(0,30,2)
# group = [0,20,40,60,80,100,120,140,160]
plt.hist(salary, group, histtype='bar', rwidth=2,color='steelblue',edgecolor='black')
# plt.hist(salary1, group, histtype='bar', rwidth=2,color='grey',edgecolor='black')
plt.legend('With CAE')

#plt.xlabel('The mean difference of pixel values between images with the polar expansion in different rotation angles')
# plt.xlabel('The mean difference of pixel values ')#between images with different rotation angles in validation set
# plt.ylabel('The number of the images')

# plt.title(u'Distribution of mean pixel differences between images rotated in different angles with and without APCT')
# plt.title(u'The histogram about the processed images in validation set')#, FontProperties=font)
# plt.xlim([0,150])
plt.show()  

plt.hist(salary1, group, histtype='bar', rwidth=2,color='grey',edgecolor='black')
plt.legend('Without CAE')
plt.show()