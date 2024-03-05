# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
#font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)  
#f = open("name2errors.txt","r",encoding="utf-8")
#f = open("pickleerrors.txt","r",encoding="utf-8")the_raw_images.txt
f = open("the_raw_images.txt","r",encoding="utf-8")
# f = open("test.txt","r",encoding="utf-8")
name = f.readline()
k = 0
diffname = []
plend = []
while name:
    name1 = name.strip().split()
    name1[4] = float(name1[4])
    if name1[4] < 0:
        name1[4] = -name1[4]
    diffname.append(name1[4])
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
plend.append('Without APCT')

f = open("the_polar_images.txt","r",encoding="utf-8")
# f = open("test.txt","r",encoding="utf-8")
name = f.readline()
k = 0
diffname1 = []
while name:
    name1 = name.strip().split()
    name1[4] = float(name1[4])
    if name1[4] < 0:
        name1[4] = -name1[4]
    diffname1.append(name1[4])
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
plend.append('With APCT')    

# group = [-1,0,1,2,3,4,5,6,7,8,9,10]#,11,12,13,14,15,16,17,18,19,20]
# group = [90,95,100,105,110,115,120,125,130,135]
group = range(0,140,5)
# group = [0,20,40,60,80,100,120,140,160]
plt.hist(salary, group, histtype='bar', rwidth=2,color='steelblue',edgecolor='black')
plt.hist(salary1, group, histtype='bar', rwidth=2,color='grey',edgecolor='black')
plt.legend(plend)

#plt.xlabel('The mean difference of pixel values between images with the polar expansion in different rotation angles')
plt.xlabel('The mean difference of pixel values ')#between images with different rotation angles in validation set
plt.ylabel('The number of the images')

plt.title(u'Distribution of mean pixel differences between images rotated in different angles with and without APCT')
# plt.title(u'The histogram about the processed images in validation set')#, FontProperties=font)
# plt.xlim([0,150])

plt.savefig('results.png',bbox_inches = 'tight',pad_inches = .1)  
plt.show()