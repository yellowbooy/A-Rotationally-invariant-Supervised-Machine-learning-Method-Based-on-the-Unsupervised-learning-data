import PIL.Image as I
import numpy as np
import glob,os
  
    
def read_pix(filepath):
    img = I.open(filepath).convert('L')
    img = np.array(img)
    h = img.shape[0]
    w = img.shape[1]
    read_pixs = []
    for i in range(h):
        for j in range(w):
            read_pixs.append(img[i][j])
    
    return read_pixs
    
if __name__=="__main__":
    k = 0
    path_lst=glob.glob(r"img\cae\\*.jpg")
    path_2st = glob.glob(r"img\nocae\\*.jpg")
    with open("the_polar_images_cae.txt","w",encoding="utf-8")as f:
        for path in path_lst:
        #print(path)
        #print(type(path))
            #if (ll+1)%200==0:print(ll)
            # stds=do_one(path)
            pixs = read_pix(path)
        # diff = sum(errors)/3
        # if diff == 0:
            # k = k+1
            # print(k,'=================')
            #exit()
            # f.write("%s\t%s\t%s\t%s\t%s\n"%(path.split("\\")[-1],str(stds[0]),str(stds[1]),str(stds[2]),sum(stds)/3))
            f.write("%s\t%s\n"%(path.split("\\")[-1],pixs))
            # if float((sum(errors)/3)) < 0.1:  
                # imag = I.open(path).convert('L')
                # imag = imag.save('alltest_Mk1\\'+path.split("\\")[-1])
            k = k + 1
            print(path.split("/")[-1],'====%s====='%k)
    k = 0
    with open("the_polar_images_nocae.txt","w",encoding="utf-8")as f:
        for path in path_2st:
        #print(path)
        #print(type(path))
            #if (ll+1)%200==0:print(ll)
            pixs = read_pix(path)
        # diff = sum(errors)/3
        # if diff == 0:
            # k = k+1
            # print(k,'=================')
            #exit()
            # f.write("%s\t%s\t%s\t%s\t%s\n"%(path.split("/")[-1],str(stds[0]),str(stds[1]),str(stds[2]),sum(stds)/3))
            f.write("%s\t%s\n"%(path.split("\\")[-1],pixs))
            # if float((sum(errors)/3)) < 0.1:  
                # imag = I.open(path).convert('L')
                # imag = imag.save('alltest_Mk1\\'+path.split("\\")[-1])
            k = k + 1
            print(path.split("/")[-1],'====%s====='%k)


#import 
# f = open("name2errors.txt","r",encoding="utf-8")
# name = f.readline()
# k = 0
# while name:
    # name1 = name.strip().split()
    # name1[4] = float(name1[4])
    # if name1[4] < 0.1:
        # img=I.open('alltest\\'+name1[0]).convert("L")
        # img=img.save('min\\'+name1[0])
        # k = k + 1
        # print(k,'===k====')
    # name = f.readline()
# f.close()
# print(k,'========')        