import PIL.Image as I
import numpy as np
import glob,os
PI=3.141592653
def get_pos(arr):
    x,y=arr.shape
    center_x,center_y=int(x/2),int(y/2)
    max_=np.max(arr)
    min_=np.min(arr)
    index=np.where(arr==max_)
    x_lst,y_lst=index
    ret=[[(x-center_x)**2+(y-center_y)**2,x,y]for x,y in zip(x_lst,y_lst)]
    ret.sort()
    max_x,max_y=ret[0][1:]
    index=np.where(arr==min_)
    x_lst,y_lst=index
    ret=[[(x-center_x)**2+(y-center_y)**2,x,y]for x,y in zip(x_lst,y_lst)]
    ret.sort()
    min_x,min_y=ret[0][1:]
    return max_x,max_y,min_x,min_y

def give_angle(x,y):
    try:
        T=np.arctan(abs(x/y))
    except:
        if x >0:
            return PI/2
        else:
            return 3/2*PI
    if x>=0 and y >=0:
        return T
    if x >=0 and y<=0:
        return PI-T
    if x<=0 and y>=0:
        return -T
    if x<=0 and y<=0:
        return PI+T
        
def mirror(arr):
    arr=np.fliplr(arr)
    x,y=arr.shape
    ret=np.zeros([x,2*y])  
    for yy in range(y):
        ret[:,yy]=arr[:,yy]        
        ret[:,(2*y-1)-yy]=arr[:,yy]

    return ret

def save_arr(arr,name,path):
    if not os.path.exists(path):
        os.makedirs(path)
    img=I.fromarray(np.uint8(arr))
    img.save(+path+name)
    
def show_arr(arr):
    img=I.fromarray(np.uint8(arr))
    img.show()
    
def polar(arr,N=10,angle_inter=0.05):
    max_x,max_y,min_x,min_y=get_pos(arr)
    img=I.fromarray(np.uint8(arr))
    raw_size=arr.shape
    raw_polar_size=[int(raw_size[0]/2*2),int(2*raw_size[0]*PI)]
    new_size=[raw_size[0]*N,raw_size[1]*N]
    X=int(new_size[0]/2)
    Y=int(2 * PI/angle_inter)
    img_resize=img.resize(new_size,resample=I.NEAREST)
    img_resize_arr=np.array(img_resize)
    max_x,max_y,min_x,min_y=max_x*N+N-1,max_y*N+N-1,min_x*N+N-1,min_y*N+N-1
    base_angle=[min_x-max_x,min_y-max_y]
    T=give_angle(base_angle[0],base_angle[1])
    polar_img = np.zeros((Y, X), dtype=np.uint8)
    for y in range(Y):
        the_angle=y*angle_inter 
        for x in range(X):
            that_y=int(x*np.cos(T+the_angle)+max_y)
            that_x=int(x*np.sin(T+the_angle)+max_x)
            try:
                polar_img[y,x]=img_resize_arr[that_x,that_y]
            except Exception as e:
                continue
    polar_img=mirror(polar_img)
    polar_img_=I.fromarray(polar_img)
    #polar_img_=polar_img_.resize(raw_polar_size,resample=I.NEAREST)
    return np.array(polar_img_)


def do_raw(path,angle=0,n=2):
    head=path.split("\\")[-1]#.split(".")[0]
    img=I.open(path).convert("L")
    img_rotate_r=img.rotate(angle)
    img_rotate_r=np.array(img_rotate_r)
    #save_arr(img_rotate_r,head+"-rotate_%s.png"%angle)
    p_1 = img_rotate_r
    #p_1=polar(img_rotate_r,N=n)
    #save_arr(p_1,head,"175_Mk2_Block1/test/")
    
    #errors=[]
    stds = []
    for an in [90,180,270]:
        img_rotate_r=img.rotate(an)
        img_rotate_r=np.array(img_rotate_r)
        p_2 = img_rotate_r
        #p_2=polar(img_rotate_r,N=n)
        #error=np.mean(p_2-p_1)
        std=np.std(p_2-p_1)
        #errors.append(error)
        stds.append(std)
        #save_arr(p_1,head,"175_Mk2_Block1/test_%s/"%an)
    # print(errors)
    # print(stds)
    # exit()
    return stds
    
def do_polar(path,angle=0,n=2):
    head=path.split("\\")[-1]#.split(".")[0]
    img=I.open(path).convert("L")
    img_rotate_r=img.rotate(angle)
    img_rotate_r=np.array(img_rotate_r)
    #save_arr(img_rotate_r,head+"-rotate_%s.png"%angle)
    p_1 = img_rotate_r
    p_1=polar(img_rotate_r,N=n)
    #save_arr(p_1,head,"175_Mk2_Block1/test/")
    
    #errors=[]
    stds = []
    for an in [90,180,270]:
        img_rotate_r=img.rotate(an)
        img_rotate_r=np.array(img_rotate_r)
        p_2 = img_rotate_r
        p_2=polar(img_rotate_r,N=n)
        #error=np.mean(p_2-p_1)
        std=np.std(p_2-p_1)
        #errors.append(error)
        stds.append(std)
        #save_arr(p_1,head,"175_Mk2_Block1/test_%s/"%an)
    # print(errors)
    # print(stds)
    # exit()
    return stds

if __name__=="__main__":
    k = 0
    #读取的文件地址
    path_lst=glob.glob(r"28_Mk2/test/*")
    path_2st = glob.glob(r"28_Mk2/train/*")
    path_st = path_lst+path_2st
    with open("the_polar_images.txt","w",encoding="utf-8")as f:
        for path in path_st:
        #print(path)
        #print(type(path))
            #if (ll+1)%200==0:print(ll)
            stds=do_polar(path)
            #print(stds)
            #exit()
        # diff = sum(errors)/3
        # if diff == 0:
            # k = k+1
            # print(k,'=================')
            #exit()
            f.write("%s\t%s\t%s\t%s\t%s\n"%(path.split("/")[-1],str(stds[0]),str(stds[1]),str(stds[2]),sum(stds)/3))
            # if float((sum(errors)/3)) < 0.1:  
                # imag = I.open(path).convert('L')
                # imag = imag.save('alltest_Mk1\\'+path.split("\\")[-1])
            k = k + 1
            print(path.split("/")[-1],'====%s====='%k)

    with open("the_raw_images.txt","w",encoding="utf-8")as f:
        for path in path_st:
        #print(path)
        #print(type(path))
            #if (ll+1)%200==0:print(ll)
            stds=do_raw(path)
            #print(stds)
            #exit()
        # diff = sum(errors)/3
        # if diff == 0:
            # k = k+1
            # print(k,'=================')
            #exit()
            f.write("%s\t%s\t%s\t%s\t%s\n"%(path.split("/")[-1],str(stds[0]),str(stds[1]),str(stds[2]),sum(stds)/3))
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