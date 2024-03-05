import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle
# import config as C
import time
import sklearn
print(sklearn.__version__)
def load_data(filename):
    name,data=[],[]
    for line in open(filename,'r',encoding="utf-8"):
    # for line in open("img_data_img",'r',encoding="utf-8"):
        l,d=line.strip().split("\t")
        
        try:
            d=[float(x)for x in d.split("<=>")]
        except:
        
            d=[int(x)for x in d.split("<=>")]
        name.append(l)
        data.append(d)
        #data = pickle.load(f_data)
    return name,data
    
if __name__=="__main__":
    
    #for j in [['img_data_conv','conv'],["img_data_ViT",'ViT'],["img_data_img",'img'],['img_data_compare','compare']]:
    for j in [["img_data_img",'img']]:
        i = j[1]
        filename = j[0]
        name,data= load_data(filename)
        print("开始%s编码图像的TSNE\n"%i)
        st = time.time()
        #T-SNE
        tsne = TSNE(n_components=2, init='pca',random_state=0,n_jobs=-1)
        # print(data)
        # exit()
        result = tsne.fit_transform(data)
        with open("%s_tsne"%filename,"w",encoding="utf-8")as f:
        # with open("img_data_tsne","w",encoding="utf-8")as f:
        # with open("img_data_tsne_img","w",encoding="utf-8")as f:
            for n,d in zip(name,result):
                f.write("%s\t%s\n"%(n,"<=>".join(str(x)for x in d)))
        ed= time.time()
        print("完成%s图像的TSNE,总计耗时 %f.4 s \n"%(i,ed-st))
