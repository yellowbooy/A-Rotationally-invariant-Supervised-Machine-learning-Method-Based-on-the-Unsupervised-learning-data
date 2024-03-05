import glob

#设置要读取的三个文件夹地址
file_names=["data_raw"]
name2label={}

#如使用的是linux系统，需要将"\\"改为"/"
for file_ in file_names:
    label_lst=glob.glob(file_+"/*")
    for label in label_lst:
        the_label=label.split("/")[-1]
        img_lst=glob.glob(label+"/*")
        for img in img_lst:
            name=img.split("/")[-1]
            if name not in name2label:
                name2label[name]={}
            name2label[name][file_]=the_label

#将读取到的结果输入进namelabel文件中
with open("namelabel","w",encoding="utf-8")as f:
    for name,labels in name2label.items():
        raw=labels["data_raw"]
        f.write("%s\t%s\n"%(name,raw))


