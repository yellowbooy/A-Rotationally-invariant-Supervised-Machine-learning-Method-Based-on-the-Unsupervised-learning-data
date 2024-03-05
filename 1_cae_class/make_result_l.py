import os,glob
from tqdm import tqdm

path = 'result_final_50/*'
all_dir = glob.glob(path)

with open('class_result_50.txt','a',encoding='utf-8') as f:
    f.write('%s\t%s\n'%('class','name'))

for i in tqdm(all_dir):
    class_ = i.split('/')[-1]
    classes = os.listdir(i)
    for i_ in classes:
        with open('class_result_50.txt','a',encoding='utf-8') as f:
            f.write('%s\t%s\n'%(class_,i_))
            #exit()
    
