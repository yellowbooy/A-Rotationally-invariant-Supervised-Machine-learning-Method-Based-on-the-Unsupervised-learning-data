#t-SNE使用 （天体任务）
依次运行0_select.py,1_t-SNE.py,get_name2label.py,2_plot_tsne.py即可，其中前三个程序在第一次运行后只要读取的数据不变，第二次画tsne图只需要运行2_plot_tsne.py。
文件夹data_hc15/中为参考的hc15工作的对无标签数据的分类结果
文件夹data_polar/中为极坐标数据训练后的The Googlenet模型对无标签数据的分类结果
文件夹data_raw/中为未进行极坐标处理的原图数据训练后的The Googlenet模型对无标签数据的分类结果

##1.读取三个文件夹中的图像数据，并把图片压缩成一维序列保存为img_data文件  
运行命令：
python 0_select.py    

需要按需求修改17行的数据读取地址

##2.读取数据img_data，利用sklearn的TSNE对数据进行t-SNE可视化降维,降维数据保存为img_data_tsne文件
运行命令：  
python 1_t-SNE.py  

##3.读取三个文件夹的数据，将读取的图像名与相应的标签存入namelabel文件。
运行命令：
python get_name2label.py

##4.读取img_data_tsne中的t-SNE编码数据，画出可视化降维图。数据保存在plot_result文件夹中。可以画相同数据在三种标签方法中的t-SNE数据分布
运行命令：
python 2_plot_tsne.py [部分数据需要配置，例如选多少个数据画图]
例：
python 2_plot_tsne.py 2000 可以得到随机抽取2000张图像的t-SNE数据的可视化分布图像
