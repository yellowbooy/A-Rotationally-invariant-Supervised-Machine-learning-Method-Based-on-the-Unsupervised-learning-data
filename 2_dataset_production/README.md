数据集划分说明：
	极坐标处理见1_cae_class\0_cae_noise\polar.py
	1.extracting_images.py：
	功能：对2_cae_class/产生的数据集进行随机划分，即随机抽取2900张图像组成验证集，剩余图像组成训练集。完成处理后会输出至自动创建的dataset/文件夹中，其中，图片名的首位为类别(0:UNC,1:IRR,2:LTD,3:ETD,4:SPH)。
	运行命令：python extracting_images.py