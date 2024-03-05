程序运行环境：
		- Anconda3(用来安装以下环境)
		- python3.6以上
		- pytorchGPU版本1.6或1.7
		- tensorboard 2.2.2

各部分文件说明如下：
			dataset/:包含了五个模型文件的数据集
		    googlenet_polar/：极坐标数据的有监督模型文件。
			googlenet_raw/: 原始数据的有监督模型文件。
			googlenet_raw_2/: 原始数据随机增强两个角度的有监督模型文件。
			googlenet_raw_5/: 原始数据随机增强五个角度的有监督模型文件。
			googlenet_raw_8/: 原始数据随机增强八个角度的有监督模型文件。
			googlenet_raw_20/: 原始数据以20°为单位增强的有监督模型文件。
			googlenet_raw_30/: 原始数据以30°为单位增强的有监督模型文件

三模型详细数据.docx和三模型详细数据.pdf记录了三个模型的详细测试数据。