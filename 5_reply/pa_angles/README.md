0   文件：
	代码调用文本：
	1661data.txt：1661个源的名称，类别，角度
	24900data.txt：有标签数据集中24900个源的名称，类别，角度
	22000data.txt：有标签数据集中训练集22000个源的名称，类别，角度
	2900data.txt：有标签数据集中验证集2900个源的名称，类别，角度

1	执行命令：
		python finding_angles.py
		
	代码：
	finding_angles.py: 生成47149源的信噪比直方图、22000源PA角直方图、2900源PA角直方图、1661源PA角直方图



原始文本：
name2label2angle.txt：全部47149个源的数据：
# ml_label 去绝对值后是我们uml+sml的结果, 0, 1, 2, 3, 4 对应UNC, IRR, LTD, ETD, SPH 
# 通过sml = 0 选出之前UML的工作的数据（24900个源） 
# 通过sml = 1 选出现在SML的工作的数据（22249个源）
train_name_lst：1661个源的名单