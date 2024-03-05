日志文件夹介绍：
	log\：各个数据的原始日志文件，里面包含了每十步的训练集准确率变化、每100步的各旋转角度验证集准确率变化、每epoch的各旋转角度验证集准确率变化（提取的日志文件是train_log4.txt）。
	acc\:  对各个数据原始日志进行提取处理后的文件，里面包含了每100步的各旋转角度验证集准确率变化。
代码作用介绍：
	errorbar_in_the_8w_step.py：对acc\中的日志进行筛查，计算出第80000步的各旋转角度验证集准确率的均值和标准差，并生成误差棒图。
	运行命令： python errorbar_in_the_8w_step.py

	make_acc_data.py：对log\中的原始日志文件进行提取，输出的结果是acc\中的文件。
	运行命令： python make_acc_data.py

	comparison_in_8w_steps.py：对acc\中的日志进行提取，计算80000步内的各个旋转角度的验证集准确率均值，同时生成对比图。
	运行命令： python comparison_in_8w_steps.py