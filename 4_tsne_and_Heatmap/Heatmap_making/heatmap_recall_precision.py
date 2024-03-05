import numpy as np
import PIL.Image as I 



#画precision的数据
attention = np.array( [[0. ,0,0,1,50], [0,1,34,1090,3], [0,27,399,28,0], [7,424,62,4,0], [740,28,1,1,0]] )
#画recall时需要把这一行注释掉
attention = attention.transpose()
densenet  = np.array( [[0. ,0,0,5,47], [0,0,37,1098,6], [0,18,403,17,0], [5,445,54,3,0], [742,17,2,1,0]] )
#画recall时需要把这一行注释掉
densenet = densenet.transpose() 
googlenet = np.array( [[0. ,0,0,1,52], [0,1,29,1105,1], [0,28,425,17,0], [9,439,42,0,0], [738,12,0,1,0]] )
#画recall时需要把这一行注释掉
googlenet = googlenet.transpose()

import matplotlib.pyplot as plt
import numpy as np 
fig, axs = plt.subplots(1,3, figsize = (13,4) )  #生成一个一行三列的图像，总大小1300x400py


fig.subplots_adjust(left = 0.08, bottom = 0.08, right = 0.93, top = 0.95, wspace = 0.25, hspace = 0.2)#确定各个图片四个方位的留白距离
matrixs = [  attention, densenet, googlenet] #三个模型的各小类详细数据
names   = [ 'Attention', 'Densenet', 'Googlenet'] #三个图片的标题

name = 0
Name = []
for kk  in range(3): 
    ax  = axs[kk]
   
    matrix = matrixs[kk]
    
    #计算各分类的图片总数
    sum_ax0 = np.sum(matrix, axis = 0 ) 

    #计算准确率
    for ii in range(5):


        matrix[:,ii]=np.round((matrix[:,ii]/sum_ax0[ii]),3)
        

        sum_acc = matrix[:,ii]

    for ii in range(5):

        if float(sum(matrix[:,ii])) != 1.00:
            #print('======')
            matrix[4-ii,ii] = matrix[4-ii,ii] + (1 - np.round(sum(matrix[:,ii]),3))

    matrix[matrix==0] = np.inf
    
    #画recall时需要把这一行注释掉
    matrix=matrix.transpose()
    
    sc = ax.imshow( matrix.T, origin = 'upper', cmap = plt.get_cmap('rainbow'), vmin  = 0, vmax = 1 ) 

    for ii in range(matrix.shape[0]): 
        for jj in range(matrix.shape[1]): 
            if matrix[ii,jj] == np.inf: continue 
            #将准确率转化为百分数形式
            ax.text(ii, jj, '%.1f'%(100*matrix[ii,jj]) + '%', va = 'center', ha = 'center', fontsize = 12 )
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    ax1 = make_axes_locatable(ax);
    cax = ax1.append_axes("right", size="5%", pad="6%")
    if kk <= 1: cax.axis('off')
    if kk == 2: 
        cb1 =  plt.colorbar(sc, cax=cax) #, orientation='vertical')
        cb1.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        cb1.set_ticklabels( [0.0, 20.0, 40.0, 60.0, 80.0, 100.0] )
        cb1.set_label('Precent [%]')
    ax.set_xticks( [0, 1, 2, 3, 4] ) 
    ax.set_yticks( [0, 1, 2, 3, 4] )
    xticklabels = ['SPH','ETD','LTD','IRR','UNC',] 
    yticklabels = ['UNC','IRR','LTD','ETD','SPH'] 
    ax.set_xticklabels( xticklabels ); ax.set_xlabel('Real labels')   
    ax.set_yticklabels( yticklabels ); ax.set_ylabel('Precdict labels')   
    ax.set_title( names[kk] )
    name = 0
    Name = 0
    if kk == 0: ax.text(0.04, 0.88, 'a)', transform = ax.transAxes, fontsize = 10 )
    if kk == 1: ax.text(0.04, 0.88, 'b)',transform = ax.transAxes, fontsize = 10 )
    if kk == 2: ax.text(0.04, 0.88,  'c)',transform = ax.transAxes, fontsize = 10 )
plt.savefig('sml_model.pdf') 
plt.show()
