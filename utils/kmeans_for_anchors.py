import glob
import xml.etree.ElementTree as ET
import numpy as np

'''
此文件单独运行即可,需要修改xml载入路径和anchors保存路径
'''

def get_data(path):
    res=[]
    #对于每一个xml找出其所有box
    for each_xml in glob.glob('{}/*xml'.format(path)):
        tree=ET.parse(each_xml)
        width=int(tree.findtext('./size/width'))
        height=int(tree.findtext('./size/height'))
        if height<=0 or width<=0:
            continue

        for obj in tree.iter('object'):
            #将box坐标转为0~1百分比
            xmin=int(float(obj.findtext('bndbox/xmin')))/width
            ymin=int(float(obj.findtext('bndbox/ymin')))/height
            xmax=int(float(obj.findtext('bndbox/xmax')))/width
            ymax=int(float(obj.findtext('bndbox/ymax')))/height

            xmin=np.float64(xmin)
            ymin=np.float64(ymin)
            xmax=np.float64(xmax)
            ymax=np.float64(ymax)

            res.append([xmax-xmin,ymax-ymin])

    return np.array(res)

def dis_iou(box,cluster):
    x=np.minimum(cluster[:,0],box[0])
    y=np.minimum(cluster[:,1],box[1])

    inter=x*y
    area1=box[0]*box[1]
    area2=cluster[:,0]*cluster[:,1]
    union=area1+area2-inter

    return inter/union


def kmeans(box,k):
    row=box.shape[0]
    #初始化每个box到每个中心（9个）的距离
    distance=np.empty((row,k))

    #最终每个box聚类位置
    last_loc=np.zeros((row,))
    np.random.seed()

    #随机选择k个box作为初始聚类中心，这里k=9
    cluster=box[np.random.choice(row,k,replace=False)]

    while True:
        for i in range(row):
            #计算每个box到k个中心的距离，这里的距离用iou度量
            distance[i]=1-dis_iou(box[i],cluster)
        #得到每个Box最近的中心点索引,范围为[0~k)
        near=np.argmin(distance,axis=1)
        if (last_loc==near).all():
            break

        for j in range(k):
            cluster[j]=np.median(box[near==j],axis=0)
        last_loc=near

    return cluster


if __name__=='__main__':
    '''
    注意：
    ==>针对SIZE大小的输入图片所生成的anchor尺寸
    ==>根据自己的实际输入尺寸来改动SIZE
    '''
    SIZE=416
    #一个九个anchor，每个anchor用长宽两个量表示。三个一组，共三组，组之间从小到大依此用来预测小，中，大物体
    anchors_num=9
    #根据自己路径来设置
    path=r'C:/AllProgram/DatasetPath/VOCdevkit/VOC2007//Annotations'

    #载入所有xml，存储格式转化相对于width，height的0~1百分比
    data=get_data(path)
    #使用k-means聚类算法，结果0~1百分比，需要转化
    out=kmeans(data,anchors_num)
    #根据面积排序
    out=out[np.argsort(out[:,0]*out[:,1])]
    data=out*SIZE

    '''
    此处路径根据自己修改，将聚类结果保存到指定位置
    '''
    with open('../model_data/test_anchors.txt','w') as f:
        for i in range(anchors_num):
            if i==0:
                x_y='%d,%d'%(data[i][0],data[i][1])
            else:
                x_y=', %d,%d'%(data[i][0],data[i][1])
            f.write(x_y)

    print('finish!')



