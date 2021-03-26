import os
from PIL import Image
import numpy as np
import torch
from config import config
from torch.utils import data
from torchvision import transforms as T
import xml.etree.ElementTree as ET

#---------------------------------------------------------------------------#
# 利用图片id获得对应的图片全路径及对应的xml文件信息，并将两个信息结合重新写入形成新的文件
# 此函数单独执行即可生成相应的文件
#---------------------------------------------------------------------------#
def voc_annotation(data_path,anno_path,file_type):
    #data_path为总路径，file_type为需要的文件类型，如取值为train,test,val,trainval等等
    #anno_path为处理结果写入路径
    imgs_id_file=os.path.join(data_path,'ImageSets','Main',file_type+'.txt')

    #获取VOC07数据集的20个类别名字
    with open(config.voc07_classes_path) as f:
        classes_names=f.readlines()
    classes_names=[e.strip() for e in classes_names]

    #获取图片的id
    with open(imgs_id_file,'r') as f:
        lines=f.readlines()
        images_ids=[line.strip() for line in lines]
    #利用图片id获得对应的图片全路径及对应的xml文件信息，并将两个信息结合重新写入
    with open(anno_path,'a') as f:
        for id in images_ids:
            image_path=os.path.join(data_path,'JPEGImages/',id+'.jpg')
            label_path=os.path.join(data_path,'Annotations',id+'.xml')
            annotations=image_path #结合图像和标签路径一起，便于后期读入

            root=ET.parse(label_path).getroot()
            objects=root.findall('object')
            new_str=''
            for obj in objects:
                difficult=obj.find('difficult').text.strip()
                #difficult表示是否容易识别，0表示容易，1表示困难
                if not config.voc07_use_difficult_bbox and int(difficult)==1:continue
                bbox=obj.find('bndbox')
                #根据找到得对象名字在类别文件中找到对应的索引
                class_id=classes_names.index(obj.find('name').text.lower().strip())
                xmin=bbox.find('xmin').text.strip()
                ymin=bbox.find('ymin').text.strip()
                xmax=bbox.find('xmax').text.strip()
                ymax=bbox.find('ymax').text.strip()

                new_str+=' '+','.join([xmin,ymin,xmax,ymax,str(class_id)])
            if new_str=='':continue
            annotations+=new_str
            annotations+='\n'
            f.write(annotations)

#暂时未利用Mosaic数据增强方法
class YoloDataset(data.Dataset):
    def __init__(self,data_path):
        super(YoloDataset,self).__init__()
        with open(data_path) as f:
            data=f.readlines()
        self.data_lines=data
        self.data_lens=len(data)

    def __getitem__(self,index):
        line=self.data_lines[index].split()
        img=Image.open(line[0])
        #img=img.convert('RGB')#如果不确定输入图像通道是否一致，需加上
        iw,ih=img.size
        #---------------------------------------------------------------------#
        # Resize一个参数和两个参数不一样，一个参数表示将最小边缩放到指定值，图像长宽比保持不变。
        # T.ToTensor()将图像转换到0~1的区间
        # T.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])将图像转化到-1~1的区间
        #--------------------------------------------------------------------#
        transform=T.Compose([T.Resize(config.model_image_w),
                             T.CenterCrop(config.model_image_w),
                             T.RandomHorizontalFlip(),
                             T.ToTensor(),
                             T.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
                             ])
        img=transform(img)
        #boxes为二维数组，每行5个数
        boxes=np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
        #-------------------------------------------------------------------------------#
        # 将输入图像最小边缩放到指定值，然后裁剪出指定尺寸的大小，此时对应的先验框也应该跟着相应变化，
        # 下面的调整方式存在缺陷！！！可以改进。
        #-------------------------------------------------------------------------------#
        if len(boxes)!=0:
            #将矩形框的左上和右下角表示法转化为中心点及宽度和高度表示
            box=np.array(boxes[:,:4],dtype=np.float32)
            box[:,2]=box[:,2]-box[:,0]
            box[:,3]=box[:,3]-box[:,1]
            box[:,0]=box[:,0]+box[:,2]/2
            box[:,1]=box[:,1]+box[:,3]/2

            #从坐标转化为相对于输入图像尺寸的0~1百分比，后面会缩放到指定大小
            box[:,0]=box[:,0]/iw
            box[:,1]=box[:,1]/ih
            box[:,2]=box[:,2]/iw
            box[:,3]=box[:,3]/ih
            boxes=np.concatenate([box,boxes[:,-1:]],axis=-1)
            #不加type(torch.FloatTensor)会变成64为浮点，内存占用太多没必要
        boxes=torch.from_numpy(boxes).type(torch.FloatTensor)

        # 返回为tensor类型，img,boxes都是归一化的结果
        return img, boxes

    def __len__(self):
        return self.data_lens

#voc_annotation(config.data_path,config.val_anno_path,'val')
#data=YoloDataset(config.train_anno_path)[1]
