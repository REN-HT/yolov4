import torch
from PIL import Image,ImageDraw,ImageFont
from torchvision import transforms as T
from utils.DecodeBox import DecodeBox
from config import config
from model.yolo4 import Yolov4
from utils.tools import non_max_suppression

#---------------------------------------#
# 将原图等比例缩放到固定大小，不足部分用灰度值填充
#---------------------------------------#
def Resize_image(image,size):
    iw,ih=image.size
    w,h=size
    scale=min(w/iw,h/ih)
    nw=int(iw*scale)
    nh=int(ih*scale)

    image=image.resize((nw,nh),Image.BICUBIC)
    #新建一个固定大小的三通道灰度图
    new_image=Image.new('RGB',size,(128,128,128))
    #复制缩放后的输入图到新建灰度图
    new_image.paste(image,((w-nw)//2,(h-nh)//2))
    return new_image

def yolo_correct_boxes(top, left, bottom, right, input_shape, image_shape):
    input_shape=torch.Tensor([input_shape[0],input_shape[1]])
    image_shape=torch.Tensor([image_shape[0],image_shape[1]])
    new_shape = image_shape*min(input_shape/image_shape)

    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape

    box_yx = torch.cat(((top+bottom)/2,(left+right)/2),-1)/input_shape
    box_hw = torch.cat((bottom-top,right-left),-1)/input_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  torch.cat([
        box_mins[:, 0:1],
        box_mins[:, 1:2],
        box_maxes[:, 0:1],
        box_maxes[:, 1:2]
    ],-1)
    boxes *= torch.cat([image_shape, image_shape],-1)
    return boxes


class Predict(object):
    def __init__(self):
        self.net=Yolov4()
        self.nums_class=config.nums_classes
        self.anchors=self._get_anchors()
        self.class_name=self._get_class_name()
        self.input_w=config.model_image_w
        self.input_h=config.model_image_h
        self.conf_thres=config.conf_thres
        self.iou_thres=config.iou_thres
        self.SIZE=config.image_size

    def _get_anchors(self):
        with open(config.anchors_path) as f:
            anchors=f.readline()
        anchors=[float(x) for x in anchors.split(',')]
        return torch.Tensor(anchors).view(-1,3,2)

    def _get_class_name(self):
        with open(config.voc07_classes_path) as f:
            class_names=f.readlines()
        return [name.strip() for name in class_names]

    def detect_image(self,root):
        image=Image.open(root)
        image_shape=image.size
        crop_img=Resize_image(image,(self.SIZE,self.SIZE))
        transform=T.Compose([T.ToTensor(),
                             T.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
                             ])
        img=transform(crop_img)
        img=img.unsqueeze(0)

        state_dict = torch.load('model_path.pth')
        self.net.load_state_dict(state_dict)
        print('model load successfully!')
        #可决定是否使用GPU
        if config.use_gpu:
            self.net=self.net.cuda()
            img=img.cuda()
        #将图像转变为tensor类型输入网络
        spred,mpred,lpred=self.net(img)
        if config.use_gpu:
            spred=spred.cpu()
            mpred=mpred.cpu()
            lpred=lpred.cpu()

        yolodecodes=[]
        for i in range(3):
            yolodecodes.append(DecodeBox(self.anchors[2-i],self.nums_class,(self.input_w,self.input_h)))
        preds=yolodecodes[0](spred)
        predm=yolodecodes[1](mpred)
        predl=yolodecodes[2](lpred)

        #非极大值抑制，第一次筛选
        all_prediction=torch.cat((preds,predm,predl),1)
        prediction_boxes=non_max_suppression(all_prediction,self.conf_thres,self.iou_thres)

        prediction_boxes=prediction_boxes[0]

        #第二次筛选,去除等于阈值的那些目标。异常处理目的在于prediction_boxes预测结果可能为0，如果为0以下操作将发生错误
        try:
            secd_mask=(prediction_boxes[:, 4]*prediction_boxes[:, 5]>self.conf_thres)
            res_boxes=prediction_boxes[secd_mask,:4]
            res_scores=prediction_boxes[secd_mask,4]*prediction_boxes[secd_mask,5]
            res_labels=prediction_boxes[secd_mask,-1].type(torch.int32)
        except:
            return image

        tl,tr,bl,br=res_boxes[:,0].unsqueeze(1),res_boxes[:,1].unsqueeze(1),res_boxes[:,2].unsqueeze(1),res_boxes[:,3].unsqueeze(1)
        boxes=yolo_correct_boxes(tl,tr,bl,br,(self.SIZE,self.SIZE),image_shape)

        for i,c in enumerate(res_labels):
            obj_name=self.class_name[c] #根据标签获取名字
            obj_score=res_scores[i]
            x1,y1,x2,y2=boxes[i]
            #可根据实际调整
            x1=x1-5; y1=y1-5
            x2=x2+5; y2=y2+5

            #防止框越界
            x1=max(0,torch.floor(x1+0.5))
            y1=max(0,torch.floor(y1+0.5))
            x2=min(image_shape[0], torch.floor(x2+0.5))
            y2=min(image_shape[1], torch.floor(y2+0.5))

            draw = ImageDraw.Draw(image)
            #font = ImageFont.truetype(font=None,size=24)
            #得到显示标签，如bike:0.89,用于显示于画框左上角
            label='{}:{:.2f}'.format(obj_name,obj_score)
            #得到显示标签的大小，也就是宽度和高度,如(23，12)
            label_size=draw.textsize(label)
            #得到显示标签的起始显示位置，一般置于画框左上角上面，如果位置不够则位于画框左上角下面
            if y1-label_size[1]>=0:
                text_origin=torch.Tensor([x1,y1-label_size[1]])
            else:
                text_origin = torch.Tensor([x1, y1 + 1])
            #以下操作依次为，画框、填充显示标签区域、显示标签文字
            draw.rectangle([x1,y1,x2,y2],outline='red',width=5)
            draw.rectangle([text_origin[0],text_origin[1],
                            text_origin[0]+label_size[0],text_origin[1]+label_size[1]],fill=(100,221,72))
            draw.text((text_origin[0],text_origin[1]),label,fill='yellow')

            del draw

        return image




