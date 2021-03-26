from torch import nn
import torch

class DecodeBox(nn.Module):
    def __init__(self,anchors,nums_classes,img_size):
        super(DecodeBox, self).__init__()
        self.anchors=anchors
        self.anchors_num=len(anchors)
        self.nums_classes=nums_classes
        self.box_attr=5+nums_classes
        self.img_size=img_size

    def forward(self, input):
        #--------------------------------#
        # bs是一次检测图片数量
        # input=size[bs,channel_nums,h,w]
        #--------------------------------#
        bs=input.size(0)
        input_h=input.size(2)
        input_w=input.size(3)

        stride_w=self.img_size[0]/input_w
        stride_h=self.img_size[1]/input_h

        #将anchor尺寸缩放到输入input对应尺寸
        self.anchors=torch.FloatTensor([[w/stride_w, h/stride_h] for w,h in self.anchors])

        prediction=input.view(bs,self.anchors_num,self.box_attr,
                              input_h,input_w).permute(0, 1, 3, 4, 2).contiguous()

        #获取所有框的中心坐标及宽度、高度的偏移量（注意这里实际应该称为偏移量，不是实际值）
        # center_x=torch.sigmoid(prediction[..., 0])
        # center_y=torch.sigmoid(prediction[..., 1])
        center_x=prediction[..., 0]
        center_y=prediction[..., 1]
        box_w=prediction[..., 2]
        box_h=prediction[..., 3]
        #获取置信度和所有类别预测结果
        conf=torch.sigmoid(prediction[..., 4])
        pred_cls=torch.sigmoid(prediction[..., 5:])

        #通过如下方式生成框的中心坐标值，后面会利用偏移量修正,,t()为转置
        grid_x=torch.linspace(0,input_w-1,input_w).repeat(input_h,1).t().repeat(
            int(self.anchors_num*bs),1,1).view(center_x.shape)
        grid_y=torch.linspace(0,input_h-1,input_h).repeat(input_w,1).repeat(
            int(self.anchors_num*bs),1,1).view(center_y.shape)

        #根据输入的anchor宽和高来生成框的宽度和高度
        anchors_w=self.anchors[:, 0]
        anchors_h=self.anchors[:, 1]

        anchors_ww=anchors_w.view(self.anchors_num,1).repeat(bs,1).view(bs,self.anchors_num,-1).repeat(1,1,input_w*input_h).view(box_w.shape)
        anchors_hh=anchors_h.view(self.anchors_num,1).repeat(bs,1).view(bs,self.anchors_num,-1).repeat(1,1,input_w*input_h).view(box_h.shape)

        #计算调整后的框的中心坐标值及宽度、高度
        pre_boxes=torch.FloatTensor(prediction[..., :4].shape)
        pre_boxes[..., 0]=grid_x+center_x
        pre_boxes[..., 1]=grid_y+center_y
        pre_boxes[..., 2]=torch.exp(box_w)*anchors_ww
        pre_boxes[..., 3]=torch.exp(box_h)*anchors_hh

        #用于将输出调整为输入图像大小尺寸,输出pre_output为二维
        _scale=torch.Tensor([stride_w,stride_w]*2)
        pre_output=torch.cat((pre_boxes.view(bs,-1,4)*_scale,conf.view(bs,-1,1),pred_cls.view(bs,-1,self.nums_classes)),-1)

        return pre_output

'''
以下代码用于测试
from model.yolo4 import Yolov4
anchors=torch.Tensor([[2,16],[19,36], [40,28]])
Box=DecodeBox(anchors,80,(416,416),0.5,0.3)

model=Yolov4()
input=torch.randn(1,3,416,416)
out1,out2,out3=model(input)
out=Box(out1)
print(out)
'''










