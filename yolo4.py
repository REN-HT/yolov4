import torch as t
from torch import nn
from config import config
from torch.nn import functional as F
from model.CSPdarknet import CSPDarkNet
from model.CSPdarknet import Mish

#SPP
class SPPNet(nn.Module):
    def __init__(self):
        super(SPPNet,self).__init__()
    def forward(self,x):
        x1=F.max_pool2d(x,5,1,2)
        x2=F.max_pool2d(x,9,1,4)
        x3=F.max_pool2d(x,13,1,6)
        x=t.cat([x1,x2,x3,x],dim=1)
        return x
#PAN
class PANet(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(PANet, self).__init__()
        self.pre=nn.Sequential(nn.Conv2d(inchannel,outchannel//2,1,1,0,bias=False),
                               nn.BatchNorm2d(outchannel//2),
                               Mish())
        self.right=nn.Sequential(nn.Conv2d(inchannel,outchannel//2,1,1,0,bias=False),
                               nn.BatchNorm2d(outchannel//2),
                               Mish())
        self.upsample=nn.Upsample(scale_factor=2, mode='nearest') #mode='bilinear'
    def forward(self,left,right):
        left=self.pre(left)
        left=self.upsample(left)
        right=self.right(right)
        #------------------------------------------------------#
        # 当输入尺寸不正确时，这里会发生错误，出现尺寸不一样的情况，
        # 比如当上层输入为19*19，此层经过下采样变为10*10,若对此
        # 层进行上采用则为20*20，当将上层结果与此层结果融合时将
        # 发生错误 ，因此建议图像原始输入为608*608或416*416(32倍数都可)
        #------------------------------------------------------#
        out=t.cat([left,right],dim=1)
        return out


class make_three_conv(nn.Module):
    def __init__(self, inchannel,outchannel):
        super(make_three_conv,self).__init__()
        self.three_conv=nn.Sequential(nn.Conv2d(inchannel,outchannel,1,1,0,bias=False),
                                      nn.BatchNorm2d(outchannel),
                                      Mish(),
                                      nn.Conv2d(outchannel,outchannel*2,3,1,1,bias=False),
                                      nn.BatchNorm2d(outchannel*2),
                                      Mish(),
                                      nn.Conv2d(outchannel*2,outchannel,1,1,0,bias=False),
                                      nn.BatchNorm2d(outchannel),
                                      Mish())
    def forward(self,x):
        x=self.three_conv(x)
        return x

class make_five_conv(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(make_five_conv,self).__init__()
        self.five_conv=nn.Sequential(make_three_conv(inchannel,outchannel),
                                      nn.Conv2d(outchannel,outchannel*2,3,1,1,bias=False),
                                      nn.BatchNorm2d(outchannel*2),
                                      Mish(),
                                      nn.Conv2d(outchannel*2,outchannel,1,1,0,bias=False),
                                      nn.BatchNorm2d(outchannel),
                                      Mish())
    def forward(self,x):
        x=self.five_conv(x)
        return x

#------------------------------------------------------------------------------------#
# 当输入大小为608*608时如下，其它尺寸依此计算即可
# 这里得到所有输出结果，类似yolov3,其中具体表示如下：
# x3=out3;           76*76*(4+1+nums_classes)*3
# x2=x3+out2+conv*5; 38*38*(4+1+nums_classes)*3
# x1=x2+out1+conv*5; 19*19*(4+1+nums_classes)*3
# note:4代表框的坐标，1为置信度，nums_classes为类别数，3代表框的数量
# 输入从这里开始，执行model=Yolov4(),output=model(input)即可得到结果output
# 当数据输入后，执行顺序为：input-->CSPdarknet-->Yolov4(SPP-->PANet-->Yolohead)-->output
#------------------------------------------------------------------------------------#
class Yolov4(nn.Module):
    def __init__(self,nums_classes=config.nums_classes):
        super(Yolov4,self).__init__()
        self.backbone=CSPDarkNet([1, 2, 8, 8, 4])
        self.three_conv1=make_three_conv(1024,512)

        self.neck=SPPNet()
        self.three_conv2=make_three_conv(2048,512)

        self.PANet1=PANet(512,512)
        self.five_conv1=make_five_conv(512,256)

        self.PANet2=PANet(256, 256)
        self.five_conv2=make_five_conv(256, 128)

        self.head3=nn.Sequential(nn.Conv2d(128,256,3,1,1,bias=False),
                                 nn.BatchNorm2d(256),
                                 Mish(),
                                 nn.Conv2d(256,(4+1+nums_classes)*3,1,1,0,bias=False))

        self.downsample1=nn.Sequential(nn.Conv2d(128,256,3,2,1,bias=False),
                                       nn.BatchNorm2d(256),
                                       Mish())
        self.five_conv3=make_five_conv(512,256)
        self.head2=nn.Sequential(nn.Conv2d(256,512,3,1,1,bias=False),
                                 nn.BatchNorm2d(512),
                                 Mish(),
                                 nn.Conv2d(512,(4+1+nums_classes)*3,1,1,0,bias=False))

        self.downsample2 = nn.Sequential(nn.Conv2d(256, 512, 3, 2, 1, bias=False),
                                         nn.BatchNorm2d(512),
                                         Mish())
        self.five_conv4 = make_five_conv(1024, 512)
        self.head1=nn.Sequential(nn.Conv2d(512,1024,3,1,1,bias=False),
                                 nn.BatchNorm2d(1024),
                                 Mish(),
                                 nn.Conv2d(1024,(4+1+nums_classes)*3,1,1,0,bias=False))
        #参数初始化
        import math
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, input):
        x3,x4,x5=self.backbone(input)
        x=self.three_conv1(x5)

        x=self.neck(x)
        out1=self.three_conv2(x)

        x=self.PANet1(out1,x4)
        out2=self.five_conv1(x)

        x=self.PANet2(out2,x3)
        out3=self.five_conv2(x)

        y3=self.head3(out3)

        out3down=self.downsample1(out3)
        newout2=t.cat([out2,out3down],dim=1)
        newout2=self.five_conv3(newout2)
        y2=self.head2(newout2)

        out2down=self.downsample2(out2)
        newout1=t.cat([out1, out2down], dim=1)
        newout1=self.five_conv4(newout1)
        y1=self.head1(newout1)
        return y1, y2, y3

# import torch
# from torch.autograd import Variable as V
# model=Yolov4()
# input=V(torch.randn(1,3,416,416))
# out=model(input)
# print(out)

