from torch import nn
from config import config
import torch
import math

class YOLO4Loss(nn.Module):
    def __init__(self,anchors):
        super(YOLO4Loss, self).__init__()
        self.anchors=anchors
        self.num_anchors=len(anchors)
        self.box_attrs=5+config.nums_classes
        self.img_size=(config.model_image_w,config.model_image_h)
        self.nums_classes=config.nums_classes
        self.ignore_threshold=config.ignore_thres

    #input:torch.Tensor表示input类型为Tensor
    def forward(self,input:torch.Tensor,targets):
        # input为bs,3*(5+num_classes),W,H
        bs=input.size(0)
        #特征层高,宽
        in_h=input.size(2)
        in_w=input.size(3)

        #相对于输入图像的缩小倍数
        stride_h=self.img_size[1]/in_h
        stride_w=self.img_size[0]/in_w

        scaled_anchors=torch.FloatTensor([[a_w/stride_w,a_h/stride_h] for a_w,a_h in self.anchors])
        predictions=input.view(bs,self.num_anchors,self.box_attrs,in_h,in_w).permute(0,1,3,4,2).contiguous()

        #获得预测框是否包含物体置信度和所有种类的置信度
        conf=torch.sigmoid(predictions[...,4])
        pred_cls=torch.sigmoid(predictions[...,5:])

        mask,noobj_mask,t_box,tconf,tcls=self.get_target(targets,scaled_anchors,in_w,in_h)

        '''
        将预测结果进行解码，判断预测结果和真实值的重合程度，如果重合程度过大则忽略，因为这些特征点属
        于预测比较准确的特征点作为负样本不合适,也就是不计算这些预测框的损失。
        '''
        noobj_mask,pred_boxes=self.get_ignore(predictions,targets,scaled_anchors,in_w,in_h,noobj_mask)
        #计算预测框的损失
        ciou=self.box_ciou(pred_boxes[mask.bool()],t_box[mask.bool()])
        loss_loc = torch.sum(ciou)

        # 计算置信度的loss,包括本身有目标的和无目标的
        loss_conf = torch.sum(self.BCELoss(conf, mask) * mask) + torch.sum(self.BCELoss(conf, mask) * noobj_mask)

        #计算预测的损失
        loss_cls = torch.sum(
            self.BCELoss(pred_cls[mask == 1], self.smooth_labels(tcls[mask == 1], config.label_smooth, self.nums_classes)))
        loss = loss_conf * config.lambda_conf + loss_cls * config.lambda_cls + loss_loc * config.lambda_loc
        #mask由0，1组成，1的个数代表多少个预测结果参与损失计算
        num_pos=torch.sum(mask)
        num_pos = torch.max(num_pos, torch.ones_like(num_pos))

        return loss,num_pos

    def box_ciou(self,b1, b2):
        # 求出预测框左上角右下角
        b1_xy = b1[..., :2]
        b1_wh = b1[..., 2:4]
        b1_wh_half = b1_wh / 2.
        b1_mins = b1_xy - b1_wh_half
        b1_maxes = b1_xy + b1_wh_half
        # 求出真实框左上角右下角
        b2_xy = b2[..., :2]
        b2_wh = b2[..., 2:4]
        b2_wh_half = b2_wh / 2.
        b2_mins = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half

        # 求真实框和预测框所有的iou
        intersect_mins = torch.max(b1_mins, b2_mins)
        intersect_maxes = torch.min(b1_maxes, b2_maxes)
        intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]
        union_area = b1_area + b2_area - intersect_area
        iou = intersect_area / torch.clamp(union_area, min=1e-6)

        # 计算中心的差距
        center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), axis=-1)

        # 找到包裹两个框的最小框的左上角和右下角
        enclose_mins = torch.min(b1_mins, b2_mins)
        enclose_maxes = torch.max(b1_maxes, b2_maxes)
        enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
        # 计算对角线距离
        enclose_diagonal = torch.sum(torch.pow(enclose_wh, 2), axis=-1)
        #根据公式来
        ciou = iou - 1.0 * (center_distance) / torch.clamp(enclose_diagonal, min=1e-6)

        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(
            b1_wh[..., 0] / torch.clamp(b1_wh[..., 1], min=1e-6)) - torch.atan(
            b2_wh[..., 0] / torch.clamp(b2_wh[..., 1], min=1e-6))), 2)
        alpha = v / torch.clamp((1.0 - iou + v), min=1e-6)
        #根据公式来
        ciou = 1-ciou + alpha * v
        return ciou

    def smooth_labels(self,y_true, label_smoothing, num_classes):
        return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes

    def clip_by_tensor(self,t, t_min, t_max):
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    def MSELoss(self,pred, target):
        return (pred - target) ** 2

    def BCELoss(self,pred, target):
        epsilon = 1e-7
        pred = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon)
        output = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output


    def get_target(self,target,anchors,in_w,in_h):
        bs=len(target)
        #创建全0或者全1阵列，mask代表有目标的特征点，noobj_mask代表无目标特征点
        mask=torch.zeros(bs,int(self.num_anchors),in_h,in_w,requires_grad=False)
        noobj_mask=torch.ones(bs,int(self.num_anchors),in_h,in_w,requires_grad=False)

        #创建与网络输出对应的target格式
        tx=torch.zeros(bs,int(self.num_anchors),in_h,in_w,requires_grad=False)
        ty=torch.zeros(bs,int(self.num_anchors),in_h,in_w,requires_grad=False)
        tw = torch.zeros(bs, int(self.num_anchors), in_h, in_w, requires_grad=False)
        th = torch.zeros(bs, int(self.num_anchors), in_h, in_w, requires_grad=False)
        t_box=torch.zeros(bs,int(self.num_anchors),in_h,in_w,4,requires_grad=False)
        tconf=torch.zeros(bs, int(self.num_anchors), in_h, in_w, requires_grad=False)
        tcls=torch.zeros(bs, int(self.num_anchors), in_h, in_w,self.nums_classes,requires_grad=False)

        for b in range(bs):
            if len(target[b]) == 0:
                continue
            #由于在数据载入时将target转化到0-1的百分比，并且从左上和右下坐标转为中心坐标和宽高。
            #此处计算target在特征图上的位置
            cxs=target[b][:,0:1]*in_w
            cys=target[b][:,1:2]*in_h
            ws=target[b][:,2:3]*in_w
            hs=target[b][:,3:4]*in_h

            #计算每个框属于哪个网格
            gis=torch.floor(cxs)
            gjs=torch.floor(cys)

            '''
            解释以下两行代码：
            对于特征图上的每一个点来说都有三个一样大小的先验框anchors.如果此特征图上存在目标，那么每一个目标会
            对应一个真实框（中心坐标和宽度、高度），此真实框(中心坐标)对应的特征图网格有三个先验框，那么到底哪一
            个先验框负责预测这个真实框呢？我们可以通过计算每个先验框和这个真实框的iou，最大的那个负责预测这个真实
            框，也就是说我们会在特征图的此网格对应的向量中填入真实框的坐标信息。由于在计算中每个网格的先验框和真实
            框的iou时（先验框和真实框的中心坐标始终一样，设置为多少都可以）都只需要宽度和高度信息，所以坐标信息可
            以用0统一表示,不影响计算结果。
            '''
            #计算真实框的位置
            gt_box=torch.FloatTensor(torch.cat([torch.zeros_like(ws),torch.zeros_like(hs),ws,hs],1))
            #计算所有先验框的位置
            anchors_box=torch.FloatTensor(torch.cat((torch.zeros(self.num_anchors,2),torch.FloatTensor(anchors)),1))
            #计算重合度
            ious=self.overlap(gt_box,anchors_box)
            #找到每个真实框最佳匹配的先验框,torch.argmax返回指定维度最大值的序号
            best_boxes_id=torch.argmax(ious,dim=-1)
            for i,best_id in enumerate(best_boxes_id):
                #best_id代表用来预测真实框的那个先验框,取值0，1，2
                gi=gis[i].long()[0]#gis[i].long(),gis[i].long()[0]这两种写法都行，以下同理
                gj=gjs[i].long()[0]
                gx=cxs[i][0]
                gy=cys[i][0]
                gw=ws[i][0]
                gh=hs[i][0]

                if gj<in_h and gi<in_w:
                    mask[b,best_id,gj,gi]=1
                    noobj_mask[b,best_id,gj,gi]=0
                    #当我们知道哪个先验框负责预测真实框后，就把真实框对应的信息填入该先验框对应的位置
                    tx[b,best_id,gj,gi]=gx
                    ty[b,best_id,gj,gi]=gy
                    tw[b,best_id,gj,gi]=gw
                    th[b,best_id,gj,gi]=gh
                    tconf[b,best_id,gj,gi]=1
                    tcls[b,best_id,gj,gi,target[b][i,4].long()]=1

        t_box[...,0] = tx
        t_box[...,1] = ty
        t_box[...,2] = tw
        t_box[...,3] = th
        return mask,noobj_mask,t_box,tconf,tcls


    def overlap(self,gt_box,anchors_box):
        box_a=torch.zeros_like(gt_box)
        box_b=torch.zeros_like(anchors_box)
        #将中心坐标及长宽转变为左上和右下，方便计算iou
        box_a[:,0]=gt_box[:,0]-gt_box[:,2]/2
        box_a[:,1]=gt_box[:,1]-gt_box[:,3]/2
        box_a[:,2]=gt_box[:,0]+gt_box[:,2]/2
        box_a[:,3]=gt_box[:,1]+gt_box[:,3]/2

        box_b[:,0]=anchors_box[:,0]-anchors_box[:,2]/2
        box_b[:,1]=anchors_box[:,1]-anchors_box[:,3]/2
        box_b[:,2]=anchors_box[:,0]+anchors_box[:,2]/2
        box_b[:,3]=anchors_box[:,1]+anchors_box[:,3]/2

        A=box_a.size(0)
        B=box_b.size(0)

        '''
        重点解释：
        例如对于一个target=[[0,0,1,1],[0,0,1,1]],与一个anchors=[[0,0,2,2],[0,0,2,2],[0,0,2,2]]
        维度不一样，我们要做的是每一个target[i]都要和整个anchors计算iou，维度不一样，处理不方便，所以以
        下操作会统一维度，再整体操作。对于target先在其第1维（0维开始）扩展一维target=[[[0,0,1,1]],[[0,0,1,1]]]
        然后对该维度重复len(anchors)倍，target=[[[0,0,1,1],[...],[...]],[[0,0,1,1],[...],[...]]]
        对于anchors来说，先在0维度扩展一维anchors=[[[0,0,2,2],[0,0,2,2],[0,0,2,2]]]再重复len(target)倍,
        anchors=[[[0,0,2,2],[0,0,2,2],[0,0,2,2]],[[...],[...],[...]]]，此时维度便一致。
        ...然后计算重合部分的两个坐标左上右下，按照左上(min_xy)取大，右下(max_xy)取小原则
        '''
        min_xy=torch.max(box_a[:,:2].unsqueeze(1).expand(A,B,2),
                         box_b[:,:2].unsqueeze(0).expand(A,B,2))
        max_xy=torch.min(box_a[:,2:].unsqueeze(1).expand(A,B,2),
                         box_b[:,2:].unsqueeze(0).expand(A,B,2))
        #torch.clamp(input, min, max, out=None) → Tensor输入映射到一个【min, max】区间
        inter=torch.clamp((max_xy-min_xy),min=0)
        #交集
        inter=inter[:,:,0]*inter[:,:,1]

        area_a=((box_a[:,2]-box_a[:,0])*(box_a[:,3]-box_a[:,1])).unsqueeze(1).expand_as(inter)
        area_b=((box_b[:,2]-box_b[:,0])*(box_b[:,3]-box_b[:,1])).unsqueeze(0).expand_as(inter)
        #并集，需先求交集
        union=area_a+area_b-inter

        #返回[A,B]，一个target[i]对应len(anchors)=B个先验框
        #所以A个target就有A*B个结果，二维表示为A行B列
        return inter/union

    def get_ignore(self,predictions,target,anchors,in_w,in_h,noobj_mask):
        bs=len(target)

        #获得预测框的中心坐标及宽度和高度
        # x=torch.sigmoid(predictions[...,0])
        # y=torch.sigmoid(predictions[...,1])
        x=predictions[...,0]
        y=predictions[...,1]
        w=predictions[...,2]
        h=predictions[...,3]

        #通过如下方式生成框的中心坐标值，后面会利用偏移量修正,,t()为转置
        grid_x=torch.linspace(0,in_w-1,in_w).repeat(in_h,1).t().repeat(
            int(bs*self.num_anchors),1,1).view(x.shape)
        grid_y=torch.linspace(0,in_h-1,in_h).repeat(in_w,1).repeat(
            int(bs*self.num_anchors),1,1).view(y.shape)

        #根据输入的anchor宽和高来生成框的宽度和高度
        anchors_w=anchors[:, 0]
        anchors_h=anchors[:, 1]
        anchors_ww = anchors_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchors_hh = anchors_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)

        # anchors_ww=anchors_w.view(self.num_anchors,1).repeat(1,in_w*in_h).view(w.shape)
        # anchors_hh=anchors_h.view(self.num_anchors,1).repeat(1,in_w*in_h).view(h.shape)

        #计算调整后的框的中心坐标值及宽度、高度
        pre_boxes=torch.FloatTensor(predictions[..., :4].shape)
        pre_boxes[..., 0]=grid_x+x
        pre_boxes[..., 1]=grid_y+y
        pre_boxes[..., 2]=torch.exp(w)*anchors_ww
        pre_boxes[..., 3]=torch.exp(h)*anchors_hh

        for i in range(bs):
            pred_boxes_for_ignore=pre_boxes[i]
            pred_boxes_for_ignore = pred_boxes_for_ignore.view(-1, 4)

            #计算真实框，并把真实框转换成相对于特征层的大小
            tglen=int(target[i][0][-1])
            if tglen > 0:
                gx = target[i][:tglen, 0:1] * in_w
                gy = target[i][:tglen, 1:2] * in_h
                gw = target[i][:tglen, 2:3] * in_w
                gh = target[i][:tglen, 3:4] * in_h
                gt_box = torch.FloatTensor(torch.cat([gx, gy, gw, gh],-1)).type(torch.FloatTensor)
                #计算交并比
                anch_ious = self.overlap(gt_box, pred_boxes_for_ignore)
                #每个先验框对应真实框的最大重合度,返回重合度及对应索引
                anch_ious_max, _ = torch.max(anch_ious,dim=0)
                anch_ious_max = anch_ious_max.view(pre_boxes[i].size()[:3])
                noobj_mask[i][anch_ious_max>self.ignore_threshold] = 0

        return noobj_mask, pre_boxes



# target=torch.Tensor([[[0.6,0.6,0.65,0.7,12],
#                      [0.55,0.6,0.7,0.8,7]]])
# anchors=torch.Tensor([[2,3],
#                       [4,5],
#                       [5,3]])
# model=YOLO4Loss(anchors=anchors)
# out=model.get_target(target,anchors,5,5)
# print(out)
