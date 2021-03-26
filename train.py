import torch
import numpy as np
from tqdm import tqdm
from config import config
from torch.utils.data import DataLoader
from dataset.yoloDataset import YoloDataset
from torch.autograd import Variable
from model.yolo4 import Yolov4
from model.yolo4_loss import YOLO4Loss

# 利用预训练的模型来更新网络参数，需要保证参数名匹配！！！否则无法完成更新
def transfer_model(pretrain_file, model):
    pretrain_dict=torch.load(pretrain_file)
    model_dict=model.state_dict()
    pretrain_dict=transfer_state_dict(pretrain_dict,model_dict)
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    return model

def transfer_state_dict(pretrain_dict, model_dict):
    state_dict={}
    count=0
    for k,v in pretrain_dict.items():
        if k in model_dict.keys():
            state_dict[k]=v
            count+=1
    if count==0:
        print('no parameters update!!!')
    else:
        print('update successfully!!!')
    return state_dict

# --------------------------------------------------------------------
# 该函数作用在于：就目标检测来说，当我们进行小批量样本训练时（也就是batch_size>1）
# 并不是每个样本都有相同数量的标记框，此时单纯使用Dataloader载入批量样本会发生错误
# 因为Dataloader默认的collate_fn函数需要保证每个样本都有相同数量的标记框，这样方
# 便放到一个Tensor中。所以在目标检测任务中，我们需要重写此函数。该函数可以批量处理数
# 据，但不需要每个样本都有相同的标记框。
# -------------------------------------------------------------------
def my_collate_fn(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array([img.numpy() for img in images], dtype=np.float32)
    # image为numpy.array类型，bboxes为 [ Tensor([[]]), ...]类型
    return images, bboxes

def train():
    # 创建模型
    net = Yolov4()
    net = transfer_model('yolo4_weights.pth', net)
    if config.use_gpu:
        net = net.cuda()
    train_set = YoloDataset(config.train_anno_path)
    val_set = YoloDataset(config.val_anno_path)

    # 获取先验框anchors，anchors[0]是对应尺度最小的输出,以此类推
    with open(config.anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = torch.Tensor(anchors).view(-1, 3, 2)

    yolo_losses = []
    for i in range(2, -1, -1):
        yolo_losses.append(YOLO4Loss(anchors[i]))

    max_epoch = config.epoch
    for epoch in range(max_epoch):
        # 训练
        train_loss = 0
        val_loss = 0

        # 冻结前Free_epoch次迭代的主干网络参数，不让其参与训练更新
        Free_epoch=max_epoch//2
        # drop_last=True可将多出来不足一个batch_size的数据丢弃
        if epoch < Free_epoch:
            lr = 1e-4
            Batch_size = 4
            optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=config.weight_decay)
            train_loader = DataLoader(train_set, shuffle=True, batch_size=Batch_size, collate_fn=my_collate_fn,
                                      num_workers=config.num_workers, drop_last=True)
            val_loader = DataLoader(val_set, shuffle=False, batch_size=Batch_size,
                                    num_workers=config.num_workers, drop_last=True)
            # 不让参数参与训练更新
            for param in net.backbone.parameters():
                param.requires_grad = False
        else:
            lr = 1e-5
            Batch_size = 2
            optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=config.weight_decay)
            train_loader = DataLoader(train_set, shuffle=True, batch_size=Batch_size,collate_fn=my_collate_fn,
                                      num_workers=config.num_workers, drop_last=True)
            val_loader = DataLoader(val_set, shuffle=False, batch_size=Batch_size,
                                    num_workers=config.num_workers, drop_last=True)
            for param in net.backbone.parameters():
                param.requires_grad = True
        # 是否使用余弦退火学习率
        if config.cosine_lr:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        for ii, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            # 重写collate_fn函数后，这里需要如下处理
            inputs = Variable(torch.from_numpy(data[0]).type(torch.FloatTensor))
            targets = [Variable(target) for target in data[1]]

            if config.use_gpu:
                inputs = inputs.cuda()
                targets = targets.cuda()
            # 梯度清零
            optimizer.zero_grad()
            outputs = net(inputs)

            loss = []
            num_pos_all = 0
            for i in range(3):
                # num_pos表示需要计算损失的先验框数量,也就是统计一个计算了多少次损失
                if config.use_gpu:
                    loss_item, num_pos = yolo_losses[i](outputs[i].cpu(), targets.cpu())
                else:
                    loss_item, num_pos = yolo_losses[i](outputs[i], targets)
                loss.append(loss_item)
                num_pos_all += num_pos
            loss = sum(loss) / num_pos_all
            train_loss += loss.item()
            # {:.2f}保留两位小数
            print('{} epoch loss:{:.2f}'.format(epoch + 1, train_loss / (ii + 1)))

            # 反向传播与梯度更新
            loss.backward()
            optimizer.step()

        # 学习率更新
        lr_scheduler.step()

        # #每训练完一个epoch验证一次：
        # net.eval()
        # print('Start Validation {} epoch...'.format(epoch+1))
        #
        # for jj ,data in tqdm(enumerate(val_loader),total=len(val_loader)):
        #     val_images=Variable(data[0])
        #     val_targets=Variable(data[1])
        #
        #     if config.use_gpu:
        #         val_images=val_images.cuda()
        #         val_targets=val_targets.cuda()
        #
        #     optimizer.zero_grad()
        #     outputs=net(val_images)
        #     loss=[]
        #     num_pos_all=0
        #     for i in range(3):
        #         if config.use_gpu:
        #             loss_item,num_pos=yolo_losses[i](outputs[i].cpu(),val_targets.cpu())
        #         else:
        #             loss_item, num_pos = yolo_losses[i](outputs[i], val_targets)
        #         loss.append(loss_item)
        #         num_pos_all+=num_pos
        #     loss=sum(loss)/num_pos_all
        #     val_loss+=loss
        #
        #     print('val_loss:{:.2f}'.format(val_loss / (jj + 1)))

        #每次模型保存
        torch.save(net.state_dict(), 'yolov4_model_weight{}.pth'.format(epoch))

if __name__=='__main__':
    train()





