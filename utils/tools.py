import torch

def non_max_suppression(pre_output,conf_thres=0.5,iou_thres=0.4):
    #求左上角和右下角
    box_corner=torch.FloatTensor(pre_output.shape)
    box_corner[...,0]=pre_output[...,0]-pre_output[...,2]/2
    box_corner[...,1]=pre_output[...,1]-pre_output[...,3]/2
    box_corner[...,2]=pre_output[...,0]+pre_output[...,2]/2
    box_corner[...,3]=pre_output[...,1]+pre_output[...,3]/2
    pre_output[...,:4]=box_corner[...,:4]

    output=[None for _ in range(len(pre_output))]
    for i,img_pre in enumerate(pre_output):
        #得到置信度
        max_score, class_pre=torch.max(img_pre[:,5:],1,keepdim=True)
        #获得掩码,用于第一轮筛选,结果为一维
        conf_mask=(img_pre[:,4]*max_score[:,0]>=conf_thres).squeeze()
        #通过掩码，去掉小于阈值的框框
        img_pre=img_pre[conf_mask]
        max_score=max_score[conf_mask]
        class_pre=class_pre[conf_mask]

        #获得的内容为(x1, y1, x2, y2, conf, class_maxscore, class_pred)
        detections=torch.cat((img_pre[:,:5],max_score.float(),class_pre.float()),1)
        #获得检测出的所有种类标签
        labels=detections[:, -1].unique()

        for cls in labels:
            detections_class=detections[detections[:,-1]==cls]

            # 使用官方自带的非极大抑制
            # from torchvision.ops import nms
            # keep = nms(
            #     detections_class[:, :4],
            #     detections_class[:, 4] * detections_class[:, 5],
            #     nums_thres
            # )

            keep=nms(detections_class[:,:4],
                               detections_class[:, 4] * detections_class[:, 5],
                               iou_thres)
            max_detections=detections_class[keep]
            output[i]=max_detections if output[i] is None else torch.cat((output[i],max_detections))
    #输出为经过非极大抑制处理的结果，三维结果，第一维表示预测图片数量，之后每一维有7个值为[x1, y1, x2, y2, conf, score, cls]
    return output


def nms(bboxs, scores, threshold):
    x1 = bboxs[:, 0]
    y1 = bboxs[:, 1]
    x2 = bboxs[:, 2]
    y2 = bboxs[:, 3]
    areas = (y2 - y1) * (x2 - x1)  # 每个bbox的面积

    # order为排序后的得分对应的原数组索引值
    _, order = scores.sort(0, descending=True)

    keep = []  # 保存所有结果框的索引值。
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
        else:
            i = order[0].item()
            keep.append(i)

        # 计算最大得分的bboxs[i]与其余各框的IOU
        xx1 = x1[order[1:]].clamp(min=int(x1[i]))
        yy1 = y1[order[1:]].clamp(min=int(y1[i]))
        xx2 = x2[order[1:]].clamp(max=int(x2[i]))
        yy2 = y2[order[1:]].clamp(max=int(y2[i]))
        inter = (yy2 - yy1).clamp(min=0) * (xx2 - xx1).clamp(min=0)
        iou = inter / (areas[i] + areas[order[1:]] - inter)  # 如果bboxs长度为N，则iou长度为N-1

        # 保留iou小于阈值的剩余bboxs,.nonzero().squeeze()转化为数字索引，可验证
        idx = (iou <= threshold).nonzero().squeeze()
        if idx.numel() == 0:
            break
        order = order[idx + 1]  # idx+1表示对其每个值加一(保证和原来的order中索引一致)，并替换原来的order

    return keep




