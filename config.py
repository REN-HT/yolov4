
class DefaultConfig(object):
    anchors_path= 'C:/AllProgram/Pytorch/yolov4/model_data/yolo_anchors.txt'
    coco_classes_path= 'C:/AllProgram/Pytorch/yolov4/model_data/coco_classes.txt'
    voc07_classes_path= 'C:/AllProgram/Pytorch/yolov4/model_data/voc_classes.txt'
    data_path='C:/AllProgram/DatasetPath/VOCdevkit/VOC2007/'
    #写入用,可以先不存在于文件夹中
    train_anno_path='C:/AllProgram/Pytorch/yolov4/model_data/train_annotations.txt'
    val_anno_path = 'C:/AllProgram/Pytorch/yolov4/model_data/val_annotations.txt'
    test_anno_path = 'C:/AllProgram/Pytorch/yolov4/model_data/test_annotations.txt'
    #模型参数载入
    model_path='C:/AllProgram/Pytorch/yolov4/model_data/yolo4_weights.pth'
    voc07_use_difficult_bbox = False
    use_gpu=False
    #此处改动需要改动predict.py中的_get_class_name函数
    nums_classes=20 
    model_image_h=416
    model_image_w=416
    image_size=416
    conf_thres = 0.6
    iou_thres = 0.5

    #训练相关参数
    cosine_lr=False #余弦退火学习率
    lr=0.01
    epoch=50
    batch_size=2
    num_workers=2
    weight_decay=5e-4
    ignore_thres=0.5
    label_smooth=0
    lambda_conf=1.0
    lambda_loc=1.0
    lambda_cls=1.0


config=DefaultConfig()
