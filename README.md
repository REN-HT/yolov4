## yolov4 简介
这是一个yolov4算法的简单实现，可用于了解其整个算法流程，在此基础上可进一步丰富，比如实现更多数据增强，调整参数用于自己实际需求。
## implementation  
1. 修改config.py中相应参数，训练、验证，测试数据目录替换为自己本地目录  
2. 自己定义的输入图片大小需要更改相应的先验框anchors,手动运行utils/kmeans_for_anchors.py完成先验框聚类  
3. 运行train.py完成训练  
4. 运行main.py可检测  
## notes
代码中暂未上传训练好的模型，训练时可去掉train.py中这句' net = transfer_model('yolo4_weights.pth', net)'
