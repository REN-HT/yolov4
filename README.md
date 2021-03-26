## yolov4 简介
这是一个yolov4算法的简单实现，可用于了解其整个算法流程，在此基础上也可进一步实现更多数据增强，调整参数等，代码中有很多注释，便于理解。
## environment  
os：windows10  
software：PyCharm Community Edition 2020.3.1  
DL framework：pytorch1.2  
interpreter：python3.7.6  
package：numpy,torch,torchvision,PIL,tqdm,
## implementation  
1. 修改config.py中相应参数，训练、验证，测试数据路径替换为自己本地路径。  
2. 自己定义的输入图片大小需要更改相应先验框anchors大小,手动运行utils/kmeans_for_anchors.py完成先验框聚类。  
3. 运行train.py完成训练，训练后需要验证代码去掉注释即可。  
4. 运行main.py可完成单张图片目标检测。  
## notes
代码中未上传训练好的模型，训练时可去掉train.py中这句' net = transfer_model('yolo4_weights.pth', net)'，当然这就需要长时间训练。
如果需要导入别人训练好的模型，最好同时替换backbone，这样保证网络参数更新时能够找到对应的参数名。
