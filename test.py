import torch
from model.yolo4 import Yolov4
from torch.autograd import Variable as V

#一个输出，一共三个输出，其余两个输出类似
# Yolov4Out = t.Tensor([[[[1,2],
#                        [2,3]],
#                       [[3,4],
#                        [1,3]]]])
#
# #model=Yolov4()
# input=t.randn(3)
# out=input.view(3,1)
# x=t.randn(2,3)
# y=t.randn(2,1)
# z=t.cat((x,y),-1)
#
# anchors=t.Tensor([[2, 16,13],[19,36,24], [40, 28,55]])
# x,y=t.max(anchors[:, 1:],1)
# print(x)
# print(y)


# x=t.Tensor([])
# y=t.Tensor([[1,5],[6,4],[1,5],[3,6]])
# try:
#     mask1=(x[:,1]>3)
#     print(x[mask1,:4])
# except:
#     print('error!')
# import torch as t
# import numpy as np
# target=t.Tensor([[1,2,3,4,5],
#                  [2,3,5,6,7]])
#
# x=t.Tensor([[0,0,2,5],
#             [0,0,3,7]])
#
# y=t.Tensor([[0,0,1,2],
#             [0,0,4,5],
#             [0,0,3,7],
#            [0,0,5,8]])

# loss=t.Tensor([0.1234])[0]
# print('loss:{:.2f} ,last_loss:{:.2f}'.format(loss,loss))
#-----------------------------------------------------------------------#
# anchor=torch.Tensor([1,2,3])
# out=torch.randn(2,3,17,17)
# anchors=anchor.view(3,1).repeat(2,1).view(2,3,-1).repeat(1,1,17*17).view(out.shape)
# print(anchors)
#-----------------------------------------------------------------------#
# import torch
# x=torch.Tensor([[[0,0,2,5],
#                 [0,0,3,7]],
#                 [[0, 1, 2, 5],
#                  [0, 1, 3, 7]]
#                 ])
#
# y=torch.Tensor([[[0,0,1,2],
#                 [0,0,4,5]],
#                 [[0, 2, 1, 2],
#                  [0, 2, 4, 5]]
#                 ])
#
# z=torch.cat((x,y),1)
# print(z)

# import numpy as np
# x=np.array([[1,2,3],
#           [4,5,6]])
# print(np.expand_dims(x[:,0],-1))

# import torch
# iou=torch.Tensor([1,2,3,4,5,6,7])
# idx = (iou <=4).nonzero().squeeze()
# print(idx)


# path='C:\AllProgram\DatasetPath\sample_submission.csv'
#
# with open(path) as f:
#     test=f.readlines()
# print(test[1].split(',')[1].split())

x=[1]
y=x[1:]
print(y)




