from torch.utils import data
import numpy as np
import torch
from _08线性回归 import synthetic_data
from torch import nn

true_w = torch.tensor([2, -3.4])
true_b = torch.tensor([4.2])
feature, labels = synthetic_data(true_w, true_b, 1000)


#创建数据迭代器
def load_array(data_array,batch_size,is_train=True):
    dataset=data.TensorDataset(*data_array)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)


if __name__=='__main__':
    batch_size = 10
    data_iter = load_array((feature, labels), batch_size)
    print(next(iter(data_iter)))

    net=nn.Sequential(nn.Linear(2,1))
    #模型初始化
    net[0].weight.data.normal_(0,0.01)
    net[0].bias.data.fill_(0)

    loss=nn.MSELoss()
    optim=torch.optim.SGD(net.parameters(),lr=0.03)

    epochs=10
    for epoch in range(epochs):
        for data,target in data_iter:
            ls=loss(net(data),target)
            optim.zero_grad()
            ls.backward()
            optim.step()
        ls=loss(net(feature),labels)
        print(f'第{epoch+1}轮时，loss的值为{ls:f}')

