import matplotlib.pyplot as plt
import torch
import random

#构造人造数据集
def synthetic_data(w,b,num_examples):
    x=torch.normal(0,1,(num_examples,len(w)))    #normal(mean,std,size)
    y=torch.matmul(x,w)+b
    print(y.shape)
    y+= torch.normal(0,0.01,y.shape)    #加入噪声，使得y值不完全等于w*x+b
    return x,y.reshape((-1,1))

true_w=torch.tensor([2,-3.4])
true_b=torch.tensor([4.2])
train_data,train_target=synthetic_data(true_w,true_b,1000)    #train_data里面有两个特征，train_data[:,0]特征1， train_data[:,1]特征2
# print(train_data,train_target)
def show_data():
    plt.figure(figsize=(5,4))
    plt.scatter(train_data[:,1].detach().numpy(),train_target.detach().numpy(),1)
    plt.show()         #根据图像可以看出呈现负相关


#设计函数读取小批量  不太会
def data_iter(batch_size,features,labels):
    num_examples=len(features)
    indices=list(range(num_examples))
    random.shuffle(indices)      #创建一个和数据等长的列表，并打乱进行随机读取
    for i in range(0,num_examples,batch_size):
        batch_indices=torch.tensor(indices[i:min(i+batch_size,num_examples)])
        yield features[batch_indices],labels[batch_indices]     #类似iter


#初始化各参数
w=torch.normal(0,0.01,size=(2,1),requires_grad=True)
b=torch.zeros(1,requires_grad=True)

#定义模型
def linreg(X,w,b):
    return torch.matmul(X,w)+b

#定义损失函数loss
def squared_loss(y_pred,y):
    loss=(y_pred-y.reshape(y_pred.shape))**2/2
    return loss

#定义优化函数  最不会
def SGD(params,lr,batch_size):      #params参数包含w，b
    with torch.no_grad():
        for param in params:
            param-=lr*param.grad/batch_size
            param.grad.zero_()          #梯度设零

#训练
if __name__=='__main__':
    batch_size = 10
    for batch_data, batch_target in data_iter(batch_size=batch_size, features=train_data, labels=train_target):
        print(batch_data)
        print(batch_target)

    lr=0.03
    batch_size=10
    num_epoch=10
    net=linreg
    loss=squared_loss
    for epoch in range(num_epoch):
        for data,target in data_iter(batch_size=batch_size,features=train_data,labels=train_target):
            l=loss(net(data,w,b),target)
            l.sum().backward()
            SGD([w,b],lr,batch_size)
        with torch.no_grad():
            train=loss(net(train_data,w,b),train_target)
            print(f"epoch:{epoch+1},loss:{float(train.mean()):f}")

    print(f'w的估计误差：{true_w-w.reshape(true_w.shape)}')
    print(f'b的估计误差：{true_b - b.reshape(true_b.shape)}')