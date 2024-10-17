#标量只有一个元素的张量
import torch
x=torch.tensor([3.0])
y=torch.tensor([2.0])

print(x+y,x-y,x*y,x/y)

#组成list
x=torch.arange(4)
print(x)
print(x[3])
print(len(x))
print(x.shape)

#创建矩阵
A=torch.arange(20).reshape((4,5))
print(A)

#矩阵的转置
print(A.T)

#对称矩阵
B=torch.tensor([[1,2,3],[2,0,4],[3,4,5]])
print(B)
print(B==B.T)

#多维矩阵
X=torch.arange(24).reshape(2,3,4)
print(X)

#常量+张量
a=2
X=torch.arange(24).reshape(2,3,4)
print(a+X,(a*X).shape)

#sum求和
print(X.sum(dim=2))    #0——2，1——3，2——4
print(X.sum(axis=[0,1]).shape)    #求和两个维度

#求平均值
X=torch.arange(24,dtype=torch.float32).reshape(2,3,4)
print(X.mean(),X.sum()/X.numel())
print(X.mean(axis=0),X.sum(axis=0)/X.shape[0])

#计算求和的时候保持轴数不变
sum_A=A.sum(axis=1,keepdim=True)   #keepdims保持维度不变，这样就可以做广播A除以sum_A,非常有用
print(A/sum_A)

#累加求和
print(A.cumsum(axis=0))   #累加每行求和第一行

#点积时相同位置相乘 -》求和
y=torch.ones(4,dtype=torch.float32)
x=torch.arange(4,dtype=torch.float32)
print(x,y,torch.dot(x,y))

#Ax=b
A=A.type(torch.long)
x=x.type(torch.long)
print(A.shape,x.shape,torch.mv(A.T,x))

#矩阵相乘Ax
B=torch.ones(4,3).type(torch.long)
print(torch.mm(A.T,B))

#范数，矩阵、向量的长度
#第二范数，平方求和开方
u=torch.tensor([3.0,-4.0])
print(torch.norm(u))

#第一范数，绝对值求和
print(torch.abs(u).sum())

#F范数，矩阵的各元素平方去和开平方
print(torch.norm(torch.ones([4,9])))