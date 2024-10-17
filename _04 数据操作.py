import torch

#张量
x=torch.arange(12)
print(x.shape)  #查看形状,Tensor的查看方式
print(x.numel())  #查看数量

#修改张量形状
X=x.reshape(3,4)
y=torch.zeros((2,3,4))

A=torch.tensor([[1,2],[2,3]])   #输入了列表
print(A)

#数值运算
x=torch.tensor([1.0,2,4,8])
y=torch.tensor([2,2,2,2])
print(x+y,x-y,x*y,x/y,x**y)

#矩阵拼接
x=torch.arange(12,dtype=torch.float32).reshape((3,4))   #reshape输入元组
y=torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
cat1=torch.cat((x,y),dim=0)   #按列拼接
cat2=torch.cat((x,y),dim=1)   #按行拼接
print(cat1)
print(cat2)

print(x==y)

#求和
print(x.sum())  #可以设置dim

#广播机制
a=torch.arange(3).reshape((3,1))
b=torch.arange(2).reshape((1,2))
print(a,b)
print(a+b)

#切片
print(x[-1],x[1:3])

#修改
x[1,2]=9
print(x)

#新结果分配新内存
before=id(y)
y=y+X

#减少内存的开销
print(id(y) == before)    #分配了新的内存

before = id(x)
x+= y
print(id(x) == before)

#转换NumPy
a=x.numpy()
print(type(a))

A=x.numpy()
B=torch.tensor(A)

a=torch.tensor([3.5])
print(a,a.item(),float(a),float(b))

