import torch

x=torch.arange(4.0,requires_grad=True)    #required_grad表示要存储梯度
print(x.grad)

y=2 * torch.dot(x,x)
#tensor(28., grad_fn=<MulBackward0>)
print(y)

y.backward()  #求导
print(x.grad)    #输出梯度(倒数)

print(x.grad==4 * x)
# df(x)/dx = 4 * x
#tensor([True, True, True, True])

x.grad.zero_()    #梯度会累积，因此清楚之前的值
y = x.sum()
y.backward()
print(x.grad)


x.grad.zero_()
y= x*x
u=y.detach()    #此时的u应当是一个常数
z=u*x       #此时对z来说变量应该只有x
z.sum().backward()
print(x.grad==u)
