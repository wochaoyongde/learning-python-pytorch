import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from d2l import torch as d2l

def show_image(dataset):
    plt.figure(figsize=(4, 3))
    dataset_iter = iter(dataset)
    img = next(dataset_iter)
    npimg = img.numpy()
    plt.imshow(npimg)
    plt.show()


def get_dataloader_workers() -> int:
    return 4

def Linear(w: object, x: object, b: object) -> object:
    return Softmax(torch.matmul(x.reshape((-1,w.shape[0])),w)+b)

def Softmax(x):
    x_exp=torch.exp(x)
    partition=x_exp.sum(1,keepdim=True)   #将行的进行求和
    return x_exp/partition

#用于验证softmax函数
# x=torch.normal(0,1,(2,5))
# x_prob=Softmax(x)
# print(x_prob,x_prob.shape)

if __name__=='__main__':
    # dataste
    trans = transforms.ToTensor()

    dataset_train = torchvision.datasets.FashionMNIST(root="../data", transform=trans, train=True, download=True)
    dataset_test = torchvision.datasets.FashionMNIST(root="../data", transform=trans, download=True, train=False)

    dataloader_train = DataLoader(dataset_train, 256, shuffle=True)
    dataloader_test = DataLoader(dataset_test, 256, shuffle=True)

    # print(len(dataset_train),len(dataset_test))
    # print(dataset_train[0][0].shape)
    # print(dataset_test.classes)

    # 可视化
    dataset_iter = iter(dataloader_train)
    img, labels = next(dataset_iter)

    img_grid = torchvision.utils.make_grid(img)
    show_image(img_grid)








