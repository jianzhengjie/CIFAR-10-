import torch
import torchvision
import torchvision.transforms as transforms
def get_data():
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                     std=(0.5, 0.5, 0.5)),
                                ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    print("数据下载完成")


get_data()