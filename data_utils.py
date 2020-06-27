import sys
import numpy
import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from BNN_Dataloader import *



def get_mnistdata(batch_size=16, num_workers=2, shuffle=True):

    transform_train = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    mnist_training = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)

    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    X_train_sampler_tensor,X_train_BNN_tensor,Y_train_sampler_tensor,Y_train_BNN_tensor,X_test_BNN_tensor,Y_test_BNN_tensor,X_test_sampler_tensor,Y_test_sampler_tensor=\
    SelectImage(mnist_training, mnist_test,train_image=[0,1,2,3,4],test_image=[5,6,7,8,9])
    MNIST_train_sampler=GroupMNIST(X_train_sampler_tensor,Y_train_sampler_tensor,group=1)
    MNIST_train_sampler_loader = DataLoader(MNIST_train_sampler, batch_size=batch_size,
                        shuffle=True)
    MNIST_test_sampler=GroupMNIST(X_test_sampler_tensor,Y_test_sampler_tensor,group=4)
    MNIST_test_sampler_loader = DataLoader(MNIST_test_sampler, batch_size=batch_size,
                        shuffle=True)

    MNIST_train_BNN=GroupMNIST(X_train_BNN_tensor,Y_train_BNN_tensor,group=2)
    MNIST_train_BNN_loader = DataLoader(MNIST_train_BNN, batch_size=batch_size,
                        shuffle=True)

    MNIST_test_BNN=GroupMNIST(X_test_BNN_tensor,Y_test_BNN_tensor,group=3)
    MNIST_test_BNN_loader = DataLoader(MNIST_test_BNN, batch_size=batch_size,
                        shuffle=True)

    return MNIST_train_sampler_loader, MNIST_test_sampler_loader, MNIST_train_BNN_loader, MNIST_test_BNN_loader





