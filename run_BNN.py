import time, os
from wrappers import NIAFNet
import torch
import torch.nn as nn
import torch.optim as optim
from BNN_Dataloader import *
from data_utils import *
from Meta_BNN import *

def meta_train(args, MNIST_train_sampler_loader, MNIST_test_sampler_loader, device):
    train_loader = MNIST_train_sampler_loader
    test_loader = MNIST_test_sampler_loader
   
    height, width, n_channels = 28, 28, 1
    input_shape = [height, width, n_channels]
    model = NIAFNet(input_shape=input_shape, flows_q=args.fq, flows_r=args.fr, activation=nn.ReLU(), nb_classes=5, thres_var=args.thres_var, flow_dim_h=args.flow_h, n_hidden = args.n_hidden)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    adaptation = False
    meta_MNIST = BNN_meta(args, model, optimizer, device, train_loader, test_loader)
    for epoch in range(1, args.epochs + 1):
        meta_MNIST.train(epoch, adaptation = False)
        meta_MNIST.test()
        model_path = "model_save/mnist_cnn_{}.pt".format(epoch)
        torch.save(meta_MNIST.model.state_dict(), model_path)     

   

def meta_test(args, MNIST_train_BNN_loader, MNIST_test_BNN_loader, device):

    train_loader = MNIST_train_BNN_loader
    test_loader = MNIST_test_BNN_loader

    model_path = "model_save/mnist_cnn_10.pt"
    height, width, n_channels = 28, 28, 1
    input_shape = [height, width, n_channels]
    model = NIAFNet(input_shape=input_shape, flows_q=args.fq, flows_r=args.fr, activation=nn.ReLU(), nb_classes=5,
                     thres_var=args.thres_var, flow_dim_h=args.flow_h, n_hidden = args.n_hidden)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model.load_state_dict(torch.load(model_path))
    print('meta adapation load model')
    meta_MNIST = BNN_meta(args, model, optimizer, device, train_loader, test_loader)
    for epoch in range(1, args.epochs + 1):
        meta_MNIST.train(epoch, adaptation = True)
 

def main():
    import argparse
    desc = "Meta_BNN"
    parser = argparse.ArgumentParser(description=desc)  
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-epzero', type=int, default=1)
    parser.add_argument('-fq', default=2, type=int)
    parser.add_argument('-fr', default=2, type=int)
    parser.add_argument('-num-sample', type=int, default=20, metavar='N',
                        help='number of test sample (default: 20)')
    parser.add_argument('-no_z', action='store_true')
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-lr', type=float, default=0.002)
    parser.add_argument('-thres_var', type=float, default=0.5)
    parser.add_argument('-n-hidden', type=int, default=0)
    parser.add_argument('-flow_h', type=int, default=50)
    parser.add_argument('-kl_w', type=float, default=1e-5)
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--lamb', type=float, default=1e-5, help='lambda for balance the two terms in the Wasserstein Gradient Flow')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    
    args = parser.parse_args()
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    MNIST_train_sampler_loader, MNIST_test_sampler_loader, MNIST_train_BNN_loader, MNIST_test_BNN_loader = get_mnistdata(
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )
    meta_train(args, MNIST_train_sampler_loader, MNIST_test_sampler_loader, device)
    meta_test(args, MNIST_train_BNN_loader, MNIST_test_BNN_loader, device)

    

if __name__ == '__main__':

    main()
