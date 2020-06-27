import numpy as np
import time, os
from wrappers import NIAFNet
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from torch.autograd import grad



class BNN_meta():

  def __init__(self, args, model, optimizer, device, train_loader, test_loader):
        self.args = args
        self.optimizer = optimizer
        self.model = model
        self.device = device
        self.count = 0
        self.accuracy = []
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lamb = args.lamb
        self.kl_w = args.kl_w


  #calculate pairwise kernel distance
  def kernal_dist(self, x, h=-1):
      
        x_numpy = x.cpu().data.numpy()
        init_dist = pdist(x_numpy)
        pairwise_dists = squareform(init_dist) 
        if h < 0:  # if h < 0, using median trick 
            h = np.median(pairwise_dists)
            h = 0.1*h ** 2 / np.log(x.shape[0] + 1)
        
        if x_numpy.shape[0]>1:
            kernal_xj_xi = torch.exp(- torch.tensor(pairwise_dists) ** 2 / h)
        else:
            kernal_xj_xi = torch.tensor([1])
        
        return kernal_xj_xi.to(x.device), h

  #calculate the Wasserstein Gradient flow for one step
  def WGF_step(self, z_gen):
           
        kernal_xj_xi, h = self.kernal_dist(z_gen, h=-1)
        kernal_xj_xi, h = kernal_xj_xi.float(), h.astype(float)
        d_kernal_xi = torch.zeros(z_gen.size()).to(z_gen.device)

        F1_d_kernal_xi = torch.zeros(z_gen.size()).to(z_gen.device)
        F2_d_kernal_xi = torch.zeros(z_gen.size()).to(z_gen.device)
        F_delta = torch.zeros(z_gen.size()).to(z_gen.device)
        x = z_gen
        part_kernel = torch.sum(kernal_xj_xi, dim = 1).unsqueeze(1).to(x.device)
        for i_index in range(x.size()[0]):  
             quot_ele = torch.div(x[i_index] - x, part_kernel)
             F1_d_kernal_xi[i_index] = torch.matmul(kernal_xj_xi[i_index], quot_ele)* 2 / h
             F2_d_kernal_xi[i_index] = torch.matmul(kernal_xj_xi[i_index], x[i_index] - x)/(torch.sum(kernal_xj_xi[i_index])) * 2 / h      
             d_kernal_xi[i_index] = torch.matmul(kernal_xj_xi[i_index], x[i_index] - x) * 2 / h
        
        for i_index in range(x.size()[0]):
             F_delta[i_index] =  self.tar_grad[i_index]  - F1_d_kernal_xi[i_index] - F2_d_kernal_xi[i_index]
       
        current_grad = (torch.matmul(kernal_xj_xi, self.tar_grad) + 0.2*d_kernal_xi)/x.size(0) + self.lamb*F_delta
    
        return current_grad


  def train(self, epoch, adaptation):
      self.model.train()
      for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data, same_noise=False)
            cross_entropy = F.nll_loss(output, target)
            regs = self.model.get_reg()
            loss = cross_entropy + self.kl_w * regs
            loss.backward(retain_graph=True)

            self.tar_grad = grad(-1.0*cross_entropy, self.model.layerout.z_gen, torch.ones(cross_entropy.data.shape).cuda(),allow_unused=False, retain_graph=True)[0]
            current_grad = self.WGF_step(self.model.layerout.z_gen)
            self.model.layerout.z_gen.backward(-1.0*current_grad, retain_graph=True)
            self.optimizer.step()
            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()))

      
  def test(self):
      self.model.eval()
      correct = 0

      with torch.no_grad():
         total = 0
         for data, target in self.test_loader:
            total = total + 1
            data, target = data.to(self.device), target.to(self.device)
            pred_count = torch.zeros(data.size(0), dtype =torch.long).to(self.device)
            for ind in range(self.args.num_sample):
                    output = self.model(data, same_noise=True)
                    pred = torch.squeeze(output.argmax(dim=1, keepdim=True)) # get the index of the max log-probability
                    pred_count+=pred # model average prediction
                    
            pred_count = pred_count/self.args.num_sample
            correct += pred_count.eq(target.view_as(pred_count)).sum().item()  #model average accuracy
		       
         print('\n accuracy {} \n'.format(
           100. * correct / len(self.test_loader.dataset)))

         acc = 1. * correct / len(self.test_loader.dataset)
         return acc
        

