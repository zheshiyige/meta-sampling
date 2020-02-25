import torch
from torchkit import flows, nn as nn_, utils
from torch import optim, nn
from torch.autograd import Variable
import os
from scipy.spatial.distance import pdist, squareform
import numpy as np
from  torch.distributions.multivariate_normal import MultivariateNormal
import argparse
from sklearn.model_selection import train_test_split
import numpy.matlib as nm
import sys
from tqdm import tqdm


class Meta_Sampler(object):
    
    def __init__(self, args):

        self.args = args
        flowtype = args.flowtype
        dimh= args.dimh
        num_hid_layers= args.num_hid_layers
        act=nn.ELU()
        num_flow_layers=args.num_flow_layers
        num_ds_dim=args.num_ds_dim
        num_ds_layers= args.num_ds_layers
        lr= args.lr
        betas=(args.beta1, args.beta2)
        self.n = args.num_particle
        self.dim= args.dim
        self.lamb = args.lamb
        self.normal = MultivariateNormal(torch.zeros(self.dim), torch.eye(self.dim))
        if flowtype == 0:
            flow = lambda **kwargs:flows.IAF_DSF(num_ds_dim=num_ds_dim,
                                                 num_ds_layers=num_ds_layers,
                                                 **kwargs)
            
        elif flowtype == 1:
            flow = lambda **kwargs:flows.IAF_DDSF(num_ds_dim=num_ds_dim,
                                                  num_ds_layers=num_ds_layers,
                                                  **kwargs)
            

        sequels = [nn_.SequentialFlow(
            flow(dim=self.dim,
                 hid_dim=dimh,
                 context_dim=1,
                 num_layers=num_hid_layers+1,
                 activation=act,
                 fixed_order=True),
            flows.FlipFlow(1)) for i in range(num_flow_layers)] + \
                  [flows.LinearFlow(self.dim, 1),]
        
        self.mdl = nn.Sequential(
                *sequels)
    
        self.optim = optim.Adam(self.mdl.parameters(), lr=lr, betas=betas)
        self.context = Variable(torch.FloatTensor(self.n, 1).zero_()) + 2.0
        self.lgd = Variable(torch.FloatTensor(self.n).zero_())
        self.zeros = Variable(torch.FloatTensor(self.n, 2).zero_())
        
    # generate samples from neural inverse autoregressive flow
    def sample_NIAF(self, spl, lgd=None, context=None, zeros=None):
        lgd = self.lgd if lgd is None else lgd
        context = self.context if context is None else context
        zeros = self.zeros if zeros is None else zeros
        z, logdet, _ = self.mdl((spl, lgd, context))
      
        return z

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
        
        return kernal_xj_xi, h

    #calculate the Wasserstein Gradient flow for one step
    def WGF_step(self, z_gen):
           
        kernal_xj_xi, h = self.kernal_dist(z_gen, h=-1)
        kernal_xj_xi, h = kernal_xj_xi.float(), h.astype(float)
        d_kernal_xi = torch.zeros(z_gen.size())

        F1_d_kernal_xi = torch.zeros(z_gen.size())
        F2_d_kernal_xi = torch.zeros(z_gen.size())
        F_delta = torch.zeros(z_gen.size())
        x = z_gen
        part_kernel = torch.sum(kernal_xj_xi, dim = 1).unsqueeze(1)
        for i_index in range(x.size()[0]):  
             quot_ele = torch.div(x[i_index] - x, part_kernel)
             F1_d_kernal_xi[i_index] = torch.matmul(kernal_xj_xi[i_index], quot_ele)* 2 / h
             F2_d_kernal_xi[i_index] = torch.matmul(kernal_xj_xi[i_index], x[i_index] - x)/(torch.sum(kernal_xj_xi[i_index])) * 2 / h      
             d_kernal_xi[i_index] = torch.matmul(kernal_xj_xi[i_index], x[i_index] - x) * 2 / h

        
        for i_index in range(x.size()[0]):
             ind_targrad = self.target_grad(x[i_index].unsqueeze(0).data.numpy())
             F_delta[i_index] = torch.tensor(ind_targrad).float() - F1_d_kernal_xi[i_index] - F2_d_kernal_xi[i_index]

        tar_grad = torch.tensor(self.target_grad(x.data.numpy())).float()
        # + self.lamb*F_delta   
        current_grad = (torch.matmul(kernal_xj_xi, tar_grad) + d_kernal_xi)/x.size(0) + self.lamb*F_delta
    
        return current_grad


    # train the NIAF and evaluate the model
    def fit(self, dlnprob, evaluation, total, stepsize):

           self.target_grad =  dlnprob
           mean = np.zeros((self.dim))
           cov = np.identity(self.dim)
           prev  = np.random.multivariate_normal(mean, cov, self.n)
           best_acc = 0
           for it in tqdm(range(total)):
              self.optim.zero_grad()       
              rand_gauss = torch.randn((self.n, self.dim))
              gen_inp = rand_gauss + torch.tensor(prev).float()    
              z_gen = self.sample_NIAF(gen_inp)
              stein_grad = self.WGF_step(z_gen) 
              self.optim.zero_grad()
              z_gen.backward(-1.0*stein_grad, retain_graph=True)
              self.optim.step()
              z_np = z_gen.cpu().data.numpy()
              prev = z_np + stepsize*stein_grad.cpu().data.numpy() 
              if ((it) % 100) == 0:
                   acc, llh = evaluation(z_np)            
                   if acc> best_acc:
                       best_acc = acc
                  
           return best_acc



