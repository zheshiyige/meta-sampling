from torchkit import flows, nn as nn_, utils
from torch import optim, nn
from torch.autograd import Variable
import torch
import os
from scipy.spatial.distance import pdist, squareform
import numpy as np
import math
import matplotlib.pyplot as plt
from toy_util import *

class Meta_Sampler(object):
    
    def __init__(self, args):
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
        self.meta_sample_path = args.meta_sample_path
        self.meta_data_path = args.meta_data_path
        self.adapt_sample_path = args.adapt_sample_path
        self.adapt_data_path = args.adapt_data_path
        #two types of neural inverse autoregressive flow
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
        
        
    # generate samples with neural inverse autoregressive flow
    def sample_NIAF(self, spl, lgd=None, context=None, zeros=None):
        lgd = self.lgd if lgd is None else lgd
        context = self.context if context is None else context
        zeros = self.zeros if zeros is None else zeros
        z, logdet, _ = self.mdl((spl, lgd, context))
        return z

    # calculate pairwise kernel distance and bandwidth
    def kernal_dist(self, x, h=-1):
        x_numpy = x.cpu().data.numpy()
        init_dist = pdist(x_numpy)
        pairwise_dists = squareform(init_dist) 

        if h < 0:  # if h < 0, using median trick
            h = np.median(pairwise_dists)
            h = 0.05*h ** 2 / np.log(x.shape[0] + 1)
        
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
            F_delta[i_index] = self.target_grad(x[i_index].unsqueeze(0)) - F1_d_kernal_xi[i_index] - F2_d_kernal_xi[i_index]

        current_grad = (torch.matmul(kernal_xj_xi, self.target_grad(x)) + d_kernal_xi) + x.size(0)*self.lamb*F_delta
        return current_grad


    #train the NIAF generator as meta sampler
    def train_meta_sampler(self, total=1000):
           range_limit = [-10, 10]  
           path = self.meta_sample_path
           sam_path = self.meta_data_path
           npoints_plot=50
           if not os.path.exists(path):
               os.makedirs(path)
           if not os.path.exists(sam_path):
               os.makedirs(sam_path)
           [os.remove(os.path.join(path, filename)) for filename in os.listdir(path) if filename.endswith('.png')]  
           [os.remove(os.path.join(sam_path, filename)) for filename in os.listdir(sam_path) if filename.endswith('.npy')] 

           
           target_mean = torch.tensor([[-6,8], [-2,8], [2,8], [6,8],
                                    [-6,4],  [6,4],
                                    [-6,0],  [6,0],
                                    [-6,-4],  [6,-4],
                                    [-6,-8], [-2,-8], [2,-8], [6,-8]]).float()
           
           num = target_mean.size(0)
           w = torch.tensor([1.0/num]*num).unsqueeze(1)
           var = 1.0
           Gauss_mix = Gauss_mixture(w, target_mean, var)
           self.target_grad = Gauss_mix.dlnprob
           self.target_den = Gauss_mix.MGprob
         
           for it in range(total):
              self.optim.zero_grad()       
              spl = 2*torch.randn((self.n, self.dim))       
              z_gen = self.sample_NIAF(spl)
              stein_grad = self.WGF_step(z_gen) 
              self.optim.zero_grad()
              z_gen.backward(-1.0*stein_grad,retain_graph=True)
              self.optim.step()
            
              if ((it) % 10) == 0:
                    print ('Iteration: [%4d/%4d]' % \
                       (it, total))
                    ax=plt.subplot(1,2,1, aspect='equal')
                    mesh_z1, mesh_z2, zv = evaluate_bivariate(range=range_limit, npoints=npoints_plot)
                    z_pp = torch.tensor(zv).float()
                    phat_z = -self.target_den(z_pp)
                    phat_z = phat_z.cpu().data.numpy()
                    phat_z = phat_z.reshape([npoints_plot,npoints_plot])
                    ax.pcolormesh(mesh_z1, mesh_z2, phat_z)
                    z_min, z_max = -np.abs(phat_z).max(), np.abs(phat_z).max()
                    plt.pcolor(mesh_z1, mesh_z2, phat_z, cmap='RdBu', vmin=z_min, vmax=z_max)
                    plt.xlim(range_limit); plt.ylim(range_limit); ax.set_title('Target distribution: $u(z)$')

                    z_np = z_gen.cpu().data.numpy()
                    ax=plt.subplot(1,2,2, aspect='equal')
                    plt.scatter(z_np[:,0], z_np[:,1], alpha= 1)
                    plt.xlim(range_limit); plt.ylim(range_limit); ax.set_title('generated particle: $p_k(z)$')
                    plt.savefig(path+str(it)+'.png')
                    np.save(sam_path+str(it), z_np)
                    plt.close()
           torch.save(self.mdl.state_dict(),'WGF_NIAF.pth')
           print('save model WGF_NIAF.pth')

    #adapt NIAF meta sampler 
    def adapt_meta_sampler(self, total=500):
           range_limit = [-10, 10]  
           path = self.adapt_sample_path
           sam_path = self.adapt_data_path
           npoints_plot=50
           if not os.path.exists(path):
               os.makedirs(path)
           if not os.path.exists(sam_path):
               os.makedirs(sam_path)

           [os.remove(os.path.join(path, filename)) for filename in os.listdir(path) if filename.endswith('.png')]  
           [os.remove(os.path.join(sam_path, filename)) for filename in os.listdir(path) if filename.endswith('.npy')] 
           self.mdl.load_state_dict(torch.load('WGF_NIAF.pth'))
           print('loading model WGF_NIAF.pth')
           
           target_mean = torch.tensor([[-6,8], [-2,8], [2,8], [6,8],
                                    [-6,4], [-2,4], [2,4], [6,4],
                                    [-6,0], [-2,0], [2,0], [6,0],
                                    [-6,-4], [-2,-4], [2,-4], [6,-4],
                                    [-6,-8], [-2,-8], [2,-8], [6,-8]]).float()

           num = target_mean.size(0)
           w = torch.tensor([1.0/num]*num).unsqueeze(1)
           var = 1.0
           Gauss_mix = Gauss_mixture(w, target_mean, var)
           self.target_grad = Gauss_mix.dlnprob
           self.target_den = Gauss_mix.MGprob
           for it in range(total):
              self.optim.zero_grad()       
              spl = 2*torch.randn((self.n, self.dim))                 
              z_gen = self.sample_NIAF(spl)
              stein_grad = self.WGF_step(z_gen) 
              self.optim.zero_grad()
              z_gen.backward(-1.0*stein_grad,retain_graph=True)
              self.optim.step()

              if ((it) % 10) == 0:
                    print ('Iteration: [%4d/%4d]' % \
                       (it, total))
                    ax=plt.subplot(1,1,1, aspect='equal')
                    mesh_z1, mesh_z2, zv = evaluate_bivariate(range=range_limit, npoints=npoints_plot)
                    z_pp = torch.tensor(zv).float()
                  
                    phat_z = -self.target_den(z_pp)
                    phat_z = phat_z.cpu().data.numpy()
                    phat_z = phat_z.reshape([npoints_plot,npoints_plot])
                    ax.pcolormesh(mesh_z1, mesh_z2, phat_z)
                    z_min, z_max = -np.abs(phat_z).max(), np.abs(phat_z).max()
                    plt.pcolor(mesh_z1, mesh_z2, phat_z, cmap='RdBu', vmin=z_min, vmax=z_max)
                    plt.xlim(range_limit); plt.ylim(range_limit); ax.set_title('Target distribution: $u(z)$')
                                   
                    
                    z_np = z_gen.cpu().data.numpy()
                    ax=plt.subplot(1,2,2, aspect='equal')
                    plt.scatter(z_np[:,0], z_np[:,1], alpha=1)
                    plt.xlim(range_limit); plt.ylim(range_limit); ax.set_title('generated particle: $p_k(z)$')
                    
                    plt.savefig(path+'_'+str(it)+'.png')
                    np.save(sam_path+'_'+str(it), z_np)
                    plt.close()
            
