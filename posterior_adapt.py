from torchkit import flows, nn as nn_, utils
from torch import optim, nn
from torch.autograd import Variable
import torch
import os
from scipy.spatial.distance import pdist, squareform
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from torch.nn.parameter import Parameter
from torch.autograd import grad
import pickle
from tqdm import tqdm

def plot_fig(it, spl, data_path, sample_path):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.hist(spl.data.numpy()[:,0],100)
        plt.xlabel('f', fontsize=15)
        ax.set_yticklabels([])
        plt.title('$f\sim q(f)$', fontsize=18)
        plt.grid()

        spl = spl.data.numpy()
        mdl_density = gaussian_kde(spl[:,0],0.05)
        xx = np.linspace(0,2,1000)

        plt.plot(xx,300*mdl_density(xx),'r')
        plt.tight_layout()
        plt.legend(['kde', 'counts'], loc=2, fontsize=20)
        plt.savefig(sample_path+str(it)+'.png',format='png')
        np.save(data_path+str(it), spl)
        plt.close()


class  Meta_Posterior_adapt(object):
    
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
        self.model_path = args.model_path
        self.sample_path = args.meta_sample_path
        self.data_path = args.meta_data_path
        self.adapt_sample_path = args.adapt_sample_path
        self.adapt_data_path = args.adapt_data_path
        self.sf = flows.SigmoidFlow(num_ds_dim=num_ds_dim)
        self.params = Parameter(torch.FloatTensor(1, 1, 3*num_ds_dim).normal_())
        
        self.optim = optim.Adam([self.params,], lr=0.01, 
                                betas=(0.9, 0.999))      
        self.n_test = args.num_test

    def sample(self, n):
        
        spl = Variable(torch.FloatTensor(n,1).normal_())
        lgd = Variable(torch.from_numpy(
            np.zeros(n).astype('float32')))
        h, logdet = self.sf.forward(spl, lgd, self.params)
        spl = nn_.sigmoid_(h) * 2.0
        return spl


    def gradient(self, freq, a0, f0, b0, data):

        x0 = utils.varify(np.array(data).astype('float32'))
        y0 = torch.mul(torch.sin(x0*2.0*np.pi*f0+b0), a0)
        f = Variable(freq.data.clone(),requires_grad=True)
        mu = torch.mul(torch.sin(x0.permute(1,0)*2.0*np.pi*f+b0), a0)
        log_likelihood = - ((mu-y0.permute(1,0))**2 * (1/0.25)).sum(1)
        dtheta_data = grad(log_likelihood, f,torch.ones(log_likelihood.data.shape),allow_unused=True,retain_graph=False)[0]

        return dtheta_data

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
            F_delta[i_index] = self.target_grad[i_index] - F1_d_kernal_xi[i_index] - F2_d_kernal_xi[i_index]

        current_grad = (torch.matmul(kernal_xj_xi, 0.3*self.target_grad) + d_kernal_xi) + x.size(0)*self.lamb*F_delta
        return current_grad


        
    def train_meta_sampler(self, total=1000):
 
        if not os.path.exists(self.data_path):
           os.makedirs(self.data_path)
        if not os.path.exists(self.sample_path):
           os.makedirs(self.sample_path)
        if not os.path.exists(self.model_path):
           os.makedirs(self.model_path)

        [os.remove(os.path.join(self.sample_path, filename)) for filename in os.listdir(self.sample_path) if filename.endswith('.png')]  
        [os.remove(os.path.join(self.data_path, filename)) for filename in os.listdir(self.data_path) if filename.endswith('.npy')] 
        [os.remove(os.path.join(self.model_path, filename)) for filename in os.listdir(self.model_path) if filename.endswith('.pickle')] 
      
        a0 = 1.0
        f0 = 5.0/4.0
        b0 = 0.0
        data = [[0.0],[2/5.],[4/5.]]

        for it in tqdm(range(total)):
              self.optim.zero_grad()       
              z_gen = self.sample(self.n)                            
              self.target_grad  = self.gradient(z_gen, a0, f0, b0, data)
              stein_grad = self.WGF_step(z_gen) 
              self.optim.zero_grad()
              z_gen.backward(-1.0*stein_grad,retain_graph=True)
              self.optim.step()
              if it%100==0:
                  spl = self.sample(self.n_test)
                  plot_fig(it, spl, self.data_path, self.sample_path)
                  with open(self.model_path +str(it) + 'gen_model.pickle', 'wb') as handle:
                      pickle.dump(self.params, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def adapt_meta_sampler(self, total=500):
           a0 = 1.0
           f0 = 5.0/6.0
           b0 = 0.0
           data = [[0.0],[3/5.],[6/5.]]

           '''
           a0 = 1.0
           f0 = 5.0/4.0
           b0 = 0.0
           data = [[0.0],[4/5.],[8/5.]]
           '''


           if not os.path.exists(self.adapt_data_path):
              os.makedirs(self.adapt_data_path)
           if not os.path.exists(self.adapt_sample_path):
              os.makedirs(self.adapt_sample_path)
           [os.remove(os.path.join(self.adapt_sample_path, filename)) for filename in os.listdir(self.adapt_sample_path) if filename.endswith('.png')]  
           [os.remove(os.path.join(self.adapt_data_path, filename)) for filename in os.listdir(self.adapt_data_path) if filename.endswith('.npy')] 
           with open(self.model_path+'2200gen_model.pickle', 'rb') as handle:
               load_params = pickle.load(handle)
           
           self.params.data = load_params.data
           for it in tqdm(range(total)):

              self.optim.zero_grad()       
              z_gen = self.sample(self.n)                            
              self.target_grad = self.gradient(z_gen, a0, f0, b0, data)
              stein_grad = self.WGF_step(z_gen) 
              self.optim.zero_grad()
              z_gen.backward(-1.0*stein_grad,retain_graph=True)
              self.optim.step()
              if it%100==0:                  
                  spl = self.sample(self.n_test)
                  plot_fig(it, spl, self.adapt_data_path, self.adapt_sample_path)

