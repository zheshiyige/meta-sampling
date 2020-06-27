# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from norm_flows import MaskedNVPFlow
from norm_flows import NAF
import torch.nn as nn

class NIAFConv2D(nn.Module):
    def __init__(self, nb_filter, nb_row, nb_col, hidden_dim, n_hidden, stack_size, n_flows_q, n_flows_r,
                 use_cuda=True, prior_var=1.0, threshold_var=0.5):
        nn.Module.__init__(self)
        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.n_hidden = n_hidden
        self.n_flows_q = n_flows_q
        self.n_flows_r = n_flows_r
        self.prior_var = prior_var
        self.threshold_var = threshold_var
        self.use_cuda = use_cuda
        self.stack_size = stack_size
        self.weight_mu = nn.Parameter(torch.Tensor(self.nb_filter, self.stack_size, self.nb_row, self.nb_col))
        self.weight_logstd = nn.Parameter(torch.Tensor(self.nb_filter, self.stack_size, self.nb_row, self.nb_col))
        self.bias_mu = nn.Parameter(torch.Tensor(self.nb_filter))
        self.bias_logvar = nn.Parameter(torch.Tensor(self.nb_filter))

        self.qzero_mu = nn.Parameter(torch.Tensor(self.nb_filter))
        self.qzero_logvar = nn.Parameter(torch.Tensor(self.nb_filter))
        self.rzero_c = nn.Parameter(torch.Tensor(self.nb_filter))
        self.rzero_b1 = nn.Parameter(torch.Tensor(self.nb_filter))
        self.rzero_b2 = nn.Parameter(torch.Tensor(self.nb_filter))

        self.flow_q = NAF(flowtype=2, n=1, dim=self.nb_filter)
        self.flow_r = MaskedNVPFlow(self.nb_filter, hidden_dim, n_hidden, n_flows_r)
        self.register_buffer('epsilon_z', torch.Tensor(self.nb_filter))
        self.reset_parameters()
        self.reset_noise()

    def reset_noise(self):
        epsilon_z = torch.randn(self.nb_filter)
        self.epsilon_z.copy_(epsilon_z)
        self.flow_r.reset_noise()

    def reset_parameters(self):

        in_stdv = np.sqrt(4.0 / self.nb_filter)
        out_size = self.nb_row * self.nb_col*self.stack_size
        out_stdv = np.sqrt(4.0 / out_size)
        stdv2 = np.sqrt(4.0 / (self.nb_filter + out_size))

        self.weight_mu.data.normal_(0, stdv2)
        self.weight_logstd.data.normal_(-9, 1e-3 * stdv2)
        self.bias_mu.data.zero_()
        self.bias_logvar.data.normal_(-9, 1e-3 * out_stdv)

        self.qzero_mu.data.normal_(1 if self.n_flows_q == 0 else 0, in_stdv)
        self.qzero_logvar.data.normal_(np.log(0.1), 1e-3 * in_stdv)
        self.rzero_c.data.normal_(0, in_stdv)
        self.rzero_b1.data.normal_(0, in_stdv)
        self.rzero_b2.data.normal_(0, in_stdv)


    def sample_particle(self, batch_size, kl=True, same_noise=False):
        assert kl == False
        z = self.qzero_mu
        z = self.flow_q(z, kl=False)
        return z

    def sample_z(self, batch_size, kl=True, same_noise=False):
        if self.training:
            if batch_size > 1:
                assert kl == False
                qzero_std = torch.exp(0.5 * self.qzero_logvar)
                qzero_std = qzero_std.expand(batch_size, self.nb_filter)
                z_mu = self.qzero_mu.expand(batch_size, self.nb_filter)
                if same_noise:
                    epsilon_z = self.epsilon_z.expand(batch_size, self.nb_filter)
                else:
                    epsilon_z = Variable(torch.randn(batch_size, self.nb_filter))
                    if self.use_cuda:
                        epsilon_z = epsilon_z.cuda()

                z = z_mu + qzero_std * epsilon_z
                z  = self.flow_q(z, kl=False)
                return z
            if batch_size ==  1:
                qzero_std = torch.exp(0.5 * self.qzero_logvar)
                z =  self.qzero_mu + qzero_std * self.epsilon_z
                z = z.unsqueeze(0)
                if kl:
                    z, logdets = self.flow_q(z, kl=True)
                    return z, logdets
                else:
                    z = self.flow_q(z, kl=False)
                    return z
        else:
            assert kl == False
            z = self.qzero_mu
            z = self.flow_q(z, kl=False)
            return z

    def forward(self, x, same_noise=False):
        batch_size = x.size()[0]
        if self.training:
            z = self.sample_z(batch_size, kl=False, same_noise=same_noise)
            self.z_gen = z
            weight_std = torch.clamp(torch.exp(self.weight_logstd), 0, self.threshold_var)
            bias_std = torch.clamp(torch.exp(0.5 * self.bias_logvar), 0, self.threshold_var)
            

            out_mu = F.conv2d(x, self.weight_mu)
            z = z[:,:,None,None]
            bias_mu = self.bias_mu[None,:,None,None]
            out_mu = out_mu*z+ bias_mu
            out_var = F.conv2d(x*x, weight_std * weight_std, bias_std)
            
            if batch_size > 1:
                if same_noise:
                    epsilon_linear = self.epsilon_linear.expand(out_var.size())
                else:
                    epsilon_linear = Variable(torch.randn(out_var.size()))
                    if self.use_cuda:
                        epsilon_linear = epsilon_linear.cuda()
            if batch_size == 1:
                epsilon_linear = self.epsilon_linear

            out = out_mu + torch.sqrt(out_var) * epsilon_linear
            return out
        else:
            z = self.sample_z(1, kl=False)
            out_mu = F.conv2d(x, self.weight_mu)
            z = z[None,:,None,None]
            bias_mu = self.bias_mu[None,:,None,None]
            out = out_mu*z+bias_mu

            return out

    def kldiv(self):
        z, logdets = self.sample_z(1, kl=True)
        z = torch.squeeze(z)
        weight_mu = z[:,None,None,None]*self.weight_mu


        kldiv_weight = 0.5 * (- 2 * self.weight_logstd + torch.exp(2 * self.weight_logstd)
                              + weight_mu * weight_mu - 1).sum()
        kldiv_bias = 0.5 * (- self.bias_logvar + torch.exp(self.bias_logvar)
                            + self.bias_mu * self.bias_mu - 1).sum()

     
        logq = - 0.5 * self.qzero_logvar.sum()
        logq -= logdets[0]
        weight_mu = weight_mu.view(weight_mu.size(0), -1)
       

        cw_mu = torch.matmul(self.rzero_c, weight_mu)
        epsilon = Variable(torch.randn(cw_mu.size()))
        if self.use_cuda:
            epsilon = epsilon.cuda()
 
        weight_logstd = self.weight_logstd.view(self.weight_logstd.size(0),-1)
        cw_var = torch.matmul(self.rzero_c * self.rzero_c, torch.exp(2 * weight_logstd))
        cw = F.tanh(cw_mu + torch.sqrt(cw_var) * epsilon)

        mu_tilde = torch.mean(self.rzero_b1.ger(cw), dim=1)
        neg_log_var_tilde = torch.mean(self.rzero_b2.ger(cw), dim=1)

        z, logr = self.flow_r(z, kl=True)

        z_mu_square = (z - mu_tilde) * (z - mu_tilde)
        logr += 0.5 * (- torch.exp(neg_log_var_tilde) * z_mu_square
                       + neg_log_var_tilde).sum()

        kldiv = kldiv_weight + kldiv_bias + logq - logr
        kldiv = torch.Tensor([kldiv]).cuda()


        return kldiv
