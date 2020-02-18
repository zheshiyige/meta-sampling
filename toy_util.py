import torch
import numpy as np
import math

"""Evaluate (possibly unnormalized) pdf over a meshgrid."""
def evaluate_bivariate(range, npoints):
    range_limit = [-9, 9]
    npoints_plot=50
    side = np.linspace(range[0], range[1], npoints)
    z1, z2 = np.meshgrid(side, side)
    zv = np.hstack([z1.reshape(-1, 1), z2.reshape(-1, 1)])
    return z1, z2, zv


""" definition of gauss mixture """
class Gauss_mixture():

    def __init__(self, omega, mean, sigma):
        self.omega = omega # mixture weights for each gauss component
        self.mean = mean # mean location vector for each gauss component
        self.sigma = sigma #variance for each gauss component
        
    #density gradient with respect to the location x
    def dlnprob(self, x):
        dim = x.size()[1]
        ind_den = [torch.exp(-1*((x - self.mean[i])**2).sum(1)/(2*self.sigma))/(np.power(2 * math.pi*self.sigma, dim/2.0)) for i in range(self.mean.size()[0])]
        ind_den = (self.omega*torch.stack(ind_den)).unsqueeze(2)
        sum_den = ind_den.sum(0)
        grad_single = [(-1 * (x - self.mean[i])/self.sigma) for i in range(self.mean.size()[0])]
        grad_single = torch.stack(grad_single)
        total_grad = ind_den * grad_single
        sum_grad = total_grad.sum(0)
        sum_grad = sum_grad/sum_den
        return sum_grad

    #density evaluation of gauss mixture with respect to the location x
    def MGprob(self, x):     
        dim = x.size()[1]
        ind_den = [torch.exp(-1*((x - self.mean[i])**2).sum(1)/(2*self.sigma))/(np.power(2 * math.pi*self.sigma, dim/2.0)) for i in range(self.mean.size()[0])]
        ind_den = torch.stack(ind_den)
        sum_den = ind_den.sum(0)
        return sum_den
