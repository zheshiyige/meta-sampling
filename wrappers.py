from CNN_NIAF import NIAFConv2D
from MLP_NIAF import NIAFLinear
import torch.nn as nn
import torch.nn.functional as F
import torch


class NIAFNet(nn.Module):
    def __init__(self, input_shape, flows_q=2, flows_r=2, activation=nn.ReLU(),
                 nb_classes=5,  layer_dims=(16, 16, 100), flow_dim_h=50, n_hidden=0, thres_var=1):
        nn.Module.__init__(self)        
        self.layer_dims = layer_dims
        self.activation = activation
        self.input_shape = input_shape
        self.activation = activation
        self.flows_q = flows_q
        self.flows_r = flows_r
        self.nb_classes = nb_classes
        self.flow_dim_h = flow_dim_h
        self.thres_var = thres_var
        self.n_hidden = n_hidden
        print('nb_classes', nb_classes)
        self.opts = 'fq{}_fr{}'.format(self.flows_q, self.flows_r)

        self.layers = []
        self.layer1 = NIAFConv2D(self.layer_dims[0], 3, 3, stack_size =1, 
                                   n_flows_q=self.flows_q, n_flows_r=self.flows_r,
                                   hidden_dim=self.flow_dim_h, n_hidden= self.n_hidden, threshold_var=self.thres_var)
        self.layers.append(self.layer1)
           
        self.layer2 = NIAFConv2D(self.layer_dims[1], 3, 3, stack_size =self.layer_dims[0],
                                   n_flows_q=self.flows_q, n_flows_r=self.flows_r, 
                                   hidden_dim=self.flow_dim_h, n_hidden= self.n_hidden, threshold_var=self.thres_var)
        self.layers.append(self.layer2)
        fcinp_dim = 400

        self.layer3 = NIAFLinear(fcinp_dim, self.layer_dims[2], n_flows_q=self.flows_q,
                                  n_flows_r=self.flows_r, hidden_dim=self.flow_dim_h, n_hidden= self.n_hidden, threshold_var=self.thres_var)
        self.layers.append(self.layer3)   
        fcinp_dim = 100
        self.layerout = NIAFLinear(fcinp_dim, self.nb_classes, n_flows_q=self.flows_q,
                                    n_flows_r=self.flows_r, hidden_dim=self.flow_dim_h, n_hidden= self.n_hidden, threshold_var=self.thres_var)
        self.layers.append(self.layerout)

    def forward(self, x, same_noise=False):

        x = self.activation(F.max_pool2d(self.layer1(x, same_noise=same_noise), (2, 2)))
        x = self.activation(F.max_pool2d(self.layer2(x, same_noise=same_noise), (2, 2)))
        x = x.view(x.size(0), -1)
        x = self.activation(self.layer3(x, same_noise=same_noise))
        x = self.layerout(x, same_noise=same_noise)
        x = F.log_softmax(x, dim=1)
        return x



    def get_reg(self):
        reg = []
        for j, layer in enumerate(self.layers):
            regi = layer.kldiv()
            reg.append(regi)  

        reg = torch.stack(reg, 0)          
        reg = torch.sum(reg)

        return reg




