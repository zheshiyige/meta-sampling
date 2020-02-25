import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from bayes_update import Meta_Sampler
from BLR_util import *
import argparse
import torch
import os


def parse_args():

    desc = "bayesian_logistic_regression"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset', type=str, default='heart',
                        help='dataset used for logistic regression, heart, german or australian')
    parser.add_argument('--train_iter', type=int, default=2000, 
                        help='The number of iterations to train')
    parser.add_argument('--batch_size', type=int, default=100, 
                        help='The number of iterations to train')
    parser.add_argument('--stepsize', type=float, default=1e-3, help='WGF learning rate')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--lamb', type=float, default=1e-4, help='lambda for balance the two terms in the Wasserstein Gradient Flow')
    parser.add_argument('--beta1', type=float, default=0.9, help='adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='adam beta2')
    parser.add_argument('--num_particle', type=int, default=100, help='number of paricles')
    parser.add_argument('--dim', type=int, default=2, help='particle dimension')
    parser.add_argument('--dimh', type=int, default=64, help='hidden size dimension of NIAF')
    parser.add_argument('--flowtype', type=int, default=0, 
                        help='0 for IAF_DSF and 1 for IAF_DDSF')
    parser.add_argument('--num_flow_layers', type=int, default=2,  help='number of flow layers')
    parser.add_argument('--num_hid_layers', type=int, default=2)
    parser.add_argument('--num_ds_dim', type=int, default=16)
    parser.add_argument('--num_ds_layers', type=int, default=1)
    parser.add_argument('--fixed_order', type=bool, default=True,
                        help='Fix the made ordering to be the given order')

    return parser.parse_args()





if __name__ == '__main__':

    args = parse_args()
    data_file = 'BLR_data/' + args.dataset + '/'
    X_input = np.load(data_file +'data.npy')
    y_input = np.load(data_file +'labels.npy')
    y_input = np.squeeze(y_input)
    y_input[y_input == 0] = -1
    
    N = X_input.shape[0]
    X_input = np.hstack([X_input, np.ones([N, 1])])
    d = X_input.shape[1]
    D = d + 1

    print('X_input.shape', X_input.shape)
    print('y_input.shape', y_input.shape)
    
    # split the dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_input, y_input, test_size=0.2, random_state=42)
    a0, b0 = 1, 0.01 #hyper-parameters
    if args.dataset == 'heart':
        npseed = 0
    if args.dataset == 'australian':
        npseed = 102
    if args.dataset == 'german':
        npseed = 166

    trseed = 0
    torch.manual_seed(trseed)
    np.random.seed(npseed)
    model = BayesianLR(X_train, y_train, X_test, y_test, args.batch_size, a0, b0) 
    
    args.dim = D

    denaf = Meta_Sampler(args)
    best_acc = denaf.fit(model.dlnprob, model.evaluation, total= args.train_iter, stepsize = args.stepsize)
    print('dataset {}, accuracy {}'.format(args.dataset, best_acc))
        
    
  

            

