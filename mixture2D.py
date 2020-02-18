import argparse, os
from  Meta_Sampling import *

def parse_args():

    desc = "sampling_example"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--train_iter', type=int, default=1000, 
                        help='The number of iterations to train')
    parser.add_argument('--adapt_iter', type=int, default=1000, 
                        help='The number of iterations to train')
    parser.add_argument('--meta_sample_path', type=str, default='WGF_trsample/',
                        help='Directory name to save the meta samples')
    parser.add_argument('--meta_data_path', type=str, default='WGF_trdata/',
                        help='Directory name to save the meta samples data')
    parser.add_argument('--adapt_sample_path', type=str, default='WGF_testsample/',
                        help='Directory name to save the meta samples')
    parser.add_argument('--adapt_data_path', type=str, default='WGF_testdata/',
                        help='Directory name to save the meta samples data')
    parser.add_argument('--seed', type=int, default=1993,
                        help='Random seed')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--lamb', type=float, default=1e-4, help='lambda for balance the two terms in the Wasserstein Gradient Flow')
    parser.add_argument('--beta1', type=float, default=0.9, help='adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='adam beta2')
    parser.add_argument('--cuda', type=bool, default=False,  help='use cuda or not')
    parser.add_argument('--num_particle', type=int, default=1000, help='number of paricles')
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

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    meta_sampler = Meta_Sampler(args)
    meta_sampler.train_meta_sampler(args.train_iter)
    meta_sampler.adapt_meta_sampler(args.adapt_iter)

if __name__ == '__main__':
    main()
















