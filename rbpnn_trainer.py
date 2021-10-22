import math, random, argparse
import rbpnn
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
import numpy as np
import matplotlib.pyplot as plt

class trainer(object):
    def __init__(self, param):
        self.p = param

    def load_data(self):
        raise NotImplementedError

    def load_module(self):
        raise NotImplementedError

    def save_module(self):
        raise NotImplementedError

    def init_centres(self):
        raise NotImplementedError

    def select_centres(self):
        raise NotImplementedError
        
    def fit(self):
        raise NotImplementedError

    def pred(self):
        raise NotImplementedError


if __name__== '__main__':

    parser = argparse.ArgumentParser(description='Radial Basis Probabilistic Neural Network Trainer')

    parser.add_argument('-data',            dest='dataset',         default='PROTEINS', type=str,           help='Dataset to use')
    parser.add_argument('-epoch',           dest='max_epochs',      default=100,        type=int,           help='Max epochs')

    parser.add_argument('-batch',           dest='batch_size',      default=128,        type=int,           help='Batch size')
    parser.add_argument('-lr',              dest='lr',              default=0.01,       type=float,         help='Learning rate')
    parser.add_argument('-sigma',           dest='sigma',           default=0,          type=float,         help='sigma')
    
    
    parser.add_argument('-folds',           dest='folds',           default=10,         type=int,           help='Cross validation folds')

    parser.add_argument('-name',            dest='name',            default='test_'+str(uuid.uuid4())[:8],  help='Name of the run')
    parser.add_argument('-gpu',             dest='gpu',             default='1',                            help='GPU to use')
    parser.add_argument('-restore',         dest='restore',         action='store_true',                    help='Model restoring')
    
    args = parser.parse_args()
    if not args.restore:
        args.name = args.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')

    
    print('Starting runs...')
    print(args)