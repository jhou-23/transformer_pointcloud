import random
from pprint import pprint
from dataset import ModelNet40
import numpy as np
import torch
import math

def pairwise_distances(x):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(2).unsqueeze(2)
    y_t = torch.transpose(x, 1, 2)
    y_norm = x_norm.permute(0,2,1)
    
    dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf) ** 0.5


pc = torch.randn((2, 4, 3)).cuda() 
print(pc)
sigma = torch.randint(1,100,(2, 4, 4)).type(torch.FloatTensor).cuda()
print(sigma)
dc = -1 / (2 * (sigma**2)).cuda()
cc = 1 / ((2 * math.pi * sigma) ** 0.5).cuda()

dists = pairwise_distances(pc).cuda()
print(dists)

pe = cc * torch.exp(dc * dists)
print(pe)