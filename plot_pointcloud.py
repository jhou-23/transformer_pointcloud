import random
from pprint import pprint
from dataset import ModelNet40
import numpy as np
import torch
import matplotlib.pyplot as plt

idx = 200
sample = ModelNet40('modelnet40', bins=8, npoints=1024)[idx]
pc, discretized, label = sample

dpx = discretized[:,0]
dpy = discretized[:,1]
dpz = discretized[:,2]
pcx = pc[:,0]
pcy = pc[:,1]
pcz = pc[:,2]

fig1 = plt.figure()
ax1 = fig1.add_subplot(111,projection='3d')
ax1.scatter(dpx, dpy, dpz, alpha=.1)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111,projection='3d')
ax2.scatter(pcx, pcy, pcz)
plt.show()