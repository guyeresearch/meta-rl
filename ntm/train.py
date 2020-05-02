#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 16:20:25 2020

@author: qiaonan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from ruamel.yaml import YAML
import qn

from tasks import *
from ntm_model import *

#%%


#%%
yaml=YAML()
with open('ntm_param.yml','r') as nf:
    param = yaml.load(nf)
p = AttrDict.from_nested_dict(param['train'])
pm = AttrDict.from_nested_dict(param['model'])
# copy training parameters required for model building
pm.batch_size = p.batch_size
pm.seq_width = p.seq_width

# (index, input_seq, target_seq)
# seq is in the shape of (seq_len, batch, seq_width)
# input seq is the shape of (seq_len+1, batch, seq_width+1) to add delimiter
# seqs in the same batch have the same seq_len
data = dataloader(p.num_batches, p.batch_size, p.seq_width,
                  p.seq_min_len, p.seq_max_len)

ntm = NTM_Head(pm)

opm = optim.RMSprop(ntm.parameters(), lr=p.rmsprop_lr, alpha=p.rmsprop_alpha,
                    momentum=p.rmsprop_momentum)


#%%
costs = []
for index, inp, target in data:
    
    seq_len = target.shape[0]
    # reset for each sequence
    ntm.reset()    
    for i in range(inp.shape[0]):
        curr = inp[i,:,:]
        ntm(curr)
    
    loss = 0
    cost = 0
    for i in range(seq_len):
        # feed delimiter
        # consider change??
        out = ntm(curr) # curr should be delimiter
        curr_tg = target[i,:,:]
        # avoid in-place operation
        loss = loss + F.binary_cross_entropy(out,curr_tg)
        cost += torch.abs(out.detach()-curr_tg.detach())
    
    loss = loss / seq_len # normalize loss for each batch
    opm.zero_grad()
    loss.backward()
    opm.step()
    
    
    if index % 100 == 0 or index == 1:
        cost = torch.mean(cost)/target.shape[0] # error per bit
        print(index, loss.detach()/target.shape[0], cost)
        costs.append([index, cost])
        plt.figure()
        plt.plot([x[0] for x in costs],
                 [x[1] for x in costs])
        plt.savefig('figures/cost.png')
        plt.close()
    
torch.save(ntm.state_dict(),'ntm_ff_copy.pt')