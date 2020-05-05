#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:55:00 2020

@author: qiaonan
"""

import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

import qn


#%%
class AttrDict(dict):
    """ Dictionary subclass whose entries can be accessed by attributes
        (as well as normally).
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
    # copied from stackflow answer
    @staticmethod
    def from_nested_dict(data):
        """ Construct nested AttrDicts from nested dictionaries. """
        if not isinstance(data, dict):
            return data
        else:
            return AttrDict({key: AttrDict.from_nested_dict(data[key])
                                for key in data})

# param = qn.load('ntm_param.yml')
# p = AttrDict.from_nested_dict(param)

class NTM_Head(nn.Module):
    # single-head version
    def __init__(self,p):
        # p for model specific parameters
        super().__init__()
        
        self.p = p
        self.recur = nn.LSTM(p.seq_width+1,p.ctrl.hidden_size,p.ctrl.n)
        
     
        #self.a_layer = nn.Linear(p.N+repr_size,M)
        
        # !!! task specific layer, remove out of this class in future
        self.out_layer = nn.Linear(p.ctrl.hidden_size,p.seq_width)
        self.reset()
    
    def reset(self):
        # reset for each batch
        self.h0 = nn.Parameter(torch.randn(self.p.ctrl.n, self.p.batch_size,
                             self.p.ctrl.hidden_size))
        self.c0 = nn.Parameter(torch.randn(self.p.ctrl.n, self.p.batch_size,
                             self.p.ctrl.hidden_size))
        
        self.count = 0
        
    
    def forward(self,x):
        
        if self.count == 0:
            x, (self.h, self.c) = self.recur(x,(self.h0,self.c0))
        else:
            x, (self.h, self.c) = self.recur(x,(self.h,self.c))
        self.count += 1
        
        x = self.out_layer(x[0,:,:])
       
        return torch.sigmoid(x)
        
        
        
class NTM_Heads(nn.Module):
    # multi-heads version
    def __init__(self,mem,ctrl,rhead_num,whead_num,shift_vec):
        super().__init__()
        self.mem, self.ctrl = mem, ctrl
        
        self.shift = Shift(shift_vec)
        self.rheads = [ReadHead(self.mem,self.shift) for _ in range(rhead_num)]
        self.wheads = [WriteHead(self.mem,self.shift) for _ in range(whead_num)]
        
    def foward(self,x):
        pass
        