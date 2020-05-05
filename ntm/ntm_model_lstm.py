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
        self.N = p.mem.N
        self.M = p.mem.M
        self.batch_size = p.batch_size
        
        # select controller
        Controller = Controllers[p.ctrl.type]
        self.mem = Memory(self.N, self.M, self.batch_size)
        # !!! p.seq_width+1 is a task specific parameter
        self.ctrl = Controller(p.batch_size, p.seq_width+1,
                p.ctrl.hidden_size, p.ctrl.n, p.ctrl.output_size)
        
        self.shift = Shift(p.shift_vec)
        self.read = ReadHead(self.mem, self.shift)
        self.write = WriteHead(self.mem, self.shift)
        
        # k + beta + g + s + gamma
        repr_size = self.M + 1 + 1 + len(p.shift_vec) + 1
        self.repr_size = repr_size
        self.read_layer = nn.Linear(p.ctrl.output_size,repr_size)
        
        
        # k + beta + g + s + gamma
        self.write_layer = nn.Linear(p.ctrl.output_size,repr_size)
        
        # e,a batch * M, cos batch * N
        # !!! repr could be changed to k, revisit, alternative implementation
        # input: cos + repr, N + repr_size, 
        # output: e + a, M + M
        self.ea_layer = nn.Linear(self.N+repr_size,self.M*2)
        #self.a_layer = nn.Linear(p.N+repr_size,M)
        
        # !!! task specific layer, remove out of this class in future
        self.out_layer = nn.Linear(self.M,p.seq_width)
    
    def reset(self):
        # reset for each batch
        self.mem.reset()
        self.read.reset()
        self.write.reset()
    
    def _split(self,represent):
        rep = represent
        k = rep[:,:self.M]
        beta = rep[:,[self.M]] # maintain batch * 1 shape
        g = rep[:,[self.M+1]] # maintian batch * 1 shape
        s = rep[:,(self.M+2):-1]
        gamma = rep[:,[-1]] # maintian batch * 1 shape
        return [k,beta,g,s,gamma]
        
    
    def forward(self,x):
        ctrl_out = F.leaky_relu(self.ctrl(x))
        
        read_vec = self.read_layer(ctrl_out)
        read_param = self._split(read_vec)
        
        write_vec = self.write_layer(ctrl_out)
        w_splits = self._split(write_vec)
        wk = w_splits[0]
        wcos = self.mem.cos(wk)
        write_param = [wcos] + w_splits[1:]
        
        write_vec_tanh = torch.tanh(write_vec) # match wcos range
        # wcos: batch*N, write_vec_tanh batch*repr_size
        ea_input = torch.cat((wcos,write_vec_tanh),dim=1) 
        ea = self.ea_layer(ea_input)
        e = ea[:,:self.M]
        a = ea[:,self.M:] # add tanh transformation for stored content
                          # as in GRU ?? also add tanh for k??
        self.write(write_param,e,a)
        
        out = self.read(read_param)
        # add leaky_relu or not??? revisit
        seq_out = self.out_layer(F.leaky_relu(out))
        
        # task specific layer
        # sigmoid output for bit copy task
        return torch.sigmoid(seq_out)
        
        
        
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
        