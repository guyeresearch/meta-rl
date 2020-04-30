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
x = torch.randn(3,2,4)
y = torch.randn(3,2)
y2 = y.unsqueeze(dim=-1)
(x*y2)*size()
x.transpose(1,2).transpose(0,1).size()

cs = nn.CosineSimilarity(dim=2)
x = torch.randn(3,2,4)
y = torch.randn(3,1,4)
cs(x,y)

x = torch.tensor([[1.,2],[3,4]])
x/x.sum(dim=1).unsqueeze(dim=-1)

class CL():
    def __init__(self,a,b,c):
        print(vars())
        x = 6
        print(vars())
        # pass
    
    def __call__(self,x):
        return 5
cl = CL(1,2,3)
#%%
class Memory():
    def __init__(self,N,M,batch_size):
        self.N, self.M, self.batch_size = N, M, batch_size
        self.reset()
        self.shape = (self.N, self.M)
        
    def reset(self):
        # bank of size (batch_size, N, M)
        self.bank = torch.zeros(self.batch_size,
                                self.N, self.M)
    
    def read(self,w):
        # w is shape of batch_size * N
        wu = w.unsqueeze(dim=-1) # batch * N * 1
        return (self.bank*wu).sum(dim=1) # batch * M
        
    
    def write(self,w,e,a):
        # w is shape of batch_size * N
        # e,a is shape of batch_size * M
        wu = w.unsqueeze(dim=-1) # batch * N * 1
        eu = e.unsqueeze(dim=1) # batch * 1 * M
        au = a.unsqueeze(dim=1) # batch * 1 * M
        
        we = wu*eu
        wa = wu*au
        self.bank -= we
        self.bank += wa
    
    def cos(self,k):
        # k is shape of batch * M
        ku = k.unsqueeze(dim=1) # batch * 1 * M
        csim = nn.CosineSimilarity(dim=2)
        res = csim(self.bank,ku)
        return res
    
#%%
class Shift():
    def __init__(self,shift_vec):
        # shift_vec in the form of [-1, 0, 1] as an example, which
        # allows back and foward shift by 1 position or no shift
        self.vec = shift_vec
    
    def __call__(self, s, wg):
        # shift wg using shift vec weighted by s
        
        pass

class Head():
    def __init__(self, mem, shift):
        # use the same batch_size as mem
        self.mem = mem
        self.batch_size = mem.batch_size
        self.shift = shift
        
        self.reset()
    
    def reset(self):
        # reset previous weight vector
        N, M = self.mem.shape
        self.w = torch.ones(self.batch_size,M)/M
    
    
    def address(self,param):
        # k: batch * M
        k, beta, g, s, gamma = param
        cos = self.mem.cos(k)
        return self.address_cos([cos,beta,g,s,gamma])
        
    def address_cos(self, param):
        # mem: memory
        # cos: batch * N
        # beta: batch * 1, 
        # g: batch * 1, range (0,1)
        # s: batch * shift_size, range softmax(0,1)
        # gamma: batch * 1, range (1,inf)
        cos, beta, g, s, gamma = param
        #TODO: where to force range of these values? 
        # before head or after?
        
        #cos = self.mem.cos(k) # batch * M
        wc = F.softmax(beta*cos,dim=1)
        
        g = F.sigmoid(g) #force range (0,1)
        wg = g*wc + (1-g)*self.w
        
        s = F.softmax(s,dim=1)
        ws = self.shift(s,wg)
        
        gamma = F.relu(gamma)+1
        w_sharp = ws.pow(gamma)
        w = w_sharp/w_sharp.sum(dim=1).unsqueeze(dim=-1)
        
        # revisit this line !!!!
        self.w = w
        return w

        
class ReadHead(Head):
    def __init__(self,mem,shift):
        super(ReadHead,self).__init__(mem,shift)
    
    def read(self,param):
        w = self.address(param)
        return self.mem.read(w)

class WriteHead(Head):
    def __init__(self,mem,shift):
        super(ReadHead,self).__init__(mem,shift)
    
    def write(self,param,e,a):
        # e range [0,1]
        e = F.sigmoid(e)
        w = self.address_cos(param)
        self.mem.write(w,e,a)

#%%

class FFController(nn.Module):
    # feedforward controller
    def __init__(self,batch_size,input_size,hidden_size,n,output_size):
        # n number of layers
        super(FFController, self).__init__()
        self.n = n
        self.layers = []
        for i in range(n):
            if i == 0:
                self.layers[i] = nn.Linear(input_size,hidden_size)
            elif i === layer_count - 1:
                self.layers[i] = nn.Linear(hidden_size,output_size)
            else:
                self.layers[i] = nn.Linear(hidden_size,hidden_size)
        
    def forward(self,x):
        for i, layer in enumerate(layers):
            if i < self.n - 1
                x = F.leaky_relu(layer(x))
            else:
                # !!!revist to see if need an activation
                x = layer(x)
        return x

#%%
class AttrDict(dict):
    """ Dictionary subclass whose entries can be accessed by attributes
        (as well as normally).
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

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
    def __init__(self,param):
        super(NTM_Head, self).__init__()
        p = AttrDict.from_nested_dict(param)
        
        self.p = p
        self.N = p.mem.N
        self.M = p.mem.M
        self.batch_size = p.batch_size
        
        self.mem = Memory(self.N, self.M, self.batch_size)
        self.ctrl = FFController(self.batch_size, p.ctrl.input_size,
                p.ctrl.hidden_size, p.ctrl.n, p.ctrl.output_size)
        
        self.shift = Shift(p.shift_vec)
        self.read = ReadHead(self.mem,self.shift)
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
        self.ea_layer = nn.Linear(p.N+repr_size,self.M*2)
        #self.a_layer = nn.Linear(p.N+repr_size,M)

    
    def _split(self,represent):
        rep = represent
        k = rep[:,:self.M]
        beta = rep[:,self.M]
        g = rep[:,(self.M+1)]
        s = rep[:,(self.M+2):-1]
        gamma = rep[:,-1]
        return [k,beta,g,s,gamma]
        
    
    def forward(self,x):
        represent = self.ctrl(x)
        
        read_repr = self.read_layer(F.leaky_relu(represent))
        read_param = self._split(read_repr)
        
        write_repr = self.write_layer(F.leaky_relu(represent))
        w_splits = self._split(write_repr)
        wk = w_splits[0]
        wcos = self.mem.cos(wk)
        write_repr_tanh = F.tanh(write_repr) # match wcos range
        # wcos: batch*N, write_repr_tanh batch*repr_size
        ea_input = concate ??? (wcos,write_repr_tanh) #TODO
        ea = self.ea_layer(ea_input)
        e = ea[:,:self.M]
        a = ea[:,self.M:]
        write_param = [wcos] + w_splits[1:]
        self.write(write_param,e,a)
        
        return self.read(read_param)
        
        
class NTM_Heads(nn.Module):
    # multi-heads version
    def __init__(self,mem,ctrl,rhead_num,whead_num,shift_vec):
        super(NTM, self).__init__()
        self.mem, self.ctrl = mem, ctrl
        
        self.shift = Shift(shift_vec)
        self.rheads = [ReadHead(mem,shift) for _ in range(rhead_num)]
        self.wheads = [WriteHead(mem,shift) for _ in range(whead_num)]
        
    def foward(self,x):
        pass
        