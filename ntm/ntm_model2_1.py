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
# change:
# 1. remove leaky_relu from out
# 2. feed memory average rather than cos to get e and a
# 3. use k
# 4. add a hidden layer before output

#%%
# x = torch.randn(3,2,4)
# y = torch.randn(3,2)
# y2 = y.unsqueeze(dim=-1)
# (x*y2).size()
# x.transpose(1,2).transpose(0,1).size()

# cs = nn.CosineSimilarity(dim=2)
# x = torch.randn(3,2,4)
# y = torch.randn(3,1,4)
# cs(x,y)

# x = torch.tensor([[1.,2],[3,4]])
# x/x.sum(dim=1).unsqueeze(dim=-1)

# class CL():
#     def __init__(self,a,b,c):
#         print(vars())
#         x = 6
#         print(vars())
#         # pass
    
#     def __call__(self,x):
#         return 5
# cl = CL(1,2,3)
# x = torch.tensor([[[1.,2,3]]])
# w = torch.tensor([[[1.,-1]]])
# F.conv1d(x,w,dilation=2)

# x = torch.empty((2,))
# y = torch.tensor([[1,2.],[3,4.]],requires_grad=True)
# x[0] = y[0,:].sum()*2
# x[1] = y[1,:].sum()
# z = x.sum()
# z.backward()

# class A():
#     def __init__(self,x):
#         self.x = x

# class B(A):
#     def go(self):
#         print(self.x)

#%%
class Memory():
    def __init__(self,N,M,batch_size):
        self.N, self.M, self.batch_size = N, M, batch_size
        self.shape = (self.N, self.M)
        self.reset()
        
    def reset(self):
        # bank of size (batch_size, N, M)
        self.bank = torch.zeros(self.batch_size,
                                self.N, self.M)
    
    def read(self,w):
        # w is shape of batch_size * N
        wu = w.unsqueeze(dim=-1) # batch * N * 1
        return (self.bank*wu).sum(dim=1) # batch * M
        
    def avg(self):
        return self.bank.mean(dim=1)
        
    def write(self,w,e,a):
        # w is shape of batch_size * N
        # e,a is shape of batch_size * M
        wu = w.unsqueeze(dim=-1) # batch * N * 1
        eu = e.unsqueeze(dim=1) # batch * 1 * M
        au = a.unsqueeze(dim=1) # batch * 1 * M
        
        we = wu*eu
        wa = wu*au
        # avoid in-place modification of self.bank
        # x += y is in-place plus operation
        # x = x + y is out-place plus operation
        self.bank = self.bank - we
        self.bank = self.bank + wa
    
    def cos(self,k):
        # k is shape of batch * M
        ku = k.unsqueeze(dim=1) # batch * 1 * M
        csim = nn.CosineSimilarity(dim=2)
        # batch * N
        res = csim(self.bank,ku)
        return res
    
#%%
class Shift():
    def __init__(self,shift_vec):
        # shift_vec in the form of [-1, 0, 1] as an example, which
        # allows back and foward shift by 1 position or no shift
        self.vec = shift_vec
        
        
    def correct_idx(self,i,N):
        return i if i < N else i-N
    
    def __call__(self, s, wg):
        # shift wg using shift vec weighted by s
        # s: batch * vec_size
        # wg: batch * N
        ws = torch.empty(wg.shape)
        N = wg.shape[1]
        
        for i in range(N):
            idx = [self.correct_idx(i+shift_val, N) for shift_val in self.vec]
            # batch * vec_size
            wg_slice = wg[:,idx]
            ws[:,i] = (wg_slice*s).sum(dim=1)
        return ws

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
        # w is the shape of batch * N
        self.w = torch.ones(self.batch_size,N)/N
    
    
    def address(self,param):
        # k: batch * M
        k, beta, g, s, gamma = param
        cos = self.mem.cos(k)
        return self.address_cos([cos,beta,g,s,gamma])
        
    def address_cos(self, param):
        # mem: memory
        # k: batch * M
        # cos: batch * N
        # beta: batch * 1, range (0, inf)
        # g: batch * 1, range (0,1)
        # s: batch * shift_size, range softmax(0,1)
        # gamma: batch * 1, range (1,inf)
        cos, beta, g, s, gamma = param
        #TODO: where to force range of these values? 
        # before head or after?
        
        #cos = self.mem.cos(k) # batch * N
        beta = F.relu(beta) + 1e-8 # beta must be strict positive
        wc = F.softmax(beta*cos,dim=1) # batch * N
        
        g = torch.sigmoid(g) #force range (0,1)
        wg = g*wc + (1-g)*self.w
        
        s = F.softmax(s,dim=1)
        ws = self.shift(s,wg)
        
        gamma = F.relu(gamma)+1 #force range (1, inf)
        w_sharp = ws.pow(gamma)
        w = w_sharp/w_sharp.sum(dim=1).unsqueeze(dim=-1)
        
        # revisit this line !!!!
        self.w = w
        return w

        
class ReadHead(Head):
    # __init__ is unnecessary as this __init__ function is 
    # exactly the sae as that of Head
    # def __init__(self,mem,shift):
    #     super(ReadHead,self).__init__(mem,shift)
    
    def __call__(self,param):
        w = self.address(param)
        return self.mem.read(w)

class WriteHead(Head):
    # def __init__(self,mem,shift):
    #     super(ReadHead,self).__init__(mem,shift)
    
    def __call__(self,param,e,a):
        # e range [0,1]
        e = torch.sigmoid(e)
        w = self.address(param)
        self.mem.write(w,e,a)

#%%

class FFController(nn.Module):
    # feedforward controller
    def __init__(self,batch_size,input_size,hidden_size,n,output_size):
        # n number of hidden layers
        # batch_size seems unused
        
        # same as super(FFController, self).__init__()
        super().__init__()
        self.n = n
        self.layers = []
        for i in range(n):
            if i == 0:
                self.layers.append(nn.Linear(input_size,hidden_size))
            else:
                self.layers.append(nn.Linear(hidden_size,hidden_size))
                
        # this is called the output layer
        self.layers.append(nn.Linear(hidden_size,output_size))
        
    def forward(self,x):
        for i, layer in enumerate(self.layers):
            # self.layers has the length of n+1
            if i < self.n:
                x = F.leaky_relu(layer(x))
            else:
                # !!!revist to see if need an activation
                # no activation for now
                x = layer(x)
        return x

Controllers = {'FFController':FFController}
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
        self.ea_layer = nn.Linear(self.M+self.M,self.M*2)
        #self.a_layer = nn.Linear(p.N+repr_size,M)
        
        # !!! task specific layer, remove out of this class in future
        self.seq_proc = nn.Linear(self.M,p.ctrl.hidden_size)
        self.out_layer = nn.Linear(p.ctrl.hidden_size,p.seq_width)
    
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
        write_param = self._split(write_vec)
        
        mem_avg = self.mem.avg() # batch * M
        ea_input = torch.cat((mem_avg,write_param[0]),dim=1)
        ea = self.ea_layer(ea_input)
        e = ea[:,:self.M]
        a = ea[:,self.M:] # add tanh transformation for stored content
                          # as in GRU ?? also add tanh for k??
        self.write(write_param,e,a)
        
        out = self.read(read_param)
        # add leaky_relu or not??? revisit
        seq_proc_out = self.seq_proc(out)
        seq_out = self.out_layer(F.leaky_relu(seq_proc_out))
        
        
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
        