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

# xchange:
# 1. constant memory initialization
# 2. tanh [-1,1] memory values. a,k are tanh transformed
# 3. softplus instead of relu, beta, gamma
# 4. complete restructure
#%%
class Memory():
    def __init__(self,N,M,batch_size):
        self.N, self.M, self.batch_size = N, M, batch_size
        self.shape = (self.N, self.M)
        self.reset()
        
    def reset(self):
        # bank of size (batch_size, N, M)
        self.bank = torch.ones(self.batch_size,
                                self.N, self.M) * 1e-6
    
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
        kt = torch.tanh(k)
        cos = self.mem.cos(kt)
        return self.address_cos([cos,beta,g,s,gamma])
        
    def address_cos(self, param):
        # mem: memory
        # k: batch * M [-1,1]
        # cos: batch * N
        # beta: batch * 1, range (0, inf)
        # g: batch * 1, range (0,1)
        # s: batch * shift_size, range softmax(0,1)
        # gamma: batch * 1, range (1,inf)
        cos, beta, g, s, gamma = param
        #TODO: where to force range of these values? 
        # before head or after?
        
        #cos = self.mem.cos(k) # batch * N
        beta = F.softplus(beta) # beta must be strict positive
        wc = F.softmax(beta*cos,dim=1) # batch * N
        
        g = torch.sigmoid(g) #force range (0,1)
        wg = g*wc + (1-g)*self.w
        
        s = F.softmax(s,dim=1)
        ws = self.shift(s,wg)
        
        gamma = F.softplus(gamma)+1 #force range (1, inf)
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
        # a range [-1,1]
        a = torch.tanh(a)
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
        self.ctrl = Controller(p.batch_size, p.seq_width+1+self.M,
                p.ctrl.hidden_size, p.ctrl.n, p.ctrl.output_size)
        
        self.shift = Shift(p.shift_vec)
        self.read = ReadHead(self.mem, self.shift)
        self.write = WriteHead(self.mem, self.shift)
        
        # (k + beta + g + s + gamma)*2 + e + a
        self.param_size = self.M + 1 + 1 + len(p.shift_vec) + 1
        repr_size = self.param_size*2 + 2*self.M
        self.repr_layer = nn.Linear(p.ctrl.output_size,repr_size)
        
        # ctrl output + read vector
        self.out_layer = nn.Linear(p.ctrl.output_size+self.M, p.seq_width)
    
    def reset(self):
        # reset for each batch
        self.mem.reset()
        self.read.reset()
        self.write.reset()
        self.prev_read = torch.zeros(self.batch_size, self.M)
    
    def _split(self,represent):
        rep = represent
        k = rep[:,:self.M]
        beta = rep[:,[self.M]] # maintain batch * 1 shape
        g = rep[:,[self.M+1]] # maintian batch * 1 shape
        s = rep[:,(self.M+2):-1]
        gamma = rep[:,[-1]] # maintian batch * 1 shape
        return [k,beta,g,s,gamma]
        
    
    def forward(self,x):
        ctrl_in = torch.cat((x,self.prev_read),axis=1)
        ctrl_out = torch.tanh(self.ctrl(ctrl_in))
        
        reprx = self.repr_layer(ctrl_out)
        read_vec = reprx[:,:self.param_size]
        # print(read_vec.shape)
        read_param = self._split(read_vec)

        
        write_vec_ea = reprx[:,self.param_size:]
        write_vec = write_vec_ea[:,:self.param_size]
        ea = write_vec_ea[:,self.param_size:]
        e = ea[:,:self.M]
        a = ea[:,self.M:]
        write_param = self._split(write_vec)
        
        read = self.read(read_param)
        self.prev_read = read
        self.write(write_param,e,a)
        
        seq_out = self.out_layer(torch.cat((ctrl_out,read),axis=1))
        
        
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
        