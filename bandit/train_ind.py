#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 10:46:39 2020

@author: qiaonan
"""
import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt

from bandit import BernoulliBanditEnv

#%% model
class AC_Net(nn.Module):
    def __init__(self, in_size, h_size, d_size, action_size):
        # d_size: output densly connected layer size
        # out_size: number of actions
        # default to one layer, one batch
        super(AC_Net, self).__init__()
        
        self.gru_num_layers = 1
        self.gru = nn.GRU(in_size, h_size, self.gru_num_layers)
        self.in_size, self.h_size, self.d_size, self.action_size \
            = in_size, h_size, d_size, action_size
        self.reset_h()
        # # value net
        # self.v1 = nn.Linear(h_size,d_size)
        # self.v2 = nn.Linear(h_size,1)
        # policy net
        self.p1 = nn.Linear(h_size,d_size)
        self.p2 = nn.Linear(d_size,action_size)
    
    def reset_h(self):
        # def
        self.h = torch.zeros((self.gru_num_layers,1,self.h_size))
    
    def forward(self,x):
        # input x shape (1,1,h_size)
        x, self.h = self.gru(x,self.h)
        x = x[0,:,:]
        
        # # value net
        # v = F.leaky_relu(self.v1(x))
        # v = F.leaky_relu(self.v2(v))
        
        # action net
        p = F.leaky_relu(self.p1(x))
        logits = self.p2(p)
        cat = Categorical(logits=logits[0,:])
        
        # return v,cat
        return cat
        
    
    
    
            
#%% material
#m = Categorical(torch.tensor([ [0.25, 0.25],[ 0.25, 0.25] ]))

#env = BernoulliBanditEnv(2)

#env.reset_task(env.sample_tasks(1)[0])

# policy_loss = (-self.log_probs * advantage.detach()).mean()
# critic_loss = 0.5 * advantage.pow(2).mean()
# entropy is summed not averaged
## coef might be 0.01
# ac_loss = actor_loss + critic_loss - coeff*entropy_loss

#%% train
# params
total_eps_count = 20000
eps_len = 100
d_size = 12

# according to paper
h_size = 48
# beta_v = 0.05
lr = 0.001
# gamma = 0.8

env = BernoulliBanditEnv(2) # two arm
ac_net = AC_Net(2,h_size,d_size,2) #two input, two output actions
optimizer = optim.Adam(ac_net.parameters(), lr=lr)
#beta_e annealing
beta_es = np.linspace(1,0,total_eps_count)

for i in range(total_eps_count):
    if (i+1) % 100 == 0:
        print(f'{i+1}th episode')
        
    beta_e = beta_es[i]
    env.reset_task(env.sample_tasks(1)[0])
    env.reset()
    # reset hidden state of ac_net
    ac_net.reset_h()
    
    a = env.action_space.sample()
    _, r, _, _ = env.step(a)
    
    eps = []
    v = 0
    for j in range(eps_len):
        item = torch.tensor([[[r,a]]],dtype=torch.float)
        cat = ac_net(item)
        a = cat.sample()
        logp = cat.log_prob(a)
        entropy = cat.entropy()
        _, r, _, _ = env.step(a.item())
        
        eps.append([r,logp,entropy])
        v += r/eps_len
    
    entropy_loss = 0
    actor_loss = 0
    # critic_arr = 0
    for k,item in enumerate(eps[::-1]):
        r, logp, entropy = item
        entropy_loss -= entropy
        adv = r - v
        actor_loss -= logp*adv
        
    loss = actor_loss.mean() + beta_e*entropy_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
#torch.save(ac_net.state_dict(),'ac_net.pt')

#%% test
def test_model(model_path,test_eps_count=300,
               h_size=48,d_size=12,eps_len=100):
    ac_net = AC_Net(2,h_size,d_size,2)
    ac_net.load_state_dict(torch.load(model_path))
    
    rec2 = []
    md = []
    for i in range(test_eps_count):
        if (i+1) % 50 == 0:
            print(f'{i+1}th episode')
        
        env.reset_task(env.sample_tasks(1)[0])
        env.reset()
        best_a = 0 if env._means[0] > env._means[1] else 1
        ac_net.reset_h()
        
        a = env.action_space.sample()
        _, r, _, _ = env.step(a)
        
        rec2_arr = []
        for j in range(eps_len):
            item = torch.tensor([[[r,a]]],dtype=torch.float)
            cat = ac_net(item)
            a = cat.sample()
            # logp = cat.log_prob(a)
            # entropy = cat.entropy()
            a = a.item()
            _, r, _, _ = env.step(a)
            
            item = 1 if a != best_a else 0
            rec2_arr.append(item)
            
        rec2.append(rec2_arr)
        md.append(abs(env._means[0]-env._means[1]))
    return rec2, md


#%% plot

env = BernoulliBanditEnv(2) # two arm
rec2, md = test_model('models/ac_net.pt',1000)

def get_recv(rec):
    rec = np.array(rec)
    recv = np.mean(rec,axis=0)
    return recv

plt.figure()
plt.plot(get_recv(rec2),label='independant model')
plt.legend()
plt.xlabel('t')
plt.ylabel('fraction of wrong pulls')
plt.title('Averaged over 1000 test episodes')

wrong_count = np.sum(rec2,axis=1)
plt.figure()
plt.scatter(md,wrong_count,s=3)
plt.xlabel('difference between the expected reward of two independant arms')
plt.ylabel('number of wrong pulls in a 100-step episode')
plt.title('1000 test episodes')
