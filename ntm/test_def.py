#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 14:53:07 2020

@author: qiaonan
"""

from ntm_model import *
import torch

def test_shift():
    shift = Shift([-1,0,1])
    s = torch.tensor([[1.,0,0],[0,0,1]])
    wg = torch.tensor([[1,2,3,4],[5,6,7,8.]])
    x = shift(s,wg)
    res = x == torch.tensor([[4., 1., 2., 3.],
        [6., 7., 8., 5.]])
    assert(torch.all(res))
