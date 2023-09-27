#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
derivatives
"""
import numpy as np

def f(x):
    return x*(x-1)

def derivative_f(x,delta):
    return (f(x+delta)-f(x))/delta

x = 1
for delta in [10**-2,10**-4,10**-6,10**-8,10**-10,10**-12,10**-14]:
    print("delta=",delta, "derivative=",derivative_f(x,delta))
    
    