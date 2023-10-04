#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uncertainty
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from math import factorial

def H(n,x):
    if n==0:
        return np.ones(x.shape)
    elif n==1:
        return 2*x
    else:
        return 2*x*H(n-1, x)-2*(n-1)*H(n-2, x)
 
    
def func(x=None):
    return (x**2*(1/np.sqrt(2**5*factorial(5)*np.sqrt(np.pi))*np.exp(-x**2/2)*H(5,x))**2)
def func_rescale(xp=None, a=None):
    x = a * (1. + xp) / (1. - xp)
    weight = 2. * a / (1. - xp)**2
    return (weight * func(x=x))  

range = np.array([-1., 1.], dtype=np.float64)
(I,none) = integrate.fixed_quad(func_rescale,range[0], range[1], args=(np.float64(1.),), n=100)
print(np.sqrt(2*I))