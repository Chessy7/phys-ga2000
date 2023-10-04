#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GH
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
#d
from numpy import ones,copy,cos,tan,pi,linspace

def gaussxw(N):

    # Initial approximation to roots of the Legendre polynomial
    a = np.linspace(3,4*N-1,N)/(4*N+2)
    x = np.cos(pi*a+1/(8*N*N*tan(a)))

    # Find roots using Newton's method
    epsilon = 1e-15
    delta = 1.0
    while delta>epsilon:
        p0 = np.ones(N,float)
        p1 = np.copy(x)
        for k in range(1,N):
            p0,p1 = p1,((2*k+1)*x*p1-k*p0)/(k+1)
        dp = (N+1)*(p0-x*p1)/(1-x*x)
        dx = p1/dp
        x -= dx
        delta = max(abs(dx))

    # Calculate the weights
    w = 2*(N+1)*(N+1)/(N*N*(1-x*x)*dp*dp)

    return x,w

def gaussxwab(N,a,b):
    x,w = gaussxw(N)
    return 0.5*(b-a)*x+0.5*(b+a),0.5*(b-a)*w
from scipy.special import roots_hermite
def gquad_hermite(b = 10, N=100):
    x,w = roots_hermite(N) # hermite polynomial roots
    a = 0
    #xp = 0.5*(b-a)*x + 0.5*(b+a) # sample points, rescaled to bounds a,b
    #wp = 0.5*(b-a)*w # rescale weights to bounds a, b
    xp, wp = gaussxwab(N,a,b)
    s = sum(integrand(xp)*wp) # add them up!
    return s
def integrand(x):
    return x**2*(1/np.sqrt(2**5*factorial(5)*np.sqrt(np.pi))*np.exp(-x**2/2)*H(5,x))**2
print(np.sqrt(2*gquad_hermite(10)))