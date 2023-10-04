#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantum uncertainty in the harmonic oscillator
"""
#a
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


x = np.linspace(-4,4, 100)
n = [0,1,2,3]

for ni in n:
    plt.plot(x,1/np.sqrt(2**ni*factorial(ni)*np.sqrt(np.pi))*np.exp(-x**2/2)*H(ni,x),label=ni)

plt.xlabel("x")
plt.ylabel(r"$\phi$")
plt.legend()
plt.savefig('p3_a.png')
plt.clf()

#b
X = np.linspace(-10,10, 100)
N = 30

plt.plot(X,1/np.sqrt(2**N*factorial(N)*np.sqrt(np.pi))*np.exp(-X**2/2)*H(N,X))

plt.xlabel("x")
plt.ylabel(r"$\phi_{30}$")
plt.legend()
plt.savefig('p3_b.png')