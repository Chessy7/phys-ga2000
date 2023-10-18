#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gamma Function
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.constants as sc

#a
x = np.linspace(0,5,100)
a_list = np.array([2,3,4])
y = np.zeros((3,100))
for i in range(3):
    a = a_list[i]
    y[i,:] = x**(a-1)*np.exp(-x)
    plt.plot(x,y[i,:],label=a)
plt.xlabel("x")
plt.ylabel(r"$x^{a-1}e^{-x}$")
plt.legend(title="a")
plt.savefig('p1_a.png')
plt.show()

#e  
def gamma(a):
    f = lambda xp:  (a-1)/(1-xp)**2*np.exp((a-1)*np.log((a-1)*xp/(1-xp))-(a-1)*xp/(1-xp))
    range = np.array([0., 1.], dtype=np.float64)
    (I,none) = integrate.fixed_quad(f,range[0], range[1], n=100)
    return I

print(gamma(3/2))

#f
print(gamma(3))
print(gamma(6))
print(gamma(10))