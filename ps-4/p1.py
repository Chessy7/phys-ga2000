#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Heat Capacity of a solid
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.constants as sc

#a, set N=50
def func(x=None):
    return x**4*np.exp(x)/(np.exp(x)-1)**2

def cv(T,N):
    V = 0.001
    rho = 6.022*10**28
    Debye = 428
    a = 0
    b = Debye/T
    (I,none) = integrate.fixed_quad(func,a,b,n=N)
    return 9*V*rho*sc.k*(T/Debye)**3*I
#b
Temp = np.linspace(5,500,100)
Cv = np.zeros(100)
for i in range(100):
    T = Temp[i]
    Cv[i] = cv(T,50)
plt.plot(Temp,Cv)
plt.xlabel("Temperature (K)")
plt.ylabel("Heat Capacity Cv")
plt.savefig('p1_b.png')
plt.show()
#c
Temp = np.linspace(5,500,100)
N_list = np.array([10,20,30,40,50,60,70])
Cv_M = np.zeros((7,100))
for j in range(7):
    N = N_list[j]
    for i in range(100):
        T = Temp[i]
        Cv_M[j,i] = cv(T,N)
    plt.plot(Temp,Cv_M[j,:],label=N_list[j])
plt.xlabel("Temperature (K)")
plt.ylabel("Heat Capacity Cv")
plt.legend()
plt.savefig('p1_c.png')
plt.show()