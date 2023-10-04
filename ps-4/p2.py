#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Period of an anharmonic oscillator--part b
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

N = 20
m = 1
def T(amplitude):
    a = 0
    b = amplitude
    f = lambda x: 1/np.sqrt(amplitude**4-x**4)
    (I,none) = integrate.fixed_quad(f,a,b,n=N)
    return np.sqrt(8*m)*I

a_list = np.linspace(0,2,100)
period = np.zeros(100)
for i in range(100):
    amplitude = a_list[i]
    period[i] = T(amplitude)
plt.plot(a_list,period)
plt.xlabel("amplitude")
plt.ylabel("Period")
plt.savefig('p2.png')
plt.show()