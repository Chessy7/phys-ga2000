#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Brent’s 1D minimization
"""
import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt

def f(x):
    return (x-0.3)**2*np.exp(x)
x_array = np.arange(-3,3,0.01)
plt.plot(x_array,f(x_array))
plt.ylabel('f(x) logscale', fontsize = 16)
plt.yscale("log")
plt.xlabel('x', fontsize = 16)
plt.savefig('P1.png')
plt.show()

def s_quad_interp(a, b, c):
    """
    inverse quadratic interpolation
    """
    epsilon = 1e-7 #for numerical stability
    s0 = a*f(b)*f(c) / (epsilon + (f(a)-f(b))*(f(a)-f(c)))
    s1 = b*f(a)*f(c) / (epsilon + (f(b)-f(a))*(f(b)-f(c)))
    s2 = c*f(a)*f(b) / (epsilon + (f(c)-f(a))*(f(c)-f(b)))
    return s0+s1+s2

def golden_section_search(f, a, b, c, tol=1e-8, lim=100):
    gsection = (3-np.sqrt(5))/2
    i = 0
    while((np.abs(c-a) > tol) & (i<lim)):
        if((b-a) > (c-b)):
            x = b
            b = b-gsection*(b-a)
        else:
            x = b+gsection*(c-b)    
        if(f(b) < f(x)):
            c = x
        else:
            a = b 
            b = x 
        i += i     
    return b

def boptimize(f,a,b,c,tol):
    if abs(f(a)) < abs(f(b)):
        a, b = b, a #swap bounds
    flag = True
    err = abs(b-a)
    err_list, b_list = [err], [b]
    while err > tol:
        s = s_quad_interp(a,b,c)
        if ((s >= b))\
            or ((flag == True) and (abs(s-b) >= abs(b-c)))\
            or ((flag == False) and (abs(s-b) >= abs(c-d))):
            s = golden_section_search(f, a, b ,c)
            flag = True
        else:
            flag = False
        c, d = b, c # d is c from previous step
        #if f(a)*f(s) < 0:
        #    b = s
        #else:
        a = s
        if abs(f(a)) < abs(f(b)):
            a, b = b, a #swap if needed
        err = abs(b-a) #update error to check for convergence
        err_list.append(err)
        b_list.append(b)
    return b_list, err_list, s

b_list, err_list, optimizer = boptimize(f,0, 0.5, 1, 1e-7)
print("Brent's:", optimizer)

minimizer = optimize.brent(f, brack=(0, 0.5, 1), tol=1e-7)
print("Scipy:", minimizer)
print("Difference =", abs(optimizer - minimizer))
