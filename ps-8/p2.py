#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lorentz
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.integrate import solve_ivp

init = [0,1,0]

def derivative_func(t, f):
    sigma = 10
    r = 28
    b = 8/3
    x = f[0]
    y = f[1]
    z = f[2]
    fx = sigma*(y-x)
    fy = r*x - y - x*z
    fz = x*y - b*z
    return [fx,fy,fz] 

t_span = [0, 50]

def numerical_traj_ex(t_span, init, t):
    sol4 = solve_ivp(derivative_func, t_span, init, t_eval = t,  \
                     method = 'LSODA')
    t = sol4.t
    f = sol4.y
    x = f[0,:]
    y = f[1,:]
    z = f[2,:]
    return t, x, y, z

exp_fps = 100 # samples per second
t = np.arange(*t_span, 1/exp_fps)
t, x, y, z = numerical_traj_ex(t_span, init, t)


plt.plot(t,y)
plt.ylabel('y', fontsize = 12)
plt.xlabel('t', fontsize = 12)
plt.savefig('P21.png')
plt.show()

plt.plot(x,z)
plt.ylabel('z', fontsize = 12)
plt.xlabel('x', fontsize = 12)
plt.savefig('P22.png')
plt.show()