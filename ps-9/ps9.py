#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schrodinger equation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar
import matplotlib.animation as animation

N = 1000
L = 1e-8
x = np.linspace(0, L, N)
x0 = L/2
sigma = 1e-10
kapa = 5e10
phi0 = np.exp(-(x-x0)**2/(2*sigma**2))*np.exp(1j*kapa*x)


h = 1e-18
m = 9.109*10**(-31)
a = L/N
a1 = 1+h*(1j*hbar)/(2*m*a**2)
a2 = -h*(1j*hbar)/(4*m*a**2)
b1 = 1-h*(1j*hbar)/(2*m*a**2)
b2 = h*(1j*hbar)/(4*m*a**2)

#v = np.zeros(N, dtype=np.csingle)
#v[0] = b1*phi0[0]+b2*(phi0[1])
#v[N-1] = b1*phi0[N-1]+b2*(phi0[N-2])
#for i in range(1,N-1):
#    v[i] = b1*phi0[i]+b2*(phi0[i+1]+phi0[i-1])

A = np.diag(np.full(N,a1),k=0)+np.diag(np.full(N-1,a2),k=1)+np.diag(np.full(N-1,a2),k=-1)
#phi = np.linalg.solve(A, v)

def next_phi(A,phi_before,b1,b2):
    v = np.zeros(N, dtype=np.csingle)
    v[0] = b1*phi_before[0]+b2*(phi_before[1])
    v[N-1] = b1*phi_before[N-1]+b2*(phi_before[N-2])
    for i in range(1,N-1):
        v[i] = b1*phi_before[i]+b2*(phi_before[i+1]+phi_before[i-1])
    phi = np.linalg.solve(A, v)
    return phi

def phi(t,phi0,A,b1,b2):
    phi_before = phi0
    for i in range(t):
        phi = next_phi(A,phi_before,b1,b2)
        phi_before = phi
    return phi

t = 3000
phi_M = np.zeros((N,t),np.complex128)
phi_before = phi0
for i in range(t):
    phi_M[:,i] = next_phi(A,phi_before,b1,b2)
    phi_before = phi_M[:,i]




fig, ax = plt.subplots()
plt.title('Crank-Nicolson Solution for 1D Particle in a box')
plt.xlabel('X (m)')
plt.ylabel('$Re(\phi(x, t))$')

line, = ax.plot([], [], lw=2)

x_min, x_max = 0, 1e-8
ax.set_xlim(x_min, x_max)
y_min, y_max = -1, 1
ax.set_ylim(y_min, y_max)

def init():
    line.set_data([], [])
    return (line,)

def frame(i):
    line.set_data(x, np.real(phi_M[:,i]))
    return (line,)

anim = animation.FuncAnimation(fig, frame, init_func=init,
                               frames=t, interval=40, blit=True)

anim.save('animation.gif', writer='pillow')



