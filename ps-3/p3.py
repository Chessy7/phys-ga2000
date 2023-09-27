#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decay Chain
P(t) from 10.3

"""

import numpy as np
import matplotlib.pyplot as plt

dt = 1
#initial atoms
N213Bi = 10000
N209Tl = 0
N209Pb = 0
N209Bi = 0
#half-lifes
h213Bi = 46*60
h209Tl = 2.2*60
h209Pb = 3.3*60
#probabilities to decay
p213Bi = 1-2**(-dt/h213Bi)
p209Tl = 1-2**(-dt/h209Tl)
p209Pb = 1-2**(-dt/h209Pb)


N213Bi_list = []
N209Tl_list = []
N209Pb_list = []
N209Bi_list = []

t = np.arange(0,2e4,dt)
for ti in t:
    N213Bi_list.append(N213Bi)
    N209Tl_list.append(N209Tl)
    N209Pb_list.append(N209Pb)
    N209Bi_list.append(N209Bi)
    for i in range(N209Pb):
        if np.random.random()<p209Pb:
            N209Pb-=1
            N209Bi+=1
    for i in range(N209Tl):
        if np.random.random()<p209Tl:
            N209Tl-=1
            N209Pb+=1
    for i in range(N213Bi):
        if np.random.random()<p213Bi:
            N213Bi-=1
            if np.random.random()<0.9791:
                N209Pb+=1
            else:
                N209Tl+=1
            
    
    

plt.plot(t,N213Bi_list,label='213Bi')
plt.plot(t,N209Tl_list,label='209Tl')
plt.plot(t,N209Pb_list,label='209Pb')
plt.plot(t,N209Bi_list,label='209Bi')
plt.legend()
plt.xlabel('time [s]')
plt.ylabel('# of atoms')
plt.savefig('p3.png')
plt.show()















