#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decay again
transformation method
"""
import numpy as np
import matplotlib.pyplot as plt

N = 1000
tau = 3.053*60

#non-uniform distribution
decaytimes = -1/(np.log(2)/tau)*np.log(1-np.random.random(N))

#find the number of decaytimes larger than the current time
t = np.arange(0,2000)
nodecay = []
for ti in t:
    idx = [i for i,v in enumerate(decaytimes) if v > ti]
    nodecay.append(len(idx))

plt.plot(t,nodecay)
plt.xlabel('time [s]')
plt.ylabel('# of atoms not decayed')
plt.savefig('p4.png')
plt.show()
