#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
signal analysis
"""

import numpy as np
import matplotlib.pyplot as plt

# source: https://stackoverflow.com/questions/46473270/import-dat-file-as-an-array
def is_float(string):
    """ True if given string is float else False"""
    try:
        return float(string)
    except ValueError:
        return False

data = []
with open('signal.dat', 'r') as f:
    d = f.readlines()
    for i in d:
        k = i.rstrip().split('|')
        for i in k:
            if is_float(i):
                data.append(float(i)) 

data = np.array(data, dtype='float')
time = data[::2]
signal = data[1::2]

#a
plt.plot(time, signal, '.')
plt.xlabel("time")
plt.ylabel("signal")
plt.savefig('p2_a.png')
plt.show()

#b
A = np.zeros((len(time), 4))
A[:, 0] = 1.
A[:, 1] = time/np.max(time) 
A[:, 2] = (time/np.max(time))**2
A[:, 3] = (time/np.max(time))**3
(u, w, vt) = np.linalg.svd(A, full_matrices=False)
ainv = vt.transpose().dot(np.diag(1. / w)).dot(u.transpose())
c = ainv.dot(signal)
ym = A.dot(c) 
plt.plot(time, signal,'.', label='data')
plt.plot(time, ym, '.', label='model')
plt.xlabel("time")
plt.ylabel("signal")
plt.legend()
plt.savefig('p2_b.png')
plt.show()

#c
plt.plot(time, signal - ym, '.', label='data - model')
plt.xlabel('time')
plt.ylabel('$\Delta$')
plt.legend()
plt.savefig('p2_c.png')
plt.show()

#d
A_high = np.zeros((len(time), 30))
for i in range(30):
    A_high[:,i]=(time/np.max(time))**i
(U, W, Vt) = np.linalg.svd(A_high, full_matrices=False)
Ainv = Vt.transpose().dot(np.diag(1. / W)).dot(U.transpose())
C = Ainv.dot(signal)
Ym = A_high.dot(C) 
plt.plot(time, signal,'.', label='data')
plt.plot(time, Ym, '.', label='model')
plt.xlabel("time")
plt.ylabel("signal")
plt.legend()
plt.savefig('p2_d.png')
plt.show()

print(np.max(w)/np.min(w))
print(np.max(W)/np.min(W))





