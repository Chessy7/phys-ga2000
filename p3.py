#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mandelbrot
"""

import numpy as np
import matplotlib.pyplot as plt

N = 1000
ITER = 100#Maximum iterations

X = np.arange(-2,2+4/N,4/N)
Y = np.arange(-2,2+4/N,4/N)
x,y = np.meshgrid(X,Y)
C = x+y*1.0j
Z = C
for i in range(ITER):
    arr = np.absolute(Z)
    Z = np.square(Z)+C
    Z[arr>2] = 1000
    
plt.imshow(arr,extent=(-2, 2, 2, -2))
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.hot()
plt.show()
