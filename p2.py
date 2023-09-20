#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Madelung
"""

import numpy as np
import timeit

#Method 1
def Madelung_forloop(L):
    M = 0
    for i in range(-L,L+1):
        for j in range(-L,L+1):
            for k in range(-L,L+1):
                if i==0 and j==0 and k==0:
                    M = M
                else:
                    M = M+(-1)**(i+j+k)/(np.sqrt(i**2+j**2+k**2))
    print("Madelung constant is",M)
    return M
#Method 2
def Madelung_meshgrid(L):
    nums = np.arange(-L,L+1,dtype=np.float32)#Integers to negative integer powers are not allowed when calculating M_matrix.
    i, j, k = np.meshgrid(nums, nums, nums)
    M_matrix = (-1)**(i+j+k)/(np.sqrt(i**2+j**2+k**2))
    M_matrix[(i == 0)*(j == 0)*(k == 0)] = 0
    M = np.sum(M_matrix)
    print("Madelung constant is",M)
    return M
#Time

n = 5
L = 100
resultfor = timeit.timeit(stmt='Madelung_forloop(L)', globals=globals(), number=n)
print(f"For loop execution time is {resultfor / n} seconds for L={L}.")
resultgrid = timeit.timeit(stmt='Madelung_meshgrid(L)', globals=globals(), number=n)
print(f"Grid execution time is {resultgrid / n} seconds for L={L}.")