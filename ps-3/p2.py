#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Matrix Multiplication
"""
import numpy as np
import timeit
from numpy import zeros
import matplotlib.pyplot as plt

def matrix_multi(A,B,N):
    C = zeros([N,N],float)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i,j] += A[i,k]*B[k,j]
    return C
            
def dot(A,B):
    return np.dot(A,B)
    
    

#Time

n = 1

N_list = np.array([10,30,50,100,150,200,300,400,500],dtype=np.float32)
timefor = zeros(len(N_list),float)
timenp = zeros(len(N_list),float)

for i in range(len(N_list)):
    N = int(N_list[i])
    A = np.random.rand(N,N)
    B = np.random.rand(N,N)
    resultfor = timeit.timeit(stmt='matrix_multi(A,B,N)', globals=globals(), number=n)
    resultnp = timeit.timeit(stmt='dot(A,B)', globals=globals(), number=n)
    timefor[i] = resultfor/n
    timenp[i] = resultnp/n
    
plt.plot(N_list,timefor,label='for loop')
plt.plot(N_list,timenp,label='numpy')
AA=np.arange(1,500,dtype=np.float32)
plt.plot(AA,(AA**3)/1000000,label=r"$\propto$N^3")
plt.legend()
plt.xlabel('size of matrix')
plt.ylabel('time')
plt.savefig('p2.png')
plt.show()










