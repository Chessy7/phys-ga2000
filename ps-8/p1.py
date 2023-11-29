#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FFT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.integrate import solve_ivp

import pandas as pd

piano = pd.read_csv('piano.txt', header = None).to_numpy()

plt.plot(np.arange(0, len(piano)), piano)
plt.xlabel('time')
plt.ylabel('amplitude')
plt.savefig('P11.png')
plt.show()



def fourier(data):
    N = len(data)
    t = np.arange(0, N)
    

    # sample spacing

    T = t[1]-t[0]

    yf = fft(data)

    xf = fftfreq(N, T)[:N//2]
    return xf, yf


xf, yf = fourier(piano)
N = len(piano)
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()
plt.xlabel('frequency')
plt.ylabel('magnitude')
plt.savefig('P12.png')
plt.show()

#do the same for trumpet data
trumpet = pd.read_csv('trumpet.txt', header = None).to_numpy()

plt.plot(np.arange(0, len(trumpet)), trumpet)
plt.xlabel('time')
plt.ylabel('amplitude')
plt.savefig('P13.png')
plt.show()

xf, yf = fourier(trumpet)
N = len(trumpet)
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()
plt.xlabel('frequency')
plt.ylabel('magnitude')
plt.savefig('P14.png')
plt.show()

