# -*- coding: utf-8 -*-
"""
Gaussian with zero mean and a standard deviation of 3 over the range [−10, +10]
"""

import numpy as np
import matplotlib.pyplot as plt


sigma = 3
mu = 0
x_axis = np.arange(-10,10.1,0.1)

plt.plot(x_axis, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x_axis - mu)**2 / (2 * sigma**2)) )
plt.xlim((-11,11))

plt.xlabel("position")
plt.ylabel("probability density")

plt.savefig("gaussian.png")