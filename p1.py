#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NumPy 32-bit floating point 100.98763
"""

import numpy as np

f = np.float32(100.98763)
int32bits = f.view(np.int32)

print('Numpy 32-bit floating point represents 100.98763 as','{:032b}'.format(int32bits))

diff = f - 100.98763

print("The round-off error of such representation is",diff)