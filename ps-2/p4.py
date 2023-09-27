#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quadratic Equation
"""
import numpy as np


def standard(a,b,c):
    x_1 = (-b+np.sqrt(b**2 - 4*a*c))/(2*a)
    x_2 = (-b-np.sqrt(b**2 - 4*a*c))/(2*a)
    return x_1,x_2

print(standard(0.001,1000,0.001))

def multi(a,b,c):
    x_1 = (2*c)/(-b-np.sqrt(b**2 - 4*a*c))
    x_2 = (2*c)/(-b+np.sqrt(b**2 - 4*a*c))
    return x_1,x_2

print(multi(0.001,1000,0.001))