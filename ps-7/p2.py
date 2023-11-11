#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
likelihood maximization
"""
import numpy as np
import matplotlib.pyplot as plt

def p(x, beta_0, beta_1):
    return 1/(1+np.exp(-(beta_0+beta_1*x)))

import pandas as pd

data = pd.read_csv('survey.csv')  

xs = data['age'].to_numpy()
ys = data['recognized_it'].to_numpy()
x_sort = np.argsort(xs)
xs = xs[x_sort]
ys = ys[x_sort]

def log_likelihood(beta, xs, ys):
    beta_0 = beta[0]
    beta_1 = beta[1]
    epsilon = 1e-16
    l_list = [ys[i]*np.log(p(xs[i], beta_0, beta_1)/(1-p(xs[i], beta_0, beta_1)+epsilon)) 
              + np.log(1-p(xs[i], beta_0, beta_1)+epsilon) for i in range(len(xs))]
    ll = np.sum(np.array(l_list), axis = -1)
    return -ll # return log likelihood

from scipy import optimize

def Covariance(hess_inv, resVariance):
    return hess_inv * resVariance

def error(hess_inv, resVariance):
    covariance = Covariance(hess_inv, resVariance)
    return np.sqrt( np.diag( covariance ))


beta = np.array([1,1]) # start with some initial beta

result = optimize.minimize(lambda beta, xs, ys:log_likelihood(beta, xs, ys),beta,args=(xs, ys))
hess_inv = result.hess_inv # inverse of hessian matrix
var = result.fun/(len(ys)-len(beta)) 
dFit = error( hess_inv,  var)
print('Optimal parameters and error:\n\tbeta: ' , result.x, '\n\tdbeta: ', dFit)
print('Covariance matrix of optimal parameters:\n\tC: ' , Covariance( hess_inv,  var))

plt.plot(xs, p(xs, result.x[0], result.x[1]))
plt.title('Probability of hearing the phrase')
plt.xlabel('Age (y)')
plt.ylabel('Probabilty')
plt.savefig('P2.png')


