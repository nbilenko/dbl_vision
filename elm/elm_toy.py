#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

########
# config

# number of hidden units
nhidden = 200

# validation fraction
v_frac = 0.1

#######
# setup

# load the data
digits = load_digits()
all_data = digits.data
all_target = digits.target
ndata, nfeatures = all_data.shape

# convert target to indicators
all_t = np.zeros((all_target.size, 10))
for i in range(all_target.size):
    all_t[i,all_target[i]] = 1.0

# partition
nval = int(v_frac * ndata)
ii = np.random.permutation(ndata)
d   = all_data[ii[nval:],:]
d_v = all_data[ii[:nval],:]
t   = all_t[ii[nval:],:]
t_v = all_t[ii[:nval],:]

##############################
# layer weights, bias (random)

w = 2.0 * np.random.rand(nfeatures, nhidden) - 1.0
b = 2.0 * np.random.rand(nhidden) - 1.0

###################################
# train (leads to output weights B)

# hidden-layer activations
H = 1.0 / (1.0 + np.exp(- np.dot(d, w) + b))

# pseudo-inverse of activations
H_dagger = np.linalg.pinv(H)

# solve for output weight matrix
B = np.dot(H_dagger, t)

## training misfit (reuse the previously computed hidden-layer activations H)

pred = np.round(np.dot(H, B))
res = np.abs(pred - t).sum(axis=1) > 0
print('Training misclassification rate: %4.1f %%' % (100 * float(res.sum()) / res.size))

## validation misfit (need to compute new activations)

H_v = 1.0 / (1.0 + np.exp(- np.dot(d_v, w) + b))
pred = np.round(np.dot(H_v, B))
res = np.abs(pred - t_v).sum(axis=1) > 0
print('Training misclassification rate: %4.1f %%' % (100 * float(res.sum()) / res.size))
