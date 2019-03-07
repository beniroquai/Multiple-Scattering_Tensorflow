#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 19:17:44 2019

@author: bene
"""
import tensorflow as tf
import numpy as np
import src.tf_helper as tf_helper 
import matplotlib.pyplot as plt

tf_myabsorb = tf.Variable(np.zeros((200,200))+1j*np.zeros((200,200)))


# Selected slice
row_start = 50
row_end = 150
col_start = 50
col_end = 150
# Make indices from meshgrid
indexes = tf.meshgrid(tf.range(row_start, row_end),
                      tf.range(col_start, col_end), indexing='ij')
indexes = tf.stack(indexes, axis=-1)
# Take slice
updates = tf_myabsorb[row_start:row_end, col_start:col_end]
# Build tensor with "filtered" gradient
tf_myabsorb_part = tf.scatter_nd(indexes, updates, tf.shape(tf_myabsorb))
tf_myabsorb_2 = tf_myabsorb_part + tf.stop_gradient(-tf_myabsorb_part + tf_myabsorb)
# Continue as before...

#%%
mysample = np.ones((70,64,64))+1j*np.ones((70,64,64))
lambda0 = .65
k0 = 2*np.pi/lambda0
Boundary = 0
mysample = np.transpose(mysample, [1,2,0])
mysample, _ = tf_helper.insertPerfectAbsorber(mysample, 0, Boundary, -1, k0);
mysample, _ = tf_helper.insertPerfectAbsorber(mysample, mysample.shape[0] - Boundary, Boundary, 1, k0);
mysample, _ = tf_helper.insertPerfectAbsorber(mysample, 0, Boundary, -2, k0);
mysample, _ = tf_helper.insertPerfectAbsorber(mysample, mysample.shape[1] - Boundary, Boundary, 2, k0);
mysample = np.transpose(mysample, [2,0,1])
plt.imshow(np.real(mysample[10,:,:])), plt.colorbar(), plt.show()
plt.imshow(np.imag(mysample[10,:,:])), plt.colorbar(), plt.show()
