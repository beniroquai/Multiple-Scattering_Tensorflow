#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 21:28:06 2017

@author: Bene
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py 
import scipy.io
import scipy as scipy
import scipy.misc

# own functions
from src import tf_helper as tf_helper


def generateObject(mysize = [100, 100, 100], obj_dim = [0.1, 0.1, 0.1], obj_type ='sphere', diameter = 1, dn = 0.1):
    ''' Function to generate a 3D RI distribution to give an artificial sample for testing the FWD system
    INPUTS:
        obj_shape - Nx, Ny, Nz e.g. [100, 100, 100]
        obj_dim - sampling in dx/dy/dz 0.1 (isotropic for now)
        obj_type - 0; One Sphere 
                 - 1; Two Spheres 
                 - 2; Two Spheres 
                 - 3; Two Spheres inside volume
                 - 4; Shepp Logan Phantom (precomputed in MATLAB 120x120x120 dim)
        diameter - 1 (parameter for diameter of sphere)
        dn - difference of RI e.g. 0.1
        
    OUTPUTS: 
        f - 3D RI distribution
            
    '''

    if(obj_type=='sphere'):
        # one spherical object inside a volume
        obj = (tf_helper.rr((mysize[0], mysize[1], mysize[2]), mode='center')<diameter)*dn
        
    elif(obj_type == 'twosphere'):
        # two spherical objects inside a volume
        sphere = dn*(tf_helper.rr((mysize[0], mysize[1], mysize[2]))* obj_dim < diameter)
        sphere1 = np.roll(np.roll(np.roll(sphere,5,0),-5,1),5,2);
        sphere2 = np.roll(np.roll(np.roll(sphere,-5,0),5,1),-5,2);
        obj = sphere1 + sphere2 
    elif(obj_type == 'eigtsphere'):
        # two spherical objects inside a volume
        sphere = dn*(tf_helper.rr(mysize[0], mysize[1], mysize[2], x_center=5, y_center = 0, z_center=10)* obj_dim < diameter)
        sphere1 = np.roll(np.roll(np.roll(sphere,5,0),5,1),5,2);
        sphere2 = np.roll(np.roll(np.roll(sphere,5,0),5,1),-5,2);
        sphere3 = np.roll(np.roll(np.roll(sphere,5,0),-5,1),5,2);
        sphere4 = np.roll(np.roll(np.roll(sphere,5,0),-5,1),-5,2);
        sphere5 = np.roll(np.roll(np.roll(sphere,-5,0),5,1),5,2);
        sphere6 = np.roll(np.roll(np.roll(sphere,-5,0),5,1),-5,2);
        sphere7 = np.roll(np.roll(np.roll(sphere,-5,0),-5,1),5,2);
        sphere8 = np.roll(np.roll(np.roll(sphere,-5,0),-5,1),-5,2);
        
        obj = sphere1 + sphere2 + sphere3 + sphere4 +sphere5 + sphere6 + sphere7 + sphere8
        
    elif(obj_type ==  'foursphere'):
        # four spherical objects inside a volume
        obj = dn*(tf_helper.rr(mysize[2], mysize[0], mysize[1], x_center=5, y_center = 0, z_center=10) * obj_dim < diameter);
        obj = obj+dn*(tf_helper.rr(mysize[2], mysize[0], mysize[1], x_center=-5, y_center = -5, z_center=-10) * obj_dim < diameter);
        obj = obj+dn*(tf_helper.rr(mysize[2], mysize[0], mysize[1], x_center=10, y_center = 0, z_center=-20) * obj_dim < diameter);
        obj = obj+dn*(tf_helper.rr(mysize[2], mysize[0], mysize[1], x_center=-10, y_center = 0, z_center=20) * obj_dim < diameter);
    #f = np.transpose(f, axes=[2, 0, 1])
    elif(obj_type=='SheppLogan'):
        print('WARNING: WRONG DIMENSIONS!!')
        inputmat_dir = './Data/SheppLogan/'
        inputmat_name = 'phantom3d_120.mat'
        mat_input = h5py.File(inputmat_dir+inputmat_name)
        obj = np.array(mat_input['obj'])*dn
    elif(obj_type=='init'):
        obj = np.ones(mysize) * (dn)

    return obj


            
