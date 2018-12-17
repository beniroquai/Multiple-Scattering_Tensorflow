# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 12:53:08 2017

@author: Bene
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py 
import scipy.io
import time
import scipy as scipy
from scipy import ndimage
import h5py 
from tensorflow.python.client import device_lib
import scipy.misc



def saveHDF5(mydata, myfilename):
     hf = h5py.File(myfilename+'.h5', 'w')
     hf.create_dataset(myfilename, data=mydata)
     hf.close()
    

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def MidPos2D(anImg):
    res=np.shape(anImg)[0:2];
    return np.floor(res)/2

def MidPos3D(anImg):
    res=np.shape(anImg)[0:3];
    return np.ceil(res)/2

# Some helpful MATLAB functions
def abssqr(inputar):
    return np.real(inputar*np.conj(inputar))
    #return tf.abs(inputar)**2

def tf_abssqr(inputar):
    return tf.real(inputar*tf.conj(inputar))
    #return tf.abs(inputar)**2

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
    
def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind

def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind # array_shape[1]
    return (rows, cols)

def rr(inputsize_x=100, inputsize_y=100, inputsize_z=100, x_center=0, y_center = 0, z_center=0):
    x = np.arange(-inputsize_x/2,inputsize_x/2, 1)
    y = np.arange(-inputsize_y/2,inputsize_y/2, 1)
    if inputsize_z<=1:
        xx, yy = np.meshgrid(x+x_center, y+y_center)
        r = np.sqrt(xx**2+yy**2)
        r = np.transpose(r, [1, 0]) #???? why that?!
    else:
        z = np.arange(-inputsize_z/2,inputsize_z/2, 1)
        xx, yy, zz = np.meshgrid(x+x_center, y+y_center, z+z_center)
        xx, yy, zz = np.meshgrid(x, y, z)
        r = np.sqrt(xx**2+yy**2+zz**2)
        r = np.transpose(r, [1, 0, 2]) #???? why that?!
        
    return (r)

def rr_freq(inputsize_x=100, inputsize_y=100, inputsize_z=100, x_center=0, y_center = 0, z_center=0):
    x = np.linspace(-inputsize_x/2,inputsize_x/2, inputsize_x)/inputsize_x
    y = np.linspace(-inputsize_y/2,inputsize_y/2, inputsize_y)/inputsize_y
    if inputsize_z<=1:
        xx, yy = np.meshgrid(x+x_center, y+y_center)
        r = np.sqrt(xx**2+yy**2)
        r = np.transpose(r, [1, 0]) #???? why that?!
    else:
        z = np.linspace(-inputsize_z/2,inputsize_z/2, inputsize_z)/inputsize_z
        xx, yy, zz = np.meshgrid(x+x_center, y+y_center, z+z_center)
        xx, yy, zz = np.meshgrid(x, y, z)
        r = np.sqrt(xx**2+yy**2+zz**2)
        r = np.transpose(r, [1, 0, 2]) #???? why that?!

        
    return (r)


def xx(inputsize_x=128, inputsize_y=128, inputsize_z=1):
    x = np.arange(-inputsize_x/2,inputsize_x/2, 1)
    y = np.arange(-inputsize_y/2,inputsize_y/2, 1)
    if inputsize_z<=1:
        xx, yy = np.meshgrid(x, y)
        xx = np.transpose(xx, [1, 0]) #???? why that?!
    else:
        z = np.arange(-inputsize_z/2,inputsize_z/2, 1)
        xx, yy, zz = np.meshgrid(x, y, z)
        xx = np.transpose(xx, [1, 0, 2]) #???? why that?!
    return (xx)

def xx_freq(inputsize_x=128, inputsize_y=128, inputsize_z=1):
    x = np.arange(-inputsize_x/2,inputsize_x/2, 1)/inputsize_x
    y = np.arange(-inputsize_y/2,inputsize_y/2, 1)/inputsize_x
    if inputsize_z<=1:
        xx, yy = np.meshgrid(x, y)
        xx = np.transpose(xx, [1, 0]) #???? why that?!
    else:
        z = np.arange(-inputsize_z/2,inputsize_z/2, 1)/inputsize_x
        xx, yy, zz = np.meshgrid(x, y, z)
        xx = np.transpose(xx, [1, 0, 2]) #???? why that?!
    return (xx)

def yy(inputsize_x=128, inputsize_y=128, inputsize_z=1):
    x = np.arange(-inputsize_x/2,inputsize_x/2, 1)
    y = np.arange(-inputsize_y/2,inputsize_y/2, 1)
    if inputsize_z<=1:
        xx, yy = np.meshgrid(x, y)
        yy = np.transpose(yy, [1, 0]) #???? why that?!
    else:
        z = np.arange(-inputsize_z/2,inputsize_z/2, 1)
        xx, yy, zz = np.meshgrid(x, y, z)
        yy = np.transpose(yy, [1, 0, 2]) #???? why that?!
    return (yy)

def yy_freq(inputsize_x=128, inputsize_y=128, inputsize_z=1):
    x = np.arange(-inputsize_x/2,inputsize_x/2, 1)/inputsize_x
    y = np.arange(-inputsize_y/2,inputsize_y/2, 1)/inputsize_x
    if inputsize_z<=1:
        xx, yy = np.meshgrid(x, y)
        yy = np.transpose(yy, [1, 0]) #???? why that?!
    else:
        z = np.arange(-inputsize_z/2,inputsize_z/2, 1)/inputsize_x
        xx, yy, zz = np.meshgrid(x, y, z)
        yy = np.transpose(xx, [1, 0, 2]) #???? why that?!
    return (yy)

def zz(inputsize_x=128, inputsize_y=128, inputsize_z=1):
    nx = np.arange(-inputsize_x/2,inputsize_x/2, 1)
    ny = np.arange(-inputsize_y/2,inputsize_y/2, 1)
    nz = np.arange(-inputsize_z/2,inputsize_z/2, 1)
    xxr, yyr, zzr = np.meshgrid(nx, ny, nz)
    zzr = np.transpose(zzr, [1, 0, 2]) #???? why that?!
    return (zzr)


def phiphi(inputsize_x, inputsize_y, inputsize_z):
    nx = np.linspace(-np.pi, np.pi, inputsize_x)
    ny = np.linspace(-np.pi, np.pi, inputsize_y)
    nz = np.linspace(-np.pi, np.pi, inputsize_z)
    xx, yy, zz = np.meshgrid(nx, ny, nz)
    phi = np.arctan2(xx, yy)
    phi = np.transpose(phi, [1, 0, 2]) #???? why that?!
    
    theta = np.arcsin(yy/np.sqrt(xx**2+yy**2+zz**2))
    theta = np.transpose(theta, [1, 0, 2]) #???? why that?!
    return phi, theta

def binary_activation(x):

    cond = tf.less(x, tf.zeros(tf.shape(x)))
    out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))

    return out

# total variation denoising
def total_variation_regularization(x, beta=1):
    #assert isinstance(x, tf.Tensor)
    wh = tf.constant([[[[ 1], [ 1], [ 1]]], [[[-1], [-1], [-1]]]], tf.float32)
    ww = tf.constant([[[[ 1], [ 1], [ 1]], [[-1], [-1], [-1]]]], tf.float32)
    tvh = lambda x: tf.nn.conv2d(x, wh, strides = [1, 1, 1, 1], padding='SAME')
    tvw = lambda x: tf.nn.conv2d(x, ww, strides = [1, 1, 1, 1], padding='SAME')
    dh = tvh(x)
    dw = tvw(x)
    tv = (tf.add(tf.reduce_sum(dh**2, [1, 2, 3]), tf.reduce_sum(dw**2, [1, 2, 3]))) ** (beta / 2.)
    return tv


def relative_error(orig, rec):
    return np.mean((orig - rec) ** 2)
    #return np.sum(np.square(np.abs(orig-rec))/np.square(np.abs(orig)))

def repmat4d(inputarr, n4dim):
    return np.tile(np.reshape(inputarr, [inputarr.shape[0], inputarr.shape[1], 1, 1]), [1, 1, 1, n4dim])
    


# %% FT

# fftshifts
def fftshift2d(tensor):
    """
    Shifts high frequency elements into the center of the filter.
    Works on last 2 dims of tensor (on both for size-2-tensors)
    Works only for even number of elements along these dims.
    """
    #warn("Only implemented for even number of elements in each axis.")
    last_dim = len(tensor.get_shape()) - 1  # from 0 to shape-1
    top, bottom = tf.split(tensor, 2, last_dim)  # split into two along last axis
    tensor = tf.concat([bottom, top], last_dim)  # concatenates along last axis
    left, right = tf.split(tensor, 2, last_dim - 1)  # split into two along second last axis
    tensor = tf.concat([right, left], last_dim - 1)  # concatenate along second last axis
    return tensor

def ifftshift2d(tensor):
    """
    Shifts high frequency elements into the center of the filter.
    Works on last 2 dims of tensor (on both for size-2-tensors)
    Works only for even number of elements along these dims.
    """
    #warn("Only implemented for even number of elements in each axis.")
    last_dim = len(tensor.get_shape()) - 1
    left, right = tf.split(tensor, 2, last_dim - 1)
    tensor = tf.concat([right, left], last_dim - 1)
    top, bottom = tf.split(tensor, 2, last_dim)
    tensor = tf.concat([bottom, top], last_dim)
    return tensor

def fftshift3d(tensor):
    """
    Shifts high frequency elements into the center of the filter.
    Works on last 3 dims of tensor (on all for size-3-tensors)
    Works only for even number of elements along these dims.
    """
    #warn("Only implemented for even number of elements in each axis.")
    top, bottom = tf.split(tensor, 2, -1)
    tensor = tf.concat([bottom, top], -1)
    left, right = tf.split(tensor, 2, -2)
    tensor = tf.concat([right, left], -2)
    front, back = tf.split(tensor, 2, -3)
    tensor = tf.concat([back, front], -3)
    return tensor

def ifftshift3d(tensor):
    """
    Shifts high frequency elements into the center of the filter.
    Works on last 3 dims of tensor (on all for size-3-tensors)
    Works only for even number of elements along these dims.
    """
    #warn("Only implemented for even number of elements in each axis.")
    left, right = tf.split(tensor, 2, -2)
    tensor = tf.concat([right, left], -2)
    top, bottom = tf.split(tensor, 2, -1)
    tensor = tf.concat([bottom, top], -1)
    front, back = tf.split(tensor, 2, -3)
    tensor = tf.concat([back, front], -3)
    return tensor 


# I would recommend to use this
def my_ft2d(tensor):
    """
    fftshift(fft(ifftshift(a)))
    
    Applies shifts to work with arrays whose "zero" is in the middle 
    instead of the first element.
    
    Uses standard normalization of fft unlike dip_image.
    """
    return fftshift2d(tf.fft2d(ifftshift2d(tensor)))

def my_ift2d(tensor):
    """
    fftshift(ifft(ifftshift(a)))
    
    Applies shifts to work with arrays whose "zero" is in the middle 
    instead of the first element.
    
    Uses standard normalization of ifft unlike dip_image.
    """
    return fftshift2d(tf.ifft2d(ifftshift2d(tensor)))

def my_ft3d(tensor):
    """
    fftshift(fft(ifftshift(a)))
    
    Applies shifts to work with arrays whose "zero" is in the middle 
    instead of the first element.
    
    Uses standard normalization of fft unlike dip_image.
    """
    return fftshift3d(tf.fft3d(ifftshift3d(tensor)))

def my_ift3d(tensor):
    """
    fftshift(ifft(ifftshift(a)))
    
    Applies shifts to work with arrays whose "zero" is in the middle 
    instead of the first element.
    
    Uses standard normalization of fft unlike dip_image.
    """
    return fftshift3d(tf.ifft3d(ifftshift3d(tensor)))
