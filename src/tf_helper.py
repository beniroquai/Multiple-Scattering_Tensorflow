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




###### CHRISTIANS STUFF
    # Copyright Christian Karras
    
def ramp(mysize=(256,256), ramp_dim=0, corner='center'):
    '''
    creates a ramp in the given direction direction 
    standart size is 256 X 256
    mode: 
        center: 0 is at cetner
                if x is even, it has one more value in the positve:
                    e.g. size_x = 100 -> goes from -50 to 49
                         size_x = 101 -> goes from -50 to 50
        negative : goes from negative size_x to 0
        positvie : goes from 0 size_x to positive
        int number: is the index where the center is!
    '''
    
    if type(mysize)== np.ndarray:
        mysize = mysize.shape;
    
    res = np.ones(mysize);
   
    if corner == 'negative':
        miniramp = np.arange(-mysize[ramp_dim]+1,1,1);
    elif corner == 'positive':
        miniramp = np.arange(0,mysize[ramp_dim],1);
    elif corner == 'freq':
        miniramp = np.arange(-mysize[ramp_dim]//2+np.mod(mysize[ramp_dim],2),mysize[ramp_dim]//2+np.mod(mysize[ramp_dim],2),1)/mysize[ramp_dim];
    elif corner == 'center':
        miniramp = np.arange(-mysize[ramp_dim]//2+np.mod(mysize[ramp_dim],2),mysize[ramp_dim]//2+np.mod(mysize[ramp_dim],2),1);
    elif (type(corner) == int or type(corner) == float):
        miniramp = np.arange(0,mysize[ramp_dim],1)-corner;
    else:
        try: 
            if np.issubdtype(corner.dtype, np.number):
                miniramp = np.arange(0,mysize[ramp_dim],1)-corner;
        except AttributeError:
           
            pass;
    minisize = list(np.ones(len(mysize)).astype(int));
    minisize[ramp_dim] = mysize[ramp_dim];
    #np.seterr(divide ='ignore');
    miniramp = np.reshape(miniramp,minisize)
    res*=miniramp;
    return(res);


def xx(mysize = (256,256), mode = 'center'):
    '''
    creates a ramp in x direction 
    standart size is 256 X 256
    mode: 
        center: 0 is at cetner
                if x is even, it has one more value in the positve:
                    e.g. size_x = 100 -> goes from -50 to 49
        negative : goes from negative size_x to 0
        positvie : goes from 0 size_x to positive
    '''
    return(ramp(mysize,1,mode))

def yy(mysize = (256,256), mode = 'center'):
    '''
    creates a ramp in y direction 
    standart size is 256 X 256
    mode: 
        center: 0 is at cetner
                if y is even, it has one more value in the positve:
                    e.g. size_y = 100 -> goes from -50 to 49
        negative : goes from negative size_y to 0
        positvie : goes from 0 size_y to positive
    '''
    return(ramp(mysize,0,mode))
 
def zz(mysize = (256,256), mode = 'center'):
    '''
    creates a ramp in z direction 
    standart size is 256 X 256
    mode: 
        center: 0 is at cetner
                if y is even, it has one more value in the positve:
                    e.g. size_y = 100 -> goes from -50 to 49
        negative : goes from negative size_y to 0
        positvie : goes from 0 size_y to positive
    '''
    return(ramp(mysize,2,mode))

def rr(mysize=(256,256), offset = (0,0,0),scale = None, mode='center'):
    '''
    creates a ramp in r direction 
    standart size is 256 X 256
    mode is always "center"
    offset -> x/y offset in pixels (number, list or tuple)
    scale is tuple, list, none or number of axis scale
    '''
    import numbers;
    if offset is None:
        scale = [0,0,0];
    elif isinstance(offset, numbers.Number):
        offset = [offset];
    elif type(offset)  == list or type(offset) == tuple:
        offset = list(offset[0:3]);
    else:
        raise TypeError('Wrong data type for offset -> must be Number, list, tuple or none')
        
    if scale is None:
        scale = [1,1,1];
    elif isinstance(scale, numbers.Integral):
        scale = [scale, scale];
    elif type(scale)  == list or type(scale) == tuple:
        scale = list(scale[0:3]);
    else:
        raise TypeError('Wrong data type for scale -> must be integer, list, tuple or none')
    return(np.sqrt(((ramp(mysize,0, mode)-offset[0])*scale[0])**2
                   +((ramp(mysize,1, mode)-offset[1])*scale[1])**2
                   +((ramp(mysize,2, mode)-offset[2])*scale[2])**2))
   
def phiphi(mysize=(256,256), offset = 0, angle_range = 1):
    '''
    creates a ramp in phi direction 
    standart size is 256 X 256
    mode is always center
    offset: angle offset in rad
    angle_range:
            1:   0 - pi for positive y, 0 - -pi for negative y
            2:   0 - 2pi for around
        
    '''
    np.seterr(divide ='ignore', invalid = 'ignore');
    x = ramp(mysize,0,'center');
    y = ramp(mysize,1,'center');
    #phi = np.arctan(y/x)+(x<0)*np.pi*((y>0)*2-1);
    if angle_range == 1:
        phi = np.mod((np.arctan(y/x)+(x<0)*np.pi*((y>0)*2-1)+offset)+np.pi, 2*np.pi) -np.pi;
    elif angle_range == 2:
        phi = np.mod((np.arctan(y/x)+(x<0)*np.pi*((y>0)*2-1)+offset), 2*np.pi);
    phi[phi.shape[0]//2,phi.shape[1]//2]=0;
    np.seterr(divide='warn', invalid = 'warn');
    return(phi)    