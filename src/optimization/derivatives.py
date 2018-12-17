#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 18:08:24 2018

@author: soenke
"""

import tensorflow as tf
#import numpy as np
import importlib.util

# waiting for depthwise_conv3d to be implemented to parallelize and 
# calculate all at once
# copy volume and ker 3 times to 4th dim and then do depthwise conv
# downside of this would be memory requirement

__all__ = ['d1x', 'd1y', 'd1z',
           'd1x_central', 'd1y_central', 'd1z_central',
           'd1x_central_shift', 'd1y_central_shift', 'd1z_central_shift',
           'd1x_central_conv', 'd1y_central_conv', 'd1z_central_conv',
           '_d1x_central_shift_sliced', '_d1y_central_shift_sliced', 
           '_d1z_central_shift_sliced',
           'd1x_fwd', 'd1y_fwd', 'd1z_fwd',
           'd1x_fwd_shift', 'd1y_fwd_shift', 'd1z_fwd_shift',
           'd1x_fwd_conv', 'd1y_fwd_conv', 'd1z_fwd_conv',
           'd1x_bwd', 'd1y_bwd', 'd1z_bwd',
           'd1x_bwd_shift', 'd1y_bwd_shift', 'd1z_bwd_shift',
           'd1x_bwd_conv', 'd1y_bwd_conv', 'd1z_bwd_conv',
           '_d1x_fwd_full', '_d1y_fwd_full', '_d1z_fwd_full',
           '_d1x_fwd_shift_full', '_d1y_fwd_shift_full', '_d1z_fwd_shift_full',
           '_d1x_fwd_conv_full', '_d1y_fwd_conv_full', '_d1z_fwd_conv_full'
           ]

# TODO: include formula in docs    
# TODO: include first order accuracy in docs


# TODO -> change to Rainer's formulation in gr and tv reg
# !!!
# NOTE
# "The main problem with the central difference method, however, 
# is that regularly oscillating functions will yield zero derivative."
# !!!

# circshift:
# https://stackoverflow.com/questions/42651714/vector-shift-roll-in-tensorflow

# circular conv:
# def _cconv(self, a, b):
#     return tf.ifft(tf.fft(a) * tf.fft(b)).real
# or use circshift from above to implement
# or use tf.pad:
# https://stackoverflow.com/questions/37659538/\
# custom-padding-for-convolutions-in-tensorflow

# %% Wrappers of preferred functions

def d1x(volume, step_size=1):
    """
    calculates 1st spatial derivative along x (the last axis) of a 3d-tensor 
    using a predefined method.
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated
        step_size (float): step size of sampling in x-direction (pixel size) 
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives. Might have
            different shape input volume depending on handling of borders.
        
    Preferred is central difference scheme - See d1x_central for reference.
    """
    return d1x_central(volume, step_size)

def d1y(volume, step_size=1):
    """
    calculates 1st spatial derivative along y (the second axis) of a 3d-tensor 
    using a predefined method.
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated
        step_size (float): step size of sampling in y-direction (pixel size) 
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives. Might have
            different shape input volume depending on handling of borders.
    
    Preferred is central difference scheme - See d1y_central for reference.
    """
    return d1y_central(volume, step_size)

def d1z(volume, step_size=1):
    """
    calculates 1st spatial derivative along z (the first axis) of a 3d-tensor 
    using a predefined method.
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated
        step_size (float): step size of sampling in z-direction
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives. Might have
            different shape input volume depending on handling of borders.
    
    Preferred is central difference scheme - See d1z_central for reference.
    """
    return d1z_central(volume, step_size)


# %% central differences wrapper (second order accuracy)
    
def d1x_central(volume, step_size=1):
    """
    calculates 1st spatial derivative along x (the last axis) of a 3d-tensor 
    using 3-pt central difference scheme implemented by a predefined method.
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated
        step_size (float): step size of sampling in x-direction (pixel size) 
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives. Might have
            different shape input volume depending on handling of borders.
    
    Preferred is implementation by shift - See d1x_central_shift for reference.
    """
    return d1x_central_shift(volume, step_size)

def d1y_central(volume, step_size=1):
    """
    calculates 1st spatial derivative along y (the second axis) of a 3d-tensor 
    using 3-pt central difference scheme implemented by a predefined method.
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated
        step_size (float): step size of sampling in y-direction (pixel size) 
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives. Might have
            different shape input volume depending on handling of borders.
    
    Preferred is implementation by shift - See d1y_central_shift for reference.
    """
    return d1y_central_shift(volume, step_size)

def d1z_central(volume, step_size=1):
    """
    calculates 1st spatial derivative along z (the first axis) of a 3d-tensor 
    using 3-pt central difference scheme implemented by a predefined method.
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated
        step_size (float): step size of sampling in z-direction
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives. Might have
            different shape input volume depending on handling of borders.
    
    Preferred is implementation by shift - See d1z_central_shift for reference.
    """
    return d1z_central_shift(volume, step_size)


# %% central differences by shifts (second order accuracy)
    
# TODO: allow padding volume or circular shift so that it can have same
# dimension as input volume

def d1x_central_shift(volume, step_size=1):
    """
    calculates 1st spatial derivative along x (the last axis) of a 3d-tensor 
    using 3-pt central difference scheme implemented by subtracting two 
    shifted arrays.
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated.
        step_size (float): step size of sampling in x-direction (pixel size) 
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives.  Output
        shape does not match input shape.
    
    Output shape: one row is cut off at the front and at the back.
    First derivative value along x corresponds to second x-value in input 
    volume.
    
    Assumes equidistant sampling.
    """
    return (volume[:, :, 2:] - volume[:, :, 0:-2]) / (2.*step_size)

def d1y_central_shift(volume, step_size=1):
    """
    calculates 1st spatial derivative along y (the second axis) of a 3d-tensor 
    using 3-pt central difference scheme implemented by subtracting 
    two shifted arrays.
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated
        step_size (float): step size of sampling in y-direction (pixel size) 
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives. Output
        shape does not match input shape.
    
    Output shape: one row is cut off at the front and at the back.
    First derivative value along y corresponds to second y-value in input 
    volume.
    
    Assumes equidistant sampling 
    """
    return (volume[:, 2:] - volume[:, 0:-2]) / (2.*step_size)

def d1z_central_shift(volume, step_size=1):
    """
    calculates 1st spatial derivative along z (the first axis) of a 3d-tensor 
    using 3-pt central difference scheme implemented by subtracting two 
    shifted arrays.
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated
        step_size (float): step size of sampling in z-direction
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives. Output
        shape does not match input shape.    
    
    Output shape: one row is cut off at the front and at the back.
    First derivative value along z corresponds to second z-value in input 
    volume.
    
    Assumes equidistant sampling.
    """
    return (volume[2:] - volume[0:-2]) / (2.*step_size)


# %% central differences by convs (second order accuracy)

# TODO: allow padding volume or circular shift so that it can have same
# dimension as input volume.  This should be easier here than for _shifts

# Notes:
# a) conv3d expects 5d-input
# vol: [batch, in_depth, in_height, in_width, in_channels]
# ker: [ker_depth, ker_height, ker_width, in_channels, out_channels]
# b) flipped ker, because what tf calls convolution is actually correlation

def d1x_central_conv(volume, step_size=1):
    """
    calculates 1st spatial derivative along x (the last axis) of a 3d-tensor 
    using 3-pt central difference scheme implemented by convolution with a 
    suitable kernel.
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated.
        step_size (float): step size of sampling in x-direction (pixel size) 
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives.  Output
        shape does not match input shape.
    
    Output shape: one row is cut off at the front and at the back.
    First derivative value along x corresponds to second x-value in input 
    volume. (Convolution just for "valid" values)
    
    Assumes equidistant sampling.
    """
    vol5d = tf.expand_dims(tf.expand_dims(volume, 0), -1)
    ker5d = tf.constant([[[[[-0.5]], [[0]], [[0.5]]]]])/step_size
    strides = [1,1,1,1,1]
    padding = "VALID"
    res5d = tf.nn.conv3d(vol5d, ker5d, strides, padding)
    return tf.squeeze(tf.squeeze(res5d, 0), -1)

def d1y_central_conv(volume, step_size=1):
    """
    calculates 1st spatial derivative along y (the second axis) of a 3d-tensor 
    using 3-pt central difference scheme implemented by convolution with a 
    suitable kernel.
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated
        step_size (float): step size of sampling in y-direction (pixel size) 
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives. Output
        shape does not match input shape.
    
    Output shape: one row is cut off at the front and at the back.
    First derivative value along y corresponds to second y-value in input 
    volume. (Convolution just for "valid" values)
    
    Assumes equidistant sampling 
    """
    vol5d = tf.expand_dims(tf.expand_dims(volume, 0), -1)
    ker5d = tf.constant([[[[[-0.5]]], [[[ 0]]], [[[0.5]]]]])/step_size
    strides = [1,1,1,1,1]
    padding = "VALID"
    res5d = tf.nn.conv3d(vol5d, ker5d, strides, padding)
    return tf.squeeze(tf.squeeze(res5d, 0), -1)

def d1z_central_conv(volume, step_size=1):
    """
    calculates 1st spatial derivative along z (the first axis) of a 3d-tensor 
    using 3-pt central difference scheme implemented by convolution with a 
    suitable kernel.
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated
        step_size (float): step size of sampling in z-direction
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives. Output
        shape does not match input shape.    
    
    Output shape: one row is cut off at the front and at the back.
    First derivative value along z corresponds to second z-value in input 
    volume. (Convolution just for "valid" values)
    
    Assumes equidistant sampling.
    """
    vol5d = tf.expand_dims(tf.expand_dims(volume, 0), -1)
    ker5d = tf.constant([[[[[ -0.5]]]], [[[[ 0]]]], [[[[ 0.5]]]]])/step_size
    strides = [1,1,1,1,1]
    padding = "VALID"
    res5d = tf.nn.conv3d(vol5d, ker5d, strides, padding)
    return tf.squeeze(tf.squeeze(res5d, 0), -1)


# %% Cutting off more than necessary to make output same size (only shifts)
# Only implemented for shifts, since there is no advantage for convs
# and you can just slice later.

def _d1x_central_shift_sliced(im, step_size=1):
    """
    same as central_shift, but slices off one row in each dimension, so
    that derivatives in all directions have the same size.
    """
    return (im[1:-1, 1:-1, 2:] - im[1:-1, 1:-1, 0:-2]) / (2*step_size)

def _d1y_central_shift_sliced(im, step_size=1):
    """
    same as central_shift, but slices off one row in each dimension, so
    that derivatives in all directions have the same size.
    """
    return (im[1:-1, 2:, 1:-1] - im[1:-1, 0:-2, 1:-1]) / (2*step_size)
  
def _d1z_central_shift_sliced(im, step_size=1):
    """
    same as central_shift, but slices off one row in each dimension, so
    that derivatives in all directions have the same size.
    """
    return (im[2:, 1:-1, 1:-1] - im[0:-2, 1:-1, 1:-1]) / (2*step_size)


# %% Fwd differences wrapper (First order accuracy).
# f_x+1 - f_x / step_size
    
def d1x_fwd(volume, step_size=1):
    """
    calculates 1st spatial derivative along x (the last axis) of a 3d-tensor 
    using 2-pt fwd difference scheme implemented by a predefined method.
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated.
        step_size (float): step size of sampling in x-direction (pixel size).
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives.  Might have
            different shape input volume depending on handling of borders.

    2-pt fwd difference scheme is 
    f_i+1 - f_i / step_size
    and has first order accuracy.

    Preferred is implementation by shift - See d1x_fwd_shift for reference.
    """
    return d1x_fwd_conv(volume, step_size)

def d1y_fwd(volume, step_size=1):
    """
    calculates 1st spatial derivative along y (the second axis) of a 3d-tensor 
    using 2-pt fwd difference scheme implemented by a predefined method.
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated
        step_size (float): step size of sampling in y-direction (pixel size) 
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives. Might have
            different shape input volume depending on handling of borders.
    
    2-pt fwd difference scheme is 
    f_i+1 - f_i / step_size
    and has first order accuracy.
    
    Preferred is implementation by shift - See d1y_fwd_shift for reference.
    """
    return d1y_fwd_conv(volume, step_size)

def d1z_fwd(volume, step_size=1):
    """
    calculates 1st spatial derivative along z (the first axis) of a 3d-tensor 
    using 2-pt fwd difference scheme implemented by a predefined method.
    
    f_z+1 - f_z / step_size
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated
        step_size (float): step size of sampling in z-direction
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives. Might have
            different shape input volume depending on handling of borders.
    
    2-pt fwd difference scheme is 
    f_i+1 - f_i / step_size
    and has first order accuracy.
    
    Preferred is implementation by shift - See d1z_fwd_shift for reference.
    """
    return d1z_fwd_conv(volume, step_size)


# %% Fwd differences by shifts (First order accuracy). 

def d1x_fwd_shift(volume, step_size=1):
    """
    calculates 1st spatial derivative along x (the last axis) of a 3d-tensor 
    using 2-pt fwd difference scheme implemented by subtracting two 
    shifted arrays.
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated.
        step_size (float): step size of sampling in x-direction (pixel size).
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives. Output
        shape does not match input shape.    
    
    2-pt fwd difference scheme is 
    f_i+1 - f_i / step_size
    and has first order accuracy.

    Output shape: one row is cut off at the front and at the back.
    First derivative value along x corresponds to second x-value in input 
    volume. Last derivative value corresponds to second last x-value.  
    
    Note that cutting off first row is not necessary for fwd-differences. It
    has been done, however, so that:
        a) there is a difference in the result of the fwd scheme and the bwd \
            scheme.
        b) output dimension matches those of the central scheme.
    Modify the function if you wish to include first row.
    
    Assumes equidistant sampling.
    """
    # using all available values (output size different from docs) would be
    # return (volume[:, :, 1:] - volume[:, :, 0:-1]) / step_size
    # see also: _d1z_fwd_full
    return (volume[:, :, 2:] - volume[:, :, 1:-1]) / step_size

def d1y_fwd_shift(volume, step_size=1):
    """
    calculates 1st spatial derivative along y (the second axis) of a 3d-tensor 
    using 2-pt fwd difference scheme implemented by subtracting two 
    shifted arrays.
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated
        step_size (float): step size of sampling in y-direction (pixel size)
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives. Output
        shape does not match input shape.
        
    2-pt fwd difference scheme is 
    f_i+1 - f_i / step_size
    and has first order accuracy.    
    
    Output shape: one row is cut off at the front and at the back.
    First derivative value along y corresponds to second y-value in input 
    volume. Last derivative value corresponds to second last y-value.  
    
    Note that cutting off first row is not necessary for fwd-differences. It
    has been done, however, so that:
        a) there is a difference in the result of the fwd scheme and the bwd \
            scheme.
        b) output dimension matches those of the central scheme.
    Modify the function if you wish to include first row.
    
    Assumes equidistant sampling.
    """
    # using all available values (output size different from docs) would be
    # return (volume[:, 1:] - volume[:, 0:-1]) / step_size
    # see also: _d1y_fwd_full
    return (volume[:, 2:] - volume[:, 1:-1]) / step_size

def d1z_fwd_shift(volume, step_size=1):
    """
    calculates 1st spatial derivative along z (the first axis) of a 3d-tensor 
    using 2-pt fwd difference scheme implemented by subtracting two 
    shifted arrays.
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated
        step_size (float): step size of sampling in z-direction
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives. Output
        shape does not match input shape.    

    2-pt fwd difference scheme is 
    f_i+1 - f_i / step_size
    and has first order accuracy.
    
    Output shape: one row is cut off at the front and at the back.
    First derivative value along z corresponds to second z-value in input 
    volume. Last derivative value corresponds to second last z-value.  
    
    Note that cutting off first row is not necessary for fwd-differences. It
    has been done, however, so that:
        a) there is a difference in the result of the fwd scheme and the bwd \
            scheme.
        b) output dimension matches those of the central scheme.
    Modify the function if you wish to include first row.
    
    Assumes equidistant sampling.
    """
    # using all available values (output size different from docs) would be
    # return (volume[1:] - volume[0:-1]) / step_size
    # see also: _d1x_fwd_full
    return (volume[2:] - volume[1:-1]) / step_size


# %% Fwd differences (First order accuracy) by convs.

def d1x_fwd_conv(volume, step_size=1):
    """
    calculates 1st spatial derivative along x (the last axis) of a 3d-tensor 
    using 2-pt fwd difference scheme implemented by convolving with a 
    suitable kernel.
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated.
        step_size (float): step size of sampling in x-direction (pixel size).
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives. Output
        shape does not match input shape.    
    
    2-pt fwd difference scheme is 
    f_i+1 - f_i / step_size
    and has first order accuracy.    
    
    Output shape: one row is cut off at the front and at the back.
    First derivative value along x corresponds to second x-value in input 
    volume. Last derivative value corresponds to second last x-value.  
    
    Note that cutting off first row is not necessary for fwd-differences. It
    has been done, however, so that:
        a) there is a difference in the result of the fwd scheme and the bwd \
            scheme.
        b) output dimension matches those of the central scheme.
    Modify the function if you wish to include first row.
    
    Assumes equidistant sampling.
    """
#    vol5d = tf.expand_dims(tf.expand_dims(volume, 0), -1)
#    ker5d = tf.constant([[[[[-1.]], [[1.]]]]])/step_size
#    strides = [1,1,1,1,1]
#    padding = "VALID"
#    res5d = tf.nn.conv3d(vol5d, ker5d, strides, padding)
#    return tf.squeeze(tf.squeeze(res5d, 0), -1)[:,:,1:]
    return _d1x_fwd_conv_full(volume, step_size)[:,:,1:]

def d1y_fwd_conv(volume, step_size=1):
    """
    calculates 1st spatial derivative along y (the second axis) of a 3d-tensor 
    using 2-pt fwd difference scheme implemented by convolving with a 
    suitable kernel.
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated
        step_size (float): step size of sampling in y-direction (pixel size)
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives. Output
        shape does not match input shape.    

    2-pt fwd difference scheme is 
    f_i+1 - f_i / step_size
    and has first order accuracy.
    
    Output shape: one row is cut off at the front and at the back.
    First derivative value along y corresponds to second y-value in input 
    volume. Last derivative value corresponds to second last y-value.  
    
    Note that cutting off first row is not necessary for fwd-differences. It
    has been done, however, so that:
        a) there is a difference in the result of the fwd scheme and the bwd \
            scheme.
        b) output dimension matches those of the central scheme.
    Modify the function if you wish to include first row.
    
    Assumes equidistant sampling.
    """
#    vol5d = tf.expand_dims(tf.expand_dims(volume, 0), -1)
#    ker5d = tf.constant([[[[[-1.]]], [[[1.]]]]])/step_size
#    strides = [1,1,1,1,1]
#    padding = "VALID"
#    res5d = tf.nn.conv3d(vol5d, ker5d, strides, padding)
#    return tf.squeeze(tf.squeeze(res5d, 0), -1)[:,1:]
    return _d1y_fwd_conv_full(volume, step_size)[:,1:]

def d1z_fwd_conv(volume, step_size=1):
    """
    calculates 1st spatial derivative along z (the first axis) of a 3d-tensor 
    using 2-pt fwd difference scheme implemented by convolving with a 
    suitable kernel.
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated
        step_size (float): step size of sampling in z-direction
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives. Output
        shape does not match input shape.    

    2-pt fwd difference scheme is 
    f_i+1 - f_i / step_size
    and has first order accuracy.
    
    Output shape: one row is cut off at the front and at the back.
    First derivative value along z corresponds to second z-value in input 
    volume. Last derivative value corresponds to second last z-value.  
    
    Note that cutting off first row is not necessary for fwd-differences. It
    has been done, however, so that:
        a) there is a difference in the result of the fwd scheme and the bwd \
            scheme.
        b) output dimension matches those of the central scheme.
    Modify the function if you wish to include first row.
    
    Assumes equidistant sampling.
    """
#    vol5d = tf.expand_dims(tf.expand_dims(volume, 0), -1)
#    ker5d = tf.constant([[[[[-1.]]]], [[[[1.]]]]])/step_size
#    strides = [1,1,1,1,1]
#    padding = "VALID"
#    res5d = tf.nn.conv3d(vol5d, ker5d, strides, padding)
#    return tf.squeeze(tf.squeeze(res5d, 0), -1)[1:]
    return _d1z_fwd_conv_full(volume, step_size)[1:]


# %% Bwd differences wrapper (First order accuracy).
    
# TODO: fwd and bwd could also be distinguished if there was a circshift version
    
def d1x_bwd(volume, step_size=1):
    """
    calculates 1st spatial derivative along x (the last axis) of a 3d-tensor 
    using 2-pt bwd difference scheme implemented by a predefined method.
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated.
        step_size (float): step size of sampling in x-direction (pixel size).
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives.  Might have
            different shape input volume depending on handling of borders.

    2-pt bwd difference scheme is 
    f_i - f_i-1 / step_size
    and has first order accuracy.
    
    Preferred is implementation by shift - See d1x_bwd_shift for reference.
    """
    return d1x_bwd_conv(volume, step_size)

def d1y_bwd(volume, step_size=1):
    """
    calculates 1st spatial derivative along y (the second axis) of a 3d-tensor 
    using 2-pt bwd difference scheme implemented by a predefined method.
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated
        step_size (float): step size of sampling in y-direction (pixel size) 
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives. Might have
            different shape input volume depending on handling of borders.

    2-pt bwd difference scheme is 
    f_i - f_i-1 / step_size
    and has first order accuracy.
    
    Preferred is implementation by shift - See d1y_bwd_shift for reference.
    """
    return d1y_bwd_conv(volume, step_size)

def d1z_bwd(volume, step_size=1):
    """
    calculates 1st spatial derivative along z (the first axis) of a 3d-tensor 
    using 2-pt bwd difference scheme implemented by a predefined method.
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated
        step_size (float): step size of sampling in z-direction
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives. Might have
            different shape input volume depending on handling of borders.

    2-pt bwd difference scheme is 
    f_i - f_i-1 / step_size
    and has first order accuracy.
    
    Preferred is implementation by shift - See d1z_bwd_shift for reference.
    """
    return d1z_bwd_conv(volume, step_size)




# %% Bwd differences (First order accuracy) by shifts
    
# TODO: consider not to cut off first row 
# -> will also avoid one slicing operation
# -> then also change below and in docs.
# -> user can take care of fwd / bwd himself.

def d1x_bwd_shift(volume, step_size=1):
    """
    calculates 1st spatial derivative along x (the last axis) of a 3d-tensor 
    using 2-pt bwd difference scheme implemented by subtracting two 
    shifted arrays.
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated.
        step_size (float): step size of sampling in x-direction (pixel size).
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives. Output
        shape does not match input shape.    
    
    2-pt bwd difference scheme is 
    f_i - f_i-1 / step_size
    and has first order accuracy.

    Output shape: one row is cut off at the front and at the back.
    First derivative value along x corresponds to second x-value in input 
    volume. Last derivative value corresponds to second last x-value.  
    
    Note that cutting off last row is not necessary for bwd-differences. It
    has been done, however, so that:
        a) there is a difference in the result of the fwd scheme and the bwd \
            scheme.
        b) output dimension matches those of the central scheme.
    Modify the function if you wish to include first row.
    
    Assumes equidistant sampling.
    """
    return (volume[:, :, 1:-1] - volume[:, :, 0:-2]) / step_size

def d1y_bwd_shift(volume, step_size=1):
    """
    calculates 1st spatial derivative along y (the second axis) of a 3d-tensor 
    using 2-pt bwd difference scheme implemented by subtracting two 
    shifted arrays.
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated
        step_size (float): step size of sampling in y-direction (pixel size)
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives. Output
        shape does not match input shape.
        
    2-pt bwd difference scheme is 
    f_i - f_i-1 / step_size
    and has first order accuracy.    
    
    Output shape: one row is cut off at the front and at the back.
    First derivative value along y corresponds to second y-value in input 
    volume. Last derivative value corresponds to second last y-value.  
    
    Note that cutting off last row is not necessary for bwd-differences. It
    has been done, however, so that:
        a) there is a difference in the result of the fwd scheme and the bwd \
            scheme.
        b) output dimension matches those of the central scheme.
    Modify the function if you wish to include first row.
    
    Assumes equidistant sampling.
    """
    return (volume[:, 1:-1] - volume[:, 0:-2]) / step_size

def d1z_bwd_shift(volume, step_size=1):
    """
    calculates 1st spatial derivative along z (the first axis) of a 3d-tensor 
    using 2-pt bwd difference scheme implemented by subtracting two 
    shifted arrays.
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated
        step_size (float): step size of sampling in z-direction
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives. Output
        shape does not match input shape.    

    2-pt bwd difference scheme is 
    f_i - f_i-1 / step_size
    and has first order accuracy.
    
    Output shape: one row is cut off at the front and at the back.
    First derivative value along z corresponds to second z-value in input 
    volume. Last derivative value corresponds to second last z-value.  
    
    Note that cutting off last row is not necessary for bwd-differences. It
    has been done, however, so that:
        a) there is a difference in the result of the fwd scheme and the bwd \
            scheme.
        b) output dimension matches those of the central scheme.
    Modify the function if you wish to include first row.
    
    Assumes equidistant sampling.
    """
    return (volume[1:-1] - volume[0:-2]) / step_size


# %% bwd differences (first order accuracy) by convs

def d1x_bwd_conv(volume, step_size=1):
    """
    calculates 1st spatial derivative along x (the last axis) of a 3d-tensor 
    using 2-pt bwd difference scheme implemented by convolving with a 
    suitable kernel.
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated.
        step_size (float): step size of sampling in x-direction (pixel size).
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives. Output
        shape does not match input shape.    
    
    2-pt bwd difference scheme is 
    f_i - f_i-1 / step_size
    and has first order accuracy.

    Output shape: one row is cut off at the front and at the back.
    First derivative value along x corresponds to second x-value in input 
    volume. Last derivative value corresponds to second last x-value.  
    
    Note that cutting off last row is not necessary for bwd-differences. It
    has been done, however, so that:
        a) there is a difference in the result of the fwd scheme and the bwd \
            scheme.
        b) output dimension matches those of the central scheme.
    Modify the function if you wish to include first row.
    
    Assumes equidistant sampling.
    """
#    vol5d = tf.expand_dims(tf.expand_dims(volume, 0), -1)
#    ker5d = tf.constant([[[[[-1.]], [[1.]]]]])/step_size
#    strides = [1,1,1,1,1]
#    padding = "VALID"
#    res5d = tf.nn.conv3d(vol5d, ker5d, strides, padding)
#    return tf.squeeze(tf.squeeze(res5d, 0), -1)[:, :, 0:-1]
    # fwd and bwd are the same for full, just shifted
    return _d1x_fwd_conv_full(volume, step_size)[:, :, 0:-1]

def d1y_bwd_conv(volume, step_size=1):
    """
    calculates 1st spatial derivative along y (the second axis) of a 3d-tensor 
    using 2-pt bwd difference scheme implemented by convolving with a 
    suitable kernel.
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated
        step_size (float): step size of sampling in y-direction (pixel size)
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives. Output
        shape does not match input shape.
        
    2-pt bwd difference scheme is 
    f_i - f_i-1 / step_size
    and has first order accuracy.    
    
    Output shape: one row is cut off at the front and at the back.
    First derivative value along y corresponds to second y-value in input 
    volume. Last derivative value corresponds to second last y-value.  
    
    Note that cutting off last row is not necessary for bwd-differences. It
    has been done, however, so that:
        a) there is a difference in the result of the fwd scheme and the bwd \
            scheme.
        b) output dimension matches those of the central scheme.
    Modify the function if you wish to include first row.
    
    Assumes equidistant sampling.
    """
#    vol5d = tf.expand_dims(tf.expand_dims(volume, 0), -1)
#    ker5d = tf.constant([[[[[-1.]]], [[[1.]]]]])/step_size
#    strides = [1,1,1,1,1]
#    padding = "VALID"
#    res5d = tf.nn.conv3d(vol5d, ker5d, strides, padding)
#    return tf.squeeze(tf.squeeze(res5d, 0), -1)[:, 0:-1]
    # fwd and bwd are the same for full, just shifted
    return _d1y_fwd_conv_full(volume, step_size)[:, 0:-1]

def d1z_bwd_conv(volume, step_size=1):
    """
    calculates 1st spatial derivative along z (the first axis) of a 3d-tensor 
    using 2-pt bwd difference scheme implemented by convolving with a 
    suitable kernel.
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated
        step_size (float): step size of sampling in z-direction
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives. Output
        shape does not match input shape.    

    2-pt bwd difference scheme is 
    f_i - f_i-1 / step_size
    and has first order accuracy.
    
    Output shape: one row is cut off at the front and at the back.
    First derivative value along z corresponds to second z-value in input 
    volume. Last derivative value corresponds to second last z-value.  
    
    Note that cutting off last row is not necessary for bwd-differences. It
    has been done, however, so that:
        a) there is a difference in the result of the fwd scheme and the bwd \
            scheme.
        b) output dimension matches those of the central scheme.
    Modify the function if you wish to include first row.
    
    Assumes equidistant sampling.
    """
#    vol5d = tf.expand_dims(tf.expand_dims(volume, 0), -1)
#    ker5d = tf.constant([[[[[-1.]]]], [[[[1.]]]]])/step_size
#    strides = [1,1,1,1,1]
#    padding = "VALID"
#    res5d = tf.nn.conv3d(vol5d, ker5d, strides, padding)
#    return tf.squeeze(tf.squeeze(res5d, 0), -1)[0:-1]
    # fwd and bwd are the same for full, just shifted
    return _d1z_fwd_conv_full(volume, step_size)[0:-1]


# %% without slicing off row that does not need to be cut off.  
# fwd / bwd are same

def _d1x_fwd_full(volume, step_size=1):
    """    
    Unlike the standard version, this does not cut off first row.
    Fwd/Bwd give the same result, but association with values is shifted 
    by one row.
    -> user can take care of fwd / bwd himself.

    Calculates 1st spatial derivative along x (the last axis) of a 3d-tensor 
    using 2-pt fwd difference scheme implemented by subtracting two 
    shifted arrays.
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated
        step_size (float): step size of sampling in z-direction
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives. Output
        shape does not match input shape.    

    2-pt fwd difference scheme is 
    f_i+1 - f_i / step_size
    and has first order accuracy.
    """
    return _d1x_fwd_conv_full(volume, step_size)

def _d1y_fwd_full(volume, step_size=1):
    """    
    Unlike the standard version, this does not cut off first row.
    Fwd/Bwd give the same result, but association with values is shifted 
    by one row.
    -> user can take care of fwd / bwd himself.

    Calculates 1st spatial derivative along y (the second axis) of a 3d-tensor 
    using 2-pt fwd difference scheme implemented by preselected method
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated
        step_size (float): step size of sampling in z-direction
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives. Output
        shape does not match input shape.    

    2-pt fwd difference scheme is 
    f_i+1 - f_i / step_size
    and has first order accuracy.
    """
    return _d1y_fwd_conv_full(volume, step_size)

def _d1z_fwd_full(volume, step_size=1):
    """    
    Unlike the standard version, this does not cut off first row.
    Fwd/Bwd give the same result, but association with values is shifted 
    by one row.
    -> user can take care of fwd / bwd himself.

    Calculates 1st spatial derivative along z (the first axis) of a 3d-tensor 
    using 2-pt fwd difference scheme implemented by preselected method
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated
        step_size (float): step size of sampling in z-direction
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives. Output
        shape does not match input shape.    

    2-pt fwd difference scheme is 
    f_i+1 - f_i / step_size
    and has first order accuracy.
    """
    return _d1z_fwd_conv_full(volume, step_size)


def _d1x_fwd_shift_full(volume, step_size=1):
    """    
    Unlike the standard version, this does not cut off first row.
    Fwd/Bwd give the same result, but association with values is shifted 
    by one row.
    -> user can take care of fwd / bwd himself.

    Calculates 1st spatial derivative along x (the last axis) of a 3d-tensor 
    using 2-pt fwd difference scheme implemented by by preselected method
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated
        step_size (float): step size of sampling in z-direction
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives. Output
        shape does not match input shape.    

    2-pt fwd difference scheme is 
    f_i+1 - f_i / step_size
    and has first order accuracy.
    """
    return (volume[:, :, 1:] - volume[:, :, 0:-1]) / step_size

def _d1y_fwd_shift_full(volume, step_size=1):
    """    
    Unlike the standard version, this does not cut off first row.
    Fwd/Bwd give the same result, but association with values is shifted 
    by one row.
    -> user can take care of fwd / bwd himself.

    Calculates 1st spatial derivative along y (the second axis) of a 3d-tensor 
    using 2-pt fwd difference scheme implemented by subtracting two 
    shifted arrays.
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated
        step_size (float): step size of sampling in z-direction
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives. Output
        shape does not match input shape.    

    2-pt fwd difference scheme is 
    f_i+1 - f_i / step_size
    and has first order accuracy.
    """
    return (volume[:, 1:] - volume[:, 0:-1]) / step_size

def _d1z_fwd_shift_full(volume, step_size=1):
    """    
    Unlike the standard version, this does not cut off first row.
    Fwd/Bwd give the same result, but association with values is shifted 
    by one row.
    -> user can take care of fwd / bwd himself.

    Calculates 1st spatial derivative along z (the first axis) of a 3d-tensor 
    using 2-pt fwd difference scheme implemented by subtracting two 
    shifted arrays.
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated
        step_size (float): step size of sampling in z-direction
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives. Output
        shape does not match input shape.    

    2-pt fwd difference scheme is 
    f_i+1 - f_i / step_size
    and has first order accuracy.
    """
    return (volume[1:] - volume[0:-1]) / step_size


def _d1x_fwd_conv_full(volume, step_size=1):
    """    
    Unlike the standard version, this does not cut off first row.
    Fwd/Bwd give the same result, but association with values is shifted 
    by one row.
    -> user can take care of fwd / bwd himself.

    Calculates 1st spatial derivative along x (the last axis) of a 3d-tensor 
    using 2-pt fwd difference scheme implemented by convolution with a 
    suitable kernel.
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated
        step_size (float): step size of sampling in z-direction
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives. Output
        shape does not match input shape.    

    2-pt fwd difference scheme is 
    f_i+1 - f_i / step_size
    and has first order accuracy.
    """
    vol5d = tf.expand_dims(tf.expand_dims(volume, 0), -1)
    ker5d = tf.constant([[[[[-1.]], [[1.]]]]])/step_size
    strides = [1,1,1,1,1]
    padding = "VALID"
    res5d = tf.nn.conv3d(vol5d, ker5d, strides, padding)
    return tf.squeeze(tf.squeeze(res5d, 0), -1)

def _d1y_fwd_conv_full(volume, step_size=1):
    """    
    Unlike the standard version, this does not cut off first row.
    Fwd/Bwd give the same result, but association with values is shifted 
    by one row.
    -> user can take care of fwd / bwd himself.

    Calculates 1st spatial derivative along y (the second axis) of a 3d-tensor 
    using 2-pt fwd difference scheme implemented by convolution with a 
    suitable kernel.
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated
        step_size (float): step size of sampling in z-direction
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives. Output
        shape does not match input shape.    

    2-pt fwd difference scheme is 
    f_i+1 - f_i / step_size
    and has first order accuracy.
    """
    vol5d = tf.expand_dims(tf.expand_dims(volume, 0), -1)
    ker5d = tf.constant([[[[[-1.]]], [[[1.]]]]])/step_size
    strides = [1,1,1,1,1]
    padding = "VALID"
    res5d = tf.nn.conv3d(vol5d, ker5d, strides, padding)
    return tf.squeeze(tf.squeeze(res5d, 0), -1)

def _d1z_fwd_conv_full(volume, step_size=1):
    """    
    Unlike the standard version, this does not cut off first row.
    Fwd/Bwd give the same result, but association with values is shifted 
    by one row.
    -> user can take care of fwd / bwd himself.

    Calculates 1st spatial derivative along z (the first axis) of a 3d-tensor 
    using 2-pt fwd difference scheme implemented by convolution with a 
    suitable kernel.
    
    Input:
        volume (tf-tensor, 3d): volume from which derivative is calculated
        step_size (float): step size of sampling in z-direction
    
    Ouput:
        out_volume (tf-tensor, 3d): volume containing derivatives. Output
        shape does not match input shape.    

    2-pt fwd difference scheme is 
    f_i+1 - f_i / step_size
    and has first order accuracy.
    """
    vol5d = tf.expand_dims(tf.expand_dims(volume, 0), -1)
    ker5d = tf.constant([[[[[-1.]]]], [[[[1.]]]]])/step_size
    strides = [1,1,1,1,1]
    padding = "VALID"
    res5d = tf.nn.conv3d(vol5d, ker5d, strides, padding)
    return tf.squeeze(tf.squeeze(res5d, 0), -1)


# %% tests

def _test_linear_ramp(d1x, d1y, d1z):
    """
    generates 3d-ramp and calculates gradient volumes in 3 directions 
    to check if the derivative is the slope
    
    input:  methods to calculate derivative
    """
    import numpy as np
    # import ../tolbox.object_creation as objs
    spec = importlib.util.spec_from_file_location("objs", 
                                                "../toolbox/create_objects.py")
    objs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(objs)
    
    # ramp volume
    shape = (3, 4, 5) # z,y,x
    slopes = (1, 2, 3) # z,y,x
    offset = 5
    
    volume = tf.constant(np.float32(
            objs.linear_ramp(shape, slopes, offset)))

    # calculate gradients and some of its attributes
    gradient_vols = [d1z(volume), d1y(volume), d1x(volume)]
    with tf.Session() as sess:
        gradient_vols = sess.run(gradient_vols)
    out_shapes = [vol.shape for vol in gradient_vols]
    gradients_min = [vol.min() for vol in gradient_vols]
    gradients_max = [vol.max() for vol in gradient_vols]    

    # check shapes
    expected_shapes = [(shape[0]-2, shape[1], shape[2]),  #d1z
                       (shape[0], shape[1]-2, shape[2]),  #d1y
                       (shape[0], shape[1], shape[2]-2)]  #d1x
    shape_match = (out_shapes == expected_shapes)
    # print("Shapes are as expected".)
    if not shape_match:
        print("Output shapes are not as expected from the docs.")
        print("Output shapes are:", out_shapes)
        print("Expected shapes are:", expected_shapes)
    
    # check if gradient min and max are the same and if they match slope
    if gradients_min == gradients_max:
        #print("gradient min and max are the same along each direction")
        print("slopes: ", slopes, " gradients: ", tuple(gradients_min))
        slope_match = (slopes == tuple(gradients_min))
        if not slope_match:
            print("gradients are not the same as slope")
    else:
        print("gradient max and min are not the same along some direction," +
              "although slope should be uniform")
        print("min gradients: ", tuple(gradients_min))
        print("max gradients: ", tuple(gradients_max))
        slope_match = False
        
    return shape_match and slope_match


def _test_square_ramp(d1x, d1y, d1z, scheme):
    """
    generates 3d-ramp and calculates gradient volumes in 3 directions 
    to check if the derivative is the slope
    
    input:  methods to calculate derivative
    scheme (string) : "central", "fwd" or "bwd"
    """
    import numpy as np
    # import ../tolbox.object_creation as objs
    spec = importlib.util.spec_from_file_location("objs", 
                                                "../toolbox/create_objects.py")
    objs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(objs)
    
    # ramp volume
    shape = (3, 4, 5) # z,y,x
    slopes_sqr = (1, 2, 3) # z,y,x
    slopes_lin = (0, 0, 0)
    offset = 5
    
    volume = tf.constant(np.float32(
            objs.square_ramp(shape, slopes_sqr, slopes_lin, offset)))

    # calculate gradients and some of its attributes
    gradient_vols = [d1z(volume), d1y(volume), d1x(volume)]
    with tf.Session() as sess:
        gradient_vols = sess.run(gradient_vols)
    out_shapes = [vol.shape for vol in gradient_vols]  

    # check shapes
    expected_shapes = [(shape[0]-2, shape[1], shape[2]),  #d1z
                       (shape[0], shape[1]-2, shape[2]),  #d1y
                       (shape[0], shape[1], shape[2]-2)]  #d1x
    shape_match = (out_shapes == expected_shapes)
    # print("Shapes are as expected".)
    if not shape_match:
        print("Output shapes are not as expected from the docs.")
        print("Output shapes are:", out_shapes)
        print("Expected shapes are:", expected_shapes)
    
    # check if gradients are the same along directions orthogonal to derivative
    d1z_num_min = np.min(gradient_vols[-3], (-2,-1))
    d1y_num_min = np.min(gradient_vols[-2], (-3,-1))
    d1x_num_min = np.min(gradient_vols[-1], (-2,-3))
    num_grads_min = [tuple(d1z_num_min), tuple(d1y_num_min), 
                     tuple(d1x_num_min)]
    
    d1z_num_max = np.max(gradient_vols[-3], (-2,-1))
    d1y_num_max = np.max(gradient_vols[-2], (-3,-1))
    d1x_num_max = np.max(gradient_vols[-1], (-2,-3))
    num_grads_max = [tuple(d1z_num_max), tuple(d1y_num_max), 
                     tuple(d1x_num_max)]
    #print(num_grads_min != num_grads_max)
    
    minmax_match = (np.all(d1z_num_min == d1z_num_max) and 
                    np.all(d1y_num_min == d1y_num_max) and
                    np.all(d1x_num_min == d1x_num_max))
    
    if minmax_match:
        # check if gradients match expected gradients
        num_grads = num_grads_min
        # 2*slopes_sqr * x + slopes_lin
        d1z_ana = 2*slopes_sqr[-3]*np.arange(shape[-3]) + slopes_lin[-3]
        d1y_ana = 2*slopes_sqr[-2]*np.arange(shape[-2]) + slopes_lin[-2]
        d1x_ana = 2*slopes_sqr[-1]*np.arange(shape[-1]) + slopes_lin[-1]
        # errors are expected for fwd and bwd.  These are constant
        d2z_ana = 2*slopes_sqr[-3]
        d2y_ana = 2*slopes_sqr[-2]
        d2x_ana = 2*slopes_sqr[-1]
        fwd_bwd_errorz = 0.5 * d2z_ana # assuming step size 1
        fwd_bwd_errory = 0.5 * d2y_ana
        fwd_bwd_errorx = 0.5 * d2x_ana
        
        if scheme == "central":
            # grads are correct (2nd order method), only cut boundary
            expected_d1z = d1z_ana[1:-1]
            expected_d1y = d1y_ana[1:-1]
            expected_d1x = d1x_ana[1:-1]
        elif scheme == "fwd":
            # error is (1/2*f'') for parabola 
            # could also shift bwd-grads and add error
            expected_d1z = d1z_ana[1:-1] + fwd_bwd_errorz
            expected_d1y = d1y_ana[1:-1] + fwd_bwd_errory
            expected_d1x = d1x_ana[1:-1] + fwd_bwd_errorx
        elif scheme == "bwd":
            # error is -(1/2*f'') for parabola 
            # could also shift fwd-grads and add error -> the same
            expected_d1z = d1z_ana[1:-1] - fwd_bwd_errorz
            expected_d1y = d1y_ana[1:-1] - fwd_bwd_errory
            expected_d1x = d1x_ana[1:-1] - fwd_bwd_errorx
        else:
            raise TypeError("Scheme can only be \"central\", \"fwd\" or " + 
                            "\"bwd\".")
        expected_grads = (tuple(expected_d1z), tuple(expected_d1y), 
                          tuple(expected_d1x))
        
        print("expected:  ", tuple(expected_grads))
        print("numerical: ", tuple(num_grads))
        gradient_match = (np.all(expected_d1z == d1z_num_min) and 
                          np.all(expected_d1y == d1y_num_min) and
                          np.all(expected_d1x == d1x_num_min))
#        gradient_match = (num_grads == expected_grads)

    else:        
        print("gradient max and min are not the same along some direction " +
              "orthogonal to derivative.")
        print("min gradient: ", num_grads_min)
        print("max gradient: ", num_grads_max)
        gradient_match = False        

    return shape_match and gradient_match


def main():
    # TODO: define "__all__" and take derivative methods from there  ...
    # TODO: change argument order for testing methods from x,y,z to z,y,x
    # change ordering above accordingly
    print("Testing derivative methods")
    print("Note that these will also fail if output shape is unexpected " + 
          "somewhere.")
    
    print("1. Does calculated slope match the given slope of a linear ramp?")
    print("\n" + "derivatives by preferred-scheme-wrappers")
    passed = _test_linear_ramp(d1x, d1y, d1z)
    
    print("\n" + "central derivatives by wrappers")
    passed = (_test_linear_ramp(d1x_central, d1y_central, d1z_central) 
              and passed)
    print("\n" + "central derivatives by shifts")
    passed = (_test_linear_ramp(d1x_central_shift, d1y_central_shift, 
                                     d1z_central_shift) 
              and passed)
    print("\n" + "central derivatives by convolution")
    passed = (_test_linear_ramp(d1x_central_conv, d1y_central_conv, 
                                d1z_central_conv) 
              and passed)
    
    print("\n" + "fwd derivatives by wrappers")
    passed = _test_linear_ramp(d1x_fwd, d1y_fwd, d1z_fwd) and passed
    print("\n" + "fwd derivatives by shifts")
    passed = (_test_linear_ramp(d1x_fwd_shift, d1y_fwd_shift, d1z_fwd_shift) 
              and passed)
    print("\n" + "fwd derivatives by convolution")
    passed = (_test_linear_ramp(d1x_fwd_conv, d1y_fwd_conv, d1z_fwd_conv) 
              and passed)
    
    print("\n" + "bwd derivatives by wrappers")
    passed = _test_linear_ramp(d1x_bwd, d1y_bwd, d1z_bwd) and passed
    print("\n" + "bwd derivatives by shifts")
    passed = (_test_linear_ramp(d1x_bwd_shift, d1y_bwd_shift, d1z_bwd_shift) 
              and passed)
    print("\n" + "bwd derivatives by convolution")
    passed = (_test_linear_ramp(d1x_bwd_conv, d1y_bwd_conv, d1z_bwd_conv) 
              and passed)
    
    # This will fail the output shape test, so it is excluded from "passed"
    print("\nThese will fail output shape test, so it is excluded from passed")
    print("central derivatives by shifts, modified output size")
    _test_linear_ramp(_d1x_central_shift_sliced, _d1y_central_shift_sliced, 
                      _d1z_central_shift_sliced)
    print("fwd derivatives by wrappers, modified output size")
    _test_linear_ramp(_d1x_fwd_full, _d1y_fwd_full, 
                      _d1z_fwd_full)
    print("fwd derivatives by shifts, modified output size")
    _test_linear_ramp(_d1x_fwd_shift_full, _d1y_fwd_shift_full, 
                      _d1z_fwd_shift_full)
    print("fwd derivatives by convs, modified output size")
    _test_linear_ramp(_d1x_fwd_conv_full, _d1y_fwd_conv_full, 
                      _d1z_fwd_conv_full)
    
    
    print("\n" + "2. Are Gradients of a square ramp as expected?")  
    print("Do fwd and bwd gradients work as expected?")
    
    print("\n" + "derivatives by preferred-scheme-wrappers")
    passed = _test_square_ramp(d1x, d1y, d1z, "central")
    
    print("\n" + "central derivatives by wrappers")
    passed = (_test_square_ramp(d1x_central, d1y_central, d1z_central, 
                                "central") 
              and passed)
    print("\n" + "central derivatives by shifts")
    passed = (_test_square_ramp(d1x_central_shift, d1y_central_shift, 
                                     d1z_central_shift, "central") 
              and passed)
    print("\n" + "central derivatives by convolution")
    passed = (_test_square_ramp(d1x_central_conv, d1y_central_conv, 
                                d1z_central_conv, "central") 
              and passed)
    
    print("\n" + "fwd derivatives by wrappers")
    passed = _test_square_ramp(d1x_fwd, d1y_fwd, d1z_fwd, "fwd") and passed
    print("\n" + "fwd derivatives by shifts")
    passed = (_test_square_ramp(d1x_fwd_shift, d1y_fwd_shift, d1z_fwd_shift,
                                "fwd") 
              and passed)
    print("\n" + "fwd derivatives by convolution")
    passed = (_test_square_ramp(d1x_fwd_conv, d1y_fwd_conv, d1z_fwd_conv,
                                "fwd") 
              and passed)
    
    print("\n" + "bwd derivatives by wrappers")
    passed = _test_square_ramp(d1x_bwd, d1y_bwd, d1z_bwd, "bwd") and passed
    print("\n" + "bwd derivatives by shifts")
    passed = (_test_square_ramp(d1x_bwd_shift, d1y_bwd_shift, d1z_bwd_shift, 
                                "bwd") 
              and passed)
    print("\n" + "bwd derivatives by convolution")
    passed = (_test_square_ramp(d1x_bwd_conv, d1y_bwd_conv, d1z_bwd_conv, 
                                "bwd") 
              and passed)

    # This will fail the output shape test, so it is excluded from "passed"
    print("\nThese will fail output shape test, so it is excluded from passed")
    print("central derivatives by shifts, modified output size")
    _test_square_ramp(_d1x_central_shift_sliced, _d1y_central_shift_sliced, 
                      _d1z_central_shift_sliced, "central")
    print("fwd derivatives by wrappers, modified output size")
    _test_square_ramp(_d1x_fwd_full, _d1y_fwd_full, 
                      _d1z_fwd_full, "fwd")
    print("fwd derivatives by shifts, modified output size")
    _test_square_ramp(_d1x_fwd_shift_full, _d1y_fwd_shift_full, 
                      _d1z_fwd_shift_full, "fwd")
    print("fwd derivatives by convs, modified output size")
    _test_square_ramp(_d1x_fwd_conv_full, _d1y_fwd_conv_full, 
                      _d1z_fwd_conv_full, "fwd")
    
    if passed:
        print(" --> Tests passed")
    else:
        print(" --> Tests not passed")
        
    # TODO: time them!

if __name__ == '__main__':
    main()
