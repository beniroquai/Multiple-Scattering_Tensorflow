#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 14:14:21 2017

@author: soenke
"""

import tensorflow as tf
import numpy as np
import src.tf_helper as tf_helper

# %% sparsity penalty

def Reg_L1(im):
    # ready for 3D!
    """
    aka sparsity penalty

    Args:
        im (tf-tensor, 2d): image

    Returns:
        l1_regularization (float, 1d):  sum of absolute differences of pixel values
    """
    # Can also be used in case some values become negative.
    # then increase values and add intensity penalty
    # in other case:  spread out values?
    return tf.reduce_mean(tf.abs(im))


# %% intesity penalty

def Reg_L2(im):
    # ready for 3D!
    """
    aka intensity penalty
    aka carrington

    Args:
        im (tf-tensor, 2d): image

    Returns:
        l2_regularization (float, 1d):  sum of square of pixel values
    """
    # Can also be used in case some values become negative.
    # then increase values and add intensity penalty
    # in other case:  spread out values?
    return tf.reduce_mean(im ** 2)


def Reg_TV_conv(im, eps=1e2, step_sizes=(1, 1, 1)):
    """
    Calculates isotropic tv penalty.
    penalty = sum ( sqrt(|grad(f)|^2+eps^2) )
    where eps serves to achieve differentiability at low gradient values.

    Arguments:
        im (tf-tensor, 3d, real): image
        eps (float): damping term for low values to avoid kink of abs fxn
        step_sizes = (3-tuple of floats): step sizes (i.e. pixel size)
                     in different directions.
                     (axis 0, 1 and 2 corresponding to z, y and x)
                     if pixel size is the same in all directions, this can
                     be kept as 1

    This implementations uses 3-pt central difference scheme for 1st
    derivative.  Gradients are calculated using convolution.

    Right now, the sum is taken over a subimage.  This can be interpreted as
    if the gradient at the image border (one-pixel-row) is just zero.

    For more info see Appendix B of Kamilov et al. - "Optical Tomographic
    Image Reconstruction Based on Beam Propagation and Sparse Regularization"
    DOI: 10.1109/TCI.2016.2519261

    "a penalty promoting joint-sparsity of the gradient components. By
     promoting signals with sparse gradients, TV minimization recovers images
     that are piecewise-smooth, which means that they consist of smooth
     regions separated by sharp edges"

    And for parameter eps see Ferreol Soulez et al. - "Blind deconvolution
    of 3D data in wide field fluorescence microscopy"

    "Parameter eps > 0 ensures differentiability of prior at 0. When
     eps is close to the quantization level, this function smooths out non-
     significant differences between adjacent pixels."
    -> TODO: what is quantization level ??
    """
    grad_z = d1z_central_conv(im, step_sizes[0])[:, 1:-1, 1:-1]
    grad_y = d1y_central_conv(im, step_sizes[1])[1:-1, :, 1:-1]
    grad_x = d1x_central_conv(im, step_sizes[2])[1:-1, 1:-1, :]

    # l2 norm of gradients
    return tf.reduce_mean(tf.sqrt(grad_z ** 2 + grad_y ** 2 + grad_x ** 2 + eps ** 2))

def Reg_GR(tfin, Eps1=1e-15,Eps2=1e-15):
    loss=0.0
    for d in range(tfin.shape.ndims):
        loss += tf.square(tf.roll(tfin,-1,d) - tf.roll(tfin,1,d))/tf.sqrt(tf.square(tfin)+Eps1)   # /4.0
    return tf.reduce_mean(tf.cast(tf.sqrt(loss+Eps2),'float64'))/2.0

def Reg_GS(tfin, Eps=1e-15):
    loss=0.0
    for d in range(0,tfin.shape.ndims):
        loss += tf.reduce_mean(tf.cast((tf.square(tfin - tf.roll(tfin,1,d))+Eps),'float64'))
    return loss



def Reg_NegSqr(toRegularize):
    mySqrt = tf.where( # Just affects the real part
                    tf.less(toRegularize , tf.zeros_like(toRegularize)),
                    tf_helper.tf_abssqr(toRegularize), tf.zeros_like(toRegularize))
    
    myReg = tf.reduce_mean(mySqrt)
    return myReg


def Reg_posiminity(im, minval=0):
    # Clip Values below zero => add to error function
    print('Regularizer: Penalize Values less then '+str(minval))
    reg = tf.reduce_mean(tf.square(tf.nn.relu(-im)))  # avdoid values smaller then zero
    return reg

def Reg_posimaxity(im, maxval=1):
    # Clip Values below zero => add to error function
    print('Regularizer: Penalize Values higher then '+str(maxval))
    reg = tf.reduce_mean(tf.square(tf.nn.relu(im - (maxval))))
    return reg

def Reg_gaussivity(im, sigma=0.1, minval=0.1):
    print('Regularizer: Try to like values around 0 and dn with gaussian distribution')
    # have guassian activation for 0 and dn - like the possible values more.
    print('Take care the regularizer is not right!!')
    reg = tf.nn.l2_loss((1 - tf.exp(-.5 * tf.square((im - 0) / sigma))))
    reg = reg + tf.nn.l2_loss(
        (1 - tf.exp(-.5 * tf.square((im - minval) / sigma))))
    return reg

def Reg_TV_sobel(toRegularize):
    with tf.name_scope("tv_reg_sobel"):
        sobel_one_direction = np.array([[[1, 2, 1],
                                         [2, 4, 2],
                                         [1, 2, 1]],

                                        [[0, 0, 0],
                                         [0, 0, 0],
                                         [0, 0, 0]],

                                        [[-1, -2, -1],
                                         [-2, -4, -2],
                                         [-1, -2, -1]]], dtype=np.float32)
        sobel = np.stack([sobel_one_direction,
                          sobel_one_direction.transpose((2, 0, 1)),
                          sobel_one_direction.transpose((1, 2, 0))],
                         axis=-1) #3x3x3x3

        sobel = tf.constant(sobel[..., np.newaxis, :]) #3x3x3x1x3
        tv_loss = tf.reduce_mean(tf.abs(tf.nn.conv3d(tf.sigmoid(toRegularize), sobel, (1,) * 5, "VALID"))) # tf.nn.conv3d filter shape: [filter_depth, filter_height, filter_width, in_channels, out_channels]
        return tv_loss

def Reg_TV(toRegularize, BetaVals = [1,1,1], epsR = 1, epsC=1e-10, is_circ = True):
    # used rainers version to realize the tv regularizer   
    #% The Regularisation modification with epsR was introduced, according to
    #% Ferreol Soulez et al. "Blind deconvolution of 3D data in wide field fluorescence microscopy
    #%
    #
    #function [myReg,myRegGrad]=RegularizeTV(toRegularize,BetaVals,epsR)
    #epsC=1e-10;

    
    if(is_circ):
        if(0): # TF>1.11
            aGradL_1 = (toRegularize - tf.roll(toRegularize, 1, 0))/BetaVals[0]
            aGradL_2 = (toRegularize - tf.roll(toRegularize, 1, 1))/BetaVals[1]
            aGradL_3 = (toRegularize - tf.roll(toRegularize, 1, 2))/BetaVals[2]
    
            aGradR_1 = (toRegularize - tf.roll(toRegularize, -1, 0))/BetaVals[0]
            aGradR_2 = (toRegularize - tf.roll(toRegularize, -1, 1))/BetaVals[1]
            aGradR_3 = (toRegularize - tf.roll(toRegularize, -1, 2))/BetaVals[2]
        else:
            aGradL_1 = (toRegularize - tf.manip.roll(toRegularize, 1, 0))/BetaVals[0]
            aGradL_2 = (toRegularize - tf.manip.roll(toRegularize, 1, 1))/BetaVals[1]
            aGradL_3 = (toRegularize - tf.manip.roll(toRegularize, 1, 2))/BetaVals[2]
    
            aGradR_1 = (toRegularize - tf.manip.roll(toRegularize, -1, 0))/BetaVals[0]
            aGradR_2 = (toRegularize - tf.manip.roll(toRegularize, -1, 1))/BetaVals[1]
            aGradR_3 = (toRegularize - tf.manip.roll(toRegularize, -1, 2))/BetaVals[2]
            
        print('We use circular shift for the TV regularizer')
    else:    
        toRegularize_sub = toRegularize[1:-2,1:-2,1:-2]
        aGradL_1 = (toRegularize_sub - toRegularize[2:-1,1:-2,1:-2])/BetaVals[0] # cyclic rotation
        aGradL_2 = (toRegularize_sub - toRegularize[1:-1-1,2:-1,1:-1-1])/BetaVals[1] # cyclic rotation
        aGradL_3 = (toRegularize_sub - toRegularize[1:-1-1,1:-1-1,2:-1])/BetaVals[2] # cyclic rotation
        
        aGradR_1 = (toRegularize_sub - toRegularize[0:-3,1:-2,1:-2])/BetaVals[0] # cyclic rotation
        aGradR_2 = (toRegularize_sub - toRegularize[1:-2,0:-3,1:-2])/BetaVals[1] # cyclic rotation
        aGradR_3 = (toRegularize_sub - toRegularize[1:-2,1:-2,0:-3])/BetaVals[2] # cyclic rotation
            
    mySqrtL = tf.sqrt(tf_helper.tf_abssqr(aGradL_1)+tf_helper.tf_abssqr(aGradL_2)+tf_helper.tf_abssqr(aGradL_3)+epsR)
    mySqrtR = tf.sqrt(tf_helper.tf_abssqr(aGradR_1)+tf_helper.tf_abssqr(aGradR_2)+tf_helper.tf_abssqr(aGradR_3)+epsR)
     
    
    
    mySqrt = mySqrtL + mySqrtR; 
    
    if(0):
        mySqrt = tf.where(
                    tf.less(mySqrt , epsC*tf.ones_like(mySqrt)),
                    epsC*tf.ones_like(mySqrt),
                    mySqrt) # To avoid divisions by zero
    else:               
        mySqrt = mySqrt
        
        
    myReg = tf.reduce_mean(mySqrt)

    return myReg