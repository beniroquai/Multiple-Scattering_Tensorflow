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
def Reg_Tichonov(tfin,regDataType=None):
    
    return tf.reduce_mean(tf.square(tfin)) # tf.math.scalar_mul(mylambda,


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


def Reg_GR(tfin, eps1=1e-15,eps2=1e-15):
    loss=0.0
    for d in range(tfin.shape.ndims):
        loss += tf.square(tf.manip.roll(tfin,-1,d) - tf.manip.roll(tfin,1,d))/tf.sqrt(tf.square(tfin)+eps1)   # /4.0
    return tf.reduce_mean(tf.cast(tf.sqrt(loss+eps2),tf.float32))/2.0

def Reg_GS(tfin, Eps=1e-15):
    loss=0.0
    for d in range(0,tfin.shape.ndims):
        loss += tf.reduce_mean(tf.cast((tf.square(tfin - tf.manip.roll(tfin,1,d))+Eps),'float64'))
    return tf.cast(loss, tf.float32)

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
        try: # TF>1.11
            aGradL_1 = (toRegularize - tf.roll(toRegularize, 1, 0))/BetaVals[0]
            aGradL_2 = (toRegularize - tf.roll(toRegularize, 1, 1))/BetaVals[1]
            aGradL_3 = (toRegularize - tf.roll(toRegularize, 1, 2))/BetaVals[2]
    
            aGradR_1 = (toRegularize - tf.roll(toRegularize, -1, 0))/BetaVals[0]
            aGradR_2 = (toRegularize - tf.roll(toRegularize, -1, 1))/BetaVals[1]
            aGradR_3 = (toRegularize - tf.roll(toRegularize, -1, 2))/BetaVals[2]
        except:
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
        mySqrt = mySqrt # tf.clip_by_value(mySqrt, 0, np.inf)    
        

        
    myReg = tf.reduce_mean(mySqrt)

    return mySqrt


def Reg_TV_RH(tfin, Eps=1e-15, doubleSided=False,regDataType=None):
    if regDataType is None:
        regDataType=tf.float32
    loss=0.0
    if doubleSided:
        for d in range(tfin.shape.ndims):   # sum over the dimensions to calculate the one norm
            loss += tf.square(tf.roll(tfin,-1,d) - tf.roll(tfin,1,d))
    else:
        for d in range(tfin.shape.ndims):   # sum over the dimensions to calculate the one norm
            loss += tf.square(tfin - tf.roll(tfin,1,d))
    if doubleSided:
        loss=loss/2.0
    return tf.reduce_mean(tf.cast(tf.sqrt(loss+Eps),regDataType)) # /2.0

    
def PreMonotonicPosNAN(tfin):
    '''
    Borrowed from Rainers: https://gitlab.com/bionanoimaging/inversemodelling/blob/master/InverseModelling/preforwardmodels.py
    applies a monotonic transform mapping the full real axis to the positive half space

    This can be used to implicitely force the reconstruction results to be all-positive. The monotinic function is defined piecewise:
        for all x>1: y=x
        else :  1/(2-tfin)
    The function is continues and once differentiable.
    This function can also be used as an activation function for neural networks.

    Parameters
    ----------
    tfin : tensorflow array
        The array to be transformed

    Returns
    -------
    tensorflow array
        The transformed array

    See also
    -------
    PreabsSqr 

    Example
    -------
    '''
    # line below: If sing(tfin) > 1 use tfin else: use 1/(2-tfin)
    monoPos=((tf.sign(tfin-1) + 1)/2)*(tfin)+((1-tf.sign(tfin-1))/2)*(1.0/(2-tfin))
    
    return monoPos;


# This monotonic positive function is based on a Hyperbola modified that one of the branches appraoches zero and the other one reaches a slope of one
def PreMonotonicPos(tfin,Eps=1e-2,b2=10.0):
    '''
    Borrowed from Rainers: https://gitlab.com/bionanoimaging/inversemodelling/blob/master/InverseModelling/preforwardmodels.py
    applies a monotonic transform mapping the full real axis to the positive half space

    This can be used to implicitely force the reconstruction results to be all-positive. The monotinic function is derived from a hyperboloid:

    The function is continues and differentiable.
    This function can also be used as an activation function for neural networks.

    Parameters
    ----------
    tfin : tensorflow array
        The array to be transformed

    Returns
    -------
    tensorflow array
        The transformed array

    See also
    -------
    PreabsSqr 

    Example
    -------
    '''
     # b2 a constant value > 0.0 0 which regulates the shape of the hyperbola. The bigger the smoother it becomes.
    return tf.sqrt(b2+tf.square(tfin)/4.0)+tfin/2.0

