#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 14:14:21 2017

@author: soenke
"""

import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    from derivatives import d1x, d1y, d1z, d1x_central_conv, d1y_central_conv, \
        d1z_central_conv, d1x_central_shift, \
        d1y_central_shift, d1z_central_shift
else:
    from .derivatives import d1x, d1y, d1z, d1x_central_conv, d1y_central_conv, \
        d1z_central_conv, d1x_central_shift, \
        d1y_central_shift, d1z_central_shift

from warnings import warn


def tf_abssqr(inputar):
    return tf.real(inputar*tf.conj(inputar))


# %% sparsity penalty

def l1_reg(im):
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
    return tf.reduce_sum(tf.abs(im))


# %% intesity penalty

def l2_reg(im):
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
    return tf.reduce_sum(im ** 2)


# %% Total Variation wrapper

# For more info see Appendix B of Kamilov et al. - "Optical Tomographic
#    Image Reconstruction Based on Beam Propagation and Sparse Regularization"
#    DOI: 10.1109/TCI.2016.2519261


# !!!
# Discuss 3pt stencil with Rainer -> could lead to high frequency noise,
# since changes such as [0,1,0,1,0,1] would give zero gradient.
# !!!


def total_variation(im, eps=1e2, step_sizes=(1, 1, 1)):
    """
    Convenience function.  Calculates isotropic tv penalty by method which is
    currentlly preferred.  3d only!

    Arguments:
        im (tf-tensor, 3d, real): image
        eps (float): damping term for low values to avoid kink of abs fxn
        step_sizes = (3-tuple of floats): step sizes (i.e. pixel size)
                     in different directions.
                     (axis 0, 1 and 2 corresponding to z, y and x)
                     if pixel size is the same in all directions, this can
                     be kept as 1

    penalty = sum ( sqrt(|grad(f)|^2+eps^2) )

    Wrapper for total_variation_iso.  See that function for more documentation.
    """
    return total_variation_iso_shift(im, eps, step_sizes)


# %% isotropic Total Variation formulations

# def total_variation_iso(im, eps=1e2, step_sizes=(1,1,1)):
#    """
#    Calculates isotropic tv penalty.
#    penalty = sum ( sqrt(|grad(f)|^2+eps^2) )
#    where eps serves to achieve differentiability at low gradient values.
#
#    Arguments:
#        im (tf-tensor, 3d, real): image
#        eps (float): damping term for low values to avoid kink of abs fxn
#        step_sizes = (3-tuple of floats): step sizes (i.e. pixel size)
#                     in different directions.
#                     (axis 0, 1 and 2 corresponding to z, y and x)
#                     if pixel size is the same in all directions, this can
#                     be kept as 1
#
#    This implementations uses 3-pt central difference scheme for 1st
#    derivative.  Gradients are calculated by derivative-wrappers.
#
#    Right now, the sum is taken over a subimage.  This can be interpreted as
#    if the gradient at the image border (one-pixel-row) is just zero.
#
#    For more info see Appendix B of Kamilov et al. - "Optical Tomographic
#    Image Reconstruction Based on Beam Propagation and Sparse Regularization"
#    DOI: 10.1109/TCI.2016.2519261
#
#    "a penalty promoting joint-sparsity of the gradient components. By
#     promoting signals with sparse gradients, TV minimization recovers images
#     that are piecewise-smooth, which means that they consist of smooth
#     regions separated by sharp edges"
#
#    And for parameter eps see Ferreol Soulez et al. - "Blind deconvolution
#    of 3D data in wide field fluorescence microscopy"
#
#    "Parameter eps > 0 ensures differentiability of prior at 0. When
#     eps is close to the quantization level, this function smooths out non-
#     significant differences between adjacent pixels."
#    -> TODO: what is quantization level ??
#    """
#    grad_z = d1z(im, step_sizes[0])[:, 1:-1, 1:-1]
#    grad_y = d1y(im, step_sizes[1])[1:-1, :, 1:-1]
#    grad_x = d1x(im, step_sizes[2])[1:-1, 1:-1, :]
#
#    # l2 norm of gradients
#    return tf.reduce_sum(tf.sqrt(grad_z**2 + grad_y**2 + grad_x**2 + eps**2))

def total_variation_iso_conv(im, eps=1e2, step_sizes=(1, 1, 1)):
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
    return tf.reduce_sum(tf.sqrt(grad_z ** 2 + grad_y ** 2 + grad_x ** 2 + eps ** 2))


def total_variation_iso_shift(im, eps=1e2, step_sizes=(1, 1, 1)):
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
    derivative.  Gradients are calculated using circshifts.

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
    # (f_(x+1) - f_(x-1))/2 * 1/ss
    # im shifted right - im shifted left
    # excludes 1 pixel at border all around

    # this saves one slicing operation compared to formulation below
    # These have been tested in derivatives-module as
    # _d1z_central_shift_sliced etc.
    grad_z = (im[2:, 1:-1, 1:-1] - im[0:-2, 1:-1, 1:-1]) / (2 * step_sizes[0])
    grad_y = (im[1:-1, 2:, 1:-1] - im[1:-1, 0:-2, 1:-1]) / (2 * step_sizes[1])
    grad_x = (im[1:-1, 1:-1, 2:] - im[1:-1, 1:-1, 0:-2]) / (2 * step_sizes[2])
    # grad_z = d1z_shift(im, step_sizes[0])[:, 1:-1, 1:-1]
    # grad_y = d1y_shift(im, step_sizes[1])[1:-1, :, 1:-1]
    # grad_x = d1x_shift(im, step_sizes[2])[1:-1, 1:-1, :]

    # l2 norm of gradients
    return tf.reduce_sum(tf.sqrt(grad_z ** 2 + grad_y ** 2 + grad_x ** 2 + eps ** 2))


def total_variation_rainer(im, eps=1e2, step_sizes=(1, 1, 1), epsC=1e-10):
    """
    Rainer's version to realize the isotropic tv regularizer using a
    combination of backward and forward 2-pt finite difference scheme and
    shifts for derivation.
    (Adapted from RegularizeTV.m)
    implementation by bene
    edited by soenke

    Note that this will not give exactly the same result, since matlab-
    implementation is using circshfits, while here the border is excluded.

    penalty = sum( sqrt(|grad(f)|^2+epsR) )

    Arguments:
        im (tf-tensor, 3d, real): image
        epsR (float): damping term for low values to avoid kink of abs fxn
        epsC (float): elevate low gradient values.
        step_sizes = (3-tuple of floats): step sizes (i.e. pixel size)
                     in different directions.
                     (axis 0, 1 and 2 corresponding to z, y and x)
                     if pixel size is the same in all directions, this can
                     be kept as 1

    for discussion of epsR see
    Ferreol Soulez et al.
    "Blind deconvolution of 3D data in wide field fluorescence microscopy"

    NOTE:  the naming of some args has been changed compared to matlab:
        toRegularize ==> im
        epsR ==> eps**2
        epsC has been kept
        BetaVals ==> step_sizes

    Restricted to real-valued 3d im opposed to Rainer's version

    assumes real image.  You should add absolut value before squaring in left,
    right for complex "images"

    |grad(f)|^2 means scalar product with cc.

    I don't understand all details of this implementation!
    TODO: why use both eps and epsC?
    TODO: the manner of adding gradients?
    """

    # original implementation:
    #    im_sub = im[1:-1,1:-1,1:-1]  # temporary helper
    #
    #    # The result of these are still 3d-images
    #    # forward difference scheme along each dimension
    #    d1z_fwd = (im[2:, 1:-1, 1:-1] - im_sub)/step_sizes[0] #(f_(z+1) - f_z)/ss0
    #    d1y_fwd = (im[1:-1, 2:, 1:-1] - im_sub)/step_sizes[1] #(f_(y+1) - f_y)/ss1
    #    d1x_fwd = (im[1:-1, 1:-1, 2:] - im_sub)/step_sizes[2] #(f_(x+1) - f_x)/ss2
    #
    #    # backward difference scheme along each dimension
    #    d1z_bwd = (im_sub - im[0:-2, 1:-1, 1:-1])/step_sizes[0] #(f_z-f_(z-1))/ss0
    #    d1y_bwd = (im_sub - im[1:-1, 0:-2, 1:-1])/step_sizes[1]
    #    d1x_bwd = (im_sub - im[1:-1, 1:-1, 0:-2])/step_sizes[2]
    #
    #    # These are still 3d-images
    #    # magn = magnitude
    #    fwd_gradient_magn = tf.sqrt(d1z_fwd**2 + d1y_fwd**2 + d1x_fwd**2 + eps**2)
    #    bwd_gradient_magn = tf.sqrt(d1z_bwd**2 + d1y_bwd**2 + d1x_bwd**2 + eps**2)
    #
    # TODO: why adding them like this?  Why not dividing by 2 or sth like that?
    #    gradient_magn = fwd_gradient_magn + bwd_gradient_magn

    #    # suggestion 1:
    #    #(f_(z+1) - f_z)/ss0 etc.
    #    d1z_fwd_full = (im[1:, 1:-1, 1:-1] - im[0:-1,1:-1,1:-1])/step_sizes[0]
    #    d1y_fwd_full = (im[1:-1, 1:, 1:-1] - im[1:-1,0:-1,1:-1])/step_sizes[1]
    #    d1x_fwd_full = (im[1:-1, 1:-1, 1:] - im[1:-1,1:-1,0:-1])/step_sizes[2]
    #
    #    # this would only be possible when circshift is available
    #    # right now I don't think dimensions fit for adding.
    #    gradient_magn_full = tf.sqrt(d1z_fwd_full**2 + d1y_fwd_full**2 +
    #                                 d1x_fwd_full**2)
    #
    #    # This corresponds to the sum of fwd and bwd gradient_magnitudes
    #    # gradient_magn_sum = gradient_magn_full[  #TODO

    # suggestion 2:
    # (f_(z+1) - f_z)/ss0 etc.
    d1z_fwd_full = (im[1:, 1:-1, 1:-1] - im[0:-1, 1:-1, 1:-1]) / step_sizes[0]
    d1y_fwd_full = (im[1:-1, 1:, 1:-1] - im[1:-1, 0:-1, 1:-1]) / step_sizes[1]
    d1x_fwd_full = (im[1:-1, 1:-1, 1:] - im[1:-1, 1:-1, 0:-1]) / step_sizes[2]
    # same as _d1z_fwd_shift_full(volume, step_size[0])[:,1:-1,1:-1]
    # but this seems a bit intransparent.

    # arrays originating from fwd and bwd gradients are the same except the
    # way they are related to coordinates.
    d1z_fwd = d1z_fwd_full[0:-1]  # first value corresponds to second z-value
    d1y_fwd = d1y_fwd_full[:, 0:-1]  # fwd cannot go to last row
    d1x_fwd = d1x_fwd_full[:, :, 0:-1]  # last value corresponds to second last

    d1z_bwd = d1z_fwd_full[1:]  # first value corresponds to second z-value
    d1y_bwd = d1y_fwd_full[:, 1:]  # bwd cannot go to first row
    d1x_bwd = d1x_fwd_full[:, :, 1:]

    fwd_gradient_magn = tf.sqrt(d1z_fwd ** 2 + d1y_fwd ** 2 + d1x_fwd ** 2 + eps ** 2)
    bwd_gradient_magn = tf.sqrt(d1z_bwd ** 2 + d1y_bwd ** 2 + d1x_bwd ** 2 + eps ** 2)

    # TODO: why adding them like this?  Why not dividing by 2 or sth like that?
    gradient_magn = fwd_gradient_magn + bwd_gradient_magn

    # TODO: why this?  To avoid divisions by zero (?).  Does this ever happen?
    # Aren't they added up anyways?  -> I think this was just needed for
    # gradient computation
    # -> might still be useful?  observe!
    # TODO: check if it ever comes into play by printing or ipdb

    # TODO: there is a bug here (!)
    # gradient_magn = tf.where(gradient_magn < epsC, epsC, gradient_magn)

    return tf.reduce_sum(gradient_magn)


def _total_variation_from_tf(im):
    """
    tv-regularization from tensorflow.  Works for 2d or 3d
    (assuming channels are mixed -- untested).
    Main problem with this is the kink at zero (eps is not added).
    This may complicate optimization.

    I'd recommend against using this unless you know what you are doing.

    Args:
        im (tf-tensor, 2d): image to be regularized

    Returns:
        penalty (float, 1d):  total variation penalty of image

    """
    if len(im.get_shape()) == 2:
        # requires 3d-input.  Add toy dimension
        im = tf.expand_dims(im, axis=2)

    return tf.image.total_variation(im)


# %% Good's roughness

# TODO docstrings
# TODO testing

# See also: Verveer et al. Journal of Microscopy, 193, 50-61

def goods_roughness(im, eps=0.1, step_sizes=(1, 1, 1)):
    """
    convenience function.  Calculates gr penalty by a predefined
    method (the method which is currently preferred)

    penalty = sum ( |grad(f)|^2 / (|f|+eps) )

    Wrapper for total_variation_iso_circshift.  See that function for more
    documentation.

    # renaming in my code compared to Rainer:
    # BetaVals ==> step_sizes (which stands for relative step size)
    # epsR ==> eps (which stands for the eps in the denominator)
    """
    return goods_roughness_shift(im, eps=eps, step_sizes=step_sizes)


def goods_roughness_conv(im, eps=0.1, step_sizes=(1, 1, 1)):
    """
    computes Good's roughness regularisation:
        penalty = sum( |Grad(f)|^2 / (|f| + eps) )
    Uses circshifts to calculate gradients.

    Input:
        im (tf-tensor, 3d) : image to be regularized
        step_sizes (tuple, 3d) : relative step_sizes (i.e. pixel sizes).
            Defaults to (1,1,1)
        eps (float) : see denominator of formula - avoid division near zero

    Returns:
        penalty  (tf-tensor, 1d  TODO or float?)

    note that eps is not squared and serves a different function
    as compared to tv

    Restricted to real-valued 3d im opposed to Rainer's version

    assumes real image.  You should add absolut value before squaring in left,
    right for complex "images"

    |grad(f)|^2 means scalar product with cc.
    """
    grad_z = d1z_central_conv(im, step_sizes[0])[:, 1:-1, 1:-1]
    grad_y = d1y_central_conv(im, step_sizes[1])[1:-1, :, 1:-1]
    grad_x = d1x_central_conv(im, step_sizes[2])[1:-1, 1:-1, :]

    # note that eps is not squared and serves a different function
    # as compared to tv
    return tf.reduce_sum((grad_z ** 2 + grad_y ** 2 + grad_x ** 2) /
                         (tf.abs(im[1:-1, 1:-1, 1:-1]) + eps))


def goods_roughness_shift(im, eps=0.1, step_sizes=(1, 1, 1)):
    """
    computes Good's roughness regularisation:
        penalty = sum( |Grad(f)|^2 / (|f| + eps) )
    Uses circshifts to calculate gradients.

    Input:
        im (tf-tensor, 3d) : image to be regularized
        step_sizes (tuple, 3d) : relative step_sizes (i.e. pixel sizes).
            Defaults to (1,1,1)
        eps (float) : see denominator of formula - avoid division near zero

    Returns:
        penalty  (tf-tensor, 1d  TODO or float?)

    note that eps is not squared and serves a different function
    as compared to tv

    Restricted to real-valued 3d im opposed to Rainer's version

    assumes real image.  You should add absolut value before squaring in left,
    right for complex "images"

    |grad(f)|^2 means scalar product with cc.
    """
    grad_z = (im[2:, 1:-1, 1:-1] - im[0:-2, 1:-1, 1:-1]) / (2 * step_sizes[0])
    grad_y = (im[1:-1, 2:, 1:-1] - im[1:-1, 0:-2, 1:-1]) / (2 * step_sizes[1])
    grad_x = (im[1:-1, 1:-1, 2:] - im[1:-1, 1:-1, 0:-2]) / (2 * step_sizes[2])

    # note that eps is not squared and serves a different function
    # as compared to tv
    return tf.reduce_sum((grad_z ** 2 + grad_y ** 2 + grad_x ** 2) /
                         (tf.abs(im[1:-1, 1:-1, 1:-1]) + eps))


def goods_roughness_rainer(im, eps=0.1, step_sizes=(1, 1, 1)):
    """
    Rainer's version to realize the good's roughness (gr) regularizer
    TODO: make more efficient -- see comment in code

    computes Good's roughness regularisation:
    penalty= sum( |Grad(f)|^2 / (|f| + eps) )

    Input:
        im (tf-tensor, 3d) : image to be regularized
        step_sizes (tuple, 3d) : relative step_sizes (i.e. pixel sizes).
            Defaults to (1,1,1)
        eps (float) : see denominator of formula - avoid division near zero

    Note:
    As compared to Rainer's code the following have been renamed:
        - toRegularize ==> im
        - BetaVals ==> relative step size
        - epsR ==> eps

    Restricted to real-valued 3d im opposed to Rainer's version

    assumes real image.  You should add absolut value before squaring in left,
    right for complex "images"

    |grad(f)|^2 means scalar product with cc.

    I don't understand the exact manner of adding fwd and bwd gradient
    """

    # original implementation:
    # TODO: allow 2d or n-d, eg. as in matlab code and allow to omit
    # regularization along one dimension by setting step_size=0 or NaN or None

    #    im_sub = im[1:-1,1:-1,1:-1]  # temporary helper
    #
    #    # forward difference scheme along each dimension
    #    right0 = (im[2:, 1:-1, 1:-1] - im_sub)/step_sizes[0] #(f_(z+1) - f_z)/ss0
    #    right1 = (im[1:-1, 2:, 1:-1] - im_sub)/step_sizes[1] #(f_(y+1) - f_y)/ss1
    #    right2 = (im[1:-1, 1:-1, 2:] - im_sub)/step_sizes[2] #(f_(x+1) - f_x)/ss2
    #
    #    # backward difference scheme along each dimension
    #    left0 = (im_sub - im[0:-2, 1:-1, 1:-1])/step_sizes[0] # (f_x - f_(x-1))/ss
    #    left1 = (im_sub - im[1:-1, 0:-2, 1:-1])/step_sizes[1]
    #    left2 = (im_sub - im[1:-1, 1:-1, 0:-2])/step_sizes[2]
    #
    #    # del im_sub
    #
    #    # The result of these are still 3d-images
    #    sqr_gradient_right = right0**2 + right1**2 + right2**2 # squared fwd grad.
    #    sqr_gradient_left = left0**2 + left1**2 + left2**2  # squared bwd gradient
    #    # TODO: why adding them like this?  Why not dividing by 2 or sth like that?
    #    sqr_gradient_sum = sqr_gradient_left + sqr_gradient_right

    # TODO fwd differences and bwd differences are actually almost the same,
    # but values they are associated with is shifted.
    # You could save some computations here !!!!
    # Needs circshift to work well
    # -> see tv suggestion 1 for starting point

    # TODO: you could omit one of fwd or bwd and shift abs-image correctly

    #    # suggestion 2:
    #    #(f_(z+1) - f_z)/ss0 etc.
    d1z_fwd_full = (im[1:, 1:-1, 1:-1] - im[0:-1, 1:-1, 1:-1]) / step_sizes[0]
    d1y_fwd_full = (im[1:-1, 1:, 1:-1] - im[1:-1, 0:-1, 1:-1]) / step_sizes[1]
    d1x_fwd_full = (im[1:-1, 1:-1, 1:] - im[1:-1, 1:-1, 0:-1]) / step_sizes[2]
    # same as _d1z_fwd_shift_full(volume, step_size[0])[:,1:-1,1:-1]
    # but this seems a bit intransparent.

    # arrays originating from fwd and bwd gradients are the same except the
    # way they are related to coordinates.
    d1z_fwd = d1z_fwd_full[0:-1]  # first value corresponds to second z-value
    d1y_fwd = d1y_fwd_full[:, 0:-1]  # fwd cannot go to last row
    d1x_fwd = d1x_fwd_full[:, :, 0:-1]  # last value corresponds to second last

    d1z_bwd = d1z_fwd_full[1:]  # first value corresponds to second z-value
    d1y_bwd = d1y_fwd_full[:, 1:]  # bwd cannot go to first row
    d1x_bwd = d1x_fwd_full[:, :, 1:]

    # The result of these are still 3d-images
    fwd_gradient_magn_sqr = d1z_fwd ** 2 + d1y_fwd ** 2 + d1x_fwd ** 2
    bwd_gradient_magn_sqr = d1z_bwd ** 2 + d1y_bwd ** 2 + d1x_bwd ** 2
    sqr_gradient_sum = fwd_gradient_magn_sqr + bwd_gradient_magn_sqr

    # note that eps is not squared and serves a different function
    # as compared to tv
    return tf.reduce_sum(sqr_gradient_sum /
                         (tf.abs(im[1:-1, 1:-1, 1:-1]) + eps))


def RegularizeNegSqr(toRegularize):
    mySqrt = tf.where( # Just affects the real part
                    tf.less(toRegularize , tf.zeros_like(toRegularize)),
                    tf_abssqr(toRegularize), tf.zeros_like(toRegularize))
    
    myReg = tf.reduce_sum(mySqrt)
    return myReg


def posiminity(im, minval=0):
    # Clip Values below zero => add to error function
    print('Regularizer: Penalize Values less then '+str(minval))
    reg = tf.reduce_sum(tf.square(tf.nn.relu(-im)))  # avdoid values smaller then zero
    return reg

def posimaxity(im, maxval=1):
    # Clip Values below zero => add to error function
    print('Regularizer: Penalize Values higher then '+str(maxval))
    reg = tf.reduce_sum(tf.square(tf.nn.relu(im - (maxval))))
    return reg

def gaussivity(im, sigma=0.1, minval=0.1):
    print('Regularizer: Try to like values around 0 and dn with gaussian distribution')
    # have guassian activation for 0 and dn - like the possible values more.
    print('Take care the regularizer is not right!!')
    reg = tf.nn.l2_loss((1 - tf.exp(-.5 * tf.square((im - 0) / sigma))))
    reg = reg + tf.nn.l2_loss(
        (1 - tf.exp(-.5 * tf.square((im - minval) / sigma))))
    return reg

def tf_total_variation_regularization_sobel(toRegularize):
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
        tv_loss = tf.reduce_sum(tf.abs(tf.nn.conv3d(tf.sigmoid(toRegularize), sobel, (1,) * 5, "VALID"))) # tf.nn.conv3d filter shape: [filter_depth, filter_height, filter_width, in_channels, out_channels]
        return tv_loss

def tf_total_variation_regularization(toRegularize, BetaVals = [1,1,1], epsR = 1, epsC=1e-10, is_circ = True):
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
            
    mySqrtL = tf.sqrt(tf_abssqr(aGradL_1)+tf_abssqr(aGradL_2)+tf_abssqr(aGradL_3)+epsR)
    mySqrtR = tf.sqrt(tf_abssqr(aGradR_1)+tf_abssqr(aGradR_2)+tf_abssqr(aGradR_3)+epsR)
     
    
    
    mySqrt = mySqrtL + mySqrtR; 
    
    if(0):
        mySqrt = tf.where(
                    tf.less(mySqrt , epsC*tf.ones_like(mySqrt)),
                    epsC*tf.ones_like(mySqrt),
                    mySqrt) # To avoid divisions by zero
    else:               
        mySqrt = mySqrt
        
        
    myReg = tf.reduce_sum(mySqrt)

    return myReg