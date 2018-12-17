#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 23:11:10 2018
@author: Soenke
"""
import tensorflow as tf


# The naming and the "reduction" parameter is a bit confusing in the
# tf.losses module, but with reduction=SUM_BY_NONZERO_WEIGHTS (default),
# they calculate a meaningful approx to the batch mean

def huberloss(y_batch, y_predicted_batch, weights=1.0, delta=1.0):
    """ docstring TODO """
    return tf.losses.huber_loss(
        y_batch, y_predicted_batch, weights=weights, delta=delta)


def l2loss(y_batch, y_predicted_batch, weights=1.0):
    """ docstring TODO """
    return tf.losses.mean_squared_error(
        y_batch, y_predicted_batch, weights=weights)


def l1loss(y_batch, y_predicted_batch, weights=1.0):
    """ docstring TODO """
    return tf.losses.absolute_difference(
        y_batch, y_predicted_batch, weights=weights)


# Not sure this makes sense
def poissonloss(y_batch, y_predicted_batch, batch_size=1, eps_num=1e-20, eps_denom=1e-2):
    """
    docstring TODO.  Does not support 'weights'.
    Not sure this makes sense for supervised learning

    eps: small constant to be added to denominator to avoid division by zero

    # y_predicted_batch -> fwd (although unblurred)
    # y_batch           -> im  (although unnoised and unblurred)
    # fwd - im - im*log(fwd) + im*log(im)

    # You could use this for unsupervised learning passing x_batch for y_batch
    # and comparing with blurred y_predicted

    will return nan, if:
        - y_batch == 0  (if eps == 0)
        - y_predicted_batch == 0  (if eps == 0)
        - y_batch>0 and y_predicted_batch<0 or y_batch<0 and y_predicted_batch>0

    Assuming y_batch >= 0 and eps != 0:
        You need to ensure that y_predicted_batch > 0
    Ideally also make y_batch != 0
    """
    # batch_size = y_batch.shape.as_list[0]  # TODO: test and remove kwarg above; batch_dim is 0
    with tf.name_scope("loss"):
        return (tf.reduce_sum(y_predicted_batch - y_batch -
                              y_batch * tf.log((y_predicted_batch + (eps_num)) / (y_batch + eps_denom))) / batch_size)