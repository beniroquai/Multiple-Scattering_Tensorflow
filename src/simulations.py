#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:39:44 2019

@author: bene
"""
import numpy as np

shiftIcY = 0*4 # has influence on the YZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
shiftIcX = 0*4 # has influence on the XZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
dn = .051
# Generic Microscope Parameters
NAc = .52
zernikefactors = np.array((0,0,0,0,0,0,0,-.0,-.0,0,0,0.0,.0))  # 7: ComaX, 8: ComaY, 11: Spherical Aberration
zernikemask = np.array(np.abs(zernikefactors)>0)*1# mask of factors that should be updated
zernikemask[0]=0 # we don't want the first one to be shifting the phase!!
is_dampic=1
mybackgroundval=-1j
if(1):
    matlab_par_file = './Data/cells/ArtificialCheek_myParameter.mat'
    matlab_par_name='myParameter'
    



