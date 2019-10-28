#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:39:44 2019

@author: bene
"""
import numpy as np
import os 

# Default
if(0):
    # sample parameters
    dn = .051

    # Generic Microscope Parameters
    NAo = .5
    NAc = .3
    lambda0 = .65 

    # Systematic parametres for microscpe 
    shiftIcY = 0*4 # has influence on the YZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
    shiftIcX = 0*4 # has influence on the XZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
    
    zernikefactors = np.array((0,0,0,0,0,0,0,-.0,-.0,0,0,0.0,.0))  # 7: ComaX, 8: ComaY, 11: Spherical Aberration
    zernikemask = np.array(np.abs(zernikefactors)>0)*1# mask of factors that should be updated
    zernikemask[0]=0 # we don't want the first one to be shifting the phase!!
    is_dampic=1
    mybackgroundval= .3+1.1j
    mysubsamplingIC = 0 #
    mysize = ((128,128,128)) # Z, X, Y
    
    matlab_par_file = './Data/cells/ArtificialCheek_myParameter.mat'
    matlab_par_name='myParameter'
    matlab_val_file = './Data/DROPLETS/RESULTS/allAmp_simu.npy'

# In-Silica Stuff for Manuscript - Compare BPM and BORN 
elif(1):
    ''' ***********************************************************************
    Experimental -Parameters - Optimization 
     ***********************************************************************'''
    # sample parameters
    print('We are taking the in-silica for manuscript parameters')
    dn = .1
    nEmbb = 1.33

    # Generic Microscope Parameters
    NAo = .95
    NAc = .32
    NAci = .2
    lambda0 = .65 
    dx = .15
    dy = .15
    dz = dx#lambda0/nEmbb/2

    is_mictype = 'BF'
    # Systematic parametres for microscpe 
    shiftIcY = 0.0 # has influence on the YZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
    shiftIcX = 0.0 # has influence on the XZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
    
    
    zernikefactors = np.array((0,0,0,0,0,0,0,-.0,-.0,0,0,0.0,.0))  # 7: ComaX, 8: ComaY, 11: Spherical Aberration
    zernikemask = np.array(np.abs(zernikefactors)>0)*1# mask of factors that should be updated
    zernikemask[0]=0 # we don't want the first one to be shifting the phase!!
    is_dampic=1
    mybackgroundval= .3+1.1j
    mysubsamplingIC = 0 #
    mysize = ((60,70,70)) # Z, X, Y
    #mysize = ((128,128,128)) # Z, X, Y    
    
    # Path's 
    mysavepath = './Data/Simulations/'
    matlab_par_name='myParameter'
    savepath_simu = mysavepath+'allAmp_simu.npy'
    
    # results after fwd model:
    result_fwd_bpm = os.path.join(mysavepath, 'allAmp_simu_BPM.npy')
    result_fwd_born = os.path.join(mysavepath, 'allAmp_simu_BORN.npy')
    
    
    ''' ***********************************************************************
    Control-Parameters - Optimization 
     ***********************************************************************'''
    my_learningrate = 1e-2  # learning rate
    
    # Regularizer 
    regularizer = 'TV'
    lambda_reg = 5e-9
    lambda_zernike = 0*1.
    lambda_icshift = 0*1.
    lambda_neg = 0*100.
    myepstvval = 1e-11 ##, 1e-12, 1e-8, 1e-6)) # - 1e-1 # smaller == more blocky
    
             
# Miroslav stuff
elif(0):
    shiftIcY = 0*4 # has influence on the YZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
    shiftIcX = 0*4 # has influence on the XZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
    dn = .051
    # Generic Microscope Parameters
    NAc = .4
    NAci = .3
    
    zernikefactors = np.array((0,0,0,0,0,0,0,-.0,-.0,0,0,0.0,.0))  # 7: ComaX, 8: ComaY, 11: Spherical Aberration
    zernikemask = np.array(np.abs(zernikefactors)>0)*1# mask of factors that should be updated
    zernikemask[0]=0 # we don't want the first one to be shifting the phase!!
    is_dampic=1
    mybackgroundval= .3+1.1j
    mysubsamplingIC = 0 #
    mysize = ((60,70,70))
    
    matlab_par_file = './Data/cells/ArtificialCheek_myParameter.mat'
    matlab_par_name='myParameter'
    matlab_val_file = './Data/DROPLETS/RESULTS/allAmp_simu.npy'
        
    
    
    
    
    
    
