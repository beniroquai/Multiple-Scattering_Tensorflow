#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:39:44 2019

@author: bene
"""

#shiftIcY = 0*4 # has influence on the YZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
#shiftIcX = 0*4 # has influence on the XZ-Plot - negative values shifts the input wave (coming from 0..end) to the left

if(0):
    # 10mum bead
    # data files for parameters and measuremets 
    matlab_val_file = './Data/cells/Beads_40x_100a_100-250.tif_allAmp.mat'
    matlab_par_file = './Data/cells/Beads_40x_100a_100-250.tif_myParameter.mat'
    matlab_par_name = 'myParameter' 
    matlab_val_name = 'allAmpSimu' 
    mybackgroundval = -.95j  
elif(0):
    # data files for parameters and measuremets 
    matlab_val_file = './Data/cells/cross_section_10x0.3_hologram_full.tif_allAmp.mat'
    matlab_par_file = './Data/cells/cross_section_10x0.3_hologram_full.tif_myParameter.mat'
    matlab_par_name = 'myParameter' 
    matlab_val_name = 'allAmpSimu'
    mybackgroundval = -1j
    shiftIcY = 4 # has influence on the YZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
    shiftIcX = 4 # has influence on the XZ-Plot - negative values shifts the input wave (coming from 0..end) to the left

elif(1):
    # data files for parameters and measuremets 
    matlab_val_file = './Data/cells/Cell_20x_100a_150-250.tif_allAmp.mat'
    matlab_par_file = './Data/cells/Cell_20x_100a_150-250.tif_myParameter.mat'
    matlab_par_name = 'myParameter' 
    matlab_val_name = 'allAmpSimu'   
    mybackgroundval = -.85j  
    shiftIcY = 10 # has influence on the YZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
    shiftIcX = 10 # has influence on the XZ-Plot - negative values shifts the input wave (coming from 0..end) to the left

elif(0):
    # data files for parameters and measuremets 
    matlab_val_file = './Data/cells/S0019-2a_zstack_dz0-02um_751planes_40x_every8thslice.tif_allAmp.mat'
    matlab_par_file = './Data/cells/S0019-2a_zstack_dz0-02um_751planes_40x_every8thslice.tif_myParameter.mat'
    matlab_par_name = 'myParameter' 
    matlab_val_name = 'allAmpSimu'
else:
    matlab_par_file = './Data/DROPLETS/S19_multiple/Parameter.mat'; matlab_par_name='myParameter'
    matlab_val_file = './Data/DROPLETS/RESULTS/allAmp_simu.npy'
    mybackgroundval = -1j
