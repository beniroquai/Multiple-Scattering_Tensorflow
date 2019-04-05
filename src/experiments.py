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
NAc = .32
zernikefactors = 0*np.array((0,0,0,0,0,-.5,0,-.5,0,0,0.0,.0))  # 7: ComaX, 8: ComaY, 11: Spherical Aberration
zernikemask = np.ones(zernikefactors.shape) #
zernikemask = np.array(np.abs(zernikefactors)>0)*1# mask of factors that should be updated
zernikemask[0]=0 # we don't want the first one to be shifting the phase!!

mybackgroundval=-1j
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
    shiftIcY = -5# 20 # has influence on the YZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
    shiftIcX = 5# 20 # has influence on the XZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
    NAc = .32
    dn=.1
elif(0):
    # data files for parameters and measuremets 
    matlab_val_file = './Data/cells/cross_section_10x0.3_hologram_full.tiffull_allAmp.mat'
    matlab_par_file = './Data/cells/cross_section_10x0.3_hologram_full.tif_full_myParameter.mat'
    matlab_par_name = 'myParameter' 
    matlab_val_name = 'allAmpSimu'
    mybackgroundval = -1j
    shiftIcY = 20 # has influence on the YZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
    shiftIcX = 20 # has influence on the XZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
    NAc = .2
elif(0):
    # data files for parameters and measuremets 
    matlab_val_file = './Data/cells/SiO2_5um_20x0.5.cdf.tif_allAmp.mat'
    matlab_par_file = './Data/cells/SiO2_5um_20x0.5.cdf.tif_myParameter.mat'
    matlab_par_name = 'myParameter' 
    matlab_val_name = 'allAmpSimu'
    mybackgroundval = -1j
    dn = 1.52-1.33
    shiftIcY = 0 # has influence on the YZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
    shiftIcX = 0 # has influence on the XZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
elif(0):
    # data files for parameters and measuremets 
    matlab_val_file = './Data/cells/Cell_20x_100a_120-270.tif_allAmp.mat'
    matlab_par_file = './Data/cells/Cell_20x_100a_120-270.tif_myParameter.mat'
    matlab_par_name = 'myParameter' 
    matlab_val_name = 'allAmpSimu'   
    mybackgroundval = -1j 
    dn = 0.05
    shiftIcY = -35 # has influence on the YZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
    shiftIcX = -35 # has influence on the XZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
    NAc = .42
elif(0):
    # data files for parameters and measuremets 
    matlab_val_file = './Data/cells/Cell_20x_100a_150-250.tif_allAmp.mat'
    matlab_par_file = './Data/cells/Cell_20x_100a_150-250.tif_myParameter.mat'
    matlab_par_name = 'myParameter' 
    matlab_val_name = 'allAmpSimu'   
    mybackgroundval = -.85j  
    shiftIcY = 8 # has influence on the YZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
    shiftIcX = 6 # has influence on the XZ-Plot - negative values shifts the input wave (coming from 0..end) to the left

elif(0):
    # data files for parameters and measuremets 
    matlab_val_file = './Data/cells/S0019-2a_zstack_dz0-02um_751planes_40x_every8thslice.tif_allAmp.mat'
    matlab_par_file = './Data/cells/S0019-2a_zstack_dz0-02um_751planes_40x_every8thslice.tif_myParameter.mat'
    matlab_par_name = 'myParameter' 
    matlab_val_name = 'allAmpSimu'
elif(0):
    matlab_val_file = './Data/cells/S0014b_zstack_dz0-02um_1301planes_40x_z160mum_Hologram.tif_allAmp.mat'
    matlab_par_file = './Data/cells/S0014b_zstack_dz0-02um_1301planes_40x_z160mum_Hologram.tifmyParameter.mat'
    matlab_par_name = 'myParameter'
    matlab_val_name = 'allAmpSimu'
    NAc = .25
    mybackgroundval = -1j
    shiftIcY =  10# has influence on the YZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
    shiftIcX =  10 # has influence on the XZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
    zernikefactors = np.array((0,0,0,0,0,-0.,-3.0,1.,0,0,.5,.0))  # 7: ComaX, 8: ComaY, 11: Spherical Aberration
    zernikemask = np.array(np.abs(zernikefactors)>0)*1# mask of factors that should be updated
elif(1):
    matlab_val_file = './Data/cells/0014a_zstack_dz0-04um_751planes_20x_every10thslice.tif_allAmp.mat'
    matlab_par_file = './Data/cells/0014a_zstack_dz0-04um_751planes_20x_every10thslice.tifmyParameter.mat'
    matlab_par_name = 'myParameter'
    matlab_val_name = 'allAmpSimu'
    NAc = .25
    mybackgroundval = -1j
    shiftIcY =  5# has influence on the YZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
    shiftIcX =  5 # has influence on the XZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
    zernikefactors = np.array((0,0,0,0,0,-0.,-.005,.001,0,0,.005,.0))  # 7: ComaX, 8: ComaY, 11: Spherical Aberration
    zernikemask = np.array(np.abs(zernikefactors)>0)*1# mask of factors that should be updated
# -0.9171849 -2.070675   0.         0.         4.0056386

  # 7: ComaX, 8: ComaY, 11: Spherical Aberration
    #zernikemask = np.array(np.abs(zernikefactors)>0)*1# mask of factors that should be updated
    #zernikemask = zernikemask*0
    zernikemask[:]=1

#Zernikes: [ 0.47486836 -0.39439008 -1.4269363  -0.37701255 -0.08931556  0.22308928
#  0.67801976  0.15227595 -0.13716215 -0.22266115 -0.24198568  0.20539095]
#ShiftX/Y: -0.073108904 / 2.4129255
else:
    matlab_par_file = './Data/DROPLETS/S19_multiple/Parameter.mat'; matlab_par_name='myParameter'
    matlab_val_file = './Data/DROPLETS/RESULTS/allAmp_simu.npy'
    mybackgroundval = 0
    NAc = .2
    



