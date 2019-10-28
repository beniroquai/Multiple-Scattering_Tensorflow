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
is_mictype = 'BF' # Brightfield
NAci = 0. # inner NA
zernikefactors = np.array((0,0,0,0,0,0,0,-.0,-.0,0,0,0.0,.0))  # 7: ComaX, 8: ComaY, 11: Spherical Aberration
zernikemask = np.array(np.abs(zernikefactors)>0)*1# mask of factors that should be updated
zernikemask[0]=0 # we don't want the first one to be shifting the phase!!
is_dampic=.01
mybackgroundval=-1j


''' Control-Parameters - Optimization '''
my_learningrate = 1e-2  # learning rate

# Regularizer 
regularizer = 'TV'
lambda_reg = 1e-3
lambda_zernike = .5
lambda_icshift = 0.
myepstvval = 1e-12##, 1e-12, 1e-8, 1e-6)) # - 1e-1 # smaller == more blocky

# Control Flow 
lambda_neg = 10000.
lambda_pos = 10000.

resultpath = ".\\Data\\DROPLETS\\RESULTS\\"

    
if(1):
    
    '''Droplets recent from Dresden! '''
    
    ''' Hologram Reconstruction Parameters '''
    myFolder = 'W:\\diederichbenedict\\Q-PHASE\\PATRICK\\';
    myFile = 'S0105k_zstack_dz0-2um_25C_40x_Ap0-52_Int63_sameAlignment.tif';
    myDarkFile = 'S0105k_zstack_dz0-2um_25C_40x_Ap0-52_Int63_sameAlignment_dark.tif';
    myBrightfieldFile = 'S0105k_zstack_dz0-2um_25C_40x_Ap0-52_Int63_sameAlignment_bright.tif';
    roi_center = (229,295)
    cc_center = np.array((637, 1418))
    cc_size = np.array((600, 600))
   
    dz = .2
    
    ''' Reconstruction Parameters '''
    # data files for parameters and measuremets 
    obj_meas_filename = './Data/cells/S0105k_zstack_dz0-2um_25C_40x_Ap0-52_Int63_sameAlignment.tif_allAmp.mat'
    matlab_par_filename = './Data/cells/S0105k_zstack_dz0-2um_25C_40x_Ap0-52_Int63_sameAlignment.tifmyParameter.mat'
    matlab_obj_file = './Data/cells/S0105k_zstack_dz0-2um_25C_40x_Ap0-52_Int63_sameAlignment.tif_mysphere.mat'
    matlab_par_name = 'myParameter' 
    matlab_val_name = 'allAmpSimu'   
    mybackgroundval = 0.
    dn = 0.1
    NAc = .3
    
    is_dampic= 0.1#.051 # smaller => more damping!
    mysubsamplingIC = 0
    
    
    ''' Control-Parameters - Optimization '''
    #my_learningrate = 1e3  # learning rate

    

    zernikefactors = np.zeros((11,)) 

    #zernikemask=1.*(np.abs(zernikefactors)>0)
    zernikemask = np.ones(zernikemask.shape)
    zernikemask[0:4] = 0 # don't care about defocus, tip, tilt 

    # worked with tv= 1e1, 1e-12, lr: 1e-2 - also in the git!
    zernikefactors = np.array((0,0,0,0, -1.2058168e-04, -2.3622499e-03, -7.7374041e-02 ,-1.4900701e-02,  -6.6282146e-04 ,-4.2013789e-04 , -1.2619525e+00))
    
    shiftIcX = 0.0143085485
    shiftIcY =-0.009161083
    #zernikefactors = 0*np.array((0.,0.,0.,0.,0.49924508,-1.162684,-0.09952152,-0.4380897,-0.6640752,0.16908956,0.860051))
    
    #for BORN
    if(0):
        ''' Control-Parameters - Optimization '''
        my_learningrate = 1e0  # learning rate
        dz =.2
        # Regularizer 
        regularizer = 'TV'
        lambda_reg = 1e1
        lambda_zernike = 0*1.
        lambda_icshift = 0*1.
        lambda_neg = 0*100.
        myepstvval = 1e-10 ##, 1e-12, 1e-8, 1e-6)) # - 1e-1 # smaller == more blocky
    #for BPM
    else:
        ''' Control-Parameters - Optimization '''
        is_mictype = 'DF'
        NAci = .1
        NAc = .3
        my_learningrate = 1e1  # learning rate
        mysubsamplingIC = 0
        dz = .3
        
        if(0):
            regularizer = 'TV'
            lambda_reg = 1e-0
            if(1):
                regularizer = 'L2'
                lambda_reg = 1e-1
                myepstvval = 1e-15 ##, 1e-12, 1e-8, 1e-6)) # - 1e-1 # smaller == more blocky
                myepstvval2 = 1e-15 ##, 1e-12, 1e-8, 1e-6)) # - 1e-1 # smaller == more blocky
            my_learningrate = 1e0  # learning rate
            lambda_zernike = 1.
            lambda_icshift = 1.
            lambda_neg = 0*100.
            myepstvval = 1e-11 ##, 1e-12, 1e-8, 1e-6)) # - 1e-1 # smaller == more blocky
            dz = .3
            zernikefactors *= 0
            NAci = .2
            NAc = .3
        else:
            # Regularizer using GR
            regularizer = 'GR'
            lambda_reg = 1e-0
            lambda_zernike = 10.
            lambda_icshift = 10.
            lambda_neg = 0*100.
            myepstvval = 1e-15 ##, 1e-12, 1e-8, 1e-6)) # - 1e-1 # smaller == more blocky
            myepstvval2 = 1e-15 ##, 1e-12, 1e-8, 1e-6)) # - 1e-1 # smaller == more blocky
            zernikefactors *= 0
            
elif(0):
    
    '''Scotch Tape from Tomas! '''
    
    ''' Hologram Reconstruction Parameters '''
    myFolder = 'W:\\diederichbenedict\\Q-PHASE\\PATRICK\\';
    myFile = 'S0105k_zstack_dz0-2um_25C_40x_Ap0-52_Int63_sameAlignment.tif';
    myDarkFile = 'S0105k_zstack_dz0-2um_25C_40x_Ap0-52_Int63_sameAlignment_dark.tif';
    myBrightfieldFile = 'S0105k_zstack_dz0-2um_25C_40x_Ap0-52_Int63_sameAlignment_bright.tif';
    roi_center = (229,295)
    cc_center = np.array((637, 1418))
    cc_size = np.array((600, 600))
   
    dz = 1.2
    
    ''' Reconstruction Parameters '''
    # data files for parameters and measuremets 
    obj_meas_filename = './Data/cells/tape_10x_10layers.tif_allAmp.mat' #Tape_10x_0.3_dz1mu.tif_allAmp.mat'
    matlab_par_filename = './Data/cells/tape_10x_10layers.tifmyParameter.mat' # Tape_10x_0.3_dz1mu.tifmyParameter.mat'
    matlab_par_name = 'myParameter' 
    matlab_val_name = 'allAmpSimu'   
    mybackgroundval = -1j
    dn = 0.1
    NAc = .22
    
    is_dampic= .051 # smaller => more damping!
    mysubsamplingIC = 0
    


    zernikefactors = np.zeros((11,)) 
    

    #zernikefactors[6]=-4.25  # coma X
    #zernikefactors[7]= 4.25 # coma y
    #zernikemask[8]=-0.00  # Trefoil X
    #zernikemask[9]=-0.00 # Trefoil y
    #zernikefactors[10]= -1.5 # defocus
    
    #zernikemask=1.*(np.abs(zernikefactors)>0)
    zernikemask = np.ones(zernikemask.shape)
    zernikemask[0:4] = 0 # don't care about defocus, tip, tilt 

    # worked with tv= 1e1, 1e-12, lr: 1e-2 - also in the git!
    zernikefactors = 0*np.array((0,0,0,0, -1.2058168e-04, -2.3622499e-03, -7.7374041e-02 ,-1.4900701e-02,  -6.6282146e-04 ,-4.2013789e-04 , -1.2619525e+00))
    
    shiftIcX = 0.0143085485*0 
    shiftIcY =-0.009161083*0
    #zernikefactors = 0*np.array((0.,0.,0.,0.,0.49924508,-1.162684,-0.09952152,-0.4380897,-0.6640752,0.16908956,0.860051))
    
    
    if(1):        
        ''' Control-Parameters - Optimization '''
        my_learningrate = 1e4  # learning rate
        
        # Regularizer 
        regularizer = 'TV'#'TV'
        lambda_reg = 1e-1
        lambda_zernike = 0*1.
        lambda_icshift = 0*1.
        lambda_neg = 0*100.
        myepstvval = 1e-8 ##, 1e-12, 1e-8, 1e-6)) # - 1e-1 # smaller == more blocky
        

elif(0):
    # data files for parameters and measuremets 
    obj_meas_filename = './Data/cells/Cell_20x_100a_150-250.tif_allAmp.mat'
    matlab_par_filename = './Data/cells/Cell_20x_100a_150-250.tif_myParameter.mat'
    matlab_par_name = 'myParameter' 
    matlab_val_name = 'allAmpSimu'   
    mybackgroundval = -.85j  
    shiftIcY = 8 # has influence on the YZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
    shiftIcX = 6 # has influence on the XZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
    

elif(0):
    # data files for parameters and measuremets 
    obj_meas_filename = './Data/cells/S0019-2a_zstack_dz0-02um_751planes_40x_every8thslice.tif_allAmp.mat'
    matlab_par_filename = './Data/cells/S0019-2a_zstack_dz0-02um_751planes_40x_every8thslice.tif_myParameter.mat'
    matlab_par_name = 'myParameter' 
    matlab_val_name = 'allAmpSimu'
elif(0):
    obj_meas_filename = './Data/cells/S0014b_zstack_dz0-02um_1301planes_40x_z160mum_Hologram.tif_allAmp.mat'
    matlab_par_filename = './Data/cells/S0014b_zstack_dz0-02um_1301planes_40x_z160mum_Hologram.tifmyParameter.mat'
    matlab_par_name = 'myParameter'
    matlab_val_name = 'allAmpSimu'
    NAc = .25
    mybackgroundval = -1j
    shiftIcY =  20# has influence on the YZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
    shiftIcX =  20 # has influence on the XZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
    zernikefactors = np.array((0,0,0,0,0,-0.,-3.0,1.,0,0,.5,.0))  # 7: ComaX, 8: ComaY, 11: Spherical Aberration
    zernikemask = np.array(np.abs(zernikefactors)>0)*1# mask of factors that should be updated
elif(0):
    obj_meas_filename = './Data/cells/0014a_zstack_dz0-04um_751planes_20x_every10thslice.tif_allAmp.mat'
    matlab_par_filename = './Data/cells/0014a_zstack_dz0-04um_751planes_20x_every10thslice.tifmyParameter.mat'
    matlab_par_name = 'myParameter'
    matlab_val_name = 'allAmpSimu'
    NAc = .25
    is_dampic = .04
    mybackgroundval = -1j
    shiftIcY =  -0.073108904# has influence on the YZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
    shiftIcX =  2.4129255 # has influence on the XZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
    zernikefactors = np.array((0,0,0,0,0,-0.,-.005,.001,0,0,.005,.0))  # 7: ComaX, 8: ComaY, 11: Spherical Aberration
    zernikemask = np.array(np.abs(zernikefactors)>0)*1# mask of factors that should be updated
# -0.9171849 -2.070675   0.         0.         4.0056386

  # 7: ComaX, 8: ComaY, 11: Spherical Aberration
    #zernikemask = np.array(np.abs(zernikefactors)>0)*1# mask of factors that should be updated
    #zernikemask = zernikemask*0
    zernikemask[:]=1
    zernikefactors = np.array((.47486836, -0.39439008, -1.4269363,  -0.37701255, -0.08931556,  0.22308928,  0.67801976,  0.15227595, -0.13716215, -0.22266115, -0.24198568,  0.20539095))
    zernikefactors = np.array((1.5145516,  -0.4922971,  -1.6731209,   0.9618724,   0.03274873,  0.0987005, 0.45747086,  0.13862132, -0.08351833, -0.11787935, -0.29825905, -0.07494219))
    shiftIcY =  2.3640773
    shiftIcX = -0.52940077
else:
    matlab_par_filename = './Data/DROPLETS/S19_multiple/Parameter.mat'; matlab_par_name='myParameter'
    obj_meas_filename = './Data/DROPLETS/RESULTS/allAmp_simu.npy'
    mybackgroundval = 0

    



