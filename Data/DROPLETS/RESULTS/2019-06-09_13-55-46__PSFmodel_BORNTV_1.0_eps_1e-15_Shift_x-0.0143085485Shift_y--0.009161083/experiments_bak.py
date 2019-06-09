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
zernikefactors = np.array((0,0,0,0,0,0,0,-.0,-.0,0,0,0.0,.0))  # 7: ComaX, 8: ComaY, 11: Spherical Aberration
zernikemask = np.array(np.abs(zernikefactors)>0)*1# mask of factors that should be updated
zernikemask[0]=0 # we don't want the first one to be shifting the phase!!
is_dampic=.01
mybackgroundval=-1j


''' Control-Parameters - Optimization '''
my_learningrate = 1e-2  # learning rate

# Regularizer 
regularizer = 'TV'
lambda_tv = 1e-3
lambda_zernike = .5
myepstvval = 1e-12##, 1e-12, 1e-8, 1e-6)) # - 1e-1 # smaller == more blocky

# Control Flow 
lambda_neg = 10000.

resultpath = ".\\Data\\DROPLETS\\RESULTS\\"

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
    '''HELA CELL GOOD! '''
    # data files for parameters and measuremets 
    matlab_val_file = './Data/cells/Cell_20x_100a_120-270.tif_allAmp.mat'
    matlab_par_file = './Data/cells/Cell_20x_100a_120-270.tifmyParameter.mat'
    matlab_par_name = 'myParameter' 
    matlab_val_name = 'allAmpSimu'   
    mybackgroundval = -.9
    dn = 0.05
    shiftIcY = -22#0*35 # has influence on the YZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
    shiftIcX = -22.#*35 # has influence on the XZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
    NAc = .32
    zernikefactors[10]=.50 # defocus
    zernikefactors[6]=1.40 # coma X
    zernikefactors[7]=1.390 # coma y
    
    
    zernikemask=1.*(np.abs(zernikefactors)>0)
    is_dampic=.05
    #zernikefactors = np.array((1.5145516,  -0.4922971,  -1.6731209,   0.9618724,   0.03274873,  0.0987005, 0.45747086,  0.13862132, -0.08351833, -0.11787935, -0.29825905, -0.07494219))
    #shiftIcY =  2.3640773
    #shiftIcX = -0.52940077
elif(0):
    
    '''CHeek CELL GOOD! '''
    # data files for parameters and measuremets 
    matlab_val_file = './Data/cells/Cheek_20x_3.tif_allAmp.mat'
    matlab_par_file = './Data/cells/Cheek_20x_3.tifmyParameter.mat'
    matlab_par_name = 'myParameter' 
    matlab_val_name = 'allAmpSimu'   
    mybackgroundval = -1.
    dn = 0.06
    shiftIcY = -12 #0*35 # has influence on the YZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
    shiftIcX = -12.#*35 # has influence on the XZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
    NAc = .5
    zernikefactors[10]=.01 # defocus
    zernikefactors[6]=-0.1  # coma X
    zernikefactors[7]=-.1 # coma y
    
    
    zernikemask=1.*(np.abs(zernikefactors)>0)
    is_dampic=.05
    #zernikefactors = np.array((1.5145516,  -0.4922971,  -1.6731209,   0.9618724,   0.03274873,  0.0987005, 0.45747086,  0.13862132, -0.08351833, -0.11787935, -0.29825905, -0.07494219))
    #shiftIcY =  2.3640773
elif(0):
    
    '''Spheres GOOD! '''
    # data files for parameters and measuremets 
    matlab_val_file = './Data/cells/uspheres_40x_1.tif_allAmp.mat'
    matlab_par_file = './Data/cells/uspheres_40x_1.tifmyParameter.mat'
    matlab_obj_file = './Data/cells/uspheres_40x_1.tif_mysphere.mat'
    matlab_par_name = 'myParameter' 
    matlab_val_name = 'allAmpSimu'   
    mybackgroundval = -1.
    dn = 0.1
    shiftIcX = -.05#*35 # has influence on the XZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
    shiftIcY =  .05 #0*35 # has influence on the YZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
    NAc = .32
    zernikefactors = np.zeros((11,)) 
    
    zernikefactors[10]= -1.5 # defocus
    zernikefactors[6]=-4.25  # coma X
    zernikefactors[7]= 4.25 # coma y
    zernikefactors = np.zeros((11,))

    zernikemask=1.*(np.abs(zernikefactors)>0)
    zernikemask = np.ones(zernikemask.shape)
    zernikemask[8]=-0.00  # Trefoil X
    zernikemask[9]=-0.00 # Trefoil y
    #zernikefactors = np.array(( 0. , 1.8749844,  -2.164156,    4.292257,    0.63288367, -0.0527322, -1.319653 ,   1.4736626,  -5.1704946 , -3.740628  ,  0.8031174 ))
    shiftIcX = -.05#-3.62768 
    shiftIcY = .05#3.7690606

   # zernikemask[0]=0
    is_dampic= .05
    #zernikefactors = np.array((1.5145516,  -0.4922971,  -1.6731209,   0.9618724,   0.03274873,  0.0987005, 0.45747086,  0.13862132, -0.08351833, -0.11787935, -0.29825905, -0.07494219))
    #shiftIcY =  2.3640773
    '''
    #%%
    is_absorption = False
    is_obj_init_tikhonov = False 
    mybordersize = 20
    my_learningrate = 1e-2  # learning rate
    NreduceLR = 1000 # when should we reduce the Learningrate? 
    lambda_tv = 5e1
    myepstvval = 1e-15##, 1e-12, 1e-8, 1e-6)) # - 1e-1 # smaller == more blocky
    lambda_neg = 10000.
    Niter =  300
    '''
elif(0):
    
    '''Spheres OLD! '''
    # data files for parameters and measuremets 
    matlab_val_file = './Data/cells/Beads_40x_100a_110_220.tif_allAmp.mat'
    matlab_par_file = './Data/cells/Beads_40x_100a_110_220.tifmyParameter.mat'
    matlab_obj_file = './Data/cells/Beads_40x_100a_110_220.tif_mysphere.mat'
    matlab_par_name = 'myParameter' 
    matlab_val_name = 'allAmpSimu'   
    mybackgroundval = -1.
    dn = 0.1
    shiftIcX = -.05#*35 # has influence on the XZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
    shiftIcY =  .05 #0*35 # has influence on the YZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
    NAc = .32
    zernikefactors = np.zeros((11,)) 
    
    zernikefactors[10]= -1.5 # defocus
    zernikefactors[6]=-4.25  # coma X
    zernikefactors[7]= 4.25 # coma y
    zernikefactors = np.zeros((11,))

    zernikemask=1.*(np.abs(zernikefactors)>0)
    zernikemask = np.ones(zernikemask.shape)
    
    zernikefactors = np.array((-1.6394907e+00, -2.7066767e-02,  9.8326124e-02,  1.1410052e+00,
                                2.4763597e-03, -2.8212504e-03, -1.2886256e-01,  3.5430852e-02,
                                -2.4610769e-03, -2.7273528e-04, -2.7424264e-01))

    #zernikemask[0]=-0.00  # Trefoil X
   # zernikemask[9]=-0.00 # Trefoil y
    #zernikefactors = np.array(( 0. , 1.8749844,  -2.164156,    4.292257,    0.63288367, -0.0527322, -1.319653 ,   1.4736626,  -5.1704946 , -3.740628  ,  0.8031174 ))
    shiftIcX = -0.04220052 
    shiftIcY = 0.013765786

   # zernikemask[0]=0
    is_dampic= .05
    #zernikefactors = np.array((1.5145516,  -0.4922971,  -1.6731209,   0.9618724,   0.03274873,  0.0987005, 0.45747086,  0.13862132, -0.08351833, -0.11787935, -0.29825905, -0.07494219))
    #shiftIcY =  2.3640773
    '''
    #%%
    is_absorption = False
    is_obj_init_tikhonov = False 
    mybordersize = 20
    my_learningrate = 1e-2  # learning rate
    NreduceLR = 1000 # when should we reduce the Learningrate? 
    lambda_tv = 5e1
    myepstvval = 1e-15##, 1e-12, 1e-8, 1e-6)) # - 1e-1 # smaller == more blocky
    lambda_neg = 10000.
    Niter =  300
    '''
    

elif(0):
    
    '''Spheres GOOD! '''
    # data files for parameters and measuremets 
    matlab_val_file = './Data/cells/cheek_40x_1.tif_allAmp.mat'
    matlab_par_file = './Data/cells/cheek_40x_1.tifmyParameter.mat'
    matlab_obj_file = './Data/cells/cheek_40x_1.tif_mysphere.mat'
    matlab_par_name = 'myParameter' 
    matlab_val_name = 'allAmpSimu'   
    mybackgroundval = -0.
    dn = 0.01
    NAc = .32
    zernikefactors = np.zeros((11,)) 
    
    zernikefactors[10]= -1.5 # defocus
    #zernikefactors[6]=-4.25  # coma X
    #zernikefactors[7]= 4.25 # coma y
    
    
    zernikemask=1.*(np.abs(zernikefactors)>0)
    zernikemask = np.ones(zernikemask.shape)
    zernikemask[8]=-0.00  # Trefoil X
    zernikemask[9]=-0.00 # Trefoil y
    #zernikefactors = np.array(( 0. , 1.8749844,  -2.164156,    4.292257,    0.63288367, -0.0527322, -1.319653 ,   1.4736626,  -5.1704946 , -3.740628  ,  0.8031174 ))
    shiftIcX = -.00#-3.62768*2
    shiftIcY = .00#3.7690606*2

    #zernikemask[0]=0
    is_dampic= .01 # smaller => more damping!
    #zernikefactors = np.array((1.5145516,  -0.4922971,  -1.6731209,   0.9618724,   0.03274873,  0.0987005, 0.45747086,  0.13862132, -0.08351833, -0.11787935, -0.29825905, -0.07494219))
    #shiftIcY =  2.3640773
    
    #%%
    '''
    is_absorption = False
    is_obj_init_tikhonov = False 
    mybordersize = 20
    my_learningrate = 1e-1  # learning rate
    NreduceLR = 1000 # when should we reduce the Learningrate? 
    lambda_tv = 1e-1
    myepstvval = 1e-8##, 1e-12, 1e-8, 1e-6)) # - 1e-1 # smaller == more blocky
    lambda_neg = 10000.
    Niter =  300
    '''
    
    
elif(0):
    
    '''HeLas from PAtrick/Yashar, Imaged in Dresden! '''
    # data files for parameters and measuremets 
    matlab_val_file = './Data/cells/HeLa_JL1_a_zstack_dz0-3um_60planes.tif_allAmp.mat'
    matlab_par_file = './Data/cells/HeLa_JL1_a_zstack_dz0-3um_60planes.tifmyParameter.mat'
    matlab_par_name = 'myParameter' 
    matlab_val_name = 'allAmpSimu'   
    mybackgroundval = -0.
    dn = 0.01
    NAc = .32
    is_dampic= .005 # smaller => more damping!
    mysubsamplingIC = 0
    
    
    ''' Control-Parameters - Optimization '''
    my_learningrate = 1e-2  # learning rate
    
    # Regularizer 
    regularizer = 'TV'
    lambda_tv = 1e1
    myepstvval = 1e-12##, 1e-12, 1e-8, 1e-6)) # - 1e-1 # smaller == more blocky
    
    # Control Flow 
    lambda_neg = 10000.
    

    zernikefactors = np.zeros((11,))     
    #zernikefactors[6]=-4.25  # coma X
    #zernikefactors[7]= 4.25 # coma y
    #zernikemask[8]=-0.00  # Trefoil X
    #zernikemask[9]=-0.00 # Trefoil y
    zernikefactors[10]= -1.75 # defocus
    
    #zernikemask=1.*(np.abs(zernikefactors)>0)
    zernikemask = np.ones(zernikemask.shape)
    zernikemask[0:4] = 0 # don't care about defocus, tip, tilt 
    
    shiftIcX = -.00#-3.62768*2
    shiftIcY = .00#3.7690606*2

    # worked with tv= 1e-1, 1e-12, lr: 1e-2 - also in the git!
    zernikefactors = np.array((0.,0.,0.,0.,0.24419421,0.13148846,1.0717771,0.9693622,-2.1087987,1.0321776,-2.6593506))
    shiftIcX = 0.08124803 
    shiftIcY = 0.05606132
elif(1):
    
    '''Droplets recent from Dresden! '''
    
    ''' Hologram Reconstruction Parameters '''
    myFolder = 'W:\\diederichbenedict\\Q-PHASE\\PATRICK\\';
    myFile = 'S0105k_zstack_dz0-2um_25C_40x_Ap0-52_Int63_sameAlignment.tif';
    myDarkFile = 'S0105k_zstack_dz0-2um_25C_40x_Ap0-52_Int63_sameAlignment_dark.tif';
    myBrightfieldFile = 'S0105k_zstack_dz0-2um_25C_40x_Ap0-52_Int63_sameAlignment_bright.tif';
    roi_center = (229,295)
    dz = .2
    

    # data files for parameters and measuremets 
    matlab_val_file = './Data/cells/S0105k_zstack_dz0-2um_25C_40x_Ap0-52_Int63_sameAlignment.tif_allAmp.mat'
    matlab_par_file = './Data/cells/S0105k_zstack_dz0-2um_25C_40x_Ap0-52_Int63_sameAlignment.tifmyParameter.mat'
    matlab_obj_file = './Data/cells/S0105k_zstack_dz0-2um_25C_40x_Ap0-52_Int63_sameAlignment.tif_mysphere.mat'
    matlab_par_name = 'myParameter' 
    matlab_val_name = 'allAmpSimu'   
    mybackgroundval = 0.
    dn = 0.1
    NAc = .52
    is_dampic= .051 # smaller => more damping!
    mysubsamplingIC = 0
    
    
    ''' Control-Parameters - Optimization '''
    my_learningrate = 1e-2  # learning rate
    
    # Regularizer 
    regularizer = 'TV'
    lambda_tv = 1e-0
    lambda_zernike = .01
    myepstvval = 1e-15 ##, 1e-12, 1e-8, 1e-6)) # - 1e-1 # smaller == more blocky
    
    # Control Flow 
    lambda_neg = 0000.
    

    zernikefactors = np.zeros((11,)) 
    

    #zernikefactors[6]=-4.25  # coma X
    #zernikefactors[7]= 4.25 # coma y
    #zernikemask[8]=-0.00  # Trefoil X
    #zernikemask[9]=-0.00 # Trefoil y
    zernikefactors[10]= -1.5 # defocus
    
    #zernikemask=1.*(np.abs(zernikefactors)>0)
    zernikemask = np.ones(zernikemask.shape)
    zernikemask[0:4] = 0 # don't care about defocus, tip, tilt 
    
    shiftIcX = -.00#-3.62768*2
    shiftIcY = .00#3.7690606*2

    # worked with tv= 1e1, 1e-12, lr: 1e-2 - also in the git!
    zernikefactors = 0*np.array((0.,0.,0.,0.,0.24419421,0.13148846,1.0717771,0.9693622,-2.1087987,1.0321776,-2.6593506))
    shiftIcX = 0.#*0.08124803 
    shiftIcY = 0.#*0.05606132
    zernikefactors = np.array((0,0,0,0, -1.2058168e-04, -2.3622499e-03, -7.7374041e-02 ,-1.4900701e-02,  -6.6282146e-04 ,-4.2013789e-04 , -1.2619525e+00))
    
    shiftIcX = 0.0143085485 
    shiftIcY =-0.009161083
    #zernikefactors = 0*np.array((0.,0.,0.,0.,0.49924508,-1.162684,-0.09952152,-0.4380897,-0.6640752,0.16908956,0.860051))
    
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
    shiftIcY =  20# has influence on the YZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
    shiftIcX =  20 # has influence on the XZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
    zernikefactors = np.array((0,0,0,0,0,-0.,-3.0,1.,0,0,.5,.0))  # 7: ComaX, 8: ComaY, 11: Spherical Aberration
    zernikemask = np.array(np.abs(zernikefactors)>0)*1# mask of factors that should be updated
elif(0):
    matlab_val_file = './Data/cells/0014a_zstack_dz0-04um_751planes_20x_every10thslice.tif_allAmp.mat'
    matlab_par_file = './Data/cells/0014a_zstack_dz0-04um_751planes_20x_every10thslice.tifmyParameter.mat'
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
    matlab_par_file = './Data/DROPLETS/S19_multiple/Parameter.mat'; matlab_par_name='myParameter'
    matlab_val_file = './Data/DROPLETS/RESULTS/allAmp_simu.npy'
    mybackgroundval = 0
    NAc = .2
    



