#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 12:04:55 2019

@author: bene

This file creates a fwd-model for the TESCAN Q-PHASE under 
multiple-scattering. It is majorly derived from  "LEarning approach for optical tomography"
U. S. Kamilov, BIG, EPFL, 2014.
"""
# %load_ext autoreload
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime

# load own functions
import src.model as mus
import src.tf_helper as tf_helper
import src.data as data
import src.tf_regularizers as reg
import src.experiments as experiments 


# Optionally, tweak styles.
mpl.rc('figure',  figsize=(8, 6))
mpl.rc('image', cmap='gray')
#plt.switch_backend('agg')
#np.set_printoptions(threshold=np.nan)


#%%
'''Define some stuff related to infrastructure'''
mytimestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
basepath = './'#'/projectnb/cislidt/diederich/muScat/Multiple-Scattering_Tensorflow/'
resultpath = 'Data/DROPLETS/RESULTS/'


''' Control-Parameters - Optimization '''
tf.reset_default_graph()

''' MODELLING StARTS HERE'''

# need to figure out why this holds somehow true - at least produces reasonable results
mysubsamplingIC = 0    
dn = .051
myfac = 1e0# 0*dn*1e-3
myabsnorm = 1e5#myfac

''' microscope parameters '''
zernikefactors = 0*np.array((0,0,0,0,0,0,-.01,-.5001,0.01,0.01,.010))  # 7: ComaX, 8: ComaY, 11: Spherical Aberration
zernikemask = np.ones(zernikefactors.shape) #np.array(np.abs(zernikefactors)>0)*1# mask of factors that should be updated

'''START CODE'''
#tf.reset_default_graph() # just in case there was an open session

# Generate Test-Object
''' File which stores the experimental parameters from the Q-PHASE setup 
    1.) Read in the parameters of the dataset ''' 
matlab_pars = data.import_parameters_mat(filename = experiments.matlab_par_file, matname = experiments.matlab_par_name)

''' 2.) Read in the parameters of the dataset ''' 
if(experiments.matlab_val_file.find('mat')==-1):
    matlab_val = np.load(experiments.matlab_val_file)
else:
    matlab_val = data.import_realdata_h5(filename = experiments.matlab_val_file, matname=experiments.matlab_val_name, is_complex=True)

# Make sure it's radix 2 along Z
if(np.mod(matlab_val.shape[0],2)==1):
    matlab_val = matlab_val[0:matlab_val.shape[0]-1,:,:]
#matlab_val = (matlab_val) - .6j

for shiftIcX in range(-25,25,5):
    for shiftIcY in range(-25,25,5):
            tf.reset_default_graph()
            ''' Create the Model'''
            muscat = mus.MuScatModel(matlab_pars, is_optimization=True)
            # Correct some values - just for the puprose of fitting in the RAM
            muscat.Nx,muscat.Ny,muscat.Nz = matlab_val.shape[1], matlab_val.shape[2], matlab_val.shape[0]
            muscat.shiftIcY=shiftIcY
            muscat.shiftIcX=shiftIcX
            muscat.dn = dn
            muscat.NAc = experiments.NAc
            #muscat.dz = muscat.lambda0/2
            
            
            ''' Adjust some parameters to fit it in the memory '''
            muscat.mysize = (muscat.Nz,muscat.Nx,muscat.Ny) # ordering is (Nillu, Nz, Nx, Ny)
            
            # introduce zernike factors here
            muscat.zernikefactors = zernikefactors
            muscat.zernikemask = zernikemask
            
            ''' Compute a first guess based on the experimental phase '''
            obj_guess =  np.zeros(matlab_val.shape)+muscat.nEmbb# np.angle(matlab_val)## 
            
            ''' Compute the systems model'''
            # Compute the System's properties (e.g. Pupil function/Illumination Source, K-vectors, etc.)¶
            muscat.computesys(obj=obj_guess, is_padding=is_padding, mysubsamplingIC=mysubsamplingIC, is_compute_psf='BORN')
            
            ''' Create Model Instance'''
            muscat.computemodel()
            
            
            
            
            #%%
            print('Convert Data to TF constant')
            matlab_val = matlab_val+1j
            TF_meas = tf.complex(np.real(matlab_val),np.imag(matlab_val))
            TF_meas = tf.cast(TF_meas, tf.complex64)
            
            #%%
            print('Start Session')
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            
            ''' Compute the ATF '''
            #print('We are precomputing the PSF')
            #myATF = sess.run(muscat.TF_ATF)
            #myASF = sess.run(muscat.TF_ASF)    
            
            #%%
            print('Start Deconvolution')
            alpha_b = np.array((1,2,5,8))
            for i in range(5):  
                if i==0:
                    alpha_i = (1*10**(-i)*alpha_b)
                else:
                    alpha_i = np.concatenate((1*10**(-i)*alpha_b,alpha_i))
            
            #alpha_i = np.array((1e-2,5e-2))
            TF_myres = muscat.computedeconv(TF_meas, alpha = 1.)
                
            if(1):
                alpha_i[0] = .1
                for iteri in range(1):# in range(np.squeeze(alpha_i.shape)): 
                    myres = sess.run(TF_myres, feed_dict={muscat.TF_alpha:alpha_i[iteri]})
                    print('Start Displaying')
                    #%
                    print(alpha_i[iteri])
                    
                    plt.figure()
                    plt.subplot(231),plt.imshow(np.real(myres[:,myres.shape[1]//2,:])),plt.colorbar()
                    plt.subplot(232),plt.imshow(np.real(myres[myres.shape[0]//2,:,:])),plt.colorbar()
                    plt.subplot(233),plt.imshow(np.real(myres[:,:,myres.shape[2]//2])),plt.colorbar()
                    
                    plt.subplot(234),plt.imshow(np.imag(myres[:,myres.shape[1]//2,:])),plt.colorbar()
                    plt.subplot(235),plt.imshow(np.imag(myres[myres.shape[0]//2,:,:])),plt.colorbar()
                    plt.subplot(236),plt.imshow(np.imag(myres[:,:,myres.shape[2]//2])),plt.colorbar()
                    
                    plt.savefig('thikonov_reg_'+str(iteri)+'_'+str(alpha_i[iteri])+'_shiftIcX_'+str(muscat.shiftIcX)+'_shiftIcY_'+str(muscat.shiftIcY)+'.png')
                
                plt.figure()
                plt.subplot(231),plt.title('real'),plt.imshow(np.real(matlab_val[:,myres.shape[1]//2,:])),plt.colorbar()
                plt.subplot(232),plt.title('real'),plt.imshow(np.real(matlab_val[myres.shape[0]//2,:,:])),plt.colorbar()
                plt.subplot(233),plt.title('real'),plt.imshow(np.real(matlab_val[:,:,myres.shape[2]//2])),plt.colorbar()
                
                plt.subplot(234),plt.title('imag'),plt.imshow(np.imag(matlab_val[:,myres.shape[1]//2,:])),plt.colorbar()
                plt.subplot(235),plt.title('imag'),plt.imshow(np.imag(matlab_val[myres.shape[0]//2,:,:])),plt.colorbar()
                plt.subplot(236),plt.title('imag'),plt.imshow(np.imag(matlab_val[:,:,myres.shape[2]//2])),plt.colorbar()
                
                
                
