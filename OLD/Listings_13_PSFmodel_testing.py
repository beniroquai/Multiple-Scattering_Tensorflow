#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 19:53:32 2017

@author: Bene

This file creates a fwd-model for the TESCAN Q-PHASE under 
multiple-scattering. It is majorly derived from  "LEarning approach for optical tomography"
U. S. Kamilov, BIG, EPFL, 2014.89    """
# %load_ext autoreload
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

# change the following to %matplotlib notebook for interactive plotting
# %matplotlib inline

# Optionally, tweak styles.
mpl.rc('figure',  figsize=(6, 4))
mpl.rc('image', cmap='gray')

# load own functions
import src.model as mus
import src.tf_generate_object as tf_go
import src.data as data

import os
from datetime import datetime


'''Define some stuff related to infrastructure'''
mytimestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
savepath = os.path.join('./Data/DROPLETS/RESULTS/')#, mytimestamp)

# Create directory
try: 
    os.mkdir(savepath)
except(FileExistsError): 
    print('Folder exists already')

# Define parameters 
is_padding = False # better don't do it, some normalization is probably incorrect
is_display = True
is_optimization = False 
is_optimization_psf = False
is_flip = False
is_measurement = False
mysubsamplingIC=0
psf_modell ='corr'#'corr' # None # 'sep'

tf.reset_default_graph()

''' File which stores the experimental parameters from the Q-PHASE setup 
    1.) Read in the parameters of the dataset ''' 
matlab_par_name = 'myParameter'  #'./Data/DROPLETS/myParameterNew.mat';matname='myParameterNew'    #'./Data/DROPLETS/myParameterNew.mat'   
matlab_par_file = './Data/DROPLETS/S19_multiple/Parameter.mat'; matname='myParameter'
matlab_pars = data.import_parameters_mat(filename = matlab_par_file, matname=matlab_par_name)

''' Create the Model'''
muscat = mus.MuScatModel(matlab_pars, is_optimization=is_optimization)
muscat.Nx,muscat.Ny = int(np.squeeze(matlab_pars['Nx'].value)), int(np.squeeze(matlab_pars['Ny'].value))
zernikefactors = np.array((0,0,0,0,0,0,0,0,0.0,0.0,0.0)) # 7: ComaX, 8: ComaY, 11: Spherical Aberration
zernikemask = np.array(np.abs(zernikefactors)>0)*1#!= np.array((0, 0, 0, 0, 0, 0, , 1, 1, 1, 1))# mask which factors should be updated
muscat.shiftIcX = 0 # has influence on the XZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
muscat.shiftIcY = 0 # has influence on the YZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
muscat.NAc = .1
muscat.NAo = .95


dn = .105 #(1.437-1.3326)#/np.pi
myfac = 1 #- 1e-6


#muscat.NAo = .95
#muscat.dz = 0.1625*2#muscat.lambda0/4
#muscat.dy = .2; muscat.dx = muscat.dy#muscat.lambda0/4
#muscat.dx = 0.1560#muscat.lambda0/4
#muscat.Nx = 50; muscat.Ny = 50; muscat.Nz = 50
#muscat.Nx = 64; muscat.Ny = 64; muscat.Nz = 70
#muscat.dz = muscat.lambda0/4
#muscat.Nz = 36

mymatfile = '/Users/bene/Dropbox/Dokumente/Promotion/PROJECTS/BOSTON/MUSCAT/MATLAB/MuScat/mtfs.mat'
matatf = data.import_realdata_h5(filename = mymatfile, matname='myatf', is_complex=True)
matasf = data.import_realdata_h5(filename = mymatfile, matname='myasf', is_complex=True)


''' Adjust some parameters to fit it in the memory '''
muscat.mysize = (muscat.Nz,muscat.Nx,muscat.Ny) # ordering is (Nillu, Nz, Nx, Ny)

''' Create a 3D Refractive Index Distributaton as a artificial sample'''
mydiameter = 5
if(1):
    obj_real= tf_go.generateObject(mysize=muscat.mysize, obj_dim=1, obj_type ='sphere', diameter = mydiameter, dn = dn, nEmbb = muscat.nEmbb)#)dn)
    obj_absorption = tf_go.generateObject(mysize=muscat.mysize, obj_dim=1, obj_type ='sphere', diameter = mydiameter, dn = .0, nEmbb = 0.01)
elif(1):
    obj_real = tf_go.generateObject(mysize=muscat.mysize, obj_dim=1, obj_type ='hollowsphere', diameter = mydiameter, dn = dn, nEmbb = muscat.nEmbb)#)dn)
    obj_absorption = tf_go.generateObject(mysize=muscat.mysize, obj_dim=muscat.dx, obj_type ='sphere', diameter = mydiameter, dn = .0)
elif(0):
    obj_real= tf_go.generateObject(mysize=muscat.mysize, obj_dim=muscat.dx, obj_type ='twosphere', diameter = mydiameter/8, dn = dn)#)dn)
    obj_absorption = tf_go.generateObject(mysize=muscat.mysize, obj_dim=muscat.dx, obj_type ='twosphere', diameter = mydiameter/8, dn = .0)
elif(0):
    obj_real= tf_go.generateObject(mysize=muscat.mysize, obj_dim=muscat.dx, obj_type ='foursphere', diameter = mydiameter/8, dn = dn)#)dn)
    obj_absorption = tf_go.generateObject(mysize=muscat.mysize, obj_dim=muscat.dx, obj_type ='foursphere', diameter = mydiameter/8, dn = .00)
elif(0):
    obj_real= tf_go.generateObject(mysize=muscat.mysize, obj_dim=muscat.dx, obj_type ='eightsphere', diameter = mydiameter/8, dn = dn)#)dn)
    obj_absorption = tf_go.generateObject(mysize=muscat.mysize, obj_dim=muscat.dx, obj_type ='eightsphere', diameter = mydiameter/8, dn = .01)
elif(0):
    # load a neuron
    obj_real= np.load('./Data/NEURON/myneuron_32_32_70.npy')*dn
    obj_absorption = obj_real*0
elif(0):
    # load the 3 bar example 
    obj_real= tf_go.generateObject(mysize=muscat.mysize, obj_dim=muscat.dx, obj_type ='bars', diameter = 3, dn = dn)#)dn)
    obj_absorption = obj_real*0
else:
    # load a phantom
    # obj_real= np.load('./Data/PHANTOM/phantom_64_64_64.npy')*dn
    obj_real =  np.load('./Data/PHANTOM/phantom_50_50_50.npy')*dn+ muscat.nEmbb
    obj_absorption = obj_real*0

obj = obj_real + obj_absorption

# introduce zernike factors here
muscat.zernikefactors = zernikefactors
#muscat.zernikefactors = np.array((0,0,0,0,0,0,.1,-1,0,0,-2)) # 7: ComaX, 8: ComaY, 11: Spherical Aberration
muscat.zernikemask = muscat.zernikefactors*0
#muscat.zernikefactors = np.array((-0.05195263 ,-0.3599817 , -0.08740465,  0.3556992  , 2.9515843 , -1.9670948 ,-0.38435063 , 0.45611984 , 3.68658  )) 
''' Compute the systems model'''
if psf_modell == None:
    #muscat.A_input = muscat.A_input*np.exp(1j*np.random.rand(muscat.A_input.shape[3])*2*np.pi)
    obj = obj - muscat.nEmbb
    muscat.computesys(obj, is_padding=is_padding, mysubsamplingIC=mysubsamplingIC, is_compute_psf=None)
    
    #muscat.A_input = muscat.A_input*np.exp(1j*np.random.rand(muscat.A_input.shape[3])*2*np.pi)
    tf_fwd = muscat.computemodel()
    plt.imshow(muscat.Ic)
    
    
    #%% Display the results
    ''' Evaluate the model '''
    sess = tf.Session()#config=tf.ConfigProto(log_device_placement=True))
    sess.run(tf.global_variables_initializer())
    
    #%% run model and measure memory/time
    start = time.time()
    myfwd = sess.run(tf_fwd)#, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True), run_metadata=run_metadata)
    end = time.time()
    print(end - start)

    
else:
    muscat.computesys(obj, is_padding=is_padding, mysubsamplingIC=mysubsamplingIC, is_compute_psf=psf_modell)

    ''' Create Model Instance'''
    muscat.computemodel()
       
    ''' Define Fwd operator'''
    tf_fwd = muscat.computeconvolution(None, myfac = myfac)

    ''' Evaluate the model '''
    sess = tf.Session()#config=tf.ConfigProto(log_device_placement=True))
    sess.run(tf.global_variables_initializer())
    
    ''' Compute the ATF '''
    myATF = sess.run(muscat.TF_ATF)
    myASF = sess.run(muscat.TF_ASF)

    #%% run model and measure memory/time
    start = time.time()
    myfwd = sess.run(tf_fwd, feed_dict={muscat.TF_ASF_placeholder:myASF,  muscat.TF_obj:obj_real, muscat.TF_obj_absorption:obj_absorption})#, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True), run_metadata=run_metadata)
    end = time.time()
    print(end - start)

#%% display the results
centerslice = myfwd.shape[0]//2
plt.figure()


#%%
if(0):
    plt.figure()    
    plt.subplot(231), plt.title('real XZ'), plt.imshow(np.real(((myASF)))[:,myASF.shape[1]//2,:]), plt.colorbar()#, plt.show()
    plt.subplot(233), plt.title('real XZ'), plt.imshow(np.real(((myASF)))[myASF.shape[0]//2,:,:]), plt.colorbar()#, plt.show()    
    plt.subplot(232), plt.title('real XZ'), plt.imshow(np.real(((myASF)))[:,:,myASF.shape[2]//2]), plt.colorbar()#
    plt.subplot(234), plt.title('imag XZ'), plt.imshow(np.imag(((myASF)))[:,myASF.shape[1]//2,:]), plt.colorbar()#, plt.show()
    plt.subplot(236), plt.title('imag XZ'), plt.imshow(np.imag(((myASF)))[myASF.shape[0]//2,:,:]), plt.colorbar()#, plt.show()    
    plt.subplot(235), plt.title('imag XZ'), plt.imshow(np.imag(((myASF)))[:,:,myASF.shape[2]//2]), plt.colorbar(), plt.show()    
    
    plt.subplot(231), plt.title('real XZ'), plt.imshow(np.real(((myATF))**.2)[:,myASF.shape[1]//2,:]), plt.colorbar()#, plt.show()
    plt.subplot(233), plt.title('real XZ'), plt.imshow(np.real(((myATF))**.2)[myASF.shape[0]//2,:,:]), plt.colorbar()#, plt.show()    
    plt.subplot(232), plt.title('real XZ'), plt.imshow(np.real(((myATF))**.2)[:,:,myASF.shape[2]//2]), plt.colorbar()#
    plt.subplot(234), plt.title('imag XZ'), plt.imshow(np.imag(((myATF))**.2)[:,myASF.shape[1]//2,:]), plt.colorbar()#, plt.show()
    plt.subplot(236), plt.title('imag XZ'), plt.imshow(np.imag(((myATF))**.2)[myASF.shape[0]//2,:,:]), plt.colorbar()#, plt.show()    
    plt.subplot(235), plt.title('imag XZ'), plt.imshow(np.imag(((myATF))**.2)[:,:,myASF.shape[2]//2]), plt.colorbar(), plt.show()    


plt.figure()    

plt.subplot(231), plt.title('real XZ'),plt.imshow(np.real(myfwd)[:,myfwd.shape[1]//2,:]), plt.colorbar()#, plt.show()
plt.subplot(232), plt.title('real YZ'),plt.imshow(np.real(myfwd)[:,:,myfwd.shape[2]//2]), plt.colorbar()#, plt.show()
plt.subplot(233), plt.title('real XY'),plt.imshow(np.real(myfwd)[myfwd.shape[0]//2,:,:]), plt.colorbar()# plt.show()
plt.subplot(234), plt.title('imag XZ'),plt.imshow(np.imag(myfwd)[:,myfwd.shape[1]//2,:]), plt.colorbar()#, plt.show()
plt.subplot(235), plt.title('imag YZ'),plt.imshow(np.imag(myfwd)[:,:,myfwd.shape[2]//2]), plt.colorbar()#, plt.show()
plt.subplot(236), plt.title('imag XY'),plt.imshow(np.imag(myfwd)[myfwd.shape[0]//2,:,:]), plt.colorbar(), plt.show()
plt.subplot(231), plt.title('abs XZ'),plt.imshow(np.abs(myfwd)[:,myfwd.shape[1]//2,:]), plt.colorbar()#, plt.show()
plt.subplot(232), plt.title('abs YZ'),plt.imshow(np.abs(myfwd)[:,:,myfwd.shape[2]//2]), plt.colorbar()#, plt.show()
plt.subplot(233), plt.title('abs XY'),plt.imshow(np.abs(myfwd)[myfwd.shape[0]//2,:,:]), plt.colorbar()# plt.show()
plt.subplot(234), plt.title('angle XZ'),plt.imshow(np.angle(myfwd)[:,myfwd.shape[1]//2,:]), plt.colorbar()#, plt.show()
plt.subplot(235), plt.title('angle YZ'),plt.imshow(np.angle(myfwd)[:,:,myfwd.shape[2]//2]), plt.colorbar()#, plt.show()
plt.subplot(236), plt.title('angle XY'),plt.imshow(np.angle(myfwd)[myfwd.shape[0]//2,:,:]), plt.colorbar(), plt.show()
