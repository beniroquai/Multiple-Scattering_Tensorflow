#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 19:53:32 2017

@author: Bene

This file creates a fwd-model for the TESCAN Q-PHASE under 
multiple-scattering. It is majorly derived from  "LEarning approach for optical tomography"
U. S. Kamilov, BIG, EPFL, 2014.
"""
# %load_ext autoreload
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# change the following to %matplotlib notebook for interactive plotting
# %matplotlib inline

# Optionally, tweak styles.
mpl.rc('figure',  figsize=(10, 6))
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

is_display = True
is_optimization = True 
is_optimization_psf = False
is_flip = False 

tf.reset_default_graph()

''' File which stores the experimental parameters from the Q-PHASE setup 
    1.) Read in the parameters of the dataset ''' 
matlab_par_file = './Data/DROPLETS/myParameterNew.mat'   
matlab_pars = data.import_parameters_mat(filename = matlab_par_file, matname='myParameterNew')

''' 2.) Read in the parameters of the dataset ''' 
matlab_val_file = './Data/DROPLETS/RESULTS/rec.npy'
matlab_val = np.load(matlab_val_file)
#matlab_val = data.import_realdata_h5(filename = matlab_val_file, matname='allAmp_red', is_complex=True)
if(is_flip):
    np_meas = np.flip(matlab_val,0)
else:
    np_meas = matlab_val
        
print('do we need to flip the data?! -> Observe FFT!!')

''' Create the Model'''
muscat = mus.MuScatModel(matlab_pars, is_optimization=is_optimization, is_optimization_psf = is_optimization_psf)
muscat.Nx,muscat.Ny = int(np.squeeze(matlab_pars['Nx'].value)), int(np.squeeze(matlab_pars['Ny'].value))
muscat.shiftIcY=0
muscat.shiftIcX=0
muscat.dn = .05

''' Adjust some parameters to fit it in the memory '''
muscat.mysize = (muscat.Nz,muscat.Nx,muscat.Ny) # ordering is (Nillu, Nz, Nx, Ny)

''' Try to get the OTF of the system'''
myatf, myapsf = muscat.computetrans()

''' Evaluate the FWD Model with an arbitary object'''
obj = tf_go.generateObject(mysize=muscat.mysize, obj_dim=muscat.dx, obj_type ='sphere', diameter = 1, dn = muscat.dn)

''' Compute the systems model'''
muscat.zernikefactors = np.array((0,0,0,0,0,0,.1,0,0))
muscat.computesys(obj, is_zernike=True)
tf_fwd = muscat.computemodel()

''' Evaluate the model '''
sess = tf.Session()
sess.run(tf.global_variables_initializer())
myfwd = sess.run(tf_fwd)
#%%
'''PErform 1st Born as and compare'''
def np_fft3d(image):
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image))) 
def np_ifft3d(image):    
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image)))

myfirstborn = np_ifft3d(np_fft3d(obj)*myatf)
plt.imshow(np.abs(myatf[:,16,:])**.1)
plt.imshow(np.abs(np_fft3d(myfwd)[:,16,:])**.1)
plt.imshow(np.abs((np_fft3d(myfwd)/myatf)[:,16,:])**.1)
plt.imshow(np.abs(wiener_filter[:,16,:])**.1)

#%% invert it using Wiener
for kwiener in np.linspace(0,10000,20):
    wiener_filter = np.conj(myatf) / (np.abs((myatf)**2) + kwiener)
    mywiener =  np.real(np_ifft3d(np_fft3d(myfwd)*wiener_filter)) 
    plt.imshow(np.abs(mywiener[:,16,:])), plt.colorbar(), plt.show()

#%%
if(is_display):     plt.imshow(np.abs(myatf[:,muscat.mysize[1]//2,:]**.1)), plt.colorbar(), plt.show()
if(is_display):     plt.imshow(np.abs(myatf[muscat.mysize[0]//2,:,:]**.1)), plt.colorbar(), plt.show()    
if(is_display):     plt.imshow(np.abs(myatf[:,:,muscat.mysize[2]//2]**.1)), plt.colorbar(), plt.show()    

if(is_display):     plt.imshow(np.abs(myapsf[:,muscat.mysize[1]//2,:]**.1)), plt.colorbar(), plt.show()
if(is_display):     plt.imshow(np.abs(myapsf[muscat.mysize[0]//2,:,:]**.1)), plt.colorbar(), plt.show()    
if(is_display):     plt.imshow(np.abs(myapsf[:,:,muscat.mysize[2]//2]**.1)), plt.colorbar(), plt.show()    


if(is_display): plt.subplot(231), plt.title('ABS XZ'),plt.imshow(np.abs(myfwd)[:,muscat.mysize[1]//2,:]), plt.colorbar()#, plt.show()
if(is_display): plt.subplot(232), plt.title('ABS XZ'),plt.imshow(np.abs(myfwd)[:,:,muscat.mysize[2]//2]), plt.colorbar()#, plt.show()
if(is_display): plt.subplot(233), plt.title('ABS XY'),plt.imshow(np.abs(myfwd)[muscat.mysize[0]//2,:,:]), plt.colorbar()#, plt.show()

if(is_display): plt.subplot(234), plt.title('Angle XZ'),plt.imshow(np.angle(myfwd)[:,muscat.mysize[1]//2,:]), plt.colorbar()#, plt.show()
if(is_display): plt.subplot(235), plt.title('Angle XZ'),plt.imshow(np.angle(myfwd)[:,:,muscat.mysize[2]//2]), plt.colorbar()#, plt.show()
if(is_display): plt.subplot(236), plt.title('Angle XY'),plt.imshow(np.angle(myfwd)[muscat.mysize[0]//2,:,:]), plt.colorbar(), plt.show()

if(is_display): plt.subplot(231), plt.title('ABS XZ'),plt.imshow(np.abs(myfirstborn)[:,muscat.mysize[1]//2,:]), plt.colorbar()#, plt.show()
if(is_display): plt.subplot(232), plt.title('ABS XZ'),plt.imshow(np.abs(myfirstborn)[:,:,muscat.mysize[2]//2]), plt.colorbar()#, plt.show()
if(is_display): plt.subplot(233), plt.title('ABS XY'),plt.imshow(np.abs(myfirstborn)[muscat.mysize[0]//2,:,:]), plt.colorbar()#, plt.show()

if(is_display): plt.subplot(234), plt.title('Angle XZ'),plt.imshow(np.angle(myfirstborn)[:,muscat.mysize[1]//2,:]), plt.colorbar()#, plt.show()
if(is_display): plt.subplot(235), plt.title('Angle XZ'),plt.imshow(np.angle(myfirstborn)[:,:,muscat.mysize[2]//2]), plt.colorbar()#, plt.show()
if(is_display): plt.subplot(236), plt.title('Angle XY'),plt.imshow(np.angle(myfirstborn)[muscat.mysize[0]//2,:,:]), plt.colorbar(), plt.show()

#%% save the results
np.save(savepath+'/rec.npy', myfwd)

data.export_realdata_h5(filename = './Data/DROPLETS/allAmp_simu.mat', matname = 'allAmp_red', data=my_res)
