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
import os
from datetime import datetime

# load own functions
import src.model as mus
import src.tf_generate_object as tf_go
import src.data as data

# change the following to %matplotlib notebook for interactive plotting
# %matplotlib inline

# Optionally, tweak styles.
#mpl.rc('figure',  figsize=(10, 6))
mpl.rc('image', cmap='gray')


'''Define some stuff related to infrastructure'''
mytimestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
savepath = os.path.join('./Data/DROPLETS/RESULTS/')#, mytimestamp)

# Create directory
try: 
    os.mkdir(savepath)
except(FileExistsError): 
    print('Folder exists already')

# Define parameters 
is_padding = False
is_display = True
is_optimization = True 
is_optimization_psf = True


# data files for parameters and measuremets 
matlab_par_file = './Data/DROPLETS/myParameterNew.mat'   

# microscope parameters
zernikefactors = np.array((0,0,0,0,0,0,.5,.5,0)) # representing the 9 first zernike coefficients in noll-writings 
dn = .075 # refractive index of the object (difference)

'''START CODE'''
tf.reset_default_graph() # just in case there was an open session

''' File which stores the experimental parameters from the Q-PHASE setup 
    1.) Read in the parameters of the dataset ''' 

matlab_pars = data.import_parameters_mat(filename = matlab_par_file, matname='myParameterNew')


''' Create the Model'''
muscat = mus.MuScatModel(matlab_pars, is_optimization=is_optimization, is_optimization_psf = is_optimization_psf)
muscat.Nx,muscat.Ny = int(np.squeeze(matlab_pars['Nx'].value)), int(np.squeeze(matlab_pars['Ny'].value))
muscat.shiftIcY=0
muscat.shiftIcX=0
muscat.dn = dn
muscat.Nx = muscat.Nx
muscat.Ny = muscat.Ny
#muscat.NAc =.4
muscat.dz = muscat.lambda0/4
print('Attention: Changed Z-sampling!!')

''' Adjust some parameters to fit it in the memory '''
muscat.mysize = (muscat.Nz,muscat.Nx,muscat.Ny) # ordering is (Nillu, Nz, Nx, Ny)

''' Create a 3D Refractive Index Distributaton as a artificial sample'''
obj = tf_go.generateObject(mysize=muscat.mysize, obj_dim=(muscat.dx, muscat.dx, muscat.dx), obj_type ='sphere', diameter = 4, dn = muscat.dn)
obj = np.roll(np.roll(obj,0,1),-5,2)
plt.title('My object'), plt.imshow(obj[:,16,:]), plt.colorbar(), plt.show()

# introduce zernike factors here
muscat.zernikefactors = zernikefactors
''' Compute the systems model'''
muscat.computesys(obj, is_zernike=True, is_padding=is_padding)
tf_fwd = muscat.computemodel()
   

#%% Display the results
''' Evaluate the model '''
sess = tf.Session()
sess.run(tf.global_variables_initializer())
my_fwd, my_res = sess.run([tf_fwd, muscat.TF_obj])
mysize = my_fwd.shape

'''Display all trainable values'''
variables_names = [v.name for v in tf.trainable_variables()]
values = sess.run(variables_names)
for k, v in zip(variables_names, values):
    print("Variable: ", k)
    print("Shape: ", v.shape)

    
if(is_display): plt.imshow(np.abs(np.fft.fftshift(np.fft.fftn(my_fwd))**.2)[:,mysize[1]//2,:]), plt.colorbar(), plt.show()
if(is_display): plt.imshow(np.abs(np.fft.fftshift(np.fft.fftn(my_fwd))**.2)[mysize[0]//2,:,:]), plt.colorbar(), plt.show()    
if(is_display): plt.imshow(np.abs(np.fft.fftshift(np.fft.fftn(my_fwd))**.2)[:,:,mysize[2]//2]), plt.colorbar(), plt.show()    


if(is_display): plt.subplot(231), plt.title('ABS XZ'),plt.imshow(np.abs(my_fwd)[:,mysize[1]//2,:]), plt.colorbar()#, plt.show()
if(is_display): plt.subplot(232), plt.title('ABS XZ'),plt.imshow(np.abs(my_fwd)[:,:,mysize[2]//2]), plt.colorbar()#, plt.show()
if(is_display): plt.subplot(233), plt.title('ABS XY'),plt.imshow(np.abs(my_fwd)[mysize[0]//2,:,:]), plt.colorbar()#, plt.show()

if(is_display): plt.subplot(234), plt.title('Angle XZ'),plt.imshow(np.angle(my_fwd*np.exp(1j*np.pi))[:,mysize[1]//2,:]), plt.colorbar()#, plt.show()
if(is_display): plt.subplot(235), plt.title('Angle XZ'),plt.imshow(np.angle(my_fwd*np.exp(1j*np.pi))[:,:,mysize[2]//2]), plt.colorbar()#, plt.show()
if(is_display): plt.subplot(236), plt.title('Angle XY'),plt.imshow(np.angle(my_fwd*np.exp(1j*np.pi))[mysize[0]//2,:,:]), plt.colorbar(), plt.show()


#%% save the results
np.save('./Data/DROPLETS/allAmp_simu.npy', my_fwd)
data.export_realdata_h5(filename = './Data/DROPLETS/allAmp_simu.mat', matname = 'allAmp_red', data=my_fwd)

if(is_display): 
    plt.subplot(131), plt.title('Ic'), plt.imshow(muscat.Ic)
    plt.subplot(132), plt.title('Po'),plt.imshow(np.fft.fftshift(np.abs(sess.run(muscat.TF_Po_aberr)))), plt.colorbar()
    plt.subplot(133), plt.title('Po'),plt.imshow(np.fft.fftshift(np.angle(sess.run(muscat.TF_Po_aberr)))), plt.colorbar(), plt.show()
