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
import matplotlib.pyplot as plt
import h5py 
import scipy.io
import scipy as scipy
import scipy.misc

# load own functions
import src.model as mus
import src.tf_helper as tf_helper
import src.tf_generate_object as tf_go
import src.data as data


tf.reset_default_graph()
is_display = True
''' File which stores the experimental parameters from the Q-PHASE setup 
    1.) Read in the parameters of the dataset ''' 
matlab_par_file = '/Users/bene/Dropbox/Dokumente/Promotion/PROJECTS/BOSTON/MUSCAT/PYTHON/muScat/Data/DROPLETS/myParameterNew.mat'   
matlab_pars = data.import_parameters_mat(filename = matlab_par_file, matname='myParameterNew')

''' 2.) Read in the parameters of the dataset ''' 
matlab_val_file = '/Users/bene/Dropbox/Dokumente/Promotion/PROJECTS/BOSTON/MUSCAT/PYTHON/muScat/Data/DROPLETS/allAmp_red.mat'   
matlab_val = data.import_realdata_h5(filename = matlab_val_file, matname='allAmp_red', is_complex=True)
matlab_val = np.flip(matlab_val,0)

''' Create the Model'''
muscat = mus.MuScatModel(matlab_pars, is_optimization=False)
muscat.Nx,muscat.Ny = int(np.squeeze(matlab_pars['Nx'].value)), int(np.squeeze(matlab_pars['Ny'].value))

# INVERTING THE MISAGLINMENT OF THE SYSTEM! Its consered to be coma and/or shifted optical axis of the illumination in Y-direction!
muscat.shiftIcX = 0 # shifts pupil along X; >0 -> shifts down (influences YZ-Plot)
muscat.shiftIcY = 0 # shifts pupil along Y; >0 -> shifts right (influences XZ-Plot)
muscat.comaX = 0 # introduces Coma in X direction 
muscat.comaY = 0 # introduces Coma in X direction 
muscat.dn = .05
 
''' Adjust some parameters to fit it in the memory '''
muscat.mysize = (muscat.Nz,muscat.Nx,muscat.Ny) # ordering is (Nillu, Nz, Nx, Ny)

''' Create a 3D Refractive Index Distributaton as a artificial sample'''
obj = tf_go.generateObject(mysize=muscat.mysize, obj_dim=muscat.dx, obj_type ='sphere', diameter = 1, dn = muscat.dn)

''' Compute the systems model'''
muscat.computesys(obj, is_zernike=True)
tf_fwd = muscat.computemodel()

plt.subplot(121), plt.title('Ic'), plt.imshow(muscat.Ic), plt.subplot(122), plt.title('Po'),plt.imshow(np.fft.fftshift(np.angle(muscat.Po))), plt.colorbar(), plt.show()

''' Evaluate the model '''
sess = tf.Session()
sess.run(tf.initialize_all_variables())
myres = sess.run(tf_fwd, feed_dict={muscat.TF_obj:obj})


#%% Display results
# add noise
myres_noise = myres# + 0.001*np.random.randn(muscat.Nz,muscat.Nx,muscat.Ny)


# dipslay spectrum
if(is_display):
    plt.subplot(121)
    plt.imshow(np.abs(np.fft.fftshift(np.fft.fftn(myres))**.2)[:,muscat.Nx//2,:]), plt.colorbar()
    plt.subplot(122)
    plt.imshow(np.abs(np.fft.fftshift(np.fft.fftn(matlab_val))**.2)[:,muscat.Nx//2,:]), plt.colorbar(), plt.show()
    plt.subplot(121)
    plt.imshow(np.abs(np.fft.fftshift(np.fft.fftn(myres))**.2)[muscat.Nz//2+1,:,:]), plt.colorbar()
    plt.subplot(122)
    plt.imshow(np.abs(np.fft.fftshift(np.fft.fftn(matlab_val))**.2)[muscat.Nz//2+1,:,:]), plt.colorbar(), plt.show()


if(is_display): 
    plt.subplot(121)
    plt.title('YZ'),plt.imshow(np.angle(myres_noise)[:,muscat.mysize[1]//2,:]), plt.colorbar()
    plt.subplot(122)
    plt.title('Experiment: YZ'),plt.imshow(np.angle(matlab_val)[:,muscat.mysize[1]//2,:]), plt.colorbar(), plt.show()    
    plt.subplot(121)
    plt.title('XZ'),plt.imshow(np.angle(myres_noise)[:,:,muscat.mysize[2]//2]), plt.colorbar()
    plt.subplot(122)
    plt.title('Experiment: XZ'),plt.imshow(np.angle(matlab_val)[:,:,muscat.mysize[2]//2]), plt.colorbar(), plt.show()
    plt.subplot(121)
    plt.title('XY'),plt.imshow(np.angle(myres_noise)[muscat.mysize[0]//2,:,:]), plt.colorbar()
    plt.subplot(122)
    plt.title('Experiment: XY'),plt.imshow(np.angle(matlab_val)[muscat.mysize[0]//2,:,:]), plt.colorbar(), plt.show()
    data.save_timeseries(np.angle(matlab_val), 'droplet_meas_angle')
    data.save_timeseries(np.angle(myres), 'droplet_simu_angle')    

    plt.subplot(121)
    plt.title('YZ'),plt.imshow(np.abs(myres_noise)[:,muscat.mysize[1]//2,:]), plt.colorbar()
    plt.subplot(122)
    plt.title('Experiment: YZ'),plt.imshow(np.abs(matlab_val)[:,muscat.mysize[1]//2,:]), plt.colorbar(), plt.show()    
    plt.subplot(121)
    plt.title('XZ'),plt.imshow(np.abs(myres_noise)[:,:,muscat.mysize[2]//2]), plt.colorbar()
    plt.subplot(122)
    plt.title('Experiment: XZ'),plt.imshow(np.abs(matlab_val)[:,:,muscat.mysize[2]//2]), plt.colorbar(), plt.show()
    plt.subplot(121)
    plt.title('XY'),plt.imshow(np.abs(myres_noise)[muscat.mysize[0]//2,:,:]), plt.colorbar()
    plt.subplot(122)
    plt.title('Experiment: XY'),plt.imshow(np.abs(matlab_val)[muscat.mysize[0]//2,:,:]), plt.colorbar(), plt.show()




# save result 
np.save('myres.npy', myres)
np.save('myobj.npy', obj)
np.save('myres_noisy.npy', myres_noise)
'''
TODO: 
- Decaying Learning Rate
- Tensorboard INtegration
- Write Samples to disc
- Load external data in the placeholder
tf_helper.saveHDF5(results, 'Obj_Guess')
muscat.sess.run(tf.assign(muscat.TF_obj, results))

'''
