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

import src.optimization.tf_lossfunctions as loss
import src.optimization.tf_regularizers as reg

from src import tf_helper as tf_helper, tf_generate_object as tf_go, data as data, model as mus

tf.reset_default_graph()

''' File which stores the experimental parameters from the Q-PHASE setup 
    1.) Read in the parameters of the dataset ''' 
matlab_par_file = './Data/BEADS/Beads_40x_100a_myParameter.mat'   
matlab_pars = data.import_parameters_mat(filename = matlab_par_file)

''' 2.) Read in the parameters of the dataset ''' 
matlab_val_file = './Data/BEADS/Beads_40x_100a_allAmp.mat'   
matlab_val = data.import_realdata_h5(filename = matlab_val_file, matname='allAmpSimu', is_complex=True)
# np_allAmpSimu = data.import_realdata_mat(filename = matlab_val_file)

''' Create the Model'''
muscat = mus.MuScatModel(matlab_pars, is_optimization=False)

''' Adjust some parameters to fit it in the memory '''
#mm.Nz, mm.Nx, mm.Ny = np.shape(np_allAmpSimu)
muscat.Nz=50#int( np.double(np.array(self.myParamter.get('Nz'))))
muscat.Nx=32#np.int(np.floor((2*self.Rsim)/self.dx)+1);
muscat.Ny=32#np.int(np.floor((2*self.Rsim)/self.dy)+1)
muscat.NAc = 0.3
muscat.dz = muscat.lambda0/4
muscat.mysize = (muscat.Nz,muscat.Nx,muscat.Ny) # ordering is (Nillu, Nz, Nx, Ny)

''' Create a 3D Refractive Index Distributaton as a artificial sample'''
obj = tf_go.generateObject(mysize=muscat.mysize, obj_dim=muscat.dx, obj_type ='sphere', diameter = .5, dn = muscat.dn)

''' Compute the systems model'''
muscat.computesys(obj)
tf_fwd = muscat.computemodel()

''' Evaluate the model '''
sess = tf.Session()
sess.run(tf.initialize_all_variables())
myres = sess.run(tf_fwd, feed_dict={muscat.TF_obj:obj})
plt.title('XZ-PLot of the result (magn)'), plt.imshow(np.abs(myres[:,muscat.mysize[1]//2,:])), plt.colorbar(), plt.show()
plt.title('YZ-PLot of the result (angle)'),plt.imshow(np.abs(myres[:,:,muscat.mysize[2]//2])), plt.colorbar(), plt.show()

# dipslay spectrum
plt.imshow(np.abs(np.fft.fftshift(np.fft.fftn(myres))**.2)[:,muscat.Nx//2,:]), plt.colorbar(), plt.show()
plt.imshow(np.abs(np.fft.fftshift(np.fft.fftn(myres))**.2)[muscat.Nz//2+1,:,:]), plt.colorbar(), plt.show()

# save result 
np.save('myres.npy', myres)
'''
TODO: 
- Decaying Learning Rate
- Tensorboard INtegration
- Write Samples to disc
- Load external data in the placeholder
tf_helper.saveHDF5(results, 'Obj_Guess')
muscat.sess.run(tf.assign(muscat.TF_obj, results))

'''
