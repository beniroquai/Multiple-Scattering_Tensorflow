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
mpl.rc('figure',  figsize=(10, 6))
mpl.rc('image', cmap='gray')

# load own functions
import src.model as mus
import src.tf_generate_object as tf_go
import src.tf_helper as tf_helper
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

tf.reset_default_graph()

''' File which stores the experimental parameters from the Q-PHASE setup 
    1.) Read in the parameters of the dataset ''' 
matlab_par_file = './Data/cells/cross_section_10x0.3_hologram.tif_myParameter.mat'
matlab_par_name = 'myParameter'  #'./Data/DROPLETS/myParameterNew.mat';matname='myParameterNew'    #'./Data/DROPLETS/myParameterNew.mat'   
matlab_par_file = './Data/DROPLETS/S14a_multiple/Parameter.mat'; matname='myParameter'
matlab_pars = data.import_parameters_mat(filename = matlab_par_file, matname=matlab_par_name)


print('do we need to flip the data?! -> Observe FFT!!')

dn = .1
mysubsamplingIC = 5

''' Create the Model1 - full'''
muscat = mus.MuScatModel(matlab_pars, is_optimization=is_optimization)
muscats = mus.MuScatModel(matlab_pars, is_optimization=is_optimization)


''' Create a 3D Refractive Index Distributaton as a artificial sample'''
mydiameter = 1
obj = tf_go.generateObject(mysize=muscat.mysize, obj_dim=muscat.dx, obj_type ='sphere', diameter = mydiameter, dn = dn)#)dn)
obj_absorption = tf_go.generateObject(mysize=muscat.mysize, obj_dim=muscat.dx, obj_type ='sphere', diameter = mydiameter, dn = .01)
obj = obj+1j*obj_absorption

muscat.zernikefactors = np.array((0,0,0,0,0,0,-2,0,0,0,-2)) # 7: ComaX, 8: ComaY, 11: Spherical Aberration
muscat.zernikemask = muscat.zernikefactors*0

''' Compute the systems model'''
muscat.computesys(obj, is_padding=is_padding, mysubsamplingIC=mysubsamplingIC)

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

  

#%% display the results
centerslice = myfwd.shape[0]//2
plt.figure()
plt.subplot(231), plt.title('ABS XZ'),plt.imshow(np.abs(muscat.obj)[:,myfwd.shape[1]//2,:]), plt.colorbar()#, plt.show()
plt.subplot(232), plt.title('ABS YZ'),plt.imshow(np.abs(muscat.obj)[:,:,myfwd.shape[2]//2]), plt.colorbar()#, plt.show()
plt.subplot(233), plt.title('ABS XY'),plt.imshow(np.abs(muscat.obj)[myfwd.shape[0]//2,:,:]), plt.colorbar(), plt.show()

plt.figure()    
plt.subplot(141), plt.imshow(np.abs(np.fft.fftshift(np.fft.fftn(myfwd))**.2)[:,myfwd.shape[1]//2,:]), plt.colorbar()#, plt.show()
plt.subplot(142), plt.imshow(np.abs(np.fft.fftshift(np.fft.fftn(myfwd))**.2)[myfwd.shape[0]//2,:,:]), plt.colorbar()#, plt.show()    
plt.subplot(143), plt.imshow(np.abs(np.fft.fftshift(np.fft.fftn(myfwd))**.2)[:,:,myfwd.shape[2]//2]), plt.colorbar()#, plt.show()    
plt.subplot(144), plt.imshow(np.sum(np.abs(np.fft.fftshift(np.fft.fftn(myfwd))**.2),0)), plt.colorbar(), plt.show()    

plt.figure()    
plt.subplot(231), plt.title('ABS XZ'),plt.imshow(np.abs(myfwd)[:,myfwd.shape[1]//2,:]), plt.colorbar()#, plt.show()
plt.subplot(232), plt.title('ABS YZ'),plt.imshow(np.abs(myfwd)[:,:,myfwd.shape[2]//2]), plt.colorbar()#, plt.show()
plt.subplot(233), plt.title('ABS XY'),plt.imshow(np.abs(myfwd)[centerslice ,:,:]), plt.colorbar()# plt.show()
#myfwd=myfwd*np.exp(1j*2)
plt.subplot(234), plt.title('Angle XZ'),plt.imshow(np.angle(myfwd)[:,myfwd.shape[1]//2,:]), plt.colorbar()#, plt.show()
plt.subplot(235), plt.title('Angle YZ'),plt.imshow(np.angle(myfwd)[:,:,myfwd.shape[2]//2]), plt.colorbar()#, plt.show()
plt.subplot(236), plt.title('Angle XY'),plt.imshow(np.angle(myfwd)[centerslice ,:,:]), plt.colorbar(), plt.show()

plt.figure()    
plt.subplot(231), plt.title('muscat.obj Real XZ'),plt.imshow(np.real(muscat.obj)[:,myfwd.shape[1]//2,:]), plt.colorbar()#, plt.show()
plt.subplot(232), plt.title('muscat.obj Real XZ'),plt.imshow(np.real(muscat.obj)[:,:,myfwd.shape[2]//2]), plt.colorbar()#, plt.show()
plt.subplot(233), plt.title('muscat.obj Real XY'),plt.imshow(np.real(muscat.obj)[myfwd.shape[0]//2,:,:]), plt.colorbar()#, plt.show()
plt.subplot(234), plt.title('muscat.obj Imag XZ'),plt.imshow(np.imag(muscat.obj)[:,myfwd.shape[1]//2,:]), plt.colorbar()#, plt.show()
plt.subplot(235), plt.title('muscat.obj Imag XZ'),plt.imshow(np.imag(muscat.obj)[:,:,myfwd.shape[2]//2]), plt.colorbar()#, plt.show()
plt.subplot(236), plt.title('muscat.obj Imag XY'),plt.imshow(np.imag(muscat.obj)[myfwd.shape[0]//2,:,:]), plt.colorbar(), plt.show()

plt.figure()
plt.subplot(231), plt.imshow(np.fft.fftshift(np.angle(sess.run(muscat.TF_Po_aberr))))
plt.subplot(232), plt.imshow((((muscat.Ic))))

#%% save the results
np.save(savepath+'allAmp_simu.npy', myfwd)
data.export_realdata_h5(filename = './Data/DROPLETS/allAmp_simu.mat', matname = 'allAmp_red', data=myfwd)
data.export_realdata_h5(filename = './Data/DROPLETS/mySample.mat', matname = 'mySample', data=np.real(muscat.obj))


#%% Get memory utilization
