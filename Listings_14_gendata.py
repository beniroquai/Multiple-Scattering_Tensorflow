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

tf.reset_default_graph()

''' File which stores the experimental parameters from the Q-PHASE setup 
    1.) Read in the parameters of the dataset ''' 
matlab_par_file = './Data/DROPLETS/myParameterNew.mat';matname='myParameterNew'    #'./Data/DROPLETS/myParameterNew.mat'   
matlab_par_file = './Data/DROPLETS/S14a_multiple/Parameter.mat'; matname='myParameter'
matlab_pars = data.import_parameters_mat(filename = matlab_par_file, matname=matname)


print('do we need to flip the data?! -> Observe FFT!!')

''' Create the Model'''
muscat = mus.MuScatModel(matlab_pars, is_optimization=is_optimization)
muscat.Nx,muscat.Ny = int(np.squeeze(matlab_pars['Nx'].value)), int(np.squeeze(matlab_pars['Ny'].value))
#muscat.shiftIcY= 0#*-.75 # has influence on the YZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
#muscat.shiftIcX= 0#*.75 # has influence on the XZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
dn = .05#(1.437-1.3326)
muscat.NAc = .252
muscat.dz = muscat.lambda0
#muscat.NAo = 1
#muscat.dz = muscat.lambda0/2 
#muscat.Nx = 50; muscat.Ny = 50; muscat.Nz = 50
#muscat.Nx = 32; muscat.Ny = 32; muscat.Nz = 70
#muscat.dz = muscat.lambdaM/4

''' Adjust some parameters to fit it in the memory '''
muscat.mysize = (muscat.Nz,muscat.Nx,muscat.Ny) # ordering is (Nillu, Nz, Nx, Ny)

''' Create a 3D Refractive Index Distributaton as a artificial sample'''
mydiameter = 5
if(1):
    obj = tf_go.generateObject(mysize=muscat.mysize, obj_dim=muscat.dx, obj_type ='sphere', diameter = mydiameter, dn = dn)#)dn)
    obj_absorption = tf_go.generateObject(mysize=muscat.mysize, obj_dim=muscat.dx, obj_type ='sphere', diameter = mydiameter, dn = .01)
elif(0):
    obj = tf_go.generateObject(mysize=muscat.mysize, obj_dim=muscat.dx, obj_type ='twosphere', diameter = mydiameter/8, dn = dn)#)dn)
    obj_absorption = tf_go.generateObject(mysize=muscat.mysize, obj_dim=muscat.dx, obj_type ='twosphere', diameter = mydiameter/8, dn = .01)
elif(0):
    obj = tf_go.generateObject(mysize=muscat.mysize, obj_dim=muscat.dx, obj_type ='foursphere', diameter = mydiameter/8, dn = dn)#)dn)
    obj_absorption = tf_go.generateObject(mysize=muscat.mysize, obj_dim=muscat.dx, obj_type ='foursphere', diameter = mydiameter/8, dn = .00)
elif(0):
    obj = tf_go.generateObject(mysize=muscat.mysize, obj_dim=muscat.dx, obj_type ='eightsphere', diameter = mydiameter/8, dn = dn)#)dn)
    obj_absorption = tf_go.generateObject(mysize=muscat.mysize, obj_dim=muscat.dx, obj_type ='eightsphere', diameter = mydiameter/8, dn = .01)
elif(1):
    # load a neuron
    obj = np.load('./Data/NEURON/myneuron_32_32_70.npy')*dn
    obj_absorption = obj*0
elif(0):
    # load the 3 bar example 
    obj = tf_go.generateObject(mysize=muscat.mysize, obj_dim=muscat.dx, obj_type ='bars', diameter = 3, dn = dn)#)dn)
    obj_absorption = obj*0
else:
    # load a phantom
    # obj = np.load('./Data/PHANTOM/phantom_64_64_64.npy')*dn
    obj = np.load('./Data/PHANTOM/phantom_50_50_50.npy')*dn
    obj_absorption = obj*0
        

obj = obj+1j*obj_absorption
#obj = np.roll(obj,-9,0)

#obj = np.load('my_res_cmplx.npy')
# introduce zernike factors here
muscat.zernikefactors = 0*np.array((0,0,0,0,0,0,.1,-1,0,0,-2)) # 7: ComaX, 8: ComaY, 11: Spherical Aberration
muscat.zernikemask = muscat.zernikefactors*0
#muscat.zernikefactors = np.array((-0.05195263 ,-0.3599817 , -0.08740465,  0.3556992  , 2.9515843 , -1.9670948 ,-0.38435063 , 0.45611984 , 3.68658  )) 
''' Compute the systems model'''
muscat.computesys(obj, is_padding=is_padding, mysubsamplingIC=mysubsamplingIC)
#muscat.A_input = muscat.A_input*np.exp(1j*np.random.rand(muscat.A_input.shape[3])*2*np.pi)
tf_fwd = muscat.computemodel()

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
centerslice = 19
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
myfwd=myfwd*np.exp(1j*2)
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
