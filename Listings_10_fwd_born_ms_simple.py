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
from datetime import datetime
import os


# change the following to %matplotlib notebook for interactive plotting
# %matplotlib inline

# Optionally, tweak styles.
mpl.rc('figure',  figsize=(9, 6))
mpl.rc('image', cmap='gray')
#plt.switch_backend('agg')

# load own functions
import src.simulations as experiments 
import src.model as mus
import src.tf_generate_object as tf_go
import src.data as data
import src.MyParameter as paras

os.environ["CUDA_VISIBLE_DEVICES"]='0'    
os.environ["TF_CUDNN_USE_AUTOTUNE"]="0" 


'''Define some stuff related to infrastructure'''
savepath = os.path.join('./Data/DROPLETS/RESULTS/')#, mytimestamp)

# Create directory
try: 
    os.mkdir(savepath)
except(FileExistsError): 
    print('Folder exists already')

''' Define parameters '''
is_padding = False # better don't do it, some normalization is probably incorrect
is_display = True
is_optimization = False 
is_optimization_psf = False
is_flip = False
is_measurement = False
mysubsamplingIC=0

tf.reset_default_graph()
'''Choose between Born (BORN) or BPM (BPM)'''
psf_modell =  'BORN' # 1st Born
#psf_modell =  'BPM' # MultiSlice


tf.reset_default_graph()
# need to figure out why this holds somehow true - at least produces reasonable results
mysubsamplingIC = 0    
   
# Generate Test-Object
''' File which stores the experimental parameters from the Q-PHASE setup 
    2.) Read in the parameters of the dataset ''' 
matlab_pars = paras.MyParameter()
matlab_pars.loadmat(mymatpath = experiments.matlab_par_file, mymatname = experiments.matlab_par_name)
matlab_pars.Nz,matlab_pars.Nx,matlab_pars.Ny =  ((128,128,128))#matlab_val.shape
matlab_pars.mysize = (matlab_pars.Nz,matlab_pars.Nx,matlab_pars.Ny) # ordering is (Nillu, Nz, Nx, Ny)
matlab_pars.shiftIcY=experiments.shiftIcY
matlab_pars.shiftIcX=experiments.shiftIcX
matlab_pars.dn = experiments.dn
matlab_pars.NAc = experiments.NAc

''' Create the Model'''
muscat = mus.MuScatModel(matlab_pars, is_optimization=is_optimization)

muscat.zernikefactors = experiments.zernikefactors
muscat.zernikemask = experiments.zernikemask
  
''' Create a 3D Refractive Index Distributaton as a artificial sample'''

mydiameter = 5
if(0):
    obj_real= tf_go.generateObject(mysize=matlab_pars.mysize, obj_dim=1, obj_type ='sphere', diameter = mydiameter, dn = experiments.dn, nEmbb = matlab_pars.nEmbb)#)dn)
    obj_absorption = tf_go.generateObject(mysize=matlab_pars.mysize, obj_dim=1, obj_type ='sphere', diameter = mydiameter, dn = .0, nEmbb = 0.0)
elif(0):
    obj_real = tf_go.generateObject(mysize=matlab_pars.mysize, obj_dim=1, obj_type ='hollowsphere', diameter = mydiameter, dn = experiments.dn, nEmbb = matlab_pars.nEmbb)#)dn)
    obj_absorption = tf_go.generateObject(mysize=matlab_pars.mysize, obj_dim=muscat.dx, obj_type ='sphere', diameter = mydiameter, dn = .0)
elif(0):
    obj_real= tf_go.generateObject(mysize=matlab_pars.mysize, obj_dim=1, obj_type ='twosphere', diameter = mydiameter, dn = experiments.dn, nEmbb = matlab_pars.nEmbb)
    obj_absorption = tf_go.generateObject(mysize=matlab_pars.mysize, obj_dim=muscat.dx, obj_type ='twosphere', diameter = mydiameter, dn = .0)
elif(0):
    obj_real= tf_go.generateObject(mysize=matlab_pars.mysize, obj_dim=muscat.dx, obj_type ='foursphere', diameter = mydiameter/8, dn = experiments.dn)#)dn)
    obj_absorption = tf_go.generateObject(mysize=matlab_pars.mysize, obj_dim=muscat.dx, obj_type ='foursphere', diameter = mydiameter/8, dn = .00)
elif(0):
    obj_real= tf_go.generateObject(mysize=matlab_pars.mysize, obj_dim=muscat.dx, obj_type ='eightsphere', diameter = mydiameter/8, dn = experiments.dn)#)dn)
    obj_absorption = tf_go.generateObject(mysize=matlab_pars.mysize, obj_dim=muscat.dx, obj_type ='eightsphere', diameter = mydiameter/8, dn = .01)
elif(0):
    # load a neuron
    obj_real= np.load('./Data/NEURON/myneuron_32_32_70.npy')*experiments.dn
    obj_absorption = obj_real*0
elif(0):
    # load the 3 bar example 
    obj_real= tf_go.generateObject(mysize=matlab_pars.mysize, obj_dim=muscat.dx, obj_type ='bars', diameter = 3, dn = experiments.dn)#)dn)
    obj_absorption = obj_real*0
elif(1):
    # Fake Cheek-Cell
    matlab_val_file = './Data/PHANTOM/HeLa_cell_mat_obj.mat'; matname='HeLa_cell_mat'
    obj_real = data.import_realdata_h5(filename = matlab_val_file, matname=matname)
    obj_absorption = obj_real*0    
else:
    # load a phantom
    # obj_real= np.load('./Data/PHANTOM/phantom_64_64_64.npy')*dn
    obj_real =  np.load('./Data/PHANTOM/phantom_50_50_50.npy')*experiments.dn+ matlab_pars.nEmbb
    obj_absorption = obj_real*0

obj = (obj_real + obj_absorption)
#obj = np.roll(obj, shift=5, axis=0)

# introduce zernike factors here
muscat.zernikefactors = experiments.zernikefactors
#muscat.zernikefactors = np.array((0,0,0,0,0,0,.1,-1,0,0,-2)) # 7: ComaX, 8: ComaY, 11: Spherical Aberration
muscat.zernikemask = experiments.zernikefactors*0

print('Start the TF-session')
sess = tf.Session()#config=tf.ConfigProto(log_device_placement=True))

''' Compute the systems model'''
if psf_modell == 'BPM':
    # Define 'BPM' model    
    tf_fwd = muscat.compute_bpm(obj,is_padding=is_padding, mysubsamplingIC=mysubsamplingIC)    

else:
    # Define Born Model 
    tf_fwd = muscat.compute_born(obj, is_padding=is_padding, mysubsamplingIC=mysubsamplingIC, is_precompute_psf=True)
    
''' Evaluate the model '''
sess.run(tf.global_variables_initializer())    

# The first call is -unfortunately- very expensive...   
start = time.time()
myfwd = sess.run(tf_fwd)
end = time.time()
print(end - start)

#%% display the results
centerslice = myfwd.shape[0]//2

if(psf_modell is not 'BPM'):
    plt.figure()    
    plt.subplot(231), plt.title('real XZ'), plt.imshow(np.real(((muscat.myASF)))[:,muscat.myASF.shape[1]//2,:]), plt.colorbar()#, plt.show()
    plt.subplot(233), plt.title('real XZ'), plt.imshow(np.real(((muscat.myASF)))[centerslice,:,:]), plt.colorbar()#, plt.show()    
    plt.subplot(232), plt.title('real XZ'), plt.imshow(np.real(((muscat.myASF)))[:,:,muscat.myASF.shape[2]//2]), plt.colorbar()#
    plt.subplot(234), plt.title('imag XZ'), plt.imshow(np.imag(((muscat.myASF)))[:,muscat.myASF.shape[1]//2,:]), plt.colorbar()#, plt.show()
    plt.subplot(236), plt.title('imag XZ'), plt.imshow(np.imag(((muscat.myASF)))[centerslice,:,:]), plt.colorbar()#, plt.show()    
    plt.subplot(235), plt.title('imag XZ'), plt.imshow(np.imag(((muscat.myASF)))[:,:,muscat.myASF.shape[2]//2]), plt.colorbar(), plt.show()    
    
    plt.subplot(231), plt.title('real XZ'), plt.imshow(np.real(((muscat.myATF))**.2)[:,muscat.myASF.shape[1]//2,:]), plt.colorbar()#, plt.show()
    plt.subplot(233), plt.title('real XZ'), plt.imshow(np.real(((muscat.myATF))**.2)[centerslice,:,:]), plt.colorbar()#, plt.show()    
    plt.subplot(232), plt.title('real XZ'), plt.imshow(np.real(((muscat.myATF))**.2)[:,:,muscat.myASF.shape[2]//2]), plt.colorbar()#
    plt.subplot(234), plt.title('imag XZ'), plt.imshow(np.imag(((muscat.myATF))**.2)[:,muscat.myASF.shape[1]//2,:]), plt.colorbar()#, plt.show()
    plt.subplot(236), plt.title('imag XZ'), plt.imshow(np.imag(((muscat.myATF))**.2)[centerslice,:,:]), plt.colorbar()#, plt.show()    
    plt.subplot(235), plt.title('imag XZ'), plt.imshow(np.imag(((muscat.myATF))**.2)[:,:,muscat.myASF.shape[2]//2]), plt.colorbar(), plt.show()    

#%%
plt.figure()    
plt.subplot(231), plt.title('real XZ'),plt.imshow(np.real(myfwd)[:,myfwd.shape[1]//2,:]), plt.colorbar()#, plt.show()
plt.subplot(232), plt.title('real YZ'),plt.imshow(np.real(myfwd)[:,:,myfwd.shape[2]//2]), plt.colorbar()#, plt.show()
plt.subplot(233), plt.title('real XY'),plt.imshow(np.real(myfwd)[centerslice,:,:]), plt.colorbar()# plt.show()
plt.subplot(234), plt.title('imag XZ'),plt.imshow(np.imag(myfwd)[:,myfwd.shape[1]//2,:]), plt.colorbar()#, plt.show()
plt.subplot(235), plt.title('imag YZ'),plt.imshow(np.imag(myfwd)[:,:,myfwd.shape[2]//2]), plt.colorbar()#, plt.show()
plt.subplot(236), plt.title('imag XY'),plt.imshow(np.imag(myfwd)[centerslice,:,:]), plt.colorbar(), plt.show()
plt.subplot(231), plt.title('abs XZ'),plt.imshow(np.abs(myfwd)[:,myfwd.shape[1]//2,:]), plt.colorbar()#, plt.show()
plt.subplot(232), plt.title('abs YZ'),plt.imshow(np.abs(myfwd)[:,:,myfwd.shape[2]//2]), plt.colorbar()#, plt.show()
plt.subplot(233), plt.title('abs XY'),plt.imshow(np.abs(myfwd)[centerslice,:,:]), plt.colorbar()# plt.show()
plt.subplot(234), plt.title('angle XZ'),plt.imshow(np.angle(myfwd)[:,myfwd.shape[1]//2,:]), plt.colorbar()#, plt.show()
plt.subplot(235), plt.title('angle YZ'),plt.imshow(np.angle(myfwd)[:,:,myfwd.shape[2]//2]), plt.colorbar()#, plt.show()
plt.subplot(236), plt.title('angle XY'),plt.imshow(np.angle(myfwd)[centerslice,:,:]), plt.colorbar(), plt.show()

#%% save the resultsl
np.save(savepath+'allAmp_simu.npy', myfwd)
#data.export_realdata_h5(filename = './Data/DROPLETS/allAmp_simu.mat', matname = 'allAmp_red', data=myfwd)
#data.export_realdata_h5(filename = './Data/DROPLETS/mySample.mat', matname = 'mySample', data=np.real(muscat.obj))
