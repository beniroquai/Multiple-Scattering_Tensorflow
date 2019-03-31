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
import NanoImagingPack as nip

# load own functions
import src.model as mus
import src.tf_generate_object as tf_go
import src.data as data
import src.tf_helper as tf_helper

os.environ["CUDA_VISIBLE_DEVICES"]='0'    
os.environ["TF_CUDNN_USE_AUTOTUNE"]="0" 
# Optionally, tweak styles.
mpl.rc('figure',  figsize=(9, 6))
mpl.rc('image', cmap='gray')
#plt.switch_backend('agg')

#tf.enable_eager_execution()

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


'''Choose between Born (BORN) or BPM (BPM)'''
psf_modell =  'BORN' # 1st Born
#psf_modell =  'BPM' # MultiSlice


#tf.reset_default_graph()

''' File which stores the experimental parameters from the Q-PHASE setup 
    1.) Read in the parameters of the dataset ''' 
matlab_par_name = 'myParameter'  #'./Data/DROPLETS/myParameterNew.mat';matname='myParameterNew'    #'./Data/DROPLETS/myParameterNew.mat'   
matlab_par_file = './Data/DROPLETS/S19_multiple/Parameter.mat'; matname='myParameter'
matlab_pars = data.import_parameters_mat(filename = matlab_par_file, matname=matlab_par_name)


''' Create a 3D Refractive Index Distributaton as a artificial sample
    2.) Read in the virtual sample ''' 
# Fake Cheek-Cell
matlab_val_file = './Data/PHANTOM/HeLa_cell_mat_obj.mat'; matname='HeLa_cell_mat'
obj_real = data.import_realdata_h5(filename = matlab_val_file, matname=matname)

if(0):
    mysize = np.array((52,50,50))
    mydiameter=1
    obj_real= tf_go.generateObject(mysize=mysize, obj_dim=1, obj_type ='sphere', diameter = mydiameter, dn = .02, nEmbb = 1.33)#)dn)
    
#TF_obj = tf.cast(tf.complex(obj_real,obj_real*0), tf.complex64)
TF_obj = tf.cast(tf.placeholder_with_default(obj_real, obj_real.shape), tf.complex64)

# etract scattering subroi-region
mysize = obj_real.shape
mysize_sub = ((32,32,32)); mycenter = ((mysize[0]//2,mysize[1]//2,mysize[1]//2)) # Region around the nuclei
TF_obj_sub = tf_helper.extract((TF_obj-1.33)*2*np.pi, mysize_sub, mycenter)


''' Create the Model'''
muscat = mus.MuScatModel(matlab_pars, is_optimization=is_optimization)
muscat_sub = mus.MuScatModel(matlab_pars, is_optimization=is_optimization)

# some adjustments of global model
muscat.NAc = .4#051
muscat.mysize = mysize # ordering is (Nillu, Nz, Nx, Ny)
muscat.Nx = muscat.mysize[1]; muscat.Ny = muscat.mysize[2]; muscat.Nz = muscat.mysize[0]

# some adjustments of local model
muscat_sub.NAc = muscat.NAc #051
muscat_sub.mysize = mysize_sub # ordering is (Nillu, Nz, Nx, Ny)
muscat_sub.Nx = muscat_sub.mysize[1]; muscat_sub.Ny = muscat_sub.mysize[2]; muscat_sub.Nz = muscat_sub.mysize[0]


print('Start the TF-session')
sess = tf.Session()#config=tf.ConfigProto(log_device_placement=True))

''' Compute the systems model'''
# Define Born Model on global field
tf_fwd_born = muscat.compute_born(TF_obj, is_padding=is_padding, mysubsamplingIC=mysubsamplingIC, is_precompute_psf=True, is_dampic=False)

# Define 'BPM' model on local subroi 
tf_fwd_bpm = muscat_sub.compute_bpm(TF_obj_sub,is_padding=False, mysubsamplingIC=mysubsamplingIC, is_dampic=False)    
print('The subroi is assumed to be in the focus of the PSF, but this is not the same as the one from BORN?!')

sess.run(tf.global_variables_initializer())    
#%%
''' Evaluate the model '''
start = time.time()
myfwd_born = sess.run(tf_fwd_born) # The first call is -unfortunately- very expensive...   
myfwd_bpm = sess.run(tf_fwd_bpm) # The first call is -unfortunately- very expensive...   
end = time.time()
print(end - start)


#% Merge the two models 

if(1):
    myfwd = np.copy(myfwd_born)
    myfwd[mycenter[0] - mysize_sub[0]//2: mycenter[0]+mysize_sub[0]//2,
          mycenter[1] - mysize_sub[1]//2: mycenter[1]+mysize_sub[1]//2,
          mycenter[2] - mysize_sub[2]//2: mycenter[2]+mysize_sub[2]//2] = myfwd_bpm
else:
    myfwd = myfwd_born
    myfwd = myfwd_bpm
#% display the results
centerslice = myfwd.shape[0]//2

if(psf_modell is not 'BPM' and False):
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

#%
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
#np.save(savepath+'allAmp_simu.npy', myfwd)
#data.export_realdata_h5(filename = './Data/DROPLETS/allAmp_simu.mat', matname = 'allAmp_red', data=myfwd)
data.export_realdata_h5(filename = './mySample_real.mat', matname = 'mySample', data=np.real(myfwd))
data.export_realdata_h5(filename = './mySample_imag.mat', matname = 'mySample', data=np.imag(myfwd))