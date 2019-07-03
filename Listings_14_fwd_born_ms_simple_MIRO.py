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

tf.reset_default_graph()
'''Choose between Born (BORN) or BPM (BPM)'''
psf_modell =  'BPM' # 1st Born
#psf_modell =  'Born' # MultiSlice
#psf_modell = None
is_mictype='BF' # BF, DF, DIC, PC


tf.reset_default_graph()
# need to figure out why this holds somehow true - at least produces reasonable results
 
   
# Generate Test-Object
''' File which stores the experimental parameters from the Q-PHASE setup 
    2.) Read in the parameters of the dataset ''' 
myparams = paras.MyParameter(Nx=128, Ny=128, NAc=.25, Nz=2, dz=3)

#myparams.loadmat(mymatpath = experiments.matlab_par_file, mymatname = experiments.matlab_par_name)
myparams.print()

''' Create the Model'''
muscat = mus.MuScatModel(myparams, is_optimization=is_optimization)
#experiments.zernikefactors = np.array((0,0,0,0, -1.2058168e-04, -2.3622499e-03, -7.7374041e-02 ,-1.4900701e-02,  -6.6282146e-04 ,-4.2013789e-04 , -1.2619525e+00))
    
obj_real = np.zeros(myparams.mysize)
obj_real[0,:,:] = 1.33+.01*tf_go.generateSpeckle(mysize=myparams.Nx, D=20)
obj_real[1,:,:]=1.33+myparams.dn*(nip.rr(mysize=(myparams.Nx,myparams.Ny))<10)*(nip.rr(mysize=(myparams.Nx,myparams.Ny),freq='ftfreq'))
obj_absorption = obj_real*0

obj = (obj_real + 1j*obj_absorption)
#obj = np.roll(obj, shift=5, axis=0)

print('Start the TF-session')
sess = tf.Session()#config=tf.ConfigProto(log_device_placement=True))

''' Compute the systems model'''
if psf_modell == 'BPM':
    # Define 'BPM' model    
    tf_fwd = muscat.compute_bpm(obj,is_padding=is_padding, mysubsamplingIC=myparams.mysubsamplingIC)    
    
elif psf_modell == 'BORN':
    # Define Born Model 
    tf_fwd = muscat.compute_born(obj, is_padding=is_padding, mysubsamplingIC=experiments.mysubsamplingIC, is_precompute_psf=True)
else:
    # This function is a wrapper to compute the Born fwd-model (convolution)
    muscat.computesys(obj, is_padding=False, mysubsamplingIC=experiments.mysubsamplingIC, is_compute_psf='BORN',is_dampic=experiments.is_dampic, is_mictype=is_mictype)
        
    # Create Model Instance
    muscat.computemodel()
    TF_ASF = muscat.TF_ASF

    # Define Fwd operator
    tf_fwd = muscat.computeconvolution(TF_ASF=TF_ASF, is_padding=is_padding)
    
    is_mictype='BF'
''' Evaluate the model '''
print('Initiliaze Variables')
sess.run(tf.global_variables_initializer())    

# The first call is -unfortunately- very expensive... 
print('Compute Result')  
start = time.time()
myfwd = sess.run(tf_fwd)
end = time.time()
print(end - start)

#%% display the results
centerslice = myfwd.shape[0]//2

if(psf_modell is 'BORN'):
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

# Display Apertures
plt.subplot(131), plt.title('Ic'),plt.imshow(muscat.Ic), plt.colorbar()#, plt.show()
plt.subplot(132), plt.title('Abs Po'),plt.imshow(np.fft.fftshift(np.abs(muscat.Po))), plt.colorbar()#, plt.show()
plt.subplot(133), plt.title('Angle Po'),plt.imshow(np.fft.fftshift(np.angle(muscat.Po))), plt.colorbar()# plt.show()

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

# plot the object RI distribution
plt.figure()    
plt.subplot(231), plt.title('obj - real XZ'),plt.imshow(np.real(obj)[:,obj.shape[1]//2,:]), plt.colorbar()#, plt.show()
plt.subplot(232), plt.title('obj - real YZ'),plt.imshow(np.real(obj)[:,:,obj.shape[2]//2]), plt.colorbar()#, plt.show()
plt.subplot(233), plt.title('obj - real XY'),plt.imshow(np.real(obj)[centerslice,:,:]), plt.colorbar()# plt.show()
plt.subplot(234), plt.title('obj - imag XZ'),plt.imshow(np.imag(obj)[:,obj.shape[1]//2,:]), plt.colorbar()#, plt.show()
plt.subplot(235), plt.title('obj - imag YZ'),plt.imshow(np.imag(obj)[:,:,obj.shape[2]//2]), plt.colorbar()#, plt.show()
plt.subplot(236), plt.title('obj - imag XY'),plt.imshow(np.imag(obj)[centerslice,:,:]), plt.colorbar(), plt.show()


#%% save the resultsl
np.save(savepath+'allAmp_simu.npy', myfwd)
data.export_realdatastack_h5(savepath+'/obj.h5', 'phase, abs', 
                        np.stack((np.real(obj),np.imag(obj)), axis=0))
data.export_realdatastack_h5(savepath+'/myfwd.h5', 'real, imag', 
                        np.stack((np.real(myfwd),
                                  np.imag(myfwd)), axis=0))
       

