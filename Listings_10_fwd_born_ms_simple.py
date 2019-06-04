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

tf.reset_default_graph()
'''Choose between Born (BORN) or BPM (BPM)'''
psf_modell =  'BORN' # 1st Born
#psf_modell =  'Born' # MultiSlice
#psf_modell = None
is_mictype='BF' # BF, DF, DIC, PC


tf.reset_default_graph()
# need to figure out why this holds somehow true - at least produces reasonable results
 
   
# Generate Test-Object
''' File which stores the experimental parameters from the Q-PHASE setup 
    2.) Read in the parameters of the dataset ''' 
myparams = paras.MyParameter()
#myparams.loadmat(mymatpath = experiments.matlab_par_file, mymatname = experiments.matlab_par_name)
myparams.print()

myparams.Nz,myparams.Nx,myparams.Ny =  experiments.mysize
myparams.mysize = (myparams.Nz,myparams.Nx,myparams.Ny) # ordering is (Nillu, Nz, Nx, Ny)
myparams.shiftIcY=experiments.shiftIcY
myparams.shiftIcX=experiments.shiftIcX
myparams.dn = experiments.dn
#myparams.NAo = .25
myparams.NAc = .1#experiments.NAc
myparams.NAci = experiments.NAci

''' Create the Model'''
muscat = mus.MuScatModel(myparams, is_optimization=is_optimization)

muscat.zernikefactors = experiments.zernikefactors
muscat.zernikemask = experiments.zernikemask
  
''' Create a 3D Refractive Index Distributaton as a artificial sample'''
mydiameter = 5
objtype = 'sphere';'cheek100' # 'sphere', 'twosphere', 'slphantom'
if(objtype == 'sphere'):
    obj_real= tf_go.generateObject(mysize=myparams.mysize, obj_dim=1, obj_type ='sphere', diameter = mydiameter, dn = experiments.dn, nEmbb = myparams.nEmbb)#)dn)
    obj_absorption = tf_go.generateObject(mysize=myparams.mysize, obj_dim=1, obj_type ='sphere', diameter = mydiameter, dn = .0, nEmbb = 0.0)
elif(0):
    obj_real = tf_go.generateObject(mysize=myparams.mysize, obj_dim=1, obj_type ='hollowsphere', diameter = mydiameter, dn = experiments.dn, nEmbb = myparams.nEmbb)#)dn)
    obj_absorption = tf_go.generateObject(mysize=myparams.mysize, obj_dim=muscat.dx, obj_type ='sphere', diameter = mydiameter, dn = .0)
elif(0):
    obj_real= tf_go.generateObject(mysize=myparams.mysize, obj_dim=1, obj_type ='twosphere', diameter = mydiameter, dn = experiments.dn, nEmbb = myparams.nEmbb)
    obj_absorption = tf_go.generateObject(mysize=myparams.mysize, obj_dim=muscat.dx, obj_type ='twosphere', diameter = mydiameter, dn = .0)
elif(0):
    obj_real= tf_go.generateObject(mysize=myparams.mysize, obj_dim=muscat.dx, obj_type ='foursphere', diameter = mydiameter/8, dn = experiments.dn)#)dn)
    obj_absorption = tf_go.generateObject(mysize=myparams.mysize, obj_dim=muscat.dx, obj_type ='foursphere', diameter = mydiameter/8, dn = .00)
elif(0):
    obj_real= tf_go.generateObject(mysize=myparams.mysize, obj_dim=muscat.dx, obj_type ='eightsphere', diameter = mydiameter/8, dn = experiments.dn)#)dn)
    obj_absorption = tf_go.generateObject(mysize=myparams.mysize, obj_dim=muscat.dx, obj_type ='eightsphere', diameter = mydiameter/8, dn = .01)
elif(0):
    # load a neuron
    obj_real= np.load('./Data/NEURON/myneuron_32_32_70.npy')*experiments.dn
    obj_absorption = obj_real*0
elif(0):
    # load the 3 bar example 
    obj_real= tf_go.generateObject(mysize=myparams.mysize, obj_dim=muscat.dx, obj_type ='bars', diameter = 3, dn = experiments.dn)#)dn)
    obj_absorption = obj_real*0
elif(objtype == 'cheek'):
    # Fake Cheek-Cell
    matlab_val_file = './Data/PHANTOM/HeLa_cell_mat_obj.mat'; matname='HeLa_cell_mat'
    obj_real = data.import_realdata_h5(filename = matlab_val_file, matname=matname)
    obj_absorption = obj_real*0  
elif(objtype == 'cheek100'):
    # Fake Cheek-Cell
    matlab_val_file = './Data/PHANTOM/HeLa_cell_mat_obj_100.mat'; matname='HeLa_cell_mat'
    obj_real = data.import_realdata_h5(filename = matlab_val_file, matname=matname)
    obj_absorption = obj_real*0
elif(objtype == 'slphantom'):
    # load a phantom
    # obj_real= np.load('./Data/PHANTOM/phantom_64_64_64.npy')*dn
    obj_real =  np.load('./Data/PHANTOM/phantom_50_50_50.npy')*experiments.dn+ myparams.nEmbb
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
    tf_fwd = muscat.compute_bpm(obj,is_padding=is_padding, mysubsamplingIC=experiments.mysubsamplingIC)    
    
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

#%% save the resultsl
np.save(savepath+'allAmp_simu.npy', myfwd)
data.export_realdatastack_h5(savepath+'/obj.h5', 'phase, abs', 
                        np.stack((np.real(obj),np.imag(obj)), axis=0))
data.export_realdatastack_h5(savepath+'/myfwd.h5', 'real, imag', 
                        np.stack((np.real(myfwd),
                                  np.imag(myfwd)), axis=0))
       

