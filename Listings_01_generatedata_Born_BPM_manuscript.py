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
import os
import NanoImagingPack as nip

# change the following to %matplotlib notebook for interactive plotting
# %matplotlib inline

# Optionally, tweak styles.
mpl.rc('figure',  figsize=(6, 4))
mpl.rc('image', cmap='gray')
#plt.switch_backend('agg')
                                                    
# load own functions
import src.simulations as experiments 
import src.model as mus
import src.tf_generate_object as tf_go
import src.data as data
import src.MyParameter as paras
import src.tf_helper as tf_helper

os.environ["CUDA_VISIBLE_DEVICES"]='0'    
os.environ["TF_CUDNN_USE_AUTOTUNE"]="0" 


'''Define some stuff related to infrastructure'''
savepath = experiments.mysavepath

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


'''Choose between Born (BORN) or BPM (BPM)'''
psf_model =  'BORN' # MultiSlice
#psf_model =  '3QDPC' # MultiSlice
#psf_model =  'BPM' # 1st Born

'''Choose your microscope type, Brightfield, Darkfield, etc. '''
is_mictype='BF' # BF, DF, DIC, PC
   
# Generate Test-Object
''' File which stores the experimental parameters from the Q-PHASE setup 
    2.) Read in the parameters of the dataset ''' 
myparams = paras.MyParameter()
myparams.loadExperiment(experiments)
myparams.print()

# reset default graph 
tf.reset_default_graph()

''' Create the Model'''
muscat = mus.MuScatModel(myparams, is_optimization=is_optimization)
  
''' Create a 3D Refractive Index Distributaton as a artificial sample'''
mydiameter = 1
objtype = 'sphere'#'cheek100' # 'sphere', 'twosphere', 'slphantom'
if(objtype == 'sphere'):
    obj_real= tf_go.generateObject(mysize=myparams.mysize, obj_dim=np.array((myparams.dz, myparams.dx, myparams.dy)), obj_type ='sphere', diameter = mydiameter, dn = experiments.dn, nEmbb = myparams.nEmbb)#)dn)
    obj_absorption= tf_go.generateObject(mysize=myparams.mysize, obj_dim=np.array((myparams.dz, myparams.dx, myparams.dy)), obj_type ='sphere', diameter = mydiameter, dn = .01, nEmbb = 0.)#)dn)
elif(objtype == 'cheek100'):
    # Fake Cheek-Cell
    matlab_val_file = './Data/PHANTOM/HeLa_cell_mat_obj_100.mat'; matname='HeLa_cell_mat'
    obj_real = data.import_realdata_h5(filename = matlab_val_file, matname=matname)
    obj_absorption = obj_real*0

obj = (obj_real + 1j*obj_absorption)
obj = np.roll(obj,5,0)
#obj = np.roll(obj, shift=5, axis=0)

# introduce zernike factors here
muscat.zernikefactors = experiments.zernikefactors
muscat.zernikemask = experiments.zernikefactors*0

print('Start the TF-session')
sess = tf.Session()#config=tf.ConfigProto(log_device_placement=True))

''' Compute the systems model'''
if psf_model == 'BPM':
    # Define 'BPM' model    
    tf_fwd = muscat.compute_bpm(obj,is_padding=is_padding, mysubsamplingIC=experiments.mysubsamplingIC)    
    
elif psf_model == 'BORN':
    # Define Born Model 
    tf_fwd = muscat.compute_born(obj, is_padding=is_padding, mysubsamplingIC=experiments.mysubsamplingIC, is_precompute_psf=True)


''' Evaluate the model '''
print('Initiliaze Variables')
sess.run(tf.global_variables_initializer())    

# The first call is -unfortunately- very expensive... 
print('Compute Result')  
start = time.time()
myfwd = sess.run(tf_fwd)
end = time.time()
print(end - start)


#nip.v5(muscat.A_input)
#%% display the results
centerslice = myfwd.shape[0]//2
#myfwd = myfwd - 1j
if(psf_model == 'BORN' or psf_model == '3QDPC'):
    #% write Freq-Support to disk
    tf_helper.plot_ASF_ATF(savepath, muscat.myATF, muscat.myASF)
    tf_helper.plot_obj_fft(savepath, myfwd)
#%%
#%% Display Apertures
plt.subplot(131), plt.title('Ic'),plt.imshow(muscat.Ic), plt.colorbar()#, plt.show()
plt.subplot(132), plt.title('Abs Po'),plt.imshow(np.abs(muscat.Po)), plt.colorbar()#, plt.show()
plt.subplot(133), plt.title('Angle Po'),plt.imshow(np.angle(muscat.Po)), plt.colorbar()# plt.show()

#%%
plt.figure()    
plt.subplot(231), plt.title('real XZ'),plt.imshow(np.real(myfwd)[:,myfwd.shape[1]//2,:]), plt.colorbar()#, plt.show()
plt.subplot(232), plt.title('real YZ'),plt.imshow(np.real(myfwd)[:,:,myfwd.shape[2]//2]), plt.colorbar()#, plt.show()
plt.subplot(233), plt.title('real XY'),plt.imshow(np.real(myfwd)[centerslice,:,:]), plt.colorbar()# plt.show()
plt.subplot(234), plt.title('imag XZ'),plt.imshow(np.imag(myfwd)[:,myfwd.shape[1]//2,:]), plt.colorbar()#, plt.show()
plt.subplot(235), plt.title('imag YZ'),plt.imshow(np.imag(myfwd)[:,:,myfwd.shape[2]//2]), plt.colorbar()#, plt.show()
plt.subplot(236), plt.title('imag XY'),plt.imshow(np.imag(myfwd)[centerslice,:,:]), plt.colorbar()
plt.savefig(savepath+'fwd_realimag'+psf_model+'.png'), plt.show()

plt.subplot(231), plt.title('abs XZ'),plt.imshow(np.abs(myfwd)[:,myfwd.shape[1]//2,:]), plt.colorbar()#, plt.show()
plt.subplot(232), plt.title('abs YZ'),plt.imshow(np.abs(myfwd)[:,:,myfwd.shape[2]//2]), plt.colorbar()#, plt.show()
plt.subplot(233), plt.title('abs XY'),plt.imshow(np.abs(myfwd)[centerslice,:,:]), plt.colorbar()# plt.show()
plt.subplot(234), plt.title('angle XZ'),plt.imshow(np.angle(myfwd)[:,myfwd.shape[1]//2,:]), plt.colorbar()#, plt.show()
plt.subplot(235), plt.title('angle YZ'),plt.imshow(np.angle(myfwd)[:,:,myfwd.shape[2]//2]), plt.colorbar()#, plt.show()
plt.subplot(236), plt.title('angle XY'),plt.imshow(np.angle(myfwd)[centerslice,:,:]), plt.colorbar()
plt.savefig(savepath+'fwd_absphase_'+psf_model+'.png'), plt.show()

# plot the object RI distribution
if(0):
    plt.figure()    
    plt.subplot(231), plt.title('obj - real XZ'),plt.imshow(np.real(obj)[:,obj.shape[1]//2,:]), plt.colorbar()#, plt.show()
    plt.subplot(232), plt.title('obj - real YZ'),plt.imshow(np.real(obj)[:,:,obj.shape[2]//2]), plt.colorbar()#, plt.show()
    plt.subplot(233), plt.title('obj - real XY'),plt.imshow(np.real(obj)[centerslice,:,:]), plt.colorbar()# plt.show()
    plt.subplot(234), plt.title('obj - imag XZ'),plt.imshow(np.imag(obj)[:,obj.shape[1]//2,:]), plt.colorbar()#, plt.show()
    plt.subplot(235), plt.title('obj - imag YZ'),plt.imshow(np.imag(obj)[:,:,obj.shape[2]//2]), plt.colorbar()#, plt.show()
    plt.subplot(236), plt.title('obj - imag XY'),plt.imshow(np.imag(obj)[centerslice,:,:]), plt.colorbar()
    plt.savefig(savepath+'obj_'+psf_model+'.png'), plt.show()


#%% save the resultsl
np.save(savepath+'allAmp_simu_'+psf_model+'.npy', myfwd)
np.save(savepath+'myObj_simu_'+psf_model+'.npy', obj)
np.save(savepath+'myPar_simu_'+psf_model+'.npy', myparams)

data.export_realdatastack_h5(savepath+'/obj'+psf_model+'.h5', 'phase, abs', 
                        np.stack((np.real(obj),np.imag(obj)), axis=0))
data.export_realdatastack_h5(savepath+'/myfwd'+psf_model+'.h5', 'real, imag', 
                        np.stack((np.real(myfwd),
                                  np.imag(myfwd)), axis=0))
       
