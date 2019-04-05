#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 12:04:55 2019

@author: bene

This file creates a fwd-model for the TESCAN Q-PHASE under 
multiple-scattering. It is majorly derived from  "LEarning approach for optical tomography"
U. S. Kamilov, BIG, EPFL, 2014.
"""
# %load_ext autoreload
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime

# load own functions
import src.model as mus
import src.tf_helper as tf_helper
import src.data as data
import src.tf_regularizers as reg
import src.experiments as experiments 


# Optionally, tweak styles.
mpl.rc('figure',  figsize=(12, 9))
mpl.rc('image', cmap='gray')
#plt.switch_backend('agg')
#np.set_printoptions(threshold=np.nan)


#%%
'''Define some stuff related to infrastructure'''
mytimestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
basepath = './'#'/projectnb/cislidt/diederich/muScat/Multiple-Scattering_Tensorflow/'
resultpath = 'Data/DROPLETS/RESULTS/'


''' Control-Parameters - Optimization '''
my_learningrate = 5e-2  # learning rate
NreduceLR = 500 # when should we reduce the Learningrate? 

# TV-Regularizer 
mylambdatv = 1e-1 ##, 1e-2, 1e-2, 1e-3)) # lambda for Total variation - 1e-1
myepstvval = 1e-10##, 1e-12, 1e-8, 1e-6)) # - 1e-1 # smaller == more blocky

# Positivity Constraint
lambda_neg = 10000

# Displaying/Saving
Niter = 200
Nsave = 25 # write info to disk
Ndisplay = Nsave

# Control Flow 
is_norm = False 
is_aberration = True
is_padding = False 
is_optimization = 1 
is_absorption = False

is_recomputemodel = False # TODO: Make it automatic! 

tf.reset_default_graph()

''' MODELLING StARTS HERE'''

# need to figure out why this holds somehow true - at least produces reasonable results
mysubsamplingIC = 0    
dn = .1
myfac = 1e0# 0*dn*1e-3
myabsnorm = 1e5#myfac


'''START CODE'''
#tf.reset_default_graph() # just in case there was an open session

# Generate Test-Object
''' File which stores the experimental parameters from the Q-PHASE setup 
    1.) Read in the parameters of the dataset ''' 
matlab_pars = data.import_parameters_mat(filename = experiments.matlab_par_file, matname = experiments.matlab_par_name)

''' 2.) Read in the parameters of the dataset ''' 
if(experiments.matlab_val_file.find('mat')==-1):
    matlab_val = np.load(experiments.matlab_val_file)+1j
else:
    matlab_val = data.import_realdata_h5(filename = experiments.matlab_val_file, matname=experiments.matlab_val_name, is_complex=True)
#matlab_val = np.conj(matlab_val)
# Make sure it's radix 2 along Z
if(np.mod(matlab_val.shape[0],2)==1):
    matlab_val = matlab_val[0:matlab_val.shape[0]-1,:,:]
matlab_val = -(matlab_val)# - .6j

''' Create the Model'''
muscat = mus.MuScatModel(matlab_pars, is_optimization=is_optimization)
# Correct some values - just for the puprose of fitting in the RAM
muscat.Nx,muscat.Ny,muscat.Nz = matlab_val.shape[1], matlab_val.shape[2], matlab_val.shape[0]
muscat.shiftIcY=experiments.shiftIcY
muscat.shiftIcX=experiments.shiftIcX
muscat.dn = dn
muscat.dz = .32
muscat.NAc = experiments.NAc

''' Adjust some parameters to fit it in the memory '''
muscat.mysize = (muscat.Nz,muscat.Nx,muscat.Ny) # ordering is (Nillu, Nz, Nx, Ny)

# introduce zernike factors here
muscat.zernikefactors = experiments.zernikefactors
muscat.zernikemask = experiments.zernikemask
#%%
''' Compute a first guess based on the experimental phase '''
obj_guess =  np.zeros(matlab_val.shape)+muscat.nEmbb# np.angle(matlab_val)## 
import src.tf_generate_object as tf_go
mydiameter=5
obj_guess= tf_go.generateObject(mysize=muscat.mysize, obj_dim=1, obj_type ='sphere', diameter = mydiameter, dn = dn, nEmbb = 1.33)#)dn)
#obj_guess = obj_guess+1j*.1*(obj_guess>1.340)

''' Compute the systems model'''
# Compute the System's properties (e.g. Pupil function/Illumination Source, K-vectors, etc.)¶
muscat.computesys(obj=obj_guess, is_padding=is_padding, mysubsamplingIC=mysubsamplingIC, is_compute_psf='BORN', is_dampic=.02)



''' Create Model Instance'''
muscat.computemodel()




#%
print('Convert Data to TF constant')
TF_meas = tf.complex(np.real(matlab_val),np.imag(matlab_val))
TF_meas = tf.cast(TF_meas, tf.complex64)

TF_myres = muscat.computeconvolution(muscat.TF_ASF,True)

#%
print('Start Session')
sess = tf.Session()
sess.run(tf.global_variables_initializer())
myres = sess.run(TF_myres)


plt.figure()    
plt.subplot(231), plt.imshow(np.real((myres))[:,myres.shape[1]//2,:]), plt.colorbar()#, plt.show()
plt.subplot(232), plt.imshow(np.real((myres))[myres.shape[0]//2,:,:]), plt.colorbar()#, plt.show()    
plt.subplot(233), plt.imshow(np.real((myres))[:,:,myres.shape[2]//2]), plt.colorbar()#, plt.show()    
plt.subplot(234), plt.imshow(np.imag((myres))[:     ,myres.shape[1]//2,:]), plt.colorbar()#, plt.show()
plt.subplot(235), plt.imshow(np.imag((myres))[myres.shape[0]//2,:,:]), plt.colorbar()#, plt.show()    
plt.subplot(236), plt.imshow(np.imag((myres))[:,:,myres.shape[2]//2]), plt.colorbar()
plt.savefig('obj_guess.png'), plt.show()    

#%%
''' Compute the ATF '''
if(1):
    #%%
    print('We are precomputing the PSF')
    myATF = sess.run(muscat.TF_ATF)
    myASF = sess.run(muscat.TF_ASF) 
       
    plt.figure()    
    plt.subplot(231), plt.imshow(np.abs(((myATF))**.2)[:,myATF.shape[1]//2,:]), plt.colorbar()#, plt.show()
    plt.subplot(232), plt.imshow(np.abs(((myATF))**.2)[myATF.shape[0]//2,:,:]), plt.colorbar()#, plt.show()    
    plt.subplot(233), plt.imshow(np.abs(((myATF))**.2)[:,:,myATF.shape[2]//2]), plt.colorbar()#, plt.show()    
    plt.subplot(234), plt.imshow(np.abs(((myASF))**.2)[:,myASF.shape[1]//2,:]), plt.colorbar()#, plt.show()
    plt.subplot(235), plt.imshow(np.abs(((myASF))**.2)[myASF.shape[0]//2,:,:]), plt.colorbar()#, plt.show()    
    plt.subplot(236), plt.imshow(np.abs(((myASF))**.2)[:,:,myASF.shape[2]//2]), plt.colorbar()
    plt.savefig('ASF_ATF.png'), plt.show() 
    
    #%
    plt.figure()
    plt.subplot(231),plt.title('real'),plt.imshow(np.real(matlab_val[:,myATF.shape[1]//2,:])),plt.colorbar()
    plt.subplot(232),plt.title('real'),plt.imshow(np.real(matlab_val[myATF.shape[0]//2,:,:])),plt.colorbar()
    plt.subplot(233),plt.title('real'),plt.imshow(np.real(matlab_val[:,:,myATF.shape[2]//2])),plt.colorbar()
    
    plt.subplot(234),plt.title('imag'),plt.imshow(np.imag(matlab_val[:,myATF.shape[1]//2,:])),plt.colorbar()
    plt.subplot(235),plt.title('imag'),plt.imshow(np.imag(matlab_val[myATF.shape[0]//2,:,:])),plt.colorbar()
    plt.subplot(236),plt.title('imag'),plt.imshow(np.imag(matlab_val[:,:,myATF.shape[2]//2])),plt.colorbar()
    
    plt.savefig('obj_GT.png')   

#%%
myobjft = np.fft.fftshift(np.fft.fftn(matlab_val))
plt.figure()    
plt.subplot(231), plt.imshow(np.abs(((myobjft))**.1)[:,myATF.shape[1]//2,:]), plt.colorbar()#, plt.show()
plt.subplot(232), plt.imshow(np.abs(((myobjft))**.1)[myATF.shape[0]//2,:,:]), plt.colorbar()#, plt.show()    
plt.subplot(233), plt.imshow(np.abs(((myobjft))**.1)[:,:,myATF.shape[2]//2]), plt.colorbar()#, plt.show()    
plt.subplot(234), plt.imshow(np.angle(((myobjft))**.1)[:,myASF.shape[1]//2,:]), plt.colorbar()#, plt.show()
plt.subplot(235), plt.imshow(np.angle(((myobjft))**.1)[myASF.shape[0]//2,:,:]), plt.colorbar()#, plt.show()    
plt.subplot(236), plt.imshow(np.angle(((myobjft))**.1)[:,:,myASF.shape[2]//2]), plt.colorbar()
plt.savefig('Obj_Ft.png'), plt.show()    

#%%
print('Start Deconvolution')
alpha_b = np.array((1,2,5,8))
for i in range(5):  
    if i==0:
        alpha_i = (1*10**(-i)*alpha_b)
    else:
        alpha_i = np.concatenate((1*10**(-i)*alpha_b,alpha_i))

#alpha_i = np.array((1e-2,5e-2))
TF_myres = muscat.computedeconv(TF_meas, alpha = 1.)
    
if(1):
    for iteri in range(np.squeeze(alpha_i.shape)): 
        myres = sess.run(TF_myres, feed_dict={muscat.TF_alpha:alpha_i[iteri]})
        print('Start Displaying')
        
        print(alpha_i[iteri])
        #%
        plt.figure()
        plt.subplot(231),plt.imshow(np.real(myres[:,myres.shape[1]//2,:])),plt.colorbar()
        plt.subplot(232),plt.imshow(np.real(myres[myres.shape[0]//2,:,:])),plt.colorbar()
        plt.subplot(233),plt.imshow(np.real(myres[:,:,myres.shape[2]//2])),plt.colorbar()
        
        plt.subplot(234),plt.imshow(np.imag(myres[:,myres.shape[1]//2,:])),plt.colorbar()
        plt.subplot(235),plt.imshow(np.imag(myres[myres.shape[0]//2,:,:])),plt.colorbar()
        plt.subplot(236),plt.imshow(np.imag(myres[:,:,myres.shape[2]//2])),plt.colorbar()
        
        plt.savefig('thikonov_reg_'+str(iteri)+'_'+str(alpha_i[iteri])+'.png')


#%%
if(0):
    #%% Rainers Test with Carrington approach
    print('Start Deconvolution')
    import InverseModelling as im
    TF_myres,_ = im.convolveCutPSFcpx(TF_meas, myASF, maxdim=3)
    matlab_val = sess.run(TF_myres)



#%%
myres = sess.run(TF_myres, feed_dict={muscat.TF_alpha:.01})
tosave = []
tosave.append(np.real(myres))
tosave.append(np.imag(myres))
tosave = np.array(tosave)
np.save('thikonovinvse.npy', myres)
data.export_realdatastack_h5('./thikonov_deconv.h5', 'temp', tosave)


#%
plt.figure()
plt.subplot(231),plt.imshow(np.real(myres[:,myATF.shape[1]//2,:])),plt.colorbar()
plt.subplot(232),plt.imshow(np.real(myres[myATF.shape[0]//2,:,:])),plt.colorbar()
plt.subplot(233),plt.imshow(np.real(myres[:,:,myATF.shape[2]//2])),plt.colorbar()

plt.subplot(234),plt.imshow(np.imag(myres[:,myATF.shape[1]//2,:])),plt.colorbar()
plt.subplot(235),plt.imshow(np.imag(myres[myATF.shape[0]//2,:,:])),plt.colorbar()
plt.subplot(236),plt.imshow(np.imag(myres[:,:,myATF.shape[2]//2])),plt.colorbar()
