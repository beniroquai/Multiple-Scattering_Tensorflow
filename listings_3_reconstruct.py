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
import os

# load own functions
import src.model as mus
import src.tf_helper as tf_helper
import src.tf_generate_object as tf_go
import src.data as data

import src.optimization.tf_lossfunctions as loss
import src.optimization.tf_regularizers as reg

from src import tf_helper as tf_helper, tf_generate_object as tf_go, data as data, model as mus

tf.reset_default_graph()

# savepath
savepath = './Data/BEADS/RESULTS/'
is_display = False

'''Define Optimization Parameters'''
my_learningrate = 1e-2  # learning rate
lambda_tv = 1e-2 # lambda for Total variation
lambda_gr = 0 # lambda for Goods Roughness 
lambda_pos = 10
lambda_neg = 10


# Create directory
try: os.mkdir(savepath)
except(FileExistsError): print('Folder exists already')


''' File which stores the experimental parameters from the Q-PHASE setup 
    1.) Read in the parameters of the dataset ''' 
matlab_par_file = './Data/BEADS/Beads_40x_100a_myParameter.mat'   
matlab_pars = data.import_parameters_mat(filename = matlab_par_file)

''' 2.) Read in the parameters of the dataset ''' 
matlab_val_file = './Data/BEADS/Beads_40x_100a_allAmp.mat'   
matlab_val = data.import_realdata_h5(filename = matlab_val_file, matname='allAmpSimu', is_complex=True)
# np_allAmpSimu = data.import_realdata_mat(filename = matlab_val_file)

''' 3.) Load simulated measurements from generator file '''
np_meas = np.load('myres.npy')#np.flip(), 2)
#np_meas = np.flip(np.flip(np_meas,2),1)

''' Create the Model'''
muscat = mus.MuScatModel(matlab_pars, is_optimization=True)

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

#%%


'''Regression + Regularization'''
tf_meas = tf.placeholder(dtype=tf.complex64, shape=muscat.mysize)
             
'''Define Cost-function'''
tf_tvloss = lambda_tv*reg.total_variation(muscat.TF_obj)
tf_posloss = lambda_neg*reg.posiminity(muscat.TF_obj, minval=0)
tf_negloss = lambda_pos*reg.posimaxity(muscat.TF_obj, maxval=.2)  
tf_fidelity = tf.reduce_sum(tf_helper.tf_abssqr(tf_meas - tf_fwd))

tf_loss = tf_fidelity +  tf_negloss + tf_posloss + tf_tvloss

 
          # tf.reduce_sum(tf_helper.tf_abssqr(tf_meas - tf_fwd)) \
          #tf.reduce_mean(1000*(tf.sign(-muscat.TF_obj)+1))
          #\    loss.l2loss(tf_meas, tf_fwd) + my_gr*reg.goods_roughness(muscat.TF_obj)     
# data fidelity
# TV regularization
# Positivity Penalty          
# eventually Goods Roughness reg
          
          
'''Define Optimizer'''
tf_optimizer = tf.train.AdamOptimizer(my_learningrate)
tf_lossop = tf_optimizer.minimize(tf_loss)

#%

#np_meas = np.flip(np_meas,0)
if(is_display): plt.title('XZ'),plt.imshow(np.abs(np_meas)[:,muscat.mysize[1]//2,:]), plt.colorbar(), plt.show()
if(is_display): plt.title('XZ'),plt.imshow(np.abs(np_meas)[:,:,muscat.mysize[2]//2]), plt.colorbar(), plt.show()
if(is_display): plt.title('XY'),plt.imshow(np.abs(np_meas)[muscat.mysize[2]//2,:,:]), plt.colorbar(), plt.show()


# this is the initial guess of the reconstruction
init_guess = np.angle(np_meas) - np.min(np.angle(np_meas))
init_guess = init_guess/np.max(init_guess)*muscat.dn

''' Evaluate the model '''
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.assign(muscat.TF_obj, init_guess)) # assign abs of measurement as initial guess of 

#myres = sess.run(tf_myres, feed_dict={muscat.TF_obj:obj})
#plt.imshow(np.abs(myres[:,50,:]))

'''
TODO: 
- Decaying Learning Rate
- Tensorboard INtegration
- Write Samples to disc
- Load external data in the placeholder
tf_helper.saveHDF5(results, 'Obj_Guess')
muscat.sess.run(tf.assign(muscat.TF_obj, results))
- Why there is this asymmetry? 
'''




#%%
''' Optimize the model '''
print('Start optimizing')
for iterx in range(1,1000):
    # try to optimize
    
    if(not np.mod(iterx, 10)):
        my_opt, my_loss, myres = sess.run([tf_lossop, tf_loss, muscat.TF_obj], feed_dict={tf_meas:np_meas})
        plt.imsave(savepath+'res_xz_'+str(iterx)+'.png', np.squeeze(np.abs(myres[:,muscat.mysize[1]//2,:])))
        plt.imsave(savepath+'res_yz_'+str(iterx)+'.png', np.squeeze(np.abs(myres[:,:,muscat.mysize[2]//2])))
        plt.imsave(savepath+'res_xy_'+str(iterx)+'.png', np.squeeze(np.abs(myres[muscat.mysize[0]//2,:,:]))) 
        print('MY loss: @'+str(iterx)+': ' + str(my_loss))
    else:
        sess.run([tf_lossop], feed_dict={tf_meas:np_meas})
        #plt.imshow(np.abs(myres[:,50,:]))
        
#%% Display the results
myfwd, mymeas, myres = sess.run([tf_fwd, tf_meas, muscat.TF_obj], feed_dict={tf_meas:np_meas})
        
if(is_display): plt.imshow(np.abs(np.fft.fftshift(np.fft.fftn(mymeas))**.2)[:,muscat.mysize[1]//2,:]), plt.colorbar(), plt.show()    
if(is_display): plt.imshow(np.abs(np.fft.fftshift(np.fft.fftn(mymeas))**.2)[muscat.mysize[1]//2,:,:]), plt.colorbar(), plt.show()   
if(is_display): plt.imshow(np.abs(np.fft.fftshift(np.fft.fftn(mymeas))**.2)[:,:,muscat.mysize[1]//2]), plt.colorbar(), plt.show()     
if(is_display): plt.imshow(np.abs(np.fft.fftshift(np.fft.fftn(myfwd))**.2)[:,muscat.mysize[1]//2,:]), plt.colorbar(), plt.show()
if(is_display): plt.imshow(np.abs(np.fft.fftshift(np.fft.fftn(myfwd))**.2)[muscat.mysize[1]//2,:,:]), plt.colorbar(), plt.show()    
if(is_display): plt.imshow(np.abs(np.fft.fftshift(np.fft.fftn(myfwd))**.2)[:,:,muscat.mysize[1]//2]), plt.colorbar(), plt.show()    
tf_helper.saveHDF5(np.abs(np.fft.fftshift(np.fft.fftn(myfwd))), 'FFT3D_FWD')

if(is_display): plt.title('XZ'),plt.imshow(np.abs(myfwd)[:,muscat.mysize[1]//2,:]), plt.colorbar(), plt.show()
if(is_display): plt.title('XZ'),plt.imshow(np.abs(myfwd)[:,:,muscat.mysize[2]//2]), plt.colorbar(), plt.show()
if(is_display): plt.title('XY'),plt.imshow(np.abs(myfwd)[muscat.mysize[2]//2,:,:]), plt.colorbar(), plt.show()

if(is_display): plt.title('XZ'),plt.imshow(np.angle(myfwd)[:,muscat.mysize[1]//2,:]), plt.colorbar(), plt.show()
if(is_display): plt.title('XZ'),plt.imshow(np.angle(myfwd)[:,:,muscat.mysize[2]//2]), plt.colorbar(), plt.show()
if(is_display): plt.title('XY'),plt.imshow(np.angle(myfwd)[muscat.mysize[2]//2,:,:]), plt.colorbar(), plt.show()


myresidual = tf_helper.abssqr(myfwd-mymeas)
if(is_display): plt.title('Residual: XZ'),plt.imshow(myresidual [:,muscat.mysize[1]//2,:]), plt.colorbar(), plt.show()
if(is_display): plt.title('Residual: XZ'),plt.imshow(myresidual [:,:,muscat.mysize[2]//2]), plt.colorbar(), plt.show()
if(is_display): plt.title('Residual: XY'),plt.imshow(myresidual [muscat.mysize[2]//2,:,:]), plt.colorbar(), plt.show()

#%% save the results
np.save(savepath+'/rec.npy', myres)
tf_helper.saveHDF5(myres, 'Obj_Orig')

