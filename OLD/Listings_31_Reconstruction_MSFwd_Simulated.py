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
import os
from datetime import datetime

# load own functions
import src.model as mus
import src.tf_helper as tf_helper
import src.tf_generate_object as tf_go
import src.data as data
import src.tf_regularizers as reg

# Optionally, tweak styles.
mpl.rc('figure',  figsize=(9, 6))
mpl.rc('image', cmap='gray')
#plt.switch_backend('agg')
#np.set_printoptions(threshold=np.nan)


#%%
'''Define some stuff related to infrastructure'''
mytimestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
basepath = './'#'/projectnb/cislidt/diederich/muScat/Multiple-Scattering_Tensorflow/'
resultpath = 'Data/DROPLETS/RESULTS/'


# Define parameters 
is_padding = False 
is_display = True
is_optimization = 1 
is_measurement = True
is_absorption = False
mysubsamplingIC = 0
NspikeLR = 25000 # try to get the system out of some local minima

'''Define Optimization Parameters'''
# these are hyperparameters
my_learningrate = 1e-3  # learning rate
NreduceLR = 2000 # when should we reduce the Learningrate? 

lambda_tv = 1e-4##, 1e-2, 1e-2, 1e-3)) # lambda for Total variation - 1e-1
eps_tv = ((1e-15))##, 1e-12, 1e-8, 1e-6)) # - 1e-1 # smaller == more blocky

# these are fixed parameters
lambda_neg = 10000
Niter = 400

Noptpsf = 0
Nsave =50 # write info to disk
Ndisplay = Nsave


# where is the data stored?
matlab_par_file = './Data/DROPLETS/S19_multiple/Parameter.mat'; matlab_par_name='myParameter'
matlab_val_file = './Data/DROPLETS/RESULTS/allAmp_simu.npy'
mybackgroundval = 0

# need to figure out why this holds somehow true - at least produces reasonable results
dn = .05
myfac = 1e0# 0*dn*1e-3
myabsnorm = 1e5#myfac

np_global_phase = 0.
np_global_abs = 1.

''' microscope parameters '''
NAc = .52
shiftIcY = 0*.8 # has influence on the YZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
shiftIcX = 0*1 # has influence on the XZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
zernikefactors = np.array((0,0,0,0,0,0,-.01,-.5001,0.01,0.01,.010))  # 7: ComaX, 8: ComaY, 11: Spherical Aberration
zernikefactors = 0*np.array(( 0.001, 0.001, 0.01, 0, 0, 0., -3.4e-03,  2.2e-03, 0.001, .001, -1.0e+00))
zernikemask = np.array(np.abs(zernikefactors)>0)*1#!= np.array((0, 0, 0, 0, 0, 0, , 1, 1, 1, 1))# mask which factors should be updated

'''START CODE'''
tf.reset_default_graph() # just in case there was an open session

# Generate Test-Object
''' File which stores the experimental parameters from the Q-PHASE setup 
    1.) Read in the parameters of the dataset ''' 
matlab_pars = data.import_parameters_mat(filename = matlab_par_file, matname=matlab_par_name)

''' 2.) Read in the parameters of the dataset ''' 
if(matlab_val_file.find('mat')==-1):
    matlab_val = np.load(matlab_val_file)
else:
    matlab_val = data.import_realdata_h5(filename = matlab_val_file, matname=matlab_val_name, is_complex=True)

if(np.mod(matlab_val.shape[0],2)==1):
    matlab_val = matlab_val[0:matlab_val.shape[0]-1,:,:]
    
    
matlab_val = matlab_val + mybackgroundval

''' Create the Model'''
muscat = mus.MuScatModel(matlab_pars, is_optimization=is_optimization)
# Correct some values - just for the puprose of fitting in the RAM
muscat.Nx,muscat.Ny,muscat.Nz = matlab_val.shape[1], matlab_val.shape[2], matlab_val.shape[0]
muscat.shiftIcY=shiftIcY
muscat.shiftIcX=shiftIcX
muscat.dn = dn
muscat.NAc = NAc
muscat.dz = muscat.lambda0/2

''' Adjust some parameters to fit it in the memory '''
muscat.mysize = (muscat.Nz,muscat.Nx,muscat.Ny) # ordering is (Nillu, Nz, Nx, Ny)

# introduce zernike factors here
muscat.zernikefactors = zernikefactors
muscat.zernikemask = zernikemask

''' Compute the systems model'''
# Compute the System's properties (e.g. Pupil function/Illumination Source, K-vectors, etc.)¶
mydiameter = 10
obj_real = tf_go.generateObject(mysize=muscat.mysize, obj_dim=muscat.dx, obj_type ='sphere', diameter = mydiameter, dn = dn, nEmbb = muscat.nEmbb)#)dn)
obj_absorption = tf_go.generateObject(mysize=muscat.mysize, obj_dim=muscat.dx, obj_type ='sphere', diameter = mydiameter, dn = .0, nEmbb = 0)
obj = obj_real + 1j*obj_absorption
muscat.computesys(obj=obj, is_padding=is_padding, mysubsamplingIC=mysubsamplingIC, is_compute_psf=None)

''' Create Model Instance'''
muscat.computemodel()
   
sess = tf.Session()
sess.run(tf.global_variables_initializer())
''' Define Fwd operator'''
tf_fwd = muscat.computemodel(is_forcepos=False)


#%%
''' Compute a first guess based on the experimental phase '''
init_guess =  np.angle(matlab_val)**3## np.zeros(matlab_val.shape)+muscat.nEmbb
init_guess = init_guess-np.min(init_guess)
init_guess = dn*init_guess/np.max(init_guess)#+muscat.nEmbb#*dn+1j*.01*np.ones(init_guess.shape)


#%%
'''# Estimate the Phase difference between Measurement and Simulation'''
# This is actually a crucial step and needs to be used with care. We don't want any phase wrappings ! 
'''Numpy to Tensorflow'''
np_meas = matlab_val
np_mean = np.mean(np_meas)
#np_meas = np_meas/np_mean
#print("Mean of the MEasurement is: "+str(np_mean))

'''Define Cost-function'''
# VERY VERY Important to add 0. and 1. - otherwise it gets converted to float!
tf_global_phase = tf.Variable(np_global_phase, tf.float32, name='var_phase') # (0.902339905500412
tf_global_abs = tf.Variable(np_global_abs, tf.float32, name='var_abs') #0.36691132
                           
'''REGULARIZER'''
# Total Variation
print('We are using TV - Regularization')
tf_tvloss = muscat.tf_lambda_tv*reg.Reg_TV(muscat.TF_obj, BetaVals = [muscat.dx,muscat.dy,muscat.dz], epsR=muscat.tf_eps, is_circ = True)  #Alernatively tf_total_variation_regularization # total_variation
tf_tvloss += muscat.tf_lambda_tv*reg.Reg_TV(muscat.TF_obj_absorption, BetaVals = [muscat.dx,muscat.dy,muscat.dz], epsR=muscat.tf_eps, is_circ = True)  #Alernatively tf_total_variation_regularization # total_variation
                                         
'''Negativity Constraint'''                                          
tf_negsqrloss = lambda_neg*reg.Reg_NegSqr(muscat.TF_obj)
tf_negsqrloss += lambda_neg*reg.Reg_NegSqr(muscat.TF_obj_absorption)

# Correc the fwd model - not good here!
if(0):
    tf_norm = tf.complex(tf_global_phase, tf_global_abs)
    tf_fwd_corrected = tf_fwd+tf_norm
else:
    tf_fwd_corrected = (tf_fwd*tf.exp(1j*tf.cast(tf_global_phase, tf.complex64)))/tf.cast(tf_global_abs, tf.complex64)


'''Define Loss-function'''
if(0):
    print('-------> ATTENTION Losstype is L1')
    tf_fidelity = tf.reduce_mean((tf.abs(muscat.tf_meas - tf_fwd_corrected))) # allow a global phase parameter to avoid unwrapping effects
else:
    print('-------> ATTENTION: Losstype is L2')
    tf_fidelity = tf.reduce_mean(tf_helper.tf_abssqr(muscat.tf_meas - tf_fwd_corrected)) # allow a global phase parameter to avoid unwrapping effects
tf_loss = tf_fidelity + tf_tvloss + tf_negsqrloss 

'''Define Optimizer'''
tf_optimizer = tf.train.AdamOptimizer(muscat.tf_learningrate)
tf_lossop_norm = tf_optimizer.minimize(tf_loss, var_list = [tf_global_abs, tf_global_phase])
tf_lossop_obj = tf_optimizer.minimize(tf_loss, var_list = [muscat.TF_obj, tf_global_abs, tf_global_phase])#, muscat.TF_obj_absorption])
tf_lossop_aberr = tf_optimizer.minimize(tf_loss, var_list = [muscat.TF_zernikefactors])
tf_lossop = tf_optimizer.minimize(tf_loss)

''' Initialize the model '''
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Assign the initial guess to the object inside the fwd-model
sess.run(tf.assign(muscat.TF_obj, np.real(init_guess))); # assign abs of measurement as initial guess of 
sess.run(tf.assign(muscat.TF_obj_absorption, np.imag(init_guess))); # assign abs of measurement as initial guess of 



#%%
    
mylambdatv = lambda_tv
myepstvval = eps_tv
   
'''Define some stuff related to infrastructure'''
mytimestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
savepath = basepath + resultpath + mytimestamp + 'tv_' + str(mylambdatv) + '_eps_' +str(myepstvval) + '_' +'Shift_x-'+str(shiftIcX)+'Shift_y-'+str(shiftIcY) + '_' 

# Create directory
try: 
    os.mkdir(savepath)
except(FileExistsError): 
    print('Folder exists already')

# assert some memory 
iter_last = 0
mylosslist = []
myfidelitylist = []
myposlosslist = []
myneglosslist = []
mytvlosslist = []
result_phaselist = []
result_absorptionlist = []
globalphaselist = []
globalabslist = []
myfwdlist = []

#%%
''' Optimize the model '''
print('Start optimizing')


for iterx in range(iter_last,Niter):
    
    # Change the learning rat - experimental
    if iterx == NreduceLR:
        print('Learning Rate has changed by factor of .1')
        my_learningrate = my_learningrate*.1
        
    
        
    # This is for debbugging purposes - writ th result to disk every n-iteration
    if(iterx==0 or not np.mod(iterx, Ndisplay)):
        my_res, my_res_absortpion, my_loss, my_fidelity, my_negloss, my_tvloss, myglobalphase, myglobalabs, myfwd =  \
            sess.run([muscat.TF_obj, muscat.TF_obj_absorption, tf_loss, tf_fidelity, tf_negsqrloss, tf_tvloss, tf_global_phase, tf_global_abs, tf_fwd_corrected], \
                     feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:my_learningrate, muscat.tf_lambda_tv:mylambdatv, muscat.tf_eps:myepstvval}) #, muscat.TF_ _placeholder:myATF
    
    if iterx>5000:
        for iternorm in range(0,1):
            # find the correct normalization parameters first 
            sess.run([tf_lossop_norm], feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:my_learningrate, muscat.tf_lambda_tv:mylambdatv, muscat.tf_eps:myepstvval})
    #else:
        
       # sess.run([tf_lossop_norm], feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:my_learningrate, muscat.tf_lambda_tv:mylambdatv, muscat.tf_eps:myepstvval})
    
            
    if(iterx==0 or not np.mod(iterx, Nsave)):
    
        print('Loss@'+str(iterx)+': ' + str(my_loss) + ' - Fid: '+str(my_fidelity)+', Neg: '+str(my_negloss)+', TV: '+str(my_tvloss)+' G-Phase:'+str(myglobalphase)+' G-ABS: '+str(myglobalabs)) 
        myfwdlist.append(myfwd)
        mylosslist.append(my_loss)
        myfidelitylist.append(my_fidelity)
        myneglosslist.append(my_negloss)
        mytvlosslist.append(my_tvloss)
        result_phaselist.append(my_res)
        result_absorptionlist.append(my_res_absortpion)
        globalphaselist.append(myglobalphase)
        globalabslist.append(myglobalabs) 
        
        # Display recovered Pupil
        plt.figure()
        myzernikes = sess.run(muscat.TF_zernikefactors)
        plt.subplot(131), plt.title('Po Phase'), plt.imshow(np.fft.fftshift(np.angle(sess.run(muscat.TF_Po_aberr)))), plt.colorbar(fraction=0.046, pad=0.04)
        plt.subplot(132), plt.title('Po abs'), plt.imshow(np.fft.fftshift(np.abs(sess.run(muscat.TF_Po_aberr)))), plt.colorbar()
        plt.subplot(133), plt.bar(np.linspace(1, np.squeeze(myzernikes.shape), np.squeeze(myzernikes.shape)), myzernikes, align='center', alpha=0.5)

        
        ''' Save Figures and Parameters '''
        muscat.saveFigures_list(savepath, myfwdlist, mylosslist, myfidelitylist, myneglosslist, mytvlosslist, result_phaselist, result_absorptionlist, 
                              globalphaselist, globalabslist, np_meas, figsuffix='Iter'+str(iterx))
            
    # Alternate between pure object optimization and aberration recovery
    if (iterx>10) & (Noptpsf>0):
        for iterobj in range(Noptpsf*3):
           sess.run([tf_lossop_obj], feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:my_learningrate, muscat.tf_lambda_tv:mylambdatv, muscat.tf_eps:myepstvval})

        for iterpsf in range(Noptpsf):
           sess.run([tf_lossop_aberr], feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:my_learningrate, muscat.tf_lambda_tv:mylambdatv, muscat.tf_eps:myepstvval})
    else:   
        sess.run([tf_lossop_obj], feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:my_learningrate, muscat.tf_lambda_tv:mylambdatv, muscat.tf_eps:myepstvval})



    iter_last = iterx
#%%        
''' Save Figures and Parameters '''
muscat.saveFigures_list(savepath, myfwdlist, mylosslist, myfidelitylist, myneglosslist, mytvlosslist, result_phaselist, result_absorptionlist, 
                              globalphaselist, globalabslist, np_meas, figsuffix='FINAL')

data.export_realdatastack_h5(savepath+'/myrefractiveindex.h5', 'temp', np.array(result_phaselist))
data.export_realdatastack_h5(savepath+'/myrefractiveindex_absorption.h5', 'temp', np.array(result_absorptionlist))
       
print('Zernikes: ' +str(np.real(sess.run(muscat.TF_zernikefactors))))

# backup current script
from shutil import copyfile
import os
src = (os.path.basename(__file__))
copyfile(src, savepath+'/script_bak.py')

#%%
plt.imshow(np.fft.ifftshift(np.angle(sess.run(muscat.TF_Po_aberr))))