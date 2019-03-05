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
mpl.rc('figure',  figsize=(8, 5.5))
mpl.rc('image', cmap='gray')
#plt.switch_backend('agg')
np.set_printoptions(threshold=np.nan)


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

# dropout parameters (experimental)
my_dropout_prob = 1
Ndropout = 20 # apply dropout to the object eery N stps in the optimization
nboundaryz = 0 # Number of pixels where the initial object get's damped at the rim in Z
'''Define Optimization Parameters'''
# these are hyperparameters
my_learningrate = 1e2  # learning rate
NreduceLR = 100 # when should we reduce the Learningrate? 

lambda_tv =((1e2))##, 1e-2, 1e-2, 1e-3)) # lambda for Total variation - 1e-1
eps_tv = ((1e-8))##, 1e-12, 1e-8, 1e-6)) # - 1e-1 # smaller == more blocky
# these are fixed parameters
lambda_neg = 10000
Niter = 3000
Ndisplay = 200
Noptpsf = 1
Nsave = 200 # write info to disk
# data files for parameters and measuremets 
matlab_val_file = './/Data//cells//Cell_20x_100a_150-250.tif_allAmp.mat'
matlab_par_file = './Data//cells//Cell_20x_100a_150-250.tif_myParameter.mat'
matlab_par_name = 'myParameter' 
matlab_val_name = 'allAmpSimu'    
''' microscope parameters '''
zernikefactors = np.array((0,0,0,0,0,0,.1,-1,0,0,-2)) # 7: ComaX, 8: ComaY, 11: Spherical Aberration
zernikefactors = np.array((0. ,        0.          ,0.      ,    0.49638072 , 0.88168746, -1.11872625  ,-0.02625366 ,-0.5901584  ,-1.0360527  , 0.92185229,  0.14297058))
zernikemask = np.array(np.abs(zernikefactors)>0)*1#!= np.array((0, 0, 0, 0, 0, 0, , 1, 1, 1, 1))# mask which factors should be updated
shiftIcY= .75 # has influence on the YZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
shiftIcX= .42 # has influence on the XZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
dn = .03#(1.437-1.3326)#/np.pi
NAc = .3


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

matlab_val = matlab_val[0:36,:,:]

''' Create the Model'''
muscat = mus.MuScatModel(matlab_pars, is_optimization=is_optimization)
# Correct some values - just for the puprose of fitting in the RAM
muscat.Nx,muscat.Ny = int(np.squeeze(matlab_pars['Nx'].value)), int(np.squeeze(matlab_pars['Ny'].value))
muscat.shiftIcY=shiftIcY
muscat.shiftIcX=shiftIcX
muscat.dn = dn
muscat.NAc = NAc
muscat.Nz = matlab_val.shape[0]
muscat.Nx = matlab_val.shape[1]
muscat.Ny = matlab_val.shape[2]
#muscat.dx = muscat.lambda0/4
#muscat.dy = muscat.lambda0/4
#muscat.dz = muscat.lambda0/2

''' Adjust some parameters to fit it in the memory '''
muscat.mysize = (muscat.Nz,muscat.Nx,muscat.Ny) # ordering is (Nillu, Nz, Nx, Ny)

# introduce zernike factors here
muscat.zernikefactors = zernikefactors
muscat.zernikemask = zernikemask

''' Compute the systems model'''
# Compute the System's properties (e.g. Pupil function/Illumination Source, K-vectors, etc.)¶
muscat.computesys(obj=None, is_padding=is_padding, dropout_prob=my_dropout_prob, mysubsamplingIC=mysubsamplingIC, is_compute_psf=True)

''' Create Model Instance'''
muscat.computemodel()
   
''' Define Fwd operator'''
tf_fwd = muscat.computeconvolution(muscat.TF_ATF)

#%%
''' Compute a first guess based on the experimental phase '''
init_guess = np.angle(matlab_val)
init_guess = init_guess-np.min(init_guess)
init_guess = dn*init_guess/np.max(init_guess)+muscat.nEmbb#*dn+1j*.01*np.ones(init_guess.shape)

#%%
'''# Estimate the Phase difference between Measurement and Simulation'''
# This is actually a crucial step and needs to be used with care. We don't want any phase wrappings ! 
'''Numpy to Tensorflow'''
np_meas = matlab_val
np_mean = np.mean((np_meas))
print("Mean of the MEasurement is: "+str(np_mean))

'''Define Cost-function'''
# VERY VERY Important to add 0. and 1. - otherwise it gets converted to float!
tf_global_phase = tf.Variable(2., tf.float32, name='var_phase') # (0.902339905500412
tf_global_abs = tf.Variable(1., tf.float32, name='var_abs') #0.36691132
                           
'''REGULARIZER'''
# Total Variation
print('We are using TV - Regularization')
tf_tvloss = muscat.tf_lambda_tv*reg.Reg_TV(muscat.TF_obj, BetaVals = [muscat.dx,muscat.dy,muscat.dz], epsR=muscat.tf_eps, is_circ = True)  #Alernatively tf_total_variation_regularization # total_variation
tf_tvloss += muscat.tf_lambda_tv*reg.Reg_TV(muscat.TF_obj_absorption, BetaVals = [muscat.dx,muscat.dy,muscat.dz], epsR=muscat.tf_eps, is_circ = True)  #Alernatively tf_total_variation_regularization # total_variation
                                         
'''Negativity Constraint'''                                          
tf_negsqrloss = lambda_neg*reg.Reg_NegSqr(muscat.TF_obj)
tf_negsqrloss += lambda_neg*reg.Reg_NegSqr(muscat.TF_obj_absorption)

# Correc the fwd model - not good here!
tf_fwd_corrected = tf_fwd/tf.cast(tf.abs(tf_global_abs), tf.complex64)*tf.exp(1j*tf.cast(tf_global_phase, tf.complex64))
'''Define Loss-function'''
if(0):
    print('-------> ATTENTION Losstype is L1')
    tf_fidelity = tf.reduce_mean((tf.abs(muscat.tf_meas - tf_fwd_corrected ))) # allow a global phase parameter to avoid unwrapping effects
else:
    print('-------> ATTENTION: Losstype is L2')
    tf_fidelity = tf.reduce_mean(tf_helper.tf_abssqr(muscat.tf_meas/np_mean - tf_fwd_corrected)) # allow a global phase parameter to avoid unwrapping effects
tf_loss = tf_fidelity + tf_tvloss + tf_negsqrloss 

'''Define Optimizer'''
tf_optimizer = tf.train.AdamOptimizer(muscat.tf_learningrate)
tf_lossop_obj_absorption = tf_optimizer.minimize(tf_loss, var_list = [muscat.TF_obj, muscat.TF_obj_absorption, tf_global_abs, tf_global_phase]) # muscat.TF_obj_absorption, 
tf_lossop_obj = tf_optimizer.minimize(tf_loss, var_list = [muscat.TF_obj, tf_global_abs, tf_global_phase]) # muscat.TF_obj_absorption, 
#tf_lossop_aberr = tf_optimizer.minimize(tf_loss, var_list = [muscat.TF_zernikefactors])
tf_lossop = tf_optimizer.minimize(tf_loss, var_list =  [muscat.TF_obj, tf_global_phase, tf_global_abs, muscat.TF_obj_absorption, muscat.TF_zernikefactors])

''' Initialize the model '''
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Assign the initial guess to the object inside the fwd-model
sess.run(tf.assign(muscat.TF_obj, np.real(init_guess))); # assign abs of measurement as initial guess of 
sess.run(tf.assign(muscat.TF_obj_absorption, np.imag(init_guess))); # assign abs of measurement as initial guess of 

''' Compute the ATF '''
if(1):
    print('We are precomputing the PSF')
    myATF = sess.run(muscat.TF_ATF)
    #%%
    #np.save('myATF.npy', myATF)
    #%%
else:
    myATF = np.load('myATF.npy')
#%%
myASF = np.fft.fftshift(np.fft.fftn(myATF))
plt.figure()    
plt.subplot(231), plt.imshow(np.abs(((myATF))**.2)[:,myATF.shape[1]//2,:]), plt.colorbar()#, plt.show()
plt.subplot(232), plt.imshow(np.abs(((myATF))**.2)[myATF.shape[0]//2,:,:]), plt.colorbar()#, plt.show()    
plt.subplot(233), plt.imshow(np.abs(((myATF))**.2)[:,:,myATF.shape[2]//2]), plt.colorbar()#, plt.show()    
plt.subplot(234), plt.imshow(np.abs(((myASF))**.2)[:,myATF.shape[1]//2,:]), plt.colorbar()#, plt.show()
plt.subplot(235), plt.imshow(np.abs(((myASF))**.2)[myATF.shape[0]//2,:,:]), plt.colorbar()#, plt.show()    
plt.subplot(236), plt.imshow(np.abs(((myASF))**.2)[:,:,myATF.shape[2]//2]), plt.colorbar()#, plt.show()    
#%%
    
mylambdatv = lambda_tv
myepstvval = eps_tv
   
'''Define some stuff related to infrastructure'''
mytimestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
savepath = basepath + resultpath + 'tv_' + str(mylambdatv) + '_eps_' +str(myepstvval) + '_' +'Shift_x-'+str(shiftIcX)+'Shift_y-'+str(shiftIcY) + '_' + mytimestamp

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
        
    if(not np.mod(iterx, NspikeLR) and iterx >0):
        print('temprarily changing LR')
        my_learningrate = 1e-1
    else:
        my_learningrate = 1e-3
    

    # Apply Dropout every N-iteration - experimental
    if(not np.mod(iterx, Ndropout) and my_dropout_prob<1):
        dropout_prob = my_dropout_prob 
        print('Applying dropout now!')
    else:
        dropout_prob = 1
        
        
    # This is for debbugging purposes - writ th result to disk every n-iteration
    if(iterx==0 or not np.mod(iterx, Ndisplay)):
        my_res, my_res_absortpion, my_loss, my_fidelity, my_negloss, my_tvloss, myglobalphase, myglobalabs, myfwd =  \
            sess.run([muscat.TF_obj, muscat.TF_obj_absorption, tf_loss, tf_fidelity, tf_negsqrloss, tf_tvloss, tf_global_phase, tf_global_abs, tf_fwd_corrected], \
                     feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:my_learningrate, muscat.tf_lambda_tv:mylambdatv, muscat.tf_eps:myepstvval, muscat.tf_dropout_prob:dropout_prob}) #, muscat.TF_ATF_placeholder:myATF
    
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
        
        ''' Save Figures and Parameters '''
        muscat.saveFigures_list(savepath, myfwdlist, mylosslist, myfidelitylist, myneglosslist, mytvlosslist, result_phaselist, result_absorptionlist, 
                              globalphaselist, globalabslist, np_meas, figsuffix='Iter'+str(iterx))
            
    # Alternate between pure object optimization and aberration recovery
    if False:# (iterx>10) & (Noptpsf>0):
        for aa in range(Noptpsf):
           sess.run([tf_lossop_aberr], feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:my_learningrate, muscat.tf_lambda_tv:mylambdatv, muscat.tf_eps:myepstvval, muscat.tf_dropout_prob:dropout_prob, muscat.TF_ATF_placeholder:myATF})
        for aa in range(Noptpsf):
           sess.run([tf_lossop], feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:my_learningrate, muscat.tf_lambda_tv:mylambdatv, muscat.tf_eps:myepstvval, muscat.tf_dropout_prob:dropout_prob, muscat.TF_ATF_placeholder:myATF})
    else:   
        sess.run([tf_lossop], feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:my_learningrate, muscat.tf_lambda_tv:mylambdatv, muscat.tf_eps:myepstvval, muscat.tf_dropout_prob:dropout_prob})


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

        #
