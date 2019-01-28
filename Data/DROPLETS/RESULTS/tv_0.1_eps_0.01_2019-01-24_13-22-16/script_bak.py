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
from os import listdir
from os.path import isfile, join
from shutil import copyfile

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
#plt.switch_backend('TKAgg')




#%%
'''Define some stuff related to infrastructure'''
mytimestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
basepath = './'#'/projectnb/cislidt/diederich/muScat/Multiple-Scattering_Tensorflow/'
resultpath = 'Data/DROPLETS/RESULTS/'

# Define parameters 
is_padding = False 
is_display = True
is_optimization = -1 # 0=keep obj constant, 1=keep obj variable, -1=keep object placeholder
is_optimization_psf = True
is_measurement = True

# data files for parameters and measuremets 
nn_meas_dir = './Data/DROPLETS/S19_multiple/NN_MEAS'     
nn_gt_dir = './Data/DROPLETS/S19_multiple/NN_GT'
matlab_par_file = './Data/DROPLETS/S19_multiple/Parameter.mat'   
matlab_val_name = 'allAmp_red'
matlab_par_name = 'myParameter' 

# get all the datapairs
my_gt_files = []
my_meas_files = []
for f in listdir(nn_gt_dir):
    if isfile(join(nn_gt_dir, f)):
        my_gt_files.append(join(nn_gt_dir, f))
        my_meas_files.append(join(nn_meas_dir, f).replace('sphere', 'subroi'))
    

# microscope parameters
zernikefactors = np.array((0,0,0,0,0,0,0.1,-0.25,0)) # representing the 9 first zernike coefficients in noll-writings 
shiftIcY=-1
shiftIcX=-0
dn = (1.437-1.3326)
NAc = .52

'''START CODE'''
tf.reset_default_graph() # just in case there was an open session

'''Define Optimization Parameters'''
# these are hyperparameters
my_learningrate = 1e-2  # learning rate
lambda_tv =  ((1e-1))##, 1e-2, 1e-3)) # lambda for Total variation - 1e-1
eps_tv = ((1e-2))#, 1e-2, 1)) # - 1e-1

# these are fixed parameters
lambda_neg = 10
Niter = 10
Nepoch = 20
Ndisplay = 5

'''START CODE'''
tf.reset_default_graph() # just in case there was an open session

# Generate Test-Object
''' File which stores the experimental parameters from the Q-PHASE setup 
    1.) Read in the parameters of the dataset ''' 
matlab_pars = data.import_parameters_mat(filename = matlab_par_file, matname=matlab_par_name)


''' Create the Model'''
muscat = mus.MuScatModel(matlab_pars, is_optimization=is_optimization, is_optimization_psf = is_optimization_psf)
muscat.Nx,muscat.Ny = int(np.squeeze(matlab_pars['Nx'].value)), int(np.squeeze(matlab_pars['Ny'].value))
muscat.shiftIcY=shiftIcY
muscat.shiftIcX=shiftIcX
muscat.dn = dn
muscat.NAc = NAc

''' Adjust some parameters to fit it in the memory '''
muscat.mysize = (muscat.Nz,muscat.Nx,muscat.Ny) # ordering is (Nillu, Nz, Nx, Ny)

''' Create a 3D Refractive Index Distributaton as a artificial sample'''
obj = tf_go.generateObject(mysize=muscat.mysize, obj_dim=(muscat.dz,muscat.dx,muscat.dy), obj_type ='sphere', diameter = 1, dn = muscat.dn)
obj = obj+1j*obj

# introduce zernike factors here
muscat.zernikefactors = zernikefactors

# Compute the System's properties (e.g. Pupil function/Illumination Source, K-vectors, etc.)¶
''' Compute the systems model'''
muscat.computesys(obj, is_zernike=True, is_padding=is_padding, dropout_prob=1)

# Generate Computational Graph (fwd model)
tf_fwd = muscat.computemodel()
if(False): # Activate this to test the resnet block in the very last layer
    with tf.variable_scope('res_real', reuse=False):
        tf_fwd_real = muscat.residual_block(tf.real(tf.expand_dims(tf.expand_dims(tf_fwd,3),0)),1,True)
    with tf.variable_scope('res_imag', reuse=False):
        tf_fwd_imag = muscat.residual_block(tf.imag(tf.expand_dims(tf.expand_dims(tf_fwd,3),0)),1,True)
    tf_fwd = tf.squeeze(tf.complex(tf_fwd_real, tf_fwd_imag))


#init_guess = np.load('my_res_cmplx.npy')
#init_guess = np.random.randn(np_meas.shape[0],init_guess.shape[1],init_guess.shape[2])*muscat.dn


# Estimate the Phase difference between Measurement and Simulation
#%%
'''Define Cost-function'''
tf_tvloss = muscat.tf_lambda_tv*reg.Reg_TV(muscat.TF_obj, BetaVals = [muscat.dx,muscat.dy,muscat.dz], epsR=muscat.tf_eps, is_circ = True)  #Alernatively tf_total_variation_regularization # total_variation
tf_tvloss += muscat.tf_lambda_tv*reg.Reg_TV(muscat.TF_obj_absorption, BetaVals = [muscat.dx,muscat.dy,muscat.dz], epsR=muscat.tf_eps, is_circ = True)  #Alernatively tf_total_variation_regularization # total_variation

tf_negsqrloss = lambda_neg*reg.Reg_NegSqr(muscat.TF_obj)
tf_negsqrloss += lambda_neg*reg.Reg_NegSqr(muscat.TF_obj_absorption)
tf_globalphase = tf.Variable(0., tf.float32, name='var_phase')
tf_globalabs = tf.Variable(1., tf.float32, name='var_abs')# 
#tf_fidelity = tf.reduce_sum((tf_helper.tf_abssqr(tf_fwd  - (tf_meas/tf.cast(tf.abs(tf_globalabs), tf.complex64)*tf.exp(1j*tf.cast(tf_globalphase, tf.complex64)))))) # allow a global phase parameter to avoid unwrapping effects
tf_fwd_corrected = tf_fwd/tf.cast(tf.abs(tf_globalabs), tf.complex64)*tf.exp(1j*tf.cast(tf_globalphase, tf.complex64))
tf_fidelity = tf.reduce_mean((tf_helper.tf_abssqr(muscat.tf_meas - tf_fwd_corrected ))) # allow a global phase parameter to avoid unwrapping effects
tf_grads = tf.gradients(tf_fidelity, [muscat.TF_obj])[0]

tf_loss = tf_fidelity +  tf_negsqrloss + tf_tvloss #tf_negloss + tf_posloss + tf_tvloss

'''Define Optimizer'''
tf_optimizer = tf.train.AdamOptimizer(muscat.tf_learningrate)
#tf_optimizer = tf.train.MomentumOptimizer(tf_learningrate, momentum = .9, use_nesterov=True)
#tf_optimizer = tf.train.ProximalGradientDescentOptimizer(tf_learningrate)
#tf_optimizer = tf.train.GradientDescentOptimizer(muscat.tf_learningrate)

tf_lossop = tf_optimizer.minimize(tf_loss)#, var_list = [muscat.TF_obj, muscat.TF_obj_absorption, tf_globalabs, tf_globalphase])

''' Evaluate the model '''
sess = tf.Session()
sess.run(tf.global_variables_initializer())


#%% assert some memory 
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


'''Define some stuff related to infrastructure'''
mytimestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
savepath = basepath + resultpath + 'tv_' + str(lambda_tv) + '_eps_' +str(eps_tv) + '_' + mytimestamp

# Create directory
try: 
    os.mkdir(savepath)
except(FileExistsError): 
    print('Folder exists already')

#%% optimize over the hyperparameters
for i_epoch in range(Nepoch):
    for i_file in range(len(my_meas_files)):
        #    for eps_tv in eps_tv:
        print('Evtl unwrap it!')
        # this is the initial guess of the reconstruction
        ''' 2.) Read in the parameters of the dataset '''
        print('Now feeding file: '+my_meas_files[i_file])
        print('Now feeding file: '+my_gt_files[i_file]) 
        np_meas = data.import_realdata_h5(filename = my_meas_files[i_file], matname='allAmp_red', is_complex=True)
        np_obj = data.import_realdata_h5(filename = my_gt_files[i_file], matname='mysphere', is_complex=False)

        ''' Create a 3D Refractive Index Distributaton as a artificial sample'''
        obj = np_obj*dn
        obj_absorption = np_obj*.01
        obj = obj+1j*obj_absorption
        init_guess = obj
        
        if(False):
            # This won't make much sense here!
            # run the fwd model once 
            '''Numpy to Tensorflow'''
            np_mean = np.mean(np_meas)
    
            my_fwd = sess.run(tf_fwd)    
            myinitphase = np.mean(np.angle(np_meas))-np.mean(np.angle(my_fwd))-1
            print('My Init Phase is :'+str(myinitphase))
            np_meas=np_meas*np.exp(-1j*(myinitphase)) # subtract globaphase - anyway we want to optimize for that, but now the global phase can be assumed to be 0 initally
            myinitabs = np.mean(np.abs(my_fwd))
            print('My Init ABS is :'+str(myinitabs))
            np_meas=np_meas/np.max(np.abs(np_meas))*myinitabs# subtract globaphase - anyway we want to optimize for that, but now the global phase can be assumed to be 0 initally
       
        #%
        ''' Optimize the model '''
        print('Start optimizing')
        for iterx in range(iter_last,Niter):
            if iterx == 100:
                #print('No change in learningrate!')
                my_learningrate = my_learningrate*.1
    
            if(iterx==0 or not np.mod(iterx, Ndisplay)):
                my_res, my_res_absortpion, my_loss, my_fidelity, my_negloss, my_tvloss, myglobalphase, myglobalabs, myfwd =  \
                    sess.run([muscat.TF_obj, muscat.TF_obj_absorption, tf_loss, tf_fidelity, tf_negsqrloss, tf_tvloss, tf_globalphase, tf_globalabs, tf_fwd_corrected], \
                             feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:my_learningrate, muscat.tf_lambda_tv:lambda_tv, muscat.tf_eps:eps_tv,
                                        muscat.TF_obj:np.real(init_guess), muscat.TF_obj_absorption:np.imag(init_guess)})
        
                print('Loss@'+str(iterx)+': ' + str(my_loss) + ' - Fid: '+str(my_fidelity)+', Neg: '+str(my_negloss)+', TV: '+str(my_tvloss)+' G-Phase:'+str(myglobalphase)+' G-ABS: '+str(myglobalabs))        
                mylosslist.append(my_loss)
                myfidelitylist.append(my_fidelity)
                myneglosslist.append(my_negloss)
                mytvlosslist.append(my_tvloss)
                result_phaselist.append(my_res)
                result_absorptionlist.append(my_res_absortpion)
                globalphaselist.append(myglobalphase)
                globalabslist.append(myglobalabs)  
    
            
            # Alternate between pure object optimization and aberration recovery

            sess.run([tf_lossop], feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:my_learningrate, muscat.tf_lambda_tv:lambda_tv, muscat.tf_eps:eps_tv,
                     muscat.TF_obj:np.real(init_guess), muscat.TF_obj_absorption:np.imag(init_guess)})

            my_fwd = sess.run([tf_fwd], feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:my_learningrate, muscat.tf_lambda_tv:lambda_tv, muscat.tf_eps:eps_tv,
                         muscat.TF_obj:np.real(init_guess), muscat.TF_obj_absorption:np.imag(init_guess)})
    
        #%%        
        ''' Save Figures and Parameters '''
        muscat.saveFigures(sess, savepath, tf_fwd_corrected, np_meas, mylosslist, myfidelitylist, myneglosslist, mytvlosslist, globalphaselist, globalabslist, 
                    result_phaselist=None, result_absorptionlist=None, init_guess=init_guess, figsuffix='Epoch_'+str(i_epoch)+'_File_'+str(i_file))
       
        muscat.writeParameterFile(my_learningrate, lambda_tv, eps_tv, filepath = savepath+'/myparameters.yml')
        
        # backup current script
        src = (os.path.basename(__file__))
        copyfile(src, savepath+'/script_bak.py')
        
    
