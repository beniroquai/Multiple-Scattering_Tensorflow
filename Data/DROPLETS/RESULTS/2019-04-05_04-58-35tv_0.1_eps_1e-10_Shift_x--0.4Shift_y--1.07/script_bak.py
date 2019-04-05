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
import src.data as data
import src.tf_regularizers as reg
import src.experiments as experiments 

import NanoImagingPack as nip

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


''' Control-Parameters - Optimization '''
my_learningrate = 5e-2  # learning rate
NreduceLR = 1000 # when should we reduce the Learningrate? 

# TV-Regularizer 
mylambdatv = 1e-1#1e1 ##, 1e-2, 1e-2, 1e-3)) # lambda for Total variation - 1e-1
myepstvval = 1e-10##, 1e-12, 1e-8, 1e-6)) # - 1e-1 # smaller == more blocky

# Positivity Constraint
lambda_neg = 1000.

# Displaying/Saving
Niter = 1000
Nsave = 50 # write info to disk
Ndisplay = Nsave

# Control Flow 
is_norm = False 
is_aberration = True
is_padding = False
is_optimization = True
is_absorption = True
is_obj_init_tikhonov = True

is_recomputemodel = True # TODO: Make it automatic! 




''' MODELLING StARTS HERE'''
if is_recomputemodel:
    tf.reset_default_graph()
    # need to figure out why this holds somehow true - at least produces reasonable results
    mysubsamplingIC = 0    
    dn = experiments.dn

    '''START CODE'''
    #tf.reset_default_graph() # just in case there was an open session
    
    # Generate Test-Object
    ''' File which stores the experimental parameters from the Q-PHASE setup 
        1.) Read in the parameters of the dataset ''' 
    matlab_pars = data.import_parameters_mat(filename = experiments.matlab_par_file, matname = experiments.matlab_par_name)
    
    ''' 2.) Read in the parameters of the dataset ''' 
    if(experiments.matlab_val_file.find('mat')==-1):
        matlab_val = np.load(experiments.matlab_val_file)
    else:
        matlab_val = data.import_realdata_h5(filename = experiments.matlab_val_file, matname=experiments.matlab_val_name, is_complex=True)

   
    # Make sure it's radix 2 along Z
    if(np.mod(matlab_val.shape[0],2)==1):
        matlab_val = matlab_val[0:matlab_val.shape[0]-1,:,:]
    matlab_val = matlab_val[:,100:200,100:200,]
    matlab_val = matlab_val + experiments.mybackgroundval
    
    ''' Create the Model'''
    muscat = mus.MuScatModel(matlab_pars, is_optimization=is_optimization)
    # Correct some values - just for the puprose of fitting in the RAM
    muscat.Nz,muscat.Nx,muscat.Ny = matlab_val.shape
    muscat.shiftIcY=experiments.shiftIcY
    muscat.shiftIcX=experiments.shiftIcX
    muscat.dn = experiments.dn
    muscat.NAc = experiments.NAc
    #muscat.dz = muscat.lambda0/2
    
    ''' Adjust some parameters to fit it in the memory '''
    muscat.mysize = (muscat.Nz,muscat.Nx,muscat.Ny) # ordering is (Nillu, Nz, Nx, Ny)
    
    # introduce zernike factors here
    muscat.zernikefactors = experiments.zernikefactors
    muscat.zernikemask = experiments.zernikemask
    
    ''' Compute a first guess based on the experimental phase '''
    if(is_obj_init_tikhonov):
        obj_guess =  np.zeros(matlab_val.shape)+muscat.nEmbb# np.angle(matlab_val)## 
        obj_guess = np.load('thikonovinvse.npy')
        obj_guess = obj_guess[:,100:200,100:200,]
        #obj_guess = obj_guess-np.min(obj_guess); obj_guess = obj_guess/np.max(obj_guess)
        obj_guess = obj_guess-(np.min(np.real(obj_guess))+1j*np.min(np.imag(obj_guess)))
        if is_absorption:
            obj_guess = dn*np.real(obj_guess)/np.max(np.real(obj_guess))+1j*dn*np.imag(obj_guess)/np.max(np.imag(obj_guess))
        else:
            obj_guess = dn*np.real(obj_guess)/np.max(np.real(obj_guess))
    else:
        obj_guess =  np.zeros(matlab_val.shape)# np.angle(matlab_val)## 
    
    obj_guess = obj_guess+muscat.nEmbb
    

    ''' Compute the systems model'''
    # Compute the System's properties (e.g. Pupil function/Illumination Source, K-vectors, etc.)¶
    muscat.computesys(obj=None, is_padding=is_padding, mysubsamplingIC=mysubsamplingIC, is_compute_psf='BORN')

    ''' Create Model Instance'''
    muscat.computemodel()
    
    
    if(1):
        # Test this Carringotn Padding to have borders at the dges where the optimizer can make pseudo-update
        #np_meas = np.pad(matlab_val,[(64, 64), (64, 64), (64, 64)], mode='constant', constant_values=0-1j)
        np_meas = matlab_val
        my_border_region = np.array((muscat.mysize[0]//2,20,20)) # border-region around the object 
        bz, bx, by = my_border_region
        obj_guess = np.pad(obj_guess,[(bz, bz), (bx, bx), (by, by)], mode='constant', constant_values=np.mean(np.real(obj_guess))+1j*np.mean(np.imag(obj_guess)))
        #muscat.tf_meas = tf.placeholder(tf.complex64, np_meas.shape, 'TF_placeholder_meas')
        muscat.TF_obj = tf.Variable(np.real(obj_guess), dtype=tf.float32, name='Object_Variable_Real')
        muscat.TF_obj_absorption = tf.Variable(np.imag(obj_guess), dtype=tf.float32, name='Object_Variable_Imag')
        tf_fwd = muscat.computeconvolution(muscat.TF_ASF, is_padding='border',border_region=my_border_region)
    else:
        ''' Define Fwd operator'''
        tf_fwd = muscat.computeconvolution(muscat.TF_ASF, is_padding=True)
    
    

    '''Define Cost-function'''
    # VERY VERY Important to add 0. and 1. - otherwise it gets converted to float!
    tf_glob_real = tf.Variable(0., tf.float32, name='var_phase') # (0.902339905500412
    tf_glob_imag = tf.Variable(0., tf.float32, name='var_abs') #0.36691132
                               
    '''REGULARIZER'''
    # Total Variation
    print('We are using TV - Regularization')
    tf_tvloss = muscat.tf_lambda_tv*reg.Reg_TV(muscat.TF_obj, BetaVals = [muscat.dx,muscat.dy,muscat.dz], epsR=muscat.tf_eps, is_circ = True)  #Alernatively tf_total_variation_regularization # total_variation
    tf_tvloss += muscat.tf_lambda_tv*reg.Reg_TV(muscat.TF_obj_absorption, BetaVals = [muscat.dx,muscat.dy,muscat.dz], epsR=muscat.tf_eps, is_circ = True)  #Alernatively tf_total_variation_regularization # total_variation
                                             
    '''Negativity Constraint'''                                          
    tf_negsqrloss = reg.Reg_NegSqr(muscat.TF_obj)#-tf.minimum(tf.reduce_min(muscat.TF_obj-1.),0) 
    tf_negsqrloss += reg.Reg_NegSqr(muscat.TF_obj_absorption)
    tf_negsqrloss *= lambda_neg
    # Correc the fwd model - not good here!
    tf_norm = tf.complex(tf_glob_real, tf_glob_imag)
    
    '''Define Loss-function'''
    if(0):
        print('-------> Losstype is L1')
        tf_fidelity = tf.reduce_mean((tf.abs((muscat.tf_meas+tf_norm) - tf_fwd))) # allow a global phase parameter to avoid unwrapping effects
    else:
        print('-------> Losstype is L2')
        tf_fidelity = tf.reduce_mean(tf_helper.tf_abssqr((muscat.tf_meas+tf_norm) - tf_fwd)) # allow a global phase parameter to avoid unwrapping effects
    tf_loss = tf_fidelity + tf_negsqrloss + tf_tvloss
    
    '''Define Optimizer'''
    tf_optimizer = tf.train.AdamOptimizer(muscat.tf_learningrate)
    tf_lossop_norm = tf_optimizer.minimize(tf_loss, var_list = [tf_glob_imag, tf_glob_real])
    #tf_lossop_tv = tf_optimizer.minimize(tf_tvloss, var_list = [muscat.TF_obj, muscat.TF_obj_absorption])
    tf_lossop_obj = tf_optimizer.minimize(tf_loss, var_list = [muscat.TF_obj])
    tf_lossop_obj_absorption = tf_optimizer.minimize(tf_loss, var_list = [muscat.TF_obj,muscat.TF_obj_absorption])
    tf_lossop_aberr = tf_optimizer.minimize(tf_loss, var_list = [muscat.TF_shiftIcX, muscat.TF_shiftIcY, muscat.TF_zernikefactors])
    tf_lossop = tf_optimizer.minimize(tf_loss) 
    
    ''' Initialize the model '''
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    ''' Compute the ATF '''
    if(0):
        print('We are precomputing the PSF')
        myATF = sess.run(muscat.TF_ATF)
        myASF = sess.run(muscat.TF_ASF)    
    
        #%
        plt.figure()    
        plt.subplot(231), plt.imshow(np.abs(((myATF))**.2)[:,myATF.shape[1]//2,:]), plt.colorbar()#, plt.show()
        plt.subplot(232), plt.imshow(np.abs(((myATF))**.2)[myATF.shape[0]//2,:,:]), plt.colorbar()#, plt.show()    
        plt.subplot(233), plt.imshow(np.abs(((myATF))**.2)[:,:,myATF.shape[2]//2]), plt.colorbar()#, plt.show()    
        plt.subplot(234), plt.imshow(np.abs(((myASF))**.2)[:,myASF.shape[1]//2,:]), plt.colorbar()#, plt.show()
        plt.subplot(235), plt.imshow(np.abs(((myASF))**.2)[myASF.shape[0]//2,:,:]), plt.colorbar()#, plt.show()    
        plt.subplot(236), plt.imshow(np.abs(((myASF))**.2)[:,:,myASF.shape[2]//2]), plt.colorbar()#, plt.show()    
    
       
    '''Define some stuff related to infrastructure'''
    mytimestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    savepath = basepath + resultpath + mytimestamp + 'tv_' + str(mylambdatv) + '_eps_' +str(myepstvval) + '_' +'Shift_x-'+str(experiments.shiftIcX)+'Shift_y-'+str(experiments.shiftIcY)
    
    # Create directory
    try: 
        os.mkdir(savepath)
    except(FileExistsError): 
        print('Folder exists already')
    
    # assert some memory 
    iter_last = 0
    mylosslist = []; myfidelitylist = []
    myposlosslist = []; myneglosslist = []
    mytvlosslist = []; result_phaselist = []
    result_absorptionlist = []; globalphaselist = []
    globalabslist = []; myfwdlist = []

else:
    # Assign the initial guess to the object inside the fwd-model
    print('Assigning Variables')
    sess.run(tf.assign(muscat.TF_obj, np.real(obj_guess))); # assign abs of measurement as initial guess of 
    sess.run(tf.assign(muscat.TF_obj_absorption, np.imag(obj_guess))); # assign abs of measurement as initial guess of 
    sess.run(tf.assign(muscat.TF_zernikefactors, muscat.zernikefactors*0))
    iter_last=0
#%%
''' Optimize the model '''
print('Start optimizing')
#import scipy.io
#scipy.io.savemat('ExperimentAsfObj.mat', dict(asf=np.array(myASF), obj=np.array(matlab_val)))

for iterx in range(iter_last,Niter):
    
    # Change the learning rat - experimental
    if iterx == NreduceLR:
        print('Learning Rate has changed by factor of .1')
        my_learningrate = my_learningrate*.1
        
    
        
    # This is for debbugging purposes - write the result to disk every n-iteration
    if(iterx==0 or not np.mod(iterx, Ndisplay)):
        my_res, my_res_absortpion, my_loss, my_fidelity, my_negloss, my_tvloss, myglobalphase, myglobalabs, myfwd =  \
            sess.run([muscat.TF_obj, muscat.TF_obj_absorption, tf_loss, tf_fidelity, tf_negsqrloss, tf_tvloss, tf_glob_real, tf_glob_imag, tf_fwd], \
                     feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:my_learningrate, muscat.tf_lambda_tv:mylambdatv, muscat.tf_eps:myepstvval}) #, muscat.TF_ATF_placeholder:myATF
    
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
    #sess.run([tf_lossop_tv], feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:my_learningrate, muscat.tf_lambda_tv:mylambdatv, muscat.tf_eps:myepstvval})

    if is_absorption:
        sess.run([tf_lossop_obj_absorption], feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:my_learningrate, muscat.tf_lambda_tv:mylambdatv, muscat.tf_eps:myepstvval})
    else:
        sess.run([tf_lossop_obj], feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:my_learningrate, muscat.tf_lambda_tv:mylambdatv, muscat.tf_eps:myepstvval})
    
    if is_aberration and (iterx > 50):
        sess.run([tf_lossop_aberr], feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:my_learningrate, muscat.tf_lambda_tv:mylambdatv, muscat.tf_eps:myepstvval})

    if is_norm:
        sess.run([tf_lossop_norm], feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:my_learningrate, muscat.tf_lambda_tv:mylambdatv, muscat.tf_eps:myepstvval})
        

    iter_last = iterx

#%%
''' Save Figures and Parameters '''
muscat.saveFigures_list(savepath, myfwdlist, mylosslist, myfidelitylist, myneglosslist, mytvlosslist, result_phaselist, result_absorptionlist, 
                              globalphaselist, globalabslist, np_meas, figsuffix='FINAL')

data.export_realdatastack_h5(savepath+'/myrefractiveindex.h5', 'phase, abs', 
                        np.stack(((nip.extract(result_phaselist[-1], muscat.mysize)),
                                 (nip.extract(result_absorptionlist[-1], muscat.mysize))), axis=0))
data.export_realdatastack_h5(savepath+'/mymeas.h5', 'real, imag', 
                        np.stack((np.real(np_meas),
                                  np.imag(np_meas)), axis=0))
       
print('Zernikes: ' +str(np.real(sess.run(muscat.TF_zernikefactors))))

# backup current script
from shutil import copyfile
import os
src = (os.path.basename(__file__))
copyfile(src, savepath+'/script_bak.py')

#%
plt.imshow(np.fft.ifftshift(np.angle(sess.run(muscat.TF_Po_aberr))))