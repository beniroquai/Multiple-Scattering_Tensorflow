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

import configurations.test as configs

# Optionally, tweak styles.
mpl.rc('figure',  figsize=(9, 6))
mpl.rc('image', cmap='gray')
if(not configs.is_display):
    plt.switch_backend('agg') # we don't want to display any images

#%%
'''Define some stuff related to infrastructure'''
mytimestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# need to figure out why this holds somehow true - at least produces reasonable results
'''START CODE'''
tf.reset_default_graph() # just in case there was an open session

# Generate Test-Object
''' File which stores the experimental parameters from the Q-PHASE setup 
    1.) Read in the parameters of the dataset ''' 
matlab_pars = data.import_parameters_mat(filename = configs.matlab_par_file, matname = configs.matlab_par_name)

''' 2.) Read in the parameters of the dataset ''' 
if(configs.matlab_val_file.find('mat')==-1):
    matlab_val = np.load(configs.matlab_val_file)
else:
    matlab_val = data.import_realdata_h5(filename = configs.matlab_val_file, matname = configs.matlab_val_name, is_complex=True)

if(np.mod(matlab_val.shape[0],2)==1):
    matlab_val = matlab_val[0:matlab_val.shape[0]-1,:,:]
    
matlab_val = matlab_val + configs.mybackgroundval
#roisize=50
#roicenter = np.array((215,201))
#matlab_val = np.flip(matlab_val,0 )#[0:100,roicenter[0]-roisize:roicenter[0]+roisize,roicenter[1]-roisize:roicenter[1]+roisize],0)

''' Create the Model'''
muscat = mus.MuScatModel(matlab_pars, is_optimization=configs.is_display)

# Correct some values - just for the puprose of fitting in the RAM
muscat.Nx,muscat.Ny,muscat.Nz = matlab_val.shape[1], matlab_val.shape[2], matlab_val.shape[0]
muscat.shiftIcY = configs.shiftIcY
muscat.shiftIcX = configs.shiftIcX
muscat.dn = configs.dn
muscat.NAc = configs.NAc

''' Adjust some parameters to fit it in the memory '''
muscat.mysize = (muscat.Nz,muscat.Nx,muscat.Ny) # ordering is (Nillu, Nz, Nx, Ny)

# introduce zernike factors here
muscat.zernikefactors = configs.zernikefactors
muscat.zernikemask = configs.zernikemask

''' Compute the systems model'''
# Compute the System's properties (e.g. Pupil function/Illumination Source, K-vectors, etc.)¶
muscat.computesys(obj=None, is_padding=configs.is_padding, mysubsamplingIC=configs.mysubsamplingIC, is_compute_psf=configs.is_compute_psf)

''' Create Model Instance'''
muscat.computemodel()

''' start TF Session '''
sess = tf.Session()
sess.run(tf.global_variables_initializer())

''' Compute either MS or Born Model'''
if configs.is_compute_psf is None:
    ''' Define Fwd operator'''
    tf_fwd = muscat.computemodel(is_forcepos=False)
else:
    ''' Define Fwd operator'''
    tf_fwd = muscat.computeconvolution(muscat.TF_ASF, myfac=configs.myfac, myabsnorm = configs.myabsnorm)

    ''' Pre-Compute the ATF and ASF '''
    myATF = sess.run(muscat.TF_ATF)
    myASF = sess.run(muscat.TF_ASF) 

#%%
''' Compute a first guess based on the experimental phase '''
init_guess =  np.zeros(matlab_val.shape)+muscat.nEmbb# np.angle(matlab_val)## 
#init_guess = init_guess-np.min(init_guess)
#init_guess = dn*init_guess/np.max(init_guess)+muscat.nEmbb#*dn+1j*.01*np.ones(init_guess.shape)


#%%
'''# Estimate the Phase difference between Measurement and Simulation'''
# This is actually a crucial step and needs to be used with care. We don't want any phase wrappings ! 
'''Numpy to Tensorflow'''
np_meas = matlab_val
#np_mean = 1#np.mean(np_meas)
#np_meas = np_meas/np_mean
#print("Mean of the MEasurement is: "+str(np_mean))

'''Define Cost-function'''
# VERY VERY Important to add 0. and 1. - otherwise it gets converted to float!
tf_global_phase = tf.Variable(configs.np_global_phase, tf.float32, name='var_phase') # (0.902339905500412
tf_global_abs = tf.Variable(configs.np_global_abs, tf.float32, name='var_abs') #0.36691132
                           
'''REGULARIZER'''
# Total Variation
print('We are using TV - Regularization')
tf_tvloss = muscat.tf_lambda_tv*reg.Reg_TV(muscat.TF_obj, BetaVals = [muscat.dx,muscat.dy,muscat.dz], epsR=muscat.tf_eps, is_circ = True)  #Alernatively tf_total_variation_regularization # total_variation
tf_tvloss += muscat.tf_lambda_tv*reg.Reg_TV(muscat.TF_obj_absorption, BetaVals = [muscat.dx,muscat.dy,muscat.dz], epsR=muscat.tf_eps, is_circ = True)  #Alernatively tf_total_variation_regularization # total_variation
                                         
'''Negativity Constraint'''                                          
tf_negsqrloss = configs.lambda_neg*reg.Reg_NegSqr(muscat.TF_obj)
tf_negsqrloss += configs.lambda_neg*reg.Reg_NegSqr(muscat.TF_obj_absorption)

# Correc the fwd model - not good here!
tf_norm = tf.complex(tf_global_phase, tf_global_abs)
tf_fwd_corrected = tf_fwd+tf_norm
#tf_fwd_corrected = (tf_fwd+1j*tf.cast(tf_global_phase, tf.complex64))/tf.cast(tf_global_abs, tf.complex64)



if(not configs.is_optimization): 
    # Here we only want to compute the fwd model
    if(0):
        mymatfile = '/Users/bene/Dropbox/Dokumente/Promotion/PROJECTS/BOSTON/MUSCAT/MATLAB/MuScat/mtfs.mat'
        matatf = data.import_realdata_h5(filename = mymatfile, matname='myatf', is_complex=True)
        matasf = data.import_realdata_h5(filename = mymatfile, matname='myasf', is_complex=True)

    
    myATF = sess.run(muscat.TF_ATF)
    myASF = sess.run(muscat.TF_ASF)
    
    #%% run model and measure memory/time
    start = time.time()
    myfwd = sess.run(myres, feed_dict={muscat.TF_ASF_placeholder:myASF,  muscat.TF_obj:obj_real, muscat.TF_obj_absorption:obj_absorption})#, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True), run_metadata=run_metadata)
    end = time.time()
    print(end - start)

else:
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
    tf_lossop_aberr = tf_optimizer.minimize(tf_loss, var_list = [muscat.TF_zernikefactors])
    tf_lossop = tf_optimizer.minimize(tf_loss)
    
    
    # Do we want to account for absorption? 
    if(configs['is_absorption']):
        tf_lossop_obj = tf_optimizer.minimize(tf_loss, var_list = [muscat.TF_obj, muscat.TF_obj_absorption])
    else:
        tf_lossop_obj = tf_optimizer.minimize(tf_loss, var_list = [muscat.TF_obj])
        
        
    ''' Initialize the model '''
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # Assign the initial guess to the object inside the fwd-model
    sess.run(tf.assign(muscat.TF_obj, np.real(init_guess))); # assign abs of measurement as initial guess of 
    sess.run(tf.assign(muscat.TF_obj_absorption, np.imag(init_guess))); # assign abs of measurement as initial guess of 
    
    #%%
    plt.figure()    
    plt.subplot(231), plt.imshow(np.abs(((myATF))**.2)[:,myATF.shape[1]//2,:]), plt.colorbar()#, plt.show()
    plt.subplot(232), plt.imshow(np.abs(((myATF))**.2)[myATF.shape[0]//2,:,:]), plt.colorbar()#, plt.show()    
    plt.subplot(233), plt.imshow(np.abs(((myATF))**.2)[:,:,myATF.shape[2]//2]), plt.colorbar()#, plt.show()    
    plt.subplot(234), plt.imshow(np.abs(((myASF))**.2)[:,myASF.shape[1]//2,:]), plt.colorbar()#, plt.show()
    plt.subplot(235), plt.imshow(np.abs(((myASF))**.2)[myASF.shape[0]//2,:,:]), plt.colorbar()#, plt.show()    
    plt.subplot(236), plt.imshow(np.abs(((myASF))**.2)[:,:,myASF.shape[2]//2]), plt.colorbar()#, plt.show()    
    
    #%%
        
    mylambdatv = configs.lambda_tv
    myepstvval = configs.eps_tv
       
    '''Define some stuff related to infrastructure'''
    mytimestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    savepath = configs.basepath + configs.resultpath + mytimestamp + 'tv_' + str(mylambdatv) + '_eps_' +str(myepstvval) + '_' +'Shift_x-'+str(configs.shiftIcX)+'Shift_y-'+str(configs.shiftIcY) + '_' 
    
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
    
    
    for iterx in range(iter_last,configs.Niter):
        
        # Change the learning rat - experimental
        if iterx == configs.NreduceLR:
            print('Learning Rate has changed by factor of .1')
            my_learningrate = configs.my_learningrate*.1
            
        # This is for debbugging purposes - writ th result to disk every n-iteration
        if(iterx==0 or not np.mod(iterx, configs.Ndisplay)):
            my_res, my_res_absortpion, my_loss, my_fidelity, my_negloss, my_tvloss, myglobalphase, myglobalabs, myfwd =  \
                sess.run([muscat.TF_obj, muscat.TF_obj_absorption, tf_loss, tf_fidelity, tf_negsqrloss, tf_tvloss, tf_global_phase, tf_global_abs, tf_fwd_corrected], \
                         feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:my_learningrate, muscat.tf_lambda_tv:mylambdatv, muscat.tf_eps:myepstvval}) #, muscat.TF_ATF_placeholder:myATF
        
        if iterx>5000:
            for iternorm in range(0,1):
                # find the correct normalization parameters first 
                sess.run([tf_lossop_norm], feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:my_learningrate, muscat.tf_lambda_tv:mylambdatv, muscat.tf_eps:myepstvval})
        #else:
            
           # sess.run([tf_lossop_norm], feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:my_learningrate, muscat.tf_lambda_tv:mylambdatv, muscat.tf_eps:myepstvval})
        
                
        if(iterx==0 or not np.mod(iterx, configs.Nsave)):
        
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
        if (iterx>10) & (configs.Noptpsf>0):
            for iterobj in range(configs.Noptpsf*3):
               sess.run([tf_lossop_obj], feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:my_learningrate, muscat.tf_lambda_tv:mylambdatv, muscat.tf_eps:myepstvval})
    
            for iterpsf in range(configs.Noptpsf):
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