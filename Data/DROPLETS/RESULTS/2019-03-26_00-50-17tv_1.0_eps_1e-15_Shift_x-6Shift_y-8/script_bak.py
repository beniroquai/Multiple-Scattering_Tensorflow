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
my_learningrate = 1e-1  # learning rate
NreduceLR = 500 # when should we reduce the Learningrate? 

# TV-Regularizer 
mylambdatv = 1e0 ##, 1e-2, 1e-2, 1e-3)) # lambda for Total variation - 1e-1
myepstvval = 1e-15##, 1e-12, 1e-8, 1e-6)) # - 1e-1 # smaller == more blocky

# Positivity Constraint
lambda_neg = 10000

# Displaying/Saving
Niter = 500
Nsave = 100 # write info to disk
Ndisplay = Nsave

# Control Flow 
is_norm = False 
is_aberration = True
is_padding = False
is_optimization = True
is_absorption = True

is_recomputemodel = True # TODO: Make it automatic! 



''' MODELLING StARTS HERE'''
if is_recomputemodel:
    tf.reset_default_graph()
    # need to figure out why this holds somehow true - at least produces reasonable results
    mysubsamplingIC = 0    
    dn = experiments.dn
    myfac = 1e0# 0*dn*1e-3
    myabsnorm = 1e5#myfac
    
    ''' microscope parameters '''
    NAc = .5
    zernikefactors = np.array((0,0,0,0,0,0,-.01,-.001,0.01,0.01,.010))  # 7: ComaX, 8: ComaY, 11: Spherical Aberration
    zernikemask = np.ones(zernikefactors.shape) #np.array(np.abs(zernikefactors)>0)*1# mask of factors that should be updated
    zernikemask[0]=0 # we don't want the first one to be shifting the phase!!
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
    matlab_val = matlab_val + experiments.mybackgroundval
    
    ''' Create the Model'''
    muscat = mus.MuScatModel(matlab_pars, is_optimization=is_optimization)
    # Correct some values - just for the puprose of fitting in the RAM
    muscat.Nx,muscat.Ny,muscat.Nz = matlab_val.shape[1], matlab_val.shape[2], matlab_val.shape[0]
    muscat.shiftIcY=experiments.shiftIcY
    muscat.shiftIcX=experiments.shiftIcX
    muscat.dn = dn
    muscat.NAc = NAc
    #muscat.dz = muscat.lambda0/2
    
    ''' Adjust some parameters to fit it in the memory '''
    muscat.mysize = (muscat.Nz,muscat.Nx,muscat.Ny) # ordering is (Nillu, Nz, Nx, Ny)
    
    # introduce zernike factors here
    muscat.zernikefactors = zernikefactors
    muscat.zernikemask = zernikemask
    
    ''' Compute a first guess based on the experimental phase '''
    obj_guess =  np.zeros(matlab_val.shape)+muscat.nEmbb# np.angle(matlab_val)## 
    obj_guess = np.real(np.load('thikonovinvse.npy'))
    obj_guess = obj_guess-np.min(obj_guess); obj_guess = obj_guess/np.max(obj_guess)
    obj_guess = obj_guess*dn+muscat.nEmbb
    
    ''' Compute the systems model'''
    # Compute the System's properties (e.g. Pupil function/Illumination Source, K-vectors, etc.)¶
    muscat.computesys(obj=obj_guess, is_padding=is_padding, mysubsamplingIC=mysubsamplingIC, is_compute_psf='corr')
    
    ''' Create Model Instance'''
    muscat.computemodel()
    
    ''' Define Fwd operator'''
    tf_fwd = muscat.computeconvolution(muscat.TF_ASF)
    
    
    #%%
    np_meas = matlab_val
    
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
    tf_negsqrloss = lambda_neg*reg.Reg_NegSqr(muscat.TF_obj)
    tf_negsqrloss += lambda_neg*reg.Reg_NegSqr(muscat.TF_obj_absorption)
    
    # Correc the fwd model - not good here!
    tf_norm = tf.complex(tf_glob_real, tf_glob_imag)
    
    '''Define Loss-function'''
    if(0):
        print('-------> Losstype is L1')
        tf_fidelity = tf.reduce_mean((tf.abs((muscat.tf_meas+tf_norm) - tf_fwd))) # allow a global phase parameter to avoid unwrapping effects
    else:
        print('-------> Losstype is L2')
        tf_fidelity = tf.reduce_mean(tf_helper.tf_abssqr((muscat.tf_meas+tf_norm) - tf_fwd)) # allow a global phase parameter to avoid unwrapping effects
    tf_loss = tf_fidelity + tf_tvloss + tf_negsqrloss 
    
    '''Define Optimizer'''
    tf_optimizer = tf.train.AdamOptimizer(muscat.tf_learningrate)
    tf_lossop_norm = tf_optimizer.minimize(tf_loss, var_list = [tf_glob_imag, tf_glob_real])
    tf_lossop_obj = tf_optimizer.minimize(tf_loss, var_list = [muscat.TF_obj])
    tf_lossop_obj_absorption = tf_optimizer.minimize(tf_loss, var_list = [muscat.TF_obj_absorption])
    tf_lossop_aberr = tf_optimizer.minimize(tf_loss, var_list = [muscat.TF_shiftIcX, muscat.TF_shiftIcY, muscat.TF_zernikefactors])
    tf_lossop = tf_optimizer.minimize(tf_loss) 
    
    ''' Initialize the model '''
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # Assign the initial guess to the object inside the fwd-model
    #print('Assigning Variables')
    #sess.run(tf.assign(muscat.TF_obj, np.real(init_guess))); # assign abs of measurement as initial guess of 
    #sess.run(tf.assign(muscat.TF_obj_absorption, np.imag(init_guess))); # assign abs of measurement as initial guess of 
    
    ''' Compute the ATF '''
    if(0):
        print('We are precomputing the PSF')
        myATF = sess.run(muscat.TF_ATF)
        myASF = sess.run(muscat.TF_ASF)    
    
        #%%
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
    sess.run([tf_lossop_obj], feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:my_learningrate, muscat.tf_lambda_tv:mylambdatv, muscat.tf_eps:myepstvval})

    if is_absorption:
        sess.run([tf_lossop_obj_absorption], feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:my_learningrate, muscat.tf_lambda_tv:mylambdatv, muscat.tf_eps:myepstvval})
        
    if is_aberration:
        sess.run([tf_lossop_aberr], feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:my_learningrate, muscat.tf_lambda_tv:mylambdatv, muscat.tf_eps:myepstvval})

    if is_norm:
        sess.run([tf_lossop_norm], feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:my_learningrate, muscat.tf_lambda_tv:mylambdatv, muscat.tf_eps:myepstvval})
        

    iter_last = iterx

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