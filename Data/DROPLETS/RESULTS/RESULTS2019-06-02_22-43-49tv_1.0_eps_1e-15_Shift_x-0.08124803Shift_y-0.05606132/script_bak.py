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
import src.MyParameter as paras

import NanoImagingPack as nip

# Optionally, tweak styles.
mpl.rc('figure',  figsize=(12, 9))
mpl.rc('image', cmap='gray')
#plt.switch_backend('agg')
#np.set_printoptions(threshold=np.nan)
#%load_ext autoreload

#%%
'''Define some stuff related to infrastructure'''
mytimestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
basepath = './'#'/projectnb/cislidt/diederich
is_aberration = True
is_aberation_iterstart = 5 # When to start optimizing for aberration?
is_padding = False
is_optimization = True   
is_absorption = False 
is_obj_init_tikhonov = False 
is_norm = False
is_recomputemodel = True  # TODO: Make it automatic! 
is_estimatepsf = False
mybordersize = 20
is_psfmodell = 'BPM' # either compute BORN or BPM ()

# Displaying/Saving
Niter =  250
Nsave = 50 # write info to disk
NreduceLR = 1000 # when should we reduce the Learningrate? 


''' MODELLING StARTS HERE''' 
if is_recomputemodel:
    tf.reset_default_graph()
    # need to figure out why this holds somehow true - at least produces reasonable results
    
    ''' 1.) Read in the parameters of the dataset ''' 
    if(experiments.matlab_val_file.find('mat')==-1):
        matlab_val = np.load(experiments.matlab_val_file)
    else:
        matlab_val = data.import_realdata_h5(filename = experiments.matlab_val_file, matname=experiments.matlab_val_name, is_complex=True)
        
    matlab_val  = matlab_val+np.random.rand(matlab_val.shape[0], matlab_val.shape[1], matlab_val.shape[2])*.001
    # If Z-is odd numbered
    if(np.mod(matlab_val.shape[0],2)==1):
        matlab_val = matlab_val[0:matlab_val.shape[0]-1,:,:]
    matlab_val = matlab_val[:,:,:,]
    if(is_psfmodell=='BORN'):
        matlab_val = matlab_val + experiments.mybackgroundval
    
    ''' 2.) Read referecne object for PSF calibration ''' 
    if(is_estimatepsf):
        obj_val = data.import_realdata_h5(filename = experiments.matlab_obj_file, matname='mysphere_mat', is_complex=False)
  
    # Generate Test-Object
    ''' File which stores the experimental parameters from the Q-PHASE setup 
        3.) Read in the parameters of the dataset ''' 
    matlab_pars = paras.MyParameter()
    matlab_pars.loadmat(mymatpath = experiments.matlab_par_file, mymatname = experiments.matlab_par_name)
    matlab_pars.Nz,matlab_pars.Nx,matlab_pars.Ny = matlab_val.shape
    matlab_pars.mysize = (matlab_pars.Nz,matlab_pars.Nx,matlab_pars.Ny) # ordering is (Nillu, Nz, Nx, Ny)
    matlab_pars.shiftIcY=experiments.shiftIcY
    matlab_pars.shiftIcX=experiments.shiftIcX
    matlab_pars.dn = experiments.dn
    matlab_pars.NAc = experiments.NAc
    
    #matlab_pars.dz = .8
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ATTENTION weird magic number to match the pixelsize')
    matlab_pars.dx /= (30/25)
    matlab_pars.dy /= (30/25)
    
    ''' Create the Model'''
    muscat = mus.MuScatModel(matlab_pars, is_optimization=is_optimization)

    muscat.zernikefactors = experiments.zernikefactors
    muscat.zernikemask = experiments.zernikemask
  
    #muscat.NAo = .8
    ''' Compute a first guess based on the experimental phase '''
    if(is_obj_init_tikhonov):
        print('Object is initialized with precomputed RI-distribution')
       
        if(0):
            print('ATTENTION: SIDELOAD a GD result!')
            obj_guess_filename = 'myrefractiveindex.h5'
            obj_guess = data.import_realdata_h5(filename = obj_guess_filename, matname='phase, abs0', is_complex=False)
        else:
             obj_guess =  np.zeros(matlab_val.shape)+muscat.params.nEmbb # np.angle(matlab_val)## 
             obj_guess = np.load('thikonovinvse.npy')
        
        #obj_guess = obj_guess-np.min(obj_guess); obj_guess = obj_guess/np.max(obj_guess)
        obj_guess = obj_guess-(np.min(np.real(obj_guess))+1j*np.min(np.imag(obj_guess)))
        if is_absorption:
            obj_guess = experiments.dn*np.real(obj_guess)/np.max(np.real(obj_guess))+1j*experiments.dn*np.imag(obj_guess)/np.max(np.imag(obj_guess))
        else:
            obj_guess = experiments.dn*np.real(obj_guess)/np.max(np.real(obj_guess))
    else:
        obj_guess =  np.zeros(matlab_val.shape)# np.angle(matlab_val)## 
    
    if is_estimatepsf:
        obj_guess =  obj_val*experiments.dn+1j*.01*obj_val
        #obj_guess = np.random.rand(matlab_val.shape[0],matlab_val.shape[1],matlab_val.shape[2])*muscat.dn/2
    obj_guess = obj_guess+muscat.params.nEmbb # add background
    np_meas = matlab_val
        
    
    ''' Define Fwd operator'''
    if(is_psfmodell=='BORN'):
        # Test this Carringotn Padding to have borders at the dges where the optimizer can make pseudo-update
        # add carrington boundary regions
        my_border_region = np.array((matlab_val.shape[0]//2,mybordersize,mybordersize)) # border-region around the object 
        bz, bx, by = my_border_region
        obj_guess_real = np.pad(np.real(obj_guess),[(bz, bz), (bx, bx), (by, by)], mode='constant', constant_values=np.mean(np.real(obj_guess)))
        obj_guess_imag = np.pad(np.imag(obj_guess),[(bz, bz), (bx, bx), (by, by)], mode='constant', constant_values=np.mean(1j*np.mean(np.imag(obj_guess))))
        obj_guess = obj_guess_real + 1j*obj_guess_imag

        ''' Compute the BORN model'''
        # Compute the System's properties (e.g. Pupil function/Illumination Source, K-vectors, etc.)¶
        muscat.computesys(obj=None, is_padding=is_padding, mysubsamplingIC=0, is_compute_psf='BORN',is_dampic=experiments.is_dampic)
        muscat.TF_obj = tf.Variable(np.real(obj_guess), dtype=tf.float32, name='Object_Variable_Real')
        muscat.TF_obj_absorption = tf.Variable(np.imag(obj_guess), dtype=tf.float32, name='Object_Variable_Imag')

        ''' Create Model Instance'''
        muscat.computemodel()
        tf_fwd = muscat.computeconvolution(muscat.TF_ASF, is_padding='border',border_region=my_border_region)
        #tf_fwd = muscat.computeconvolution(muscat.TF_ASF, is_padding=True)    
    if(is_psfmodell=='BPM'):
        ''' Compute the Multiple Scattering model'''
        muscat.computesys(obj=None, is_padding=is_padding, mysubsamplingIC=experiments.mysubsamplingIC, is_compute_psf='BPM', is_dampic=experiments.is_dampic)
        muscat.TF_obj = tf.Variable(np.real(obj_guess), dtype=tf.float32, name='Object_Variable_Real')
        muscat.TF_obj_absorption = tf.Variable(np.imag(obj_guess), dtype=tf.float32, name='Object_Variable_Imag')
        tf_fwd = muscat.computemodel()
        tf_fwd = tf_fwd + 1j
       
    '''experiments.regularizer'''
    # Total Variation
    if(experiments.regularizer=='TV'):
        # TV experiments.regularizer
        print('We are using TV - Regularization') # (tfin, Eps=1e-15, doubleSided=False,regDataType=None)
       # TF_obj_tmp = tf_helper.extract(tf.cast(muscat.TF_obj, tf.float32), muscat.mysize)
        #TF_obj_absorption_tmp = tf_helper.extract(tf.cast(muscat.TF_obj_absorption, tf.float32), muscat.mysize)
        
        #tf_regloss =  muscat.tf_lambda_tv*reg.Reg_TV_RH(muscat.TF_obj, Eps=muscat.tf_eps)  #Alernatively tf_total_variation_regularization # total_variation
        #tf_regloss += muscat.tf_lambda_tv*reg.Reg_TV_RH(muscat.TF_obj_absorption, Eps=muscat.tf_eps)  #Alernatively tf_total_variation_regularization # total_variation
         
        #  [muscat.dx,muscat.dy,muscat.dz]
        mysqrt_real = reg.Reg_TV(muscat.TF_obj, BetaVals = [1,1,1], epsR=muscat.tf_eps, is_circ = True)  #Alernatively tf_total_variation_regularization # total_variation
        tf_regloss_real = tf.reduce_mean(mysqrt_real)
        mysqrt_imag = reg.Reg_TV(muscat.TF_obj_absorption, BetaVals = [1,1,1], epsR=muscat.tf_eps, is_circ = True)  #Alernatively tf_total_variation_regularization # total_variation
        tf_regloss_imag = tf.reduce_mean(mysqrt_imag)
        tf_regloss = muscat.tf_lambda_tv*(tf_regloss_real + tf_regloss_imag)
        #tf_regloss += muscat.tf_lambda_tv*reg.Reg_TV(muscat.TF_obj_absorption, BetaVals = [1,1,1], epsR=muscat.tf_eps, is_circ = True)  #Alernatively tf_total_variation_regularization # total_variation
        #lkk 
    elif(experiments.regularizer=='GR'):
        # Goods roughness rgularizer
        print('We are using GR - Regularization')
        tf_regloss =  reg.Reg_GR(muscat.TF_obj)#, eps1=muscat.tf_eps, eps2=muscat.tf_eps)  #Alernatively tf_total_variation_regularization # total_variation
        tf_regloss += reg.Reg_GR(muscat.TF_obj_absorption)#,  eps1=muscat.tf_eps, eps2=muscat.tf_eps)  #Alernatively tf_total_variation_regularization # total_variation
        tf_regloss *= muscat.tf_lambda_tv
    elif(experiments.regularizer=='L1'):
        # L1 rgularizer
        print('We are using GR - Regularization')
        tf_regloss =  reg.Reg_L1(muscat.TF_obj)
        tf_regloss += reg.Reg_L1(muscat.TF_obj_absorption)
        tf_regloss *= muscat.tf_lambda_tv   
    elif(experiments.regularizer=='L2'):
        # L1 rgularizer
        print('We are using GR - Regularization')
        tf_regloss =  reg.Reg_L2(muscat.TF_obj)
        tf_regloss += reg.Reg_L2(muscat.TF_obj_absorption)
        tf_regloss *= muscat.tf_lambda_tv   
                                                
    '''Negativity Constraint'''                                          
    #tf_negsqrloss = reg.Reg_NegSqr(tf_helper.extract(tf.cast(muscat.TF_obj, tf.float32), muscat.mysize))#-tf.minimum(tf.reduce_min(muscat.TF_obj-1.),0) 
    tf_negsqrloss = reg.Reg_NegSqr(muscat.TF_obj)#-tf.minimum(tf.reduce_min(muscat.TF_obj-1.),0) 
    tf_negsqrloss += reg.Reg_NegSqr(muscat.TF_obj_absorption)
    tf_negsqrloss *= experiments.lambda_neg
    # Correc the fwd model - not good here!
    #tf_norm = tf.complex(tf_glob_real, tf_glob_imag)
    
    '''Define Loss-function'''
    if(0):
        print('-------> Losstype is L1')
        tf_fidelity = tf.reduce_mean((tf.abs((muscat.tf_meas) - tf_fwd))) # allow a global phase parameter to avoid unwrapping effects
    else:
        print('-------> Losstype is L2')
        tf_fidelity = tf.reduce_mean(tf_helper.tf_abssqr(muscat.tf_meas - tf_fwd)) # allow a global phase parameter to avoid unwrapping effects
    tf_loss = tf_fidelity + tf_negsqrloss + tf_regloss
   
    tf_optimizer = tf.train.AdamOptimizer(muscat.tf_learningrate)
    '''Define Optimizer'''
    if not is_estimatepsf:
      tf_lossop_obj = tf_optimizer.minimize(tf_loss, var_list = [muscat.TF_obj])
      tf_lossop_obj_absorption = tf_optimizer.minimize(tf_loss, var_list = [muscat.TF_obj,muscat.TF_obj_absorption])
      tf_lossop = tf_optimizer.minimize(tf_loss, var_list = [muscat.TF_obj,muscat.TF_obj_absorption, muscat.TF_shiftIcX, muscat.TF_shiftIcY, muscat.TF_zernikefactors])
      tf_lossop_aberr = tf_optimizer.minimize(tf_loss, var_list = [muscat.TF_shiftIcX, muscat.TF_shiftIcY, muscat.TF_zernikefactors])
    
    else:
        TF_zernloss = .005*reg.Reg_L2(muscat.TF_zernikefactors)
        tf_lossop_aberr = tf_optimizer.minimize(tf_loss+TF_zernloss, var_list = [muscat.TF_shiftIcX, muscat.TF_shiftIcY, muscat.TF_zernikefactors])
    
    ''' Initialize the model '''
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
       
    '''Define some stuff related to infrastructure'''
    mytimestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    savepath = basepath + experiments.resultpath + mytimestamp + 'tv_' + str(experiments.lambda_tv) + '_eps_' +str(experiments.myepstvval) + '_' +'Shift_x-'+str(experiments.shiftIcX)+'Shift_y-'+str(experiments.shiftIcY)
   
        # Create directory
    try: 
        os.mkdir(savepath)
    except(FileExistsError): 
        print('Folder exists already')
    
    ''' Compute the ATF '''
    if(is_psfmodell=='BORN'):
        #%
        print('We are precomputing the PSF')
        myATF = sess.run(muscat.TF_ATF)
        myASF = sess.run(muscat.TF_ASF)    
    
        #%
        plt.figure()    
        plt.subplot(331), plt.imshow(np.log(1+np.abs(((myATF))**.2)[:,myATF.shape[1]//2,:])), plt.colorbar()#, plt.show()
        plt.subplot(332), plt.imshow(np.log(1+np.abs(((myATF))**.2)[myATF.shape[0]//2,:,:])), plt.colorbar()#, plt.show()    
        plt.subplot(333), plt.imshow(np.log(1+np.abs(((myATF))**.2)[:,:,myATF.shape[2]//2])), plt.colorbar()#, plt.show()    
        plt.subplot(334), plt.imshow(np.real(((myASF))**.2)[:,myASF.shape[1]//2,:]), plt.colorbar()#, plt.show()
        plt.subplot(335), plt.imshow(np.real(((myASF))**.2)[myASF.shape[0]//2,:,:]), plt.colorbar()#, plt.show()    
        plt.subplot(336), plt.imshow(np.real(((myASF))**.2)[:,:,myASF.shape[2]//2]), plt.colorbar()#, plt.show()    
        plt.subplot(337), plt.imshow(np.imag(((myASF))**.2)[:,myASF.shape[1]//2,:]), plt.colorbar()#, plt.show()
        plt.subplot(338), plt.imshow(np.imag(((myASF))**.2)[myASF.shape[0]//2,:,:]), plt.colorbar()#, plt.show()    
        plt.subplot(339), plt.imshow(np.imag(((myASF))**.2)[:,:,myASF.shape[2]//2]), plt.colorbar()#, plt.show()    
        plt.savefig(savepath+'/ASFATF.png'), plt.show()
        data.export_realdatastack_h5(savepath+'/myatf.h5', 'real, imag', 
                        np.stack((np.real(myASF), np.imag(myASF)), axis=0))

        plt.figure()    
        myobjft = np.fft.fftshift(np.fft.fftn(np_meas))
        plt.subplot(331), plt.imshow(np.log(1+np.abs(((myobjft))**.2)[:,myATF.shape[1]//2,:])), plt.colorbar()#, plt.show()
        plt.subplot(332), plt.imshow(np.log(1+np.abs(((myobjft))**.2)[myATF.shape[0]//2,:,:])), plt.colorbar()#, plt.show()    
        plt.subplot(333), plt.imshow(np.log(1+np.abs(((myobjft))**.2)[:,:,myATF.shape[2]//2])), plt.colorbar()#, plt.show()    
        plt.subplot(334), plt.imshow(np.real(((np_meas))**.2)[:,myASF.shape[1]//2,:]), plt.colorbar()#, plt.show()
        plt.subplot(335), plt.imshow(np.real(((np_meas))**.2)[myASF.shape[0]//2,:,:]), plt.colorbar()#, plt.show()    
        plt.subplot(336), plt.imshow(np.real(((np_meas))**.2)[:,:,myASF.shape[2]//2]), plt.colorbar()#, plt.show()    
        plt.subplot(337), plt.imshow(np.imag(((np_meas))**.2)[:,myASF.shape[1]//2,:]), plt.colorbar()#, plt.show()
        plt.subplot(338), plt.imshow(np.imag(((np_meas))**.2)[myASF.shape[0]//2,:,:]), plt.colorbar()#, plt.show()    
        plt.subplot(339), plt.imshow(np.imag(((np_meas))**.2)[:,:,myASF.shape[2]//2]), plt.colorbar()#, plt.show()    
        plt.savefig(savepath+'/ATF_Support.png'), plt.show()    
        
        data.export_realdatastack_h5(savepath+'/myatf.h5', 'real, imag', 
                np.stack((np.abs(myATF)/np.max(np.abs(myATF)), np.abs(myobjft)/np.max(np.abs(myobjft))), axis=0))


else:
    # Assign the initial guess to the object inside the fwd-model
    print('Assigning Variables')
    sess.run(tf.assign(muscat.TF_obj, np.real(obj_guess))); # assign abs of measurement as initial guess of 
    sess.run(tf.assign(muscat.TF_obj_absorption, np.imag(obj_guess))); # assign abs of measurement as initial guess of 
    muscat.zernikefactors *= 0
    muscat.zernikefactors[10]= +1.0
    sess.run(tf.assign(muscat.TF_zernikefactors, muscat.zernikefactors))
    
    #TODO: THE ZERNIKES count differently for BPM

# assert some memory or reset the lists
iter_last = 0
mylosslist = []; myfidelitylist = []
myposlosslist = []; myneglosslist = []
mytvlosslist = []; result_phaselist = []
result_absorptionlist = []; globalphaselist = []
globalabslist = []; myfwdlist = []
#%%
''' Optimize the model '''
print('Start optimizing')
#import scipy.io
#scipy.io.savemat('ExperimentAsfObj.mat', dict(asf=np.array(myASF), obj=np.array(matlab_val)))

for iterx in range(iter_last,Niter):
    
    # Change the learning rat - experimental
    if iterx == NreduceLR:
        print('Learning Rate has changed by factor of .1')
        experiments.my_learningrate = experiments.my_learningrate*.1
        
    #% This is for debbugging purposes - write the result to disk every n-iteration
    if(iterx==0 or not np.mod(iterx, Nsave)):
        my_res, my_res_absortpion, my_loss, my_fidelity, my_negloss, my_tvloss, myfwd =  \
            sess.run([muscat.TF_obj, muscat.TF_obj_absorption, tf_loss, tf_fidelity, tf_negsqrloss, tf_regloss, tf_fwd], \
                     feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:experiments.my_learningrate, muscat.tf_lambda_tv:experiments.lambda_tv, muscat.tf_eps:experiments.myepstvval}) #, muscat.TF_ATF_placeholder:myATF

        print('Loss@'+str(iterx)+': ' + str(my_loss) + ' - Fid: '+str(my_fidelity)+', Neg: '+str(my_negloss)+', TV: '+str(my_tvloss))
        myfwdlist.append(myfwd)
        mylosslist.append(my_loss)
        myfidelitylist.append(my_fidelity)
        myneglosslist.append(my_negloss)
        mytvlosslist.append(my_tvloss)
        result_phaselist.append(my_res)
        result_absorptionlist.append(my_res_absortpion)

        #% Display recovered Pupil
        plt.figure()
        myzernikes = sess.run(muscat.TF_zernikefactors)
        myshiftX = sess.run(muscat.TF_shiftIcX)
        myshiftY = sess.run(muscat.TF_shiftIcY)
        
        plt.subplot(141), plt.title('Po Phase'), plt.imshow(np.fft.fftshift(np.angle(sess.run(muscat.TF_Po_aberr)))), plt.colorbar(fraction=0.046, pad=0.04)
        plt.subplot(142), plt.title('Po abs'), plt.imshow(np.fft.fftshift(np.abs(sess.run(muscat.TF_Po_aberr)))), plt.colorbar(fraction=0.046, pad=0.04)
        plt.subplot(143), plt.title('Ic, shiftX: '+str(myshiftX)+' myShiftY: '+str(myshiftY)), plt.imshow(np.abs(sess.run(muscat.TF_Ic_shift))), plt.colorbar(fraction=0.046, pad=0.04)
        plt.subplot(144), plt.bar(np.linspace(1, np.squeeze(myzernikes.shape), np.squeeze(myzernikes.shape)), myzernikes, align='center', alpha=0.5)
        plt.savefig(savepath+'/Aberrations_'+str(iterx)+'.png'), plt.show()
        
        ''' Save Figures and Parameters '''
        muscat.saveFigures_list(savepath, myfwdlist, mylosslist, myfidelitylist, myneglosslist, mytvlosslist, result_phaselist, result_absorptionlist, 
                              globalphaselist, globalabslist, np_meas, figsuffix='Iter'+str(iterx))
        #%    
    # Alternate between pure object optimization and aberration recovery
    #sess.run([tf_lossop_tv], feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:experiments.my_learningrate, muscat.tf_lambda_tv:experiments.lambda_tv, muscat.tf_eps:experiments.myepstvval})

    # Do not optimize the object if we try to estimate the PSF (object is known!)
    if is_absorption and not is_estimatepsf:
        sess.run(tf_lossop_obj_absorption  , feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:experiments.my_learningrate, muscat.tf_lambda_tv:experiments.lambda_tv, muscat.tf_eps:experiments.myepstvval})
   
    elif(not is_estimatepsf):
        sess.run(tf_lossop_obj, feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:experiments.my_learningrate, muscat.tf_lambda_tv:experiments.lambda_tv, muscat.tf_eps:experiments.myepstvval})
   
    # print('Attetntion: Generalized costfunction1')
    if is_aberration and (iterx > is_aberation_iterstart) or is_estimatepsf:
        sess.run([tf_lossop_aberr], feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:experiments.my_learningrate, muscat.tf_lambda_tv:experiments.lambda_tv, muscat.tf_eps:experiments.myepstvval})


    if is_estimatepsf:
        zern_error = sess.run([TF_zernloss], feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:experiments.my_learningrate, muscat.tf_lambda_tv:experiments.lambda_tv, muscat.tf_eps:experiments.myepstvval})
        print(zern_error)
        
    if is_norm:
        sess.run([tf_lossop_norm], feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:experiments.my_learningrate, muscat.tf_lambda_tv:experiments.lambda_tv, muscat.tf_eps:experiments.myepstvval})
        

    iter_last = iterx

#%%
''' Save Figures and Parameters '''
muscat.saveFigures_list(savepath, myfwdlist, mylosslist, myfidelitylist, myneglosslist, mytvlosslist, result_phaselist, result_absorptionlist, 
                              globalphaselist, globalabslist, np_meas, figsuffix='FINAL')

data.export_realdatastack_h5(savepath+'/myrefractiveindex.h5', 'phase, abs', 
                        np.stack(((nip.extract(result_phaselist[-1], muscat.mysize,None,None)),
                                 (nip.extract(result_absorptionlist[-1], muscat.mysize,None,None))), axis=0))
data.export_realdatastack_h5(savepath+'/mymeas.h5', 'real, imag', 
                        np.stack((np.real(np_meas),
                                  np.imag(np_meas)), axis=0))
       
print('Zernikes: ' +str(np.real(sess.run(muscat.TF_zernikefactors))))
print('ShiftX/Y: '+ str(sess.run(muscat.TF_shiftIcX))+' / ' + str(sess.run(muscat.TF_shiftIcY)))


# backup current script
from shutil import copyfile
import os
src = (os.path.basename(__file__))
copyfile(src, savepath+'/script_bak.py')

#%
plt.imshow(np.fft.ifftshift(np.angle(sess.run(muscat.TF_Po_aberr))))
