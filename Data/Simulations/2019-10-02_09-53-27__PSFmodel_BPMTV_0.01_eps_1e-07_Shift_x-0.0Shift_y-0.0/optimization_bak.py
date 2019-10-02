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
import src.simulations as experiments 
import src.MyParameter as paras
import NanoImagingPack as nip

# Optionally, tweak styles.
mpl.rc('figure',  figsize=(8, 6))
mpl.rc('image', cmap='gray')

if(1):
    if(1):
        var_tvc = 1e-7
        var_tv =  1e-2
#for var_tv in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0, 1e1, 1e2):
    #for var_tvc in (1e-10, 1e-8, 1e-6, 1e-4, 1e-2):
        np_meas_file = './Data/PHANTOM/HeLa_cell_mat_obj_100.mat'; matname='HeLa_cell_mat'
        obj_real = data.import_realdata_h5(filename = np_meas_file, matname=matname)
        obj_absorption = obj_real*0
        obj_guess = (obj_real + 1j*obj_absorption)
        #%%
        '''Define some stuff related to infrastructure'''
        mytimestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        basepath = ""
        is_aberration = False
        is_aberation_iterstart = 0 # When to start optimizing for aberration?
        is_padding = False
        is_optimization = True   
        is_absorption = False
        is_obj_init_tikhonov = False # intialize the 
        is_norm = False # Want to have a floating value for the background?
        is_recomputemodel = True  # TODO: Make it automatic! 
        is_estimatepsf = False
        mybordersize = 20
        psf_model = 'BPM'
        #psf_model = 'BORN' # either compute BORN or BPM ()
        is_debugging = True # don't write all data to disk
        
        
        # Displaying/Saving
        Niter =  250
        Nsave = 25 # write info to disk
        NreduceLR = 1000 # when should we reduce the Learningrate? 
        
        experiments.lambda_reg = var_tv
        experiments.myepstvval = var_tvc
        
        '''Define some stuff related to infrastructure'''
        mytimestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        savepath = experiments.mysavepath +  mytimestamp + '_' + '_PSFmodel_' + psf_model + experiments.regularizer + '_' + str(experiments.lambda_reg) + '_eps_' +str(experiments.myepstvval) + '_' +'Shift_x-'+str(experiments.shiftIcX)+'Shift_y-' +str(experiments.shiftIcY) 
        tf_helper.mkdir(savepath)
        print('My path is: '+savepath )

        
        
        ''' 1.) Read in the parameters of the dataset ''' 
        experiments.result_fwd_bpm = '.\Data\Simulations\allAmp_simu_BPM.npy'
        np_meas = np.load(experiments.savepath_simu)
        
        # Generate Test-Object
        ''' File which stores the experimental parameters from the Q-PHASE setup 
            3.) Read in the parameters of the dataset ''' 
        myparams = paras.MyParameter()
        myparams.loadExperiment(experiments)
        myparams.print()
        
        ''' MODELLING StARTS HERE''' 
        tf.reset_default_graph()
        
        ''' Create the Model'''
        muscat = mus.MuScatModel(myparams, is_optimization=is_optimization)
        
        muscat.zernikefactors = experiments.zernikefactors
        muscat.zernikemask = experiments.zernikemask
          
        ''' Compute a first guess based on the experimental phase '''
        if(is_obj_init_tikhonov):
            print('Object is initialized with precomputed RI-distribution')
           
            if(0):
                print('ATTENTION: SIDELOAD a GD result!')
                obj_guess_filename = 'myrefractiveindex.h5'
                obj_guess = data.import_realdata_h5(filename = obj_guess_filename, matname='phase, abs0', is_complex=False)
            else:
                 obj_guess =  np.zeros(np_meas.shape)+muscat.params.nEmbb # np.angle(np_meas)## 
                 obj_guess = np.load('thikonovinvse.npy')
            
            #obj_guess = obj_guess-np.min(obj_guess); obj_guess = obj_guess/np.max(obj_guess)
            obj_guess = obj_guess-(np.min(np.real(obj_guess))+1j*np.min(np.imag(obj_guess)))
            if is_absorption:
                obj_guess = experiments.dn*np.real(obj_guess)/np.max(np.real(obj_guess))+1j*experiments.dn*np.imag(obj_guess)/np.max(np.imag(obj_guess))
            else:
                obj_guess = experiments.dn*np.real(obj_guess)/np.max(np.real(obj_guess))
        else:
            obj_guess =  np.zeros(np_meas.shape)# np.angle(np_meas)## 
            #obj_guess = obj_guess
        
        if is_estimatepsf:
            obj_guess =  -obj_val*experiments.dn+experiments.dn+1j*.1*obj_val
            #obj_guess = np.random.rand(np_meas.shape[0],np_meas.shape[1],np_meas.shape[2])*muscat.dn/2
        obj_guess = obj_guess+muscat.params.nEmbb # add background
        
        np_meas = np_meas
            
        
        ''' Define Fwd operator'''
        if(psf_model=='BORN'):
            # Test this Carringotn Padding to have borders at the dges where the optimizer can make pseudo-update
            # add carrington boundary regions
            my_border_region = np.array((np_meas.shape[0]//2,mybordersize,mybordersize)) # border-region around the object 
            bz, bx, by = my_border_region
            obj_guess_real = np.pad(np.real(obj_guess),[(bz, bz), (bx, bx), (by, by)], mode='constant', constant_values=np.mean(np.real(obj_guess)))
            obj_guess_imag = np.pad(np.imag(obj_guess),[(bz, bz), (bx, bx), (by, by)], mode='constant', constant_values=np.mean(1j*np.mean(np.imag(obj_guess))))
            obj_guess = obj_guess_real + 1j*obj_guess_imag
        
            ''' Compute the BORN model'''
            # Compute the System's properties (e.g. Pupil function/Illumination Source, K-vectors, etc.)¶
            muscat.computesys(obj=None, is_padding=is_padding, mysubsamplingIC=0, is_compute_psf=psf_model,is_dampic=experiments.is_dampic)
            muscat.TF_obj = tf.Variable(np.real(obj_guess), dtype=tf.float32, name='Object_Variable_Real')
            muscat.TF_obj_absorption = tf.Variable(np.imag(obj_guess), dtype=tf.float32, name='Object_Variable_Imag')
        
            ''' Create Model Instance'''
            muscat.computemodel()
            tf_fwd = muscat.computeconvolution(muscat.TF_ASF, is_padding='border',border_region=my_border_region)
            #tf_fwd = muscat.computeconvolution(muscat.TF_ASF, is_padding=True)    
        
        elif(psf_model=='BPM'):
            ''' Compute the Multiple Scattering model'''
            muscat.computesys(obj=None, is_padding=is_padding, mysubsamplingIC=experiments.mysubsamplingIC, is_compute_psf='BPM', is_dampic=experiments.is_dampic)
            muscat.TF_obj = tf.Variable(np.real(obj_guess), dtype=tf.float32, name='Object_Variable_Real')
            muscat.TF_obj_absorption = tf.Variable(np.imag(obj_guess), dtype=tf.float32, name='Object_Variable_Imag')
        
            tf_fwd = muscat.computemodel()
            tf_fwd = tf_fwd #+ 1j
           
        '''experiments.regularizer'''
        # Total Variation
        tf_regloss = 0.
        if(experiments.regularizer=='TV'):
            # TV experiments.regularizer
            print('We are using TV - Regularization') # (tfin, Eps=1e-15, doubleSided=False,regDataType=None)
           # TF_obj_tmp = tf_helper.extract(tf.cast(muscat.TF_obj, tf.float32), muscat.mysize)
            #TF_obj_absorption_tmp = tf_helper.extract(tf.cast(muscat.TF_obj_absorption, tf.float32), muscat.mysize)
            
            #tf_regloss =  muscat.tf_lambda_reg*reg.Reg_TV_RH(muscat.TF_obj, Eps=muscat.tf_eps)  #Alernatively tf_total_variation_regularization # total_variation
            #tf_regloss += muscat.tf_lambda_reg*reg.Reg_TV_RH(muscat.TF_obj_absorption, Eps=muscat.tf_eps)  #Alernatively tf_total_variation_regularization # total_variation
             
            #  [muscat.dx,muscat.dy,muscat.dz]
            mysqrt_real = reg.Reg_TV(muscat.TF_obj, BetaVals = [1,1,1], epsR=muscat.tf_eps, is_circ = True)  #Alernatively tf_total_variation_regularization # total_variation
            tf_regloss_real = tf.reduce_mean(mysqrt_real)
            mysqrt_imag = reg.Reg_TV(muscat.TF_obj_absorption, BetaVals = [1,1,1], epsR=muscat.tf_eps, is_circ = True)  #Alernatively tf_total_variation_regularization # total_variation
            tf_regloss_imag = tf.reduce_mean(mysqrt_imag)
            tf_regloss = muscat.tf_lambda_reg*(tf_regloss_real + tf_regloss_imag)
            #tf_regloss += muscat.tf_lambda_reg*reg.Reg_TV(muscat.TF_obj_absorption, BetaVals = [1,1,1], epsR=muscat.tf_eps, is_circ = True)  #Alernatively tf_total_variation_regularization # total_variation
            #lkk 
        elif(experiments.regularizer=='GR'):
            # Goods roughness rgularizer
            print('We are using GR - Regularization')
            tf_regloss += muscat.tf_lambda_reg*(reg.Reg_GR(muscat.TF_obj) + reg.Reg_GR(muscat.TF_obj_absorption))#, eps1=muscat.tf_eps, eps2=muscat.tf_eps)  #Alernatively tf_total_variation_regularization # total_variation
        elif(experiments.regularizer=='L1'):
            # L1 rgularizer
            print('We are using L1 - Regularization')
            tf_regloss += muscat.tf_lambda_reg*(reg.Reg_L1(muscat.TF_obj)+reg.Reg_L1(muscat.TF_obj_absorption))
        elif(experiments.regularizer=='L2'):
            # L1 rgularizer
            print('We are using L2 - Regularization')
            tf_regloss = muscat.tf_lambda_reg*(reg.Reg_L2(muscat.TF_obj)+reg.Reg_L2(muscat.TF_obj_absorption))
        
        tf_zernloss = experiments.lambda_zernike*reg.Reg_L2(muscat.TF_zernikefactors) # general loss on the zernike modes - don't over estimate them by value!
        tf_icshiftloss = experiments.lambda_icshift*reg.Reg_L2(muscat.TF_shiftIcX+muscat.TF_shiftIcY)
        
        # fuse all loss functions
        tf_regloss += tf_zernloss
        tf_regloss += tf_icshiftloss
        
        '''Define Optimizer'''
                                                
        '''Negativity Constraint'''                                          
        #tf_negsqrloss = reg.Reg_NegSqr(tf_helper.extract(tf.cast(muscat.TF_obj, tf.float32), muscat.mysize))#-tf.minimum(tf.reduce_min(muscat.TF_obj-1.),0) 
        tf_negsqrloss = reg.Reg_NegSqr(muscat.TF_obj-myparams.nEmbb-myparams.dn)#-tf.minimum(tf.reduce_min(muscat.TF_obj-1.),0) 
        tf_negsqrloss += reg.Reg_NegSqr(muscat.TF_obj_absorption)
        tf_negsqrloss *= experiments.lambda_neg
        
        
        # Correc the fwd model - not good here!
        tf_glob_real = tf.constant(0.,name='tf_glob_real')
        tf_glob_imag = tf.constant(0.,name='tf_glob_imag')
        tf_norm = tf.complex(tf_glob_real, tf_glob_imag) 
        
        '''Define Loss-function'''
        if(0):
            print('-------> Losstype is L1')
            tf_fidelity = tf.reduce_mean((tf.abs((muscat.tf_meas) - tf_fwd))) # allow a global phase parameter to avoid unwrapping effects
        elif(0):
            print('-------> Losstype mixed L2 ')
            tf_fidelity = tf.reduce_mean(tf_helper.tf_abssqr(tf.real(muscat.tf_meas) - tf.real(tf_fwd))+tf_helper.tf_abssqr(tf.imag(muscat.tf_meas) - tf.imag(tf_fwd))) # allow a global phase parameter to avoid unwrapping effects
        elif(1):
            print('-------> Losstype is L2')
            tf_fidelity = tf.reduce_mean(tf_helper.tf_abssqr(muscat.tf_meas - tf_fwd + tf_norm)) # allow a global phase parameter to avoid unwrapping effects
        elif(0):
            print('-------> Losstype is L2')
            tf_fidelity_real = tf.reduce_mean(tf.losses.mean_squared_error(tf.real(muscat.tf_meas), tf.real(tf_fwd))) # allow a global phase parameter to avoid unwrapping effects
            tf_fidelity_imag = tf.reduce_mean(tf.losses.mean_squared_error(tf.imag(muscat.tf_meas), tf.imag(tf_fwd))) # allow a global phase parameter to avoid unwrapping effects
            tf_fidelity = tf_fidelity_real + tf_fidelity_imag 
        else:
            print('-------> Losstype is L2')
            tf_fidelity_real = tf.reduce_mean(tf.losses.mean_squared_error(tf.abs(muscat.tf_meas), tf.abs(tf_fwd))) # allow a global phase parameter to avoid unwrapping effects
            tf_fidelity_imag = tf.reduce_mean(tf.losses.mean_squared_error(tf_helper.angle(muscat.tf_meas), tf_helper.angle(tf_fwd))) # allow a global phase parameter to avoid unwrapping effects
            tf_fidelity = tf_fidelity_real + tf_fidelity_imag 
        
        tf_loss = tf_fidelity + tf_negsqrloss + tf_regloss
           
        #tf_optimizer = tf.train.AdamOptimizer(muscat.tf_learningrate)
        tf_optimizer = tf.train.MomentumOptimizer(1e1, momentum=.9, use_nesterov=True)
        if not is_estimatepsf:
            # ordinary case - we want to optimize for the object                
            tf_lossop_obj = tf_optimizer.minimize(tf_loss, var_list = [muscat.TF_obj])
            tf_lossop_obj_absorption = tf_optimizer.minimize(tf_loss, var_list = [muscat.TF_obj,muscat.TF_obj_absorption])
            tf_lossop = tf_optimizer.minimize(tf_loss, var_list = [muscat.TF_obj,muscat.TF_obj_absorption, muscat.TF_shiftIcX, muscat.TF_shiftIcY, muscat.TF_zernikefactors])
            tf_lossop_aberr = tf_optimizer.minimize(tf_loss, var_list = [muscat.TF_shiftIcX, muscat.TF_shiftIcY, muscat.TF_zernikefactors])
          
        else:
            
            # unordinary case - we want to optimize for the system only
            tf_lossop_aberr = tf_optimizer.minimize(tf_loss, var_list = [muscat.TF_shiftIcX, muscat.TF_shiftIcY, muscat.TF_zernikefactors])
            tf_lossop_norm = tf_optimizer.minimize(tf_loss, var_list = [tf_glob_real,tf_glob_imag])
        ''' Initialize the model '''
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
           
        ''' Compute the ATF '''
        if(psf_model=='BORN' or psf_model == '3QDPC'):
            #%%
            print('We are precomputing the PSF')
            myATF = sess.run(muscat.TF_ATF)
            myASF = sess.run(muscat.TF_ASF)    
            
           
            if(is_debugging):
                #% write Freq-Support to disk
                tf_helper.plot_ASF_ATF(savepath, myATF, myASF)
                tf_helper.plot_obj_fft(savepath, np_meas)
        
        
        # assert some memory or reset the lists
        iter_last = 0
        mylosslist = []; myfidelitylist = []
        myposlosslist = []; myneglosslist = []
        mytvlosslist = []; result_phaselist = []
        result_absorptionlist = []; globareallist = []
        globalimaglist = []; myfwdlist = []
        #%%
        ''' Optimize the model '''
        print('Start optimizing')
        #import scipy.io
        #scipy.io.savemat('ExperimentAsfObj.mat', dict(asf=np.array(myASF), obj=np.array(np_meas)))
        
        for iterx in range(iter_last,Niter):
            

            
            # Change the learning rat - experimental
            if iterx == NreduceLR:
                print('Learning Rate has changed by factor of .1')
                experiments.my_learningrate = experiments.my_learningrate*.1
                
            #% This is for debbugging purposes - write the result to disk every n-iteration
            if(iterx==0 or not np.mod(iterx, Nsave)):
                my_res, my_res_absortpion, my_loss, my_fidelity, my_negloss, my_tvloss, myfwd, myglobalreal, myglobalimag =  \
                    sess.run([muscat.TF_obj, muscat.TF_obj_absorption, tf_loss, tf_fidelity, tf_negsqrloss, tf_regloss, tf_fwd, tf_glob_real, tf_glob_imag], \
                             feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:experiments.my_learningrate, muscat.tf_lambda_reg:experiments.lambda_reg, muscat.tf_eps:experiments.myepstvval}) #, muscat.TF_ATF_placeholder:myATF
        
                print('Loss@'+str(iterx)+': ' + str(my_loss) + ' - Fid: '+str(my_fidelity)+', Neg: '+str(my_negloss)+', TV: '+str(my_tvloss))
                myfwdlist.append(myfwd)
                mylosslist.append(my_loss)
                myfidelitylist.append(my_fidelity)
                myneglosslist.append(my_negloss)
                mytvlosslist.append(my_tvloss)
                result_phaselist.append(my_res)
                result_absorptionlist.append(my_res_absortpion)
                globareallist.append(myglobalreal)
                globalimaglist.append(myglobalimag)
                #% Display recovered Pupil
                plt.figure()
                myzernikes = sess.run(muscat.TF_zernikefactors)
                myshiftX = sess.run(muscat.TF_shiftIcX)
                myshiftY = sess.run(muscat.TF_shiftIcY)
                
                plt.subplot(141), plt.title('Po Phase'), plt.imshow(np.angle(sess.run(muscat.TF_Po_aberr))), plt.colorbar(fraction=0.046, pad=0.04)
                plt.subplot(142), plt.title('Po abs'), plt.imshow(np.abs(sess.run(muscat.TF_Po_aberr))), plt.colorbar(fraction=0.046, pad=0.04)
                plt.subplot(143), plt.title('Ic, shiftX: '+str(myshiftX)+' myShiftY: '+str(myshiftY)), plt.imshow(np.abs(sess.run(muscat.TF_Ic_shift))), plt.colorbar(fraction=0.046, pad=0.04)
                plt.subplot(144), plt.bar(np.linspace(1, np.squeeze(myzernikes.shape), np.squeeze(myzernikes.shape)), myzernikes, align='center', alpha=0.5)
                plt.savefig(savepath+'/Aberrations_'+str(iterx)+'.png'), plt.show()
                
                ''' Save Figures and Parameters '''
                muscat.saveFigures_list(savepath, myfwdlist, mylosslist, myfidelitylist, myneglosslist, mytvlosslist, result_phaselist, result_absorptionlist, 
                                      globareallist, globalimaglist, np_meas, figsuffix='Iter'+str(iterx))
                #%    
            # Alternate between pure object optimization and aberration recovery
            #sess.run([tf_lossop_tv], feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:experiments.my_learningrate, muscat.tf_lambda_reg:experiments.lambda_reg, muscat.tf_eps:experiments.myepstvval})
        
            # Do not optimize the object if we try to estimate the PSF (object is known!)
            if is_absorption and not is_estimatepsf:
                sess.run(tf_lossop_obj_absorption  , feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:experiments.my_learningrate, muscat.tf_lambda_reg:experiments.lambda_reg, muscat.tf_eps:experiments.myepstvval})
           
            elif(not is_estimatepsf):
                sess.run(tf_lossop_obj, feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:experiments.my_learningrate, muscat.tf_lambda_reg:experiments.lambda_reg, muscat.tf_eps:experiments.myepstvval})
           
            # print('Attetntion: Generalized costfunction1')
            if is_aberration and (iterx > is_aberation_iterstart) or is_estimatepsf:
                sess.run([tf_lossop_aberr], feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:experiments.my_learningrate, muscat.tf_lambda_reg:experiments.lambda_reg, muscat.tf_eps:experiments.myepstvval})
        
        
            #if is_estimatepsf:
            #zern_error = sess.run([tf_zernloss], feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:experiments.my_learningrate, muscat.tf_lambda_reg:experiments.lambda_reg, muscat.tf_eps:experiments.myepstvval})
            #print(zern_error)
                
            if is_norm:
                sess.run([tf_lossop_norm], feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:experiments.my_learningrate, muscat.tf_lambda_reg:experiments.lambda_reg, muscat.tf_eps:experiments.myepstvval})
                
        
            iter_last = iterx
        
        #%%
        ''' Save Figures and Parameters '''
        muscat.saveFigures_list(savepath, myfwdlist, mylosslist, myfidelitylist, myneglosslist, mytvlosslist, result_phaselist, result_absorptionlist, 
                                      globareallist, globalimaglist, np_meas, figsuffix='FINAL')
        
        data.export_realdatastack_h5(savepath+'/myrefractiveindex.h5', 'phase, abs', 
                                np.stack(((nip.extract(result_phaselist[-1], muscat.mysize,None,None)),
                                         (nip.extract(result_absorptionlist[-1], muscat.mysize,None,None))), axis=0))
        data.export_realdatastack_h5(savepath+'/mymeas.h5', 'real, imag', 
                                np.stack((np.real(np_meas),
                                          np.imag(np_meas)), axis=0))
               
        print('Zernikes: ' +str(np.real(sess.run(muscat.TF_zernikefactors))))
        print('ShiftX/Y: '+ str(sess.run(muscat.TF_shiftIcX))+' / ' + str(sess.run(muscat.TF_shiftIcY)))
        
        #nip.v5(nip.cat(np.stack((np.flip(nip.extract(result_phaselist[-1], muscat.mysize,None,None),0),np.real(np_meas), np.imag(np_meas)), axis=0)))
        nip.v5(nip.cat(np.stack(((nip.extract(result_phaselist[-1], muscat.mysize,None,None)),np.real(np_meas), np.imag(np_meas)), axis=0)))
        nip.v5(nip.cat(np.stack(((nip.extract(result_absorptionlist[-1], muscat.mysize,None,None)),np.real(np_meas), np.imag(np_meas)), axis=0)))
        
        import tifffile as tif
        tif.imsave('real_tif.tif', nip.extract(result_phaselist[-1], muscat.mysize,None,None), dtype='float32')
        # backup current script
        from shutil import copyfile
        src = (os.path.basename(__file__))
        folder = os.path.dirname
        copyfile(src, savepath+'/optimization_bak.py')
        copyfile('./src/experiments.py', savepath+'/experiments_bak.py')
        
