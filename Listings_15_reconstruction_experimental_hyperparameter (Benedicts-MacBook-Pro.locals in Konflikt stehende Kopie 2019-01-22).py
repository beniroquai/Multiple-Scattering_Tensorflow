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
mpl.rc('figure',  figsize=(10, 7))
mpl.rc('image', cmap='gray')
plt.switch_backend('agg')


#%%
'''Define some stuff related to infrastructure'''
mytimestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
basepath = './'#'/projectnb/cislidt/diederich/muScat/Multiple-Scattering_Tensorflow/'
resultpath = 'Data/DROPLETS/RESULTS/'
savepath = basepath + resultpath + mytimestamp

# Create directory
print('Start the code and try to mkdir at:')
print(savepath)
try: 
    os.mkdir(savepath)
except(FileExistsError): 
    print('Folder exists already')

# Define parameters 
is_padding = False 
is_display = True
is_optimization = True 
is_optimization_psf = False
is_flip = False
is_measurement = True

# data files for parameters and measuremets 
if is_measurement:
    if(False):
        matlab_val_file = './Data/DROPLETS/allAmp_red.mat'      #'./Data/DROPLETS/allAmp_simu.npy' #
        matlab_par_file = './Data/DROPLETS/myParameterNew.mat'   
        matlab_val_name = 'allAmp_red'
        matlab_par_name = 'myParameterNew'  
    if(True):
        matlab_val_file = './Data/DROPLETS/S19_multiple/S19_subroi5.mat'      #'./Data/DROPLETS/allAmp_simu.npy' #
        matlab_par_file = './Data/DROPLETS/S19_multiple/Parameter.mat'   
        matlab_val_name = 'allAmp_red'
        matlab_par_name = 'myParameter' 
        
else:
    matlab_val_file = './Data/DROPLETS/RESULTS/allAmp_simu.npy' #'./Data/DROPLETS/allAmp_simu.mat'      
    matlab_par_file = './Data/DROPLETS/myParameterNew.mat' 
    matlab_val_name = 'allAmp_red'
    matlab_par_name = 'myParameterNew'

# microscope parameters
zernikefactors = np.array((0,0,0,0,0,0,0.5,-0.5,0)) # representing the 9 first zernike coefficients in noll-writings 
shiftIcY=-1
shiftIcX=-0
dn = 1.437-1.3326
NAc = .52

'''START CODE'''
tf.reset_default_graph() # just in case there was an open session

'''Define Optimization Parameters'''
# these are hyperparameters
my_learningrate = 1e-2  # learning rate
lambda_tv =  1e-1 # ((100, 10, 1, 1e-1)) # lambda for Total variation
eps_tv = 1e-1 #  ((1e-2, 1e-1, 1, 10))

# these are fixed parameters
lambda_neg = 10
Niter = 1000
Ndisplay = 15

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

if(is_flip):
    np_meas_raw = np.flip(matlab_val,0)
    print('Attention: We are flipping the data!')
else:
    np_meas_raw = matlab_val
    print('do we need to flip the data?! -> Observe FFT!!')

''' Create the Model'''
muscat = mus.MuScatModel(matlab_pars, is_optimization=is_optimization, is_optimization_psf = is_optimization_psf)
muscat.Nx,muscat.Ny = int(np.squeeze(matlab_pars['Nx'].value)), int(np.squeeze(matlab_pars['Ny'].value))
muscat.shiftIcY=shiftIcY
muscat.shiftIcX=shiftIcX
muscat.dn = dn
muscat.NAc = NAc
muscat.dz = muscat.lambda0/4

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
print(muscat.Ic.shape)

# Generate Computational Graph (fwd model)
tf_fwd = muscat.computemodel()

print('Evtl unwrap it!')

# this is the initial guess of the reconstruction
np_meas=np_meas_raw#*np.exp(1j*np.pi)


if(False): # good for the experiment
    init_guess = np.angle(np_meas)
    init_guess = init_guess - np.min(init_guess)
    init_guess = init_guess**2
    init_guess = init_guess/np.max(init_guess)*muscat.dn
elif(False): # good for the simulation
    init_guess = -np.abs(np_meas)
    init_guess = init_guess - np.min(init_guess)
    init_guess = init_guess**2
    init_guess = init_guess/np.max(init_guess)*muscat.dn
    init_guess = np.flip(init_guess,0)
elif(False):
    init_guess = np.ones(np_meas.shape)*muscat.dn
elif(False):
    init_guess = tf_go.generateObject(mysize=muscat.mysize, obj_dim=muscat.dx, obj_type ='sphere', diameter = 7, dn = muscat.dn)
else:
    init_guess = np_meas
    ''' Create a 3D Refractive Index Distributaton as a artificial sample'''
    mydiameter = 7
    obj = tf_go.generateObject(mysize=muscat.mysize, obj_dim=muscat.dx, obj_type ='sphere', diameter = mydiameter, dn = .4)
    obj_absorption = tf_go.generateObject(mysize=muscat.mysize, obj_dim=muscat.dx, obj_type ='sphere', diameter = mydiameter, dn = 1)
    obj = obj+1j*obj_absorption*.1
    obj = np.roll(obj,12,0)
    init_guess = obj
    #init_guess = np.load('my_res_cmplx.npy')
    #init_guess = np.random.randn(np_meas.shape[0],init_guess.shape[1],init_guess.shape[2])*muscat.dn


# Estimate the Phase difference between Measurement and Simulation
#%%
@tf.custom_gradient
def clip_grad_layer(x):
    def grad(dy):
        return tf.clip_by_value(dy, -5, 5)
    return tf.identity(x), grad

'''Numpy to Tensorflow'''
np_meas = matlab_val
np_mean = np.mean(np_meas)

'''Regression + Regularization'''
tf_meas = tf.placeholder(dtype=tf.complex64, shape=init_guess.shape)
             
'''Define Cost-function'''
tf_lambda_tv = tf.placeholder(tf.float32, [])
tf_eps = tf.placeholder(tf.float32, [])
tf_tvloss = tf_lambda_tv*reg.Reg_TV(muscat.TF_obj, BetaVals = [muscat.dx,muscat.dy,muscat.dz], epsR=tf_eps, is_circ = True)  #Alernatively tf_total_variation_regularization # total_variation
tf_tvloss += tf_lambda_tv*reg.Reg_TV(muscat.TF_obj_absorption, BetaVals = [muscat.dx,muscat.dy,muscat.dz], epsR=tf_eps, is_circ = True)  #Alernatively tf_total_variation_regularization # total_variation
#tf_tvloss = tf_lambda_tv*reg.total_variation_iso_conv(muscat.TF_obj,  eps=epsTV, step_sizes = [muscat.dx,muscat.dy,muscat.dz])  #Alernatively tf_total_variation_regularization # total_variation
#tf_tvloss += tf_lambda_tv*reg.total_variation_iso_conv(muscat.TF_obj_absorption,  eps=epsTV, step_sizes = [muscat.dx,muscat.dy,muscat.dz])  #Alernatively tf_total_variation_regularization # total_variation

tf_negsqrloss = lambda_neg*reg.Reg_NegSqr(muscat.TF_obj)
tf_globalphase = tf.Variable(0., tf.float32, name='var_phase')
tf_globalabs = tf.Variable(1., tf.float32, name='var_abs')# 
#tf_fidelity = tf.reduce_sum((tf_helper.tf_abssqr(tf_fwd  - (tf_meas/tf.cast(tf.abs(tf_globalabs), tf.complex64)*tf.exp(1j*tf.cast(tf_globalphase, tf.complex64)))))) # allow a global phase parameter to avoid unwrapping effects
tf_fidelity = tf.reduce_mean((tf_helper.tf_abssqr(tf_fwd  - (tf_meas/tf.cast(tf.abs(tf_globalabs), tf.complex64)*tf.exp(1j*tf.cast(tf_globalphase, tf.complex64)))))) # allow a global phase parameter to avoid unwrapping effects
#tf_fidelity = clip_grad_layer(tf_fidelity)
tf_grads = tf.gradients(tf_fidelity, [muscat.TF_obj])[0]

tf_loss = tf_fidelity/np_mean +  tf_negsqrloss + tf_tvloss #tf_negloss + tf_posloss + tf_tvloss

'''Define Optimizer'''
tf_learningrate = tf.placeholder(tf.float32, []) 
tf_optimizer = tf.train.AdamOptimizer(tf_learningrate)
if(False):
    gradients, variables = zip(*tf_optimizer.compute_gradients(tf_loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    tf_lossop = tf_optimizer.apply_gradients(zip(gradients, variables))

#tf_optimizer = tf.train.MomentumOptimizer(tf_learningrate, momentum = .9, use_nesterov=True)
#tf_optimizer = tf.train.ProximalGradientDescentOptimizer(tf_learningrate)
else:
    tf_lossop = tf_optimizer.minimize(tf_loss)



# optimize over the hyperparameters
#for mytvregval in lambda_tv:
#    for myepstvval in eps_tv:
if(1):
    if(1):
        mytvregval = lambda_tv
        myepstvval = eps_tv
       
        '''Define some stuff related to infrastructure'''
        mytimestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        savepath = os.path.join('./Data/DROPLETS/RESULTS/', mytimestamp+'_TV_'+str(mytvregval)+'_eps_'+str(myepstvval))
        
        # Create directory
        try: 
            os.mkdir(savepath)
        except(FileExistsError): 
            print('Folder exists already')
        
        ''' Save Parameters '''
        muscat.writeParameterFile(my_learningrate, mytvregval, myepstvval, filepath = savepath+'/myparameters.yml')
        
        ''' Evaluate the model '''
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        if is_optimization:
            if is_padding:
                # Pad object with zeros along X/Y
                init_guess_tmp= np.zeros(muscat.mysize)# + 1j*np.zeros(muscat.mysize)
                init_guess_tmp[:,muscat.Nx//2-muscat.Nx//4:muscat.Nx//2+muscat.Nx//4, muscat.Ny//2-muscat.Ny//4:muscat.Ny//2+muscat.Ny//4] =init_guess
                init_guess = init_guess_tmp
        
            sess.run(tf.assign(muscat.TF_obj, np.real(init_guess))); # assign abs of measurement as initial guess of 
            sess.run(tf.assign(muscat.TF_obj_absorption, np.imag(init_guess))); # assign abs of measurement as initial guess of 
        
        my_fwd = sess.run(tf_fwd)
        mysize = my_fwd.shape
        
        
        # We assume, that there is a global phase mismatch between measurment and first estimate of the fwd model, this can be estimated by the difference of mean phase of the two
        # subtracting the mean phase from either measurement or the fwd model could help to speed up the optimization
        # this is the initial guess of the reconstruction
        np_meas = matlab_val
        myinitphase = np.mean(np.angle(np_meas))-np.mean(np.angle(my_fwd))
        print('My Init Phase is :'+str(myinitphase))
        np_meas=np_meas*np.exp(-1j*(myinitphase+2)) # subtract globaphase - anyway we want to optimize for that, but now the global phase can be assumed to be 0 initally
        if(is_display): plt.subplot(231), plt.title('Angle XZ - Measurement'),plt.imshow(np.angle(np_meas)[:,mysize[1]//2,:]), plt.colorbar()#, plt.show()
        if(is_display): plt.subplot(232), plt.title('Angle YZ - Measurement'),plt.imshow(np.angle(np_meas)[:,:,mysize[2]//2]), plt.colorbar()#, plt.show()
        if(is_display): plt.subplot(233), plt.title('Angle XY - Measurement'),plt.imshow(np.angle(np_meas)[mysize[0]//2,:,:]), plt.colorbar(), plt.show()
        
        if(is_display): plt.subplot(234), plt.title('Angle YZ - Simulation'),plt.imshow(np.angle(my_fwd)[:,mysize[1]//2,:]), plt.colorbar()#, plt.show()
        if(is_display): plt.subplot(235), plt.title('Angle XZ - Simulation'),plt.imshow(np.angle(my_fwd)[:,:,mysize[2]//2]), plt.colorbar()#, plt.show()
        if(is_display): plt.subplot(236), plt.title('Angle XY - Simulation'),plt.imshow(np.angle(my_fwd)[mysize[0]//2,:,:]), plt.colorbar(), plt.show()
        
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
        
        #%%
        ''' Optimize the model '''
        print('Start optimizing')
        np_meas = matlab_val # use the previously simulated data
        for iterx in range(iter_last,Niter):
            if iterx == 1000:
                my_learningrate = my_learningrate#*.1
            # try to optimize
            mylambdatv = mytvregval#/((iterx+1)/100)
            if(iterx==0 or not np.mod(iterx, Ndisplay)):
                my_res, my_res_absortpion, my_loss, my_fidelity, my_negloss, my_tvloss, myglobalphase, myglobalabs, myfwd =  \
                    sess.run([muscat.TF_obj, muscat.TF_obj_absorption, tf_loss, tf_fidelity, tf_negsqrloss, tf_tvloss, tf_globalphase, tf_globalabs, tf_fwd], \
                             feed_dict={tf_meas:np_meas, tf_learningrate:my_learningrate, tf_lambda_tv:mylambdatv, tf_eps:myepstvval})
        
                print('Loss@'+str(iterx)+': ' + str(my_loss) + ' - Fid: '+str(my_fidelity)+', Neg: '+str(my_negloss)+', TV: '+str(my_tvloss)+' G-Phase:'+str(myglobalphase)+' G-ABS: '+str(myglobalabs))        
                mylosslist.append(my_loss)
                myfidelitylist.append(my_fidelity)
                myneglosslist.append(my_negloss)
                mytvlosslist.append(my_tvloss)
                result_phaselist.append(my_res)
                result_absorptionlist.append(my_res_absortpion)
                globalphaselist.append(myglobalphase)
                globalabslist.append(myglobalabs)  

            else:
                _, mygrads = sess.run([tf_lossop,tf_grads], feed_dict={tf_meas:np_meas, tf_learningrate:my_learningrate, tf_lambda_tv:mylambdatv, tf_eps:myepstvval})
                #plt.imshow(np.abs(my_res[:,50,:]))
                #print(mygrads)
                #print(mygrads.shape)
                #_, myfwd = sess.run([tf_lossop,tf_fwd], feed_dict={tf_meas:np_meas, tf_learningrate:my_learningrate, tf_lambda_tv:mylambdatv})

        # Save everything to disk
        iter_last = iterx
        
        is_display=True
        myfwd, mymeas, my_res, myres_absorption = sess.run([tf_fwd, tf_meas, muscat.TF_obj, muscat.TF_obj_absorption], feed_dict={tf_meas:np_meas})
        
        # This is the reconstruction
        plt.fig()
        plt.subplot(231), plt.title('ABS XZ'),plt.imshow(np.abs(myfwd)[:,myfwd.shape[1]//2,:]), plt.colorbar()#, plt.show()
        plt.subplot(232), plt.title('ABS YZ'),plt.imshow(np.abs(myfwd)[:,:,myfwd.shape[2]//2]), plt.colorbar()#, plt.show()
        plt.subplot(233), plt.title('ABS XY'),plt.imshow(np.abs(myfwd)[myfwd.shape[0]//2,:,:]), plt.colorbar()#, plt.show()
        plt.subplot(234), plt.title('Angle XZ'),plt.imshow(np.angle(myfwd)[:,myfwd.shape[1]//2,:]), plt.colorbar()#, plt.show()
        plt.subplot(235), plt.title('Angle XZ'),plt.imshow(np.angle(myfwd)[:,:,myfwd.shape[2]//2]), plt.colorbar()#, plt.show()
        plt.subplot(236), plt.title('Angle XY'),plt.imshow(np.angle(myfwd)[myfwd.shape[0]//2,:,:]), plt.colorbar()#, plt.show()
        plt.savefig(savepath+'/res_myfwd.png'), plt.show()
    
        # This is the measurment
        plt.fig()
        plt.subplot(231), plt.title('ABS XZ'),plt.imshow(np.abs(np_meas)[:,np_meas.shape[1]//2,:]), plt.colorbar()#, plt.show()
        plt.subplot(232), plt.title('ABS YZ'),plt.imshow(np.abs(np_meas)[:,:,np_meas.shape[2]//2]), plt.colorbar()#, plt.show()
        plt.subplot(233), plt.title('ABS XY'),plt.imshow(np.abs(np_meas)[np_meas.shape[0]//2,:,:]), plt.colorbar()#, plt.show()
        plt.subplot(234), plt.title('Angle XZ'),plt.imshow(np.angle(np_meas)[:,np_meas.shape[1]//2,:]), plt.colorbar()#, plt.show()
        plt.subplot(235), plt.title('Angle XZ'),plt.imshow(np.angle(np_meas)[:,:,np_meas.shape[2]//2]), plt.colorbar()#, plt.show()
        plt.subplot(236), plt.title('Angle XY'),plt.imshow(np.angle(np_meas)[np_meas.shape[0]//2,:,:]), plt.colorbar()#, plt.show()
        plt.savefig(savepath+'/res_mymeas.png'), plt.show()
    
        # This is the residual
        plt.fig()
        plt.subplot(231), plt.title('Residual ABS XZ'),plt.imshow((np.abs(myfwd)-np.abs(np_meas))[:,myfwd.shape[1]//2,:]), plt.colorbar()#, plt.show()
        plt.subplot(232), plt.title('Residual ABS YZ'),plt.imshow((np.abs(myfwd)-np.abs(np_meas))[:,:,myfwd.shape[2]//2]), plt.colorbar()#, plt.show()
        plt.subplot(233), plt.title('Residual ABS XY'),plt.imshow((np.abs(myfwd)-np.abs(np_meas))[myfwd.shape[0]//2,:,:]), plt.colorbar()#, plt.show()
        plt.subplot(234), plt.title('Residual Angle XZ'),plt.imshow((np.angle(myfwd)-np.angle(np_meas))[:,myfwd.shape[1]//2,:]), plt.colorbar()#, plt.show()
        plt.subplot(235), plt.title('Residual Angle XZ'),plt.imshow((np.angle(myfwd)-np.angle(np_meas))[:,:,myfwd.shape[2]//2]), plt.colorbar()#, plt.show()
        plt.subplot(236), plt.title('Residual Angle XY'),plt.imshow((np.angle(myfwd)-np.angle(np_meas))[myfwd.shape[0]//2,:,:]), plt.colorbar()#, plt.show()
        plt.savefig(savepath+'/res_myresidual.png'), plt.show()
        
        
        # diplay the error over time
        plt.fig()
        plt.subplot(231), plt.title('Error/Cost-function'), plt.semilogy((np.array(mylosslist)))#, plt.show()
        plt.subplot(232), plt.title('Fidelity-function'), plt.semilogy((np.array(myfidelitylist)))#, plt.show()
        plt.subplot(233), plt.title('Neg-loss-function'), plt.plot(np.array(myneglosslist))#, plt.show()
        plt.subplot(234), plt.title('TV-loss-function'), plt.semilogy(np.array(mytvlosslist))#, plt.show()
        plt.subplot(235), plt.title('Global Phase'), plt.plot(np.array(globalphaselist))#, plt.show()
        plt.subplot(236), plt.title('Global ABS'), plt.plot(np.array(globalabslist))#, plt.show()
        plt.savefig(savepath+'/myplots.png'), plt.show()
        
        plt.fig()
        plt.subplot(131), plt.title('Result Phase: XZ'),plt.imshow(my_res[:,my_res.shape[1]//2,:]), plt.colorbar()#, plt.show()
        plt.subplot(132), plt.title('Result Phase: XZ'),plt.imshow(my_res[:,:,my_res.shape[2]//2]), plt.colorbar()#, plt.show()
        plt.subplot(133), plt.title('Result Phase: XY'),plt.imshow(my_res[my_res.shape[0]//2,:,:]), plt.colorbar()
        plt.savefig(savepath+'/RI_result.png'), plt.show()
        
        plt.fig()
        plt.subplot(131), plt.title('Result Abs: XZ'),plt.imshow(my_res_absortpion[:,my_res.shape[1]//2,:]), plt.colorbar()#, plt.show()
        plt.subplot(132), plt.title('Result abs: XZ'),plt.imshow(my_res_absortpion[:,:,my_res.shape[2]//2]), plt.colorbar()#, plt.show()
        plt.subplot(133), plt.title('Result abs: XY'),plt.imshow(my_res_absortpion[my_res.shape[0]//2,:,:]), plt.colorbar()
        plt.savefig(savepath+'/RI_abs_result.png'), plt.show()
        
        print(np.real(sess.run(muscat.TF_zernikefactors)))
        plt.fig()
        plt.subplot(121), plt.title('Po Phase'), plt.imshow(np.fft.fftshift(np.angle(sess.run(muscat.TF_Po_aberr)))), plt.colorbar()
        plt.subplot(122), plt.title('Po abs'), plt.imshow(np.fft.fftshift(np.abs(sess.run(muscat.TF_Po_aberr)))), plt.colorbar()
        plt.savefig(savepath+'/recovered_pupil.png'), plt.show()
    
        data.export_realdatastack_h5(savepath+'/myrefractiveindex.h5', 'temp', np.array(result_phaselist))
        data.export_realdatastack_h5(savepath+'/myrefractiveindex_absorption.h5', 'temp', np.array(result_absorptionlist))
        myobj = my_res+1j*my_res_absortpion
        np.save('my_res_cmplx', myobj)