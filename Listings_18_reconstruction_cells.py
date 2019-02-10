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


#make both devices visible to tensorflow 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

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

'''Define Optimization Parameters'''
# these are hyperparameters
my_learningrate = 1e-3  # learning rate
lambda_tv =  100#5e-1#((1e0, 1e-1, 1e-2)) # lambda for Total variation - 1e-1
eps_tv = 1e-8#((1e-2, 1e-1, 1e-0)) # - 1e-1 # smaller == more blocky
# these are fixed parameters
lambda_neg = 10000
Niter = 1500
Ndisplay = 15
Noptpsf = 1
# data files for parameters and measuremets 
matlab_val_file = './Data/cells/Hologramm.tif_allAmp.mat' #cells_20x_100a.tif_allAmp.mat' #'./Data/DROPLETS/S19_multiple/Spheres/S19_subroi48.mat'  #5 # 20 has this severe shadow    #'./Data/DROPLETS/allAmp_simu.npy' #
matlab_par_file = './Data/cells/Hologramm.tif_myParameter.mat' #Hologramm.tif_allAmp.mat' #cells_20x_100a.tif_myParameter.mat'   
matlab_val_name = 'allAmpSimu'
matlab_par_name = 'myParameter' 
        
# microscope parameters
#zernikefactors = np.array((0,0,0,0,0,0,0.,-0.,0)) # representing the 9 first zernike coefficients in noll-writings 
#zernikefactors = np.array((-0.13543801 ,-1.8246844 , -0.7559651 ,  0.2754147 ,  2.322039 ,  -2.872361, -0.28803617, -0.25946134,  4.9388413 ))
#zernikefactors = np.array((0.10448612, -0.08286186,  0.18136881 ,-0.11662757, -0.09957132,  0.14661853, -0.14000118, -0.29074576,  0.11014813))

zernikefactors = 0*np.array((0, 0, 0, 0.001 , .22, -.31, .02, -1.0001,  .02, -0.02, -0.149230834))
#zernikefactors = np.array((0,0,0,0,0,0,-1,-1,0,0,1)) # representing the 9 first zernike coefficients in noll-writings 
zernikemask = np.array(np.abs(zernikefactors)>0)*1#!= np.array((0, 0, 0, 0, 0, 0, , 1, 1, 1, 1))# mask which factors should be updated
shiftIcY=.72
shiftIcX= .42
dn = .05#(1.437-1.3326)#/np.pi
NAc = .32
#Nx = 50
#Ny = 50
#Nz = 30

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

#%%
matlab_val.shape
#%%
mysize_old = matlab_val.shape
#matlab_val = matlab_val[mysize_old[0]//2-Nz//2:mysize_old[0]//2+Nz//2,mysize_old[1]//2-Ny//2:mysize_old[1]//2+Ny//2,mysize_old[2]//2-Nx//2:mysize_old[2]//2+Nx//2]
print('.....> ATTENTION: Taking ocnjugated of object"')
''' Create the Model'''
muscat = mus.MuScatModel(matlab_pars, is_optimization=is_optimization)
muscat.Nx,muscat.Ny = int(np.squeeze(matlab_pars['Nx'].value)), int(np.squeeze(matlab_pars['Ny'].value))
muscat.shiftIcY=shiftIcY
muscat.shiftIcX=shiftIcX
muscat.dn = dn
muscat.NAc = NAc
muscat.lambda0 = .65/10
muscat.lambdaM = muscat.lambda0/muscat.dn
#muscat.dy = 10.
#muscat.dx = 10.
#muscat.Nx = Nx
#muscat.Ny = Ny
#muscat.Nz = Nz
muscat.dz = 10

#%%

print(muscat.dx)
print(muscat.dy)
print(muscat.dz)
print(muscat.lambda0)
#%%
''' Adjust some parameters to fit it in the memory '''
muscat.mysize = (muscat.Nz,muscat.Nx,muscat.Ny) # ordering is (Nillu, Nz, Nx, Ny)

''' Create a 3D Refractive Index Distributaton as a artificial sample'''
obj = tf_go.generateObject(mysize=muscat.mysize, obj_dim=(muscat.dz,muscat.dx,muscat.dy), obj_type ='sphere', diameter = 1, dn = muscat.dn)
obj = obj+1j*obj

# introduce zernike factors here
muscat.zernikefactors = zernikefactors
muscat.zernikemask = zernikemask

# Compute the System's properties (e.g. Pupil function/Illumination Source, K-vectors, etc.)¶
''' Compute the systems model'''
muscat.computesys(obj, is_padding=is_padding, dropout_prob=1)
print(muscat.Ic.shape)
plt.imshow(muscat.Ic), plt.show()
plt.imshow(np.abs(np.fft.fftshift(muscat.Po))),plt.colorbar(), plt.show()
#%%
# Generate Computational Graph (fwd model)
tf_fwd = muscat.computemodel(is_forcepos=False)


print('Evtl unwrap it!')
# this is the initial guess of the reconstruction
np_meas=matlab_val#*np.exp(1j*np.pi)


''' Create a 3D Refractive Index Distributaton as a artificial sample'''
init_guess = np.angle(np_meas)
init_guess = ((init_guess)-np.min(init_guess))**2
init_guess = dn*init_guess/np.max(init_guess)#*dn+1j*.01*np.ones(init_guess.shape)
plt.imshow(np.real(init_guess[:,15,:])), plt.colorbar, plt.show()

# Estimate the Phase difference between Measurement and Simulation
#%%
'''Numpy to Tensorflow'''
np_meas = np.conj(matlab_val)
np_mean = np.mean(np_meas)

'''Define Cost-function'''
tf_tvloss = muscat.tf_lambda_tv*reg.Reg_TV(muscat.TF_obj, BetaVals = [muscat.dx,muscat.dy,muscat.dz], epsR=muscat.tf_eps, is_circ = True)  #Alernatively tf_total_variation_regularization # total_variation
tf_tvloss += muscat.tf_lambda_tv*reg.Reg_TV(muscat.TF_obj_absorption, BetaVals = [muscat.dx,muscat.dy,muscat.dz], epsR=muscat.tf_eps, is_circ = True)  #Alernatively tf_total_variation_regularization # total_variation
tf_tvloss += .1*reg.Reg_L1(muscat.TF_obj)
tf_negsqrloss = lambda_neg*reg.Reg_NegSqr(muscat.TF_obj)
tf_negsqrloss += lambda_neg*reg.Reg_NegSqr(muscat.TF_obj_absorption)
print('-------> ATTENTION: CONSTANT Prefactors!')
tf_globalphase = tf.Variable(0., tf.float32, name='var_phase')
tf_globalabs = tf.Variable(.7, tf.float32, name='var_abs')# 
#tf_fidelity = tf.reduce_sum((tf_helper.tf_abssqr(tf_fwd  - (tf_meas/tf.cast(tf.abs(tf_globalabs), tf.complex64)*tf.exp(1j*tf.cast(tf_globalphase, tf.complex64)))))) # allow a global phase parameter to avoid unwrapping effects
tf_fwd_corrected = tf_fwd/tf.cast(tf.abs(tf_globalabs), tf.complex64)*tf.exp(1j*tf.cast(tf_globalphase, tf.complex64))
#tf_fidelity = tf.reduce_mean((tf_helper.tf_abssqr(muscat.tf_meas - tf_fwd_corrected ))) # allow a global phase parameter to avoid unwrapping effects
print('-------> ATTENTION L1')
tf_fidelity = tf.reduce_mean((tf.abs(muscat.tf_meas - tf_fwd_corrected ))) # allow a global phase parameter to avoid unwrapping effects
tf_grads = tf.gradients(tf_fidelity, [muscat.TF_obj])[0]

tf_loss = tf_fidelity/np_mean + tf_tvloss + tf_negsqrloss 

'''Define Optimizer'''
tf_optimizer = tf.train.AdamOptimizer(muscat.tf_learningrate)
#tf_optimizer = tf.train.MomentumOptimizer(tf_learningrate, momentum = .9, use_nesterov=True)
#tf_optimizer = tf.train.ProximalGradientDescentOptimizer(tf_learningrate)
#tf_optimizer = tf.train.GradientDescentOptimizer(muscat.tf_learningrate)
tf_lossop_obj_absorption = tf_optimizer.minimize(tf_loss, var_list = [muscat.TF_obj, muscat.TF_obj_absorption, tf_globalabs, tf_globalphase]) # muscat.TF_obj_absorption, 
tf_lossop_obj = tf_optimizer.minimize(tf_loss, var_list = [muscat.TF_obj, tf_globalabs, tf_globalphase]) # muscat.TF_obj_absorption, 
tf_lossop_aberr = tf_optimizer.minimize(tf_loss, var_list = [muscat.TF_zernikefactors])


''' Evaluate the model '''
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.global_variables_initializer())
if is_optimization:
    sess.run(tf.assign(muscat.TF_obj, np.real(init_guess))); # assign abs of measurement as initial guess of 
    sess.run(tf.assign(muscat.TF_obj_absorption, np.imag(init_guess))); # assign abs of measurement as initial guess of 

my_fwd = sess.run(tf_fwd)
mysize = my_fwd.shape

#%% We assume, that there is a global phase mismatch between measurment and first estimate of the fwd model, this can be estimated by the difference of mean phase of the two
# subtracting the mean phase from either measurement or the fwd model could help to speed up the optimization
# this is the initial guess of the reconstruction
np_meas=matlab_val#*np.exp(1j*np.pi)
myinitphase = np.mean(np.angle(np_meas))-np.mean(np.angle(my_fwd))-1
print('My Init Phase is :'+str(myinitphase))
np_meas=np_meas*np.exp(-1j*(myinitphase)) # subtract globaphase - anyway we want to optimize for that, but now the global phase can be assumed to be 0 initally
if(is_display): plt.subplot(231), plt.title('Angle XZ - Measurement'),plt.imshow(np.angle(np_meas)[:,mysize[1]//2,:]), plt.colorbar()#, plt.show()
if(is_display): plt.subplot(232), plt.title('Angle YZ - Measurement'),plt.imshow(np.angle(np_meas)[:,:,mysize[2]//2]), plt.colorbar()#, plt.show()
if(is_display): plt.subplot(233), plt.title('Angle XY - Measurement'),plt.imshow(np.angle(np_meas)[mysize[0]//2,:,:]), plt.colorbar()#, plt.show()
if(is_display): plt.subplot(234), plt.title('Angle YZ - Simulation'),plt.imshow(np.angle(my_fwd)[:,mysize[1]//2,:]), plt.colorbar()#, plt.show()
if(is_display): plt.subplot(235), plt.title('Angle XZ - Simulation'),plt.imshow(np.angle(my_fwd)[:,:,mysize[2]//2]), plt.colorbar()#, plt.show()
if(is_display): plt.subplot(236), plt.title('Angle XY - Simulation'),plt.imshow(np.angle(my_fwd)[mysize[0]//2,:,:]), plt.colorbar(), plt.show()

myinitabs = np.mean(np.abs(my_fwd))
print('My Init ABS is :'+str(myinitabs))
np_meas=np_meas/np.max(np.abs(np_meas))*myinitabs# subtract globaphase - anyway we want to optimize for that, but now the global phase can be assumed to be 0 initally
if(is_display): plt.subplot(231), plt.title('Abs XZ - Measurement'),plt.imshow(np.abs(np_meas)[:,mysize[1]//2,:]), plt.colorbar()#, plt.show()
if(is_display): plt.subplot(232), plt.title('Abs YZ - Measurement'),plt.imshow(np.abs(np_meas)[:,:,mysize[2]//2]), plt.colorbar()#, plt.show()
if(is_display): plt.subplot(233), plt.title('Abs XY - Measurement'),plt.imshow(np.abs(np_meas)[mysize[0]//2,:,:]), plt.colorbar()#, plt.show()
if(is_display): plt.subplot(234), plt.title('Abs YZ - Simulation'),plt.imshow(np.abs(my_fwd)[:,mysize[1]//2,:]), plt.colorbar()#, plt.show()
if(is_display): plt.subplot(235), plt.title('Abs XZ - Simulation'),plt.imshow(np.abs(my_fwd)[:,:,mysize[2]//2]), plt.colorbar()#, plt.show()
if(is_display): plt.subplot(236), plt.title('Abs XY - Simulation'),plt.imshow(np.abs(my_fwd)[mysize[0]//2,:,:]), plt.colorbar(), plt.show()


#%% optimize over the hyperparameters
#for mylambdatv in lambda_tv:
#    for myepstvval in eps_tv:
         
if(1):
    if(1):
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
        
        #%%
        ''' Optimize the model '''
        print('Start optimizing')
        np_meas = matlab_val # use the previously simulated data
        for iterx in range(iter_last,Niter):
            if iterx == 100:
                #print('No change in learningrate!')
                my_learningrate = my_learningrate*.1
            # try to optimize
            
            if(iterx==0 or not np.mod(iterx, Ndisplay)):
                my_res, my_res_absortpion, my_loss, my_fidelity, my_negloss, my_tvloss, myglobalphase, myglobalabs, myfwd =  \
                    sess.run([muscat.TF_obj, muscat.TF_obj_absorption, tf_loss, tf_fidelity, tf_negsqrloss, tf_tvloss, tf_globalphase, tf_globalabs, tf_fwd_corrected], \
                             feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:my_learningrate, muscat.tf_lambda_tv:mylambdatv, muscat.tf_eps:myepstvval})
        
                print('Loss@'+str(iterx)+': ' + str(my_loss) + ' - Fid: '+str(my_fidelity)+', Neg: '+str(my_negloss)+', TV: '+str(my_tvloss)+' G-Phase:'+str(myglobalphase)+' G-ABS: '+str(myglobalabs))        
                mylosslist.append(my_loss)
                myfidelitylist.append(my_fidelity)
                myneglosslist.append(my_negloss)
                mytvlosslist.append(my_tvloss)
                result_phaselist.append(my_res)
                result_absorptionlist.append(my_res_absortpion)
                globalphaselist.append(myglobalphase)
                globalabslist.append(myglobalabs) 
                
                ''' Save Figures and Parameters '''
                muscat.saveFigures(sess, savepath, tf_fwd_corrected, np_meas, mylosslist, myfidelitylist, myneglosslist, mytvlosslist, globalphaselist, globalabslist, 
                            result_phaselist=None, result_absorptionlist=None, init_guess=None, figsuffix='Iter'+str(iterx))
               

            
            # Alternate between pure object optimization and aberration recovery
            if iterx>150 & Noptpsf>1:
                for aa in range(Noptpsf):
                   sess.run([tf_lossop_aberr], feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:my_learningrate, muscat.tf_lambda_tv:mylambdatv, muscat.tf_eps:myepstvval})


            for aa in range(Noptpsf):
                if iterx<1500:
                    sess.run([tf_lossop_obj], feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:my_learningrate, muscat.tf_lambda_tv:mylambdatv, muscat.tf_eps:myepstvval})
                else:
                    sess.run([tf_lossop_obj_absorption], feed_dict={muscat.tf_meas:np_meas, muscat.tf_learningrate:my_learningrate, muscat.tf_lambda_tv:mylambdatv, muscat.tf_eps:myepstvval})

            iter_last = iterx
        #%%        
        ''' Save Figures and Parameters '''
        muscat.saveFigures(sess, savepath, tf_fwd_corrected, np_meas, mylosslist, myfidelitylist, myneglosslist, mytvlosslist, globalphaselist, globalabslist, 
                    result_phaselist, result_absorptionlist)
   
        muscat.writeParameterFile(my_learningrate, mylambdatv, myepstvval, filepath = savepath+'/myparameters.yml')#, figsuffix = 'Shift_x-'+str(shiftIcX)+'Shift_y-'+str(shiftIcY))
        
        print('Zernikes: ' +str(np.real(sess.run(muscat.TF_zernikefactors))))
        
        # backup current script
        from shutil import copyfile
        import os
        src = (os.path.basename(__file__))
        copyfile(src, savepath+'/script_bak.py')
        

