# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:08:21 2019

@author: diederichbenedict
"""


import NanoImagingPack as nip
from NanoImagingPack import v5
import numpy as np
import InverseModelling as im
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime

# load own functions
import src.model as mus
import src.data as data
import src.experiments as experiments 


# Optionally, tweak styles.
mpl.rc('figure',  figsize=(12, 9))
mpl.rc('image', cmap='gray')


#%%
'''Define some stuff related to infrastructure'''
mytimestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
basepath = './'#'/projectnb/cislidt/diederich/muScat/Multiple-Scattering_Tensorflow/'
resultpath = 'Data/DROPLETS/RESULTS/'

# optional - reset tensorflow graph
tf.reset_default_graph()

# Generate Test-Object
''' File which stores the experimental parameters from the Q-PHASE setup 
    1.) Read in the parameters of the dataset ''' 
matlab_pars = data.import_parameters_mat(filename = experiments.matlab_par_file, matname = experiments.matlab_par_name)

''' 2.) Read in the parameters of the dataset ''' 
if(experiments.matlab_val_file.find('mat')==-1):
    matlab_val = np.load(experiments.matlab_val_file)+1j
else:
    matlab_val = data.import_realdata_h5(filename = experiments.matlab_val_file, matname=experiments.matlab_val_name, is_complex=True)

# Make sure it's even numberalong Z
if(np.mod(matlab_val.shape[0],2)==1):
    matlab_val = matlab_val[0:matlab_val.shape[0]-1,:,:]
matlab_val = (matlab_val)# - .6j
matlab_val=matlab_val[:,0:50,0:50]

''' Create the Model'''
muscat = mus.MuScatModel(matlab_pars, is_optimization=True)
# Correct some values - just for the puprose of fitting in the RAM
muscat.Nx,muscat.Ny,muscat.Nz = matlab_val.shape[1], matlab_val.shape[2], matlab_val.shape[0]
muscat.shiftIcY=experiments.shiftIcY
muscat.shiftIcX=experiments.shiftIcX
muscat.NAc = experiments.NAc

''' Adjust some parameters to fit it in the memory '''
muscat.mysize = (muscat.Nz,muscat.Nx,muscat.Ny) # ordering is (Nillu, Nz, Nx, Ny)

# introduce zernike factors here
muscat.zernikefactors = experiments.zernikefactors
muscat.zernikemask = experiments.zernikemask

''' Compute a first guess based on the experimental phase '''
obj_guess =  np.zeros(matlab_val.shape)+muscat.nEmbb# np.angle(matlab_val)## 
import src.tf_generate_object as tf_go
mydiameter=5
obj_guess= tf_go.generateObject(mysize=muscat.mysize, obj_dim=1, obj_type ='sphere', diameter = mydiameter, dn = .05, nEmbb = 1.33)#)dn)
n_r = obj_guess#obj.astype("float32") / np.max(obj) * NumPhot
k02 = (2*np.pi*muscat.nEmbb/muscat.lambda0)**2
V = (k02/(4*np.pi))*(n_r**2-muscat.nEmbb**2)

''' Compute the systems model'''
# Compute the System's properties (e.g. Pupil function/Illumination Source, K-vectors, etc.)¶
muscat.computesys(obj=obj_guess, is_compute_psf='BORN', is_dampic=.02)

''' Create Model Instance'''
muscat.computemodel()

print('Start Session')
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('We are precomputing the PSF')
atf = sess.run(muscat.TF_ATF)
asf = sess.run(muscat.TF_ASF) 



#v5(obj);v5(obj);v5(obj)
#nip.close()
#%%  simulate an image
print('We are conpmuting the convolution')
pimg = nip.convolve(V, asf,full_fft=True)
pimg[pimg<0.0]=0.0
nimg = nip.noise.poisson(pimg,seed=0,dtype='complex64')  # Christian: default should be seed=0

#%%  Here the actual models are defined
im.Init()

print("contructing inverse obj model")
objfwdinv_real =im.Variable("preObj_real")
objfwdinv_imag =im.Variable("preObj_imag")
#    objfwdinv=im.PreBorderRegion(objfwdinv,nimg,border=myborder) # The cut operation allows the data to be smaller than the reconstructed region
#    objfwdinv=im.PreAbsSqr(objfwdinv)
objfwdinv_real=im.PreMonotonicPos(objfwdinv_real)
obj_real=im.PreObjInit(objfwdinv_real,np.mean(V)*np.ones(V.shape))
objfwdinv_imag=im.PreMonotonicPos(objfwdinv_real)
obj_imag=im.PreObjInit(objfwdinv_imag,0*np.ones(V.shape))
TF_nr = tf.complex(obj_real,obj_imag)
V = (k02/(4*np.pi))*(TF_nr**2-(muscat.nEmbb)**2)


# here the measured values are baked into the model:
#       res=im.Convolve(obj,psf.astype("float32"))
mypimg=im.convolveCpx(V, asf)


#%%  Here the actual models are defined
mylambdaTi=0.1
mylambdaGR=0.001
mylambdaTV=0.01
mylambdaGuided=1e-8
mylambdaGuidedTV=4e-6

myloss = im.Loss_FixedGaussian(mypimg,nimg,checkScaling=False)
#loss = myloss + mylambdaGuided*im.Reg_Guided(obj,myguide,1e-7,gradType=gradType)
#    myloss = myloss + mylambdaGuidedTV*im.Reg_Guided(obj,myguide,1e-2,doTV=True)
myloss = myloss + mylambdaTV*im.Reg_TV(TF_nr,1e-5)
#    myloss = myloss + mylambdaTi*im.Reg_Tichonov(obj)
#    myloss = myloss +  mylambdaGR*im.Reg_GR(obj)


#%% Optimize 
learningRate = 30.0
NIter = 80

# tfinit=nip.convROTF(nimg,rotf).astype('float32')  # start with a convolved version
resVars=[TF_nr]  # generates the graph and variables to solve for

myoptimizer = im.optimizer(myloss,otype="adam",oparam={"learning_rate": 30.0},NIter=1)
# myoptimizer = im.optimizer(loss,otype="L-BFGS-B",NIter=150)

resObj = im.Optimize(myoptimizer,loss=myloss,resVars=resVars,TBSummary=True) # ,resVars=(ObjFwd,PupilFwd)


#%%

# here the measured values are baked into the model:
#       res=im.Convolve(obj,psf.astype("float32"))
mypimg=im.convolveROTF(obj, rotf)



im.Init()

mylambdaTi=0.5; mylambdaTV=0.0001; mylambdaGR=0.08
myborder=3 # better: [3 3]
objinit=nip.convROTF(nimg,np.conj(rotf)).astype('float32')  # start with a convolved version
#objinit=obj.astype('float32')  # start with a convolved version

PhaseOnly=True

objfwdinv=im.Variable("preObj") # ,1.0/np.mean(objinit)
objfwdinv=im.PreMonotonicPos(objfwdinv)
objfwd=im.PreObjInit(objfwdinv,objinit)

# now the otf part
if PhaseOnly:
    pupilfwdinv=im.PrePhase2Cpx(im.Variable("prePhases"))  # Attention! Eps has to stay low for this!
else:
    pupilfwdinv=im.PrePackCpx(im.Variable("prePhases"))
pupilfwdinv=im.PreFillROI(pupilfwdinv,pupilmask)
pupilfwd=im.PreObjInit(pupilfwdinv,-(0.0+1.0j)*pupilmask.astype(np.complex64))  # start pupil with assigning 1.0

print("running inverse pupil model")        

myobj=objfwd
#    myobj=tf.constant(objinit.astype("float32"))
#    myobj=tf.constant(obj.astype("float32")) # yields perfect phase reconstructions
#    pupil=tf.constant(mypupil.astype("complex64"))
pupil=pupilfwd
# psf part
mypsf = im.PSFfromPupil(pupil) #  
#    rotf = im.ROTFfromPupil(pupil) 
mypsf = mypsf/tf.reduce_sum(mypsf)
#    mypsf=tf.constant(psf.astype("float32"))
# obj part
# both 
#    fwd=im.ConvolveROTF(myobj,10*rotf) #   !!! The gradient of the rotf is not defined!
fwd=im.convolveReal(myobj, mypsf)
#    fwd=im.ConvolveReal(10*mypsf,myobj) #   !!! The gradient is not defined!
#    fwd=mypsf
#    myloss=im.Loss_Poisson(fwd,nimg)+ mylambdaTV*im.Reg_TV(myobj,1e-15)
#    myloss=im.Loss_ScaledGaussianReadNoise(fwd,nimg,1.0) + mylambdaTi*im.Reg_Tichonov(myobj)
#    myloss=im.Loss_ScaledGaussianReadNoise(fwd,nimg,1.0) + mylambdaTV*im.Reg_TV(myobj,1e-15)
#    myloss=im.Loss_ScaledGaussianReadNoise(fwd,nimg,1.0) + mylambdaGR*im.Reg_GR(myobj)
#    myloss=im.Loss_ScaledGaussianReadNoise(fwd,nimg,1.0)
loss=im.Loss_FixedGaussian(fwd,nimg.astype("float32"))

# (loss,ObjFwd,PupilFwd)=Loss()

#%% Check the gradients

ObjTests=[[0,0],[400,400],[401,400],[400,410]];
PhaseTests=[0,1,500,1000,2000,2800]
myresult=im.GradCheck(loss,Idx=[ObjTests,PhaseTests],Eps=0.1,TestVars=tf.trainable_variables())  # real data

#toshow=nip.image.image(nip.cat([nimg,myresult[0],obj,myresult[1]],3))
#nip.v5(toshow)
#raise ValueError("stopped")

#%%
resVars=(myobj,pupil) # generates the graph and variables to solve for

AllVars=tf.trainable_variables()
# optimizeBoth= im.optimizer(loss,otype="L-BFGS-B",NIter=100) # not so good here. Adam is much better
optimizeBoth= im.optimizer(loss,otype="adam",oparam={"learning_rate": 1},NIter=100)
resObj,resPupil=im.Optimize(optimizeBoth,loss=loss,resVars=resVars) # ,resVars=(ObjFwd,PupilFwd)

# raise ValueError("stopped")

# start with 10 pupil-only iterations
optimizeObj= im.optimizer(loss,otype="adam",oparam={"learning_rate": 1.0},NIter=10,var_list=AllVars[0]) # continue with alteranting updates
optimizePupil= im.optimizer(loss,otype="adam",oparam={"learning_rate": 1.0},NIter=10,var_list=AllVars[1])
optimizeBothFinal= im.optimizer(loss,otype="adam",oparam={"learning_rate": 0.1},NIter=10,var_list=AllVars,verbose=True)  # This massively reduced learning rate is necessary to tame the adam
alternatingMain = im.alternatingOptimizer((optimizePupil,optimizeObj),4) # same amount of iterations, but slower than adam
alternatingOptimizer = im.alternatingOptimizer((optimizePupil,alternatingMain,optimizeBothFinal)) # same amount of iterations, but slower than adam

resObj2,resPupil2=im.Optimize(alternatingOptimizer,loss=loss,resVars=resVars) # ,resVars=(ObjFwd,PupilFwd)
#resObj,resPupil=im.Optimize(myoptimizer,resVars=resVars) # ,resVars=(ObjFwd,PupilFwd)
# myresult=im.Optimize(Loss,myoptimizer,Eager=True)

toshow=nip.image.image(nip.cat([nimg,resObj,resObj2,obj],-4))
toshowC=nip.cat([resPupil,resPupil2,mypupil],-4)
# nip.view(toshow)
v=v5(toshow)
v=v5(nip.cat((resPupil,resPupil2),-1),showPhases=True)
v=v5(mypupil,showPhases=True)
# np.angle(toshowC)
#v=v5(toshow)
#vc=v5(np.angle(toshowC))
