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
obj_real=im.PreObjInit(objfwdinv_real,muscat.nEmbb*np.ones(V.shape))
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
learningRate = 1e-0
NIter = 100

# tfinit=nip.convROTF(nimg,rotf).astype('float32')  # start with a convolved version
resVars=[obj_real,obj_imag]  # generates the graph and variables to solve for

myoptimizer = im.optimizer(myloss,otype="adam",oparam={"learning_rate": 30.0},NIter=NIter)
# myoptimizer = im.optimizer(loss,otype="L-BFGS-B",NIter=150)

resObj = im.Optimize(myoptimizer,loss=myloss,resVars=resVars,TBSummary=True) # ,resVars=(ObjFwd,PupilFwd)
resObj[1][:,25,:]

print("running inverse pupil model")        

