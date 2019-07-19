# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 12:32:46 2018

@author: pi96doc
"""

import NanoImagingPack as nip
from NanoImagingPack import v5
import numpy as np
import InverseModelling as im
import tensorflow as tf

#import matplotlib
#matplotlib.use('Qt5Agg', warn=False)

obj=nip.readim()
obj.pixelsize = [100,62.5, 62.5]
# obj=nip.image(np.rot90(obj))
if False:
    pupilrad = 30.0
    pupilmask = nip.rr(obj.shape)  < pupilrad
    perfectAperture = pupilmask
    mypupil = pupilmask.astype("complex64") * np.exp(1j*nip.xx(obj)/10)  # oblique phase to shift the PSF
else:
    psfParam = nip.PSF_PARAMS()
    perfectAperture = nip.aberratedPupil(obj, psfParam).astype("complex64")
    psfParam.aberration_types = [psfParam.aberration_zernikes.spheric, (3, 5)] #, aber_map]  # define list of aberrations (select from choice, zernike coeffs, or aberration map
    psfParam.aberration_strength = [0.5, 0.8] # , 0.2]
    mypupil = nip.aberratedPupil(obj, psfParam)
    pupilmask = nip.pupilAperture(obj, psfParam)

# v=v5(obj,1500,1500)
#v.ProcessKeyMainWindow("c");v.ProcessKeyMainWindow("c");v.ProcessKeyMainWindow("c");v.ProcessKeyMainWindow("c");v.ProcessKeyMainWindow("c");v.UpdatePanels()

asf = nip.ift(mypupil)
psf = np.real(asf * np.conj(asf))
psf = psf/psf.sum()
# psf=nip.psf2d(obj)
NumPhot = 100
obj = obj.astype("float32") / np.max(obj) * NumPhot

#v5(obj);v5(obj);v5(obj)
#nip.close()
#%%  simulate an image
rotf=nip.PSF2ROTF(psf)
pimg = nip.convROTF(obj, rotf)
pimg[pimg<0.0]=0.0
nimg = nip.noise.poisson(pimg, seed=0, dtype='float32')  # Christian: default should be seed=0

#%%  Here the actual models are defined

im.Init()

mylambdaTi=0.5; mylambdaTV=0.0001; mylambdaGR=0.08
myborder=3 # better: [3 3]
objinit=nip.convROTF(nimg, np.conj(rotf)).astype('float32')  # start with a convolved version
#objinit=obj.astype('float32')  # start with a convolved version

objfwdinv=im.Variable("preObj") # ,1.0/np.mean(objinit)
objfwdinv=im.PreMonotonicPos(objfwdinv)
objfwd=im.PreObjInit(objfwdinv, objinit)

PhaseOnly = True
# now the otf part
if PhaseOnly:
    pupilfwdinv = im.PrePhase2Cpx(im.Variable("prePhases"))  # Attention! Eps has to stay low for this!
else:
    pupilfwdinv = im.PrePackCpx(im.Variable("prePhases"))
pupilfwdinv = im.PreFillROI(pupilfwdinv, pupilmask)
pupilfwd = im.PreObjInit(pupilfwdinv, -(0.0+1.0j)*pupilmask.astype(np.complex64)) * perfectAperture  # start pupil with assigning 1.0

print("running inverse pupil model")        

if True:
    myobj = objfwd
else:
#    myobj=tf.constant(objinit.astype("float32"))
    myobj = tf.constant(obj.astype("float32")) # yields perfect phase reconstructions

if False:
    pupil=tf.constant(mypupil.astype("complex64"))
else:
    pupil=pupilfwd
# psf part
mypsf = im.PSFfromPupil(pupil)#, "corner") #
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
loss=im.Loss_FixedGaussian(fwd, nimg.astype("float32"))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
myctf_np = nip.ft2d(sess.run(mypsf))
mypsf_np = (sess.run(mypsf))

nip.ft2d(sess.run(mypsf))
# (loss,ObjFwd,PupilFwd)=Loss()

#%% Check the gradients

if False:
    ObjTests=[[0,0],[400,400],[401,400],[400,410]];
    PhaseTests=[0,1,500,1000,2000,2800]
    myresult=im.GradCheck(loss,Idx=[ObjTests,PhaseTests],Eps=0.1,TestVars=tf.trainable_variables())  # real data

#toshow=nip.image.image(nip.cat([nimg,myresult[0],obj,myresult[1]],3))
#nip.v5(toshow)
# raise ValueError("stopped")

#%%
resVars=(myobj, pupil, mypsf) # generates the graph and variables to solve for

AllVars=tf.trainable_variables()
optimizeBoth= im.optimizer(loss, otype="L-BFGS-B", NIter=100) # not so good here. Adam is much better
#optimizeBoth= im.optimizer(loss, otype="adam", oparam={"learning_rate": 1}, NIter=200)
resObj, resPupil, respsf = im.Optimize(optimizeBoth, loss=loss, resVars=resVars) # ,resVars=(ObjFwd,PupilFwd)

# raise ValueError("stopped")

# start with 10 pupil-only iterations
if len(im.findVars(loss)) >= 2:
    optimizeObj= im.optimizer(loss, otype="adam", oparam={"learning_rate": 1.0}, NIter=10, var_list=AllVars[0]) # continue with alteranting updates
#    optimizeObj= im.optimizer(loss, otype="L-BFGS-B", NIter=10, var_list=AllVars[0]) # continue with alternating updates
    optimizePupil= im.optimizer(loss, otype="adam", oparam={"learning_rate": 1.0}, NIter=10, var_list=AllVars[1:], verbose=True)
    optimizeBothFinal= im.optimizer(loss, otype="adam", oparam={"learning_rate": 0.1}, NIter=40, var_list=AllVars, verbose=True)  # This massively reduced learning rate is necessary to tame the adam
    alternatingMain = im.alternatingOptimizer((optimizePupil, optimizeObj), 8) # same amount of iterations, but slower than adam
    alternatingOptimizer = im.alternatingOptimizer((optimizePupil, alternatingMain, optimizeBothFinal)) # same amount of iterations, but slower than adam
else:
    alternatingOptimizer = im.optimizer(loss, otype="adam", oparam={"learning_rate": 1.0}, NIter=200, var_list=AllVars[1], verbose=True)

resObjAlternating, resPupil2, respsf2=im.Optimize(alternatingOptimizer, loss=loss, resVars=resVars) # ,resVars=(ObjFwd,PupilFwd)
#resObj,resPupil=im.Optimize(myoptimizer,resVars=resVars) # ,resVars=(ObjFwd,PupilFwd)
# myresult=im.Optimize(Loss,myoptimizer,Eager=True)

toshow=nip.image(nip.catE([nimg, resObj, resObjAlternating, obj]))
toshowC=nip.catE([resPupil, resPupil2, mypupil])
# nip.view(toshow)
v=v5(toshow)
v=v5(nip.cat((mypupil, resPupil, resPupil2),-1), showPhases=True)
v=v5(nip.catE(psf, np.fft.fftshift(respsf), np.fft.fftshift(respsf2)), showPhases=True)

#v=v5(mypupil, showPhases=True)
#np.angle(toshowC)
#v=v5(toshow)
#vc=v5(np.angle(toshowC))
input('stop?')
