# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 07:34:27 2019

@author: diederichbenedit
@author: bene
"""
# #load_ext autoreload
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from datetime import datetime
import tifffile as tif

# load own functions
import src.model as mus
import src.tf_helper as tf_helper
import src.data as data
import src.tf_regularizers as reg
import src.experiments as experiments 
import src.MyParameter as paras
import src.qphase_helper as qh

import NanoImagingPack as nip
nip.setDefault('IMG_VIEWER', 'NIP_VIEW')

if(1):
    # Optionally, tweak styles.
    mpl.rc('figure',  figsize=(12, 9))
    mpl.rc('image', cmap='gray')
    #%%
    #plt.switch_backend('agg')
    #np.set_printoptions(threshold=np.nan)
    ##load_ext autoreload
    ###
    '''Read the images from disk'''
    print('Reading the dataset from disc. This can take a while..\n')
    try:
        AllDataRaw = tif.imread(experiments.myFolder + experiments.myFile)
        AllDark = tif.imread(experiments.myFolder + experiments.myDarkFile)
        AllBright = tif.imread(experiments.myFolder + experiments.myBrightfieldFile)    
        MyDark = np.mean(AllDark,0);
        MyBright = AllBright#;# mean(AllBright,[],3);
    except():
        print('Something went wrong')
        AllData = tif.imread(experiments.myFolder + experiments.myFile)
    
    print('Done reading the TIF-files!')
    
    ###
    
    ''' Flat field correction '''
    print('Applying Flatfielding!')
    AllData = (AllDataRaw-0.*MyDark)
    AllBright = (MyBright-0.*MyDark)
    
    ''' Remove ripples '''
    correctioncurve = np.mean(AllData, (1,2));
    correctioncurve = np.expand_dims(np.expand_dims(correctioncurve,axis=-1),-1)
    AllData_aligned = AllData/correctioncurve*np.mean(correctioncurve);##*mean(correctioncurve);
    
    ###
    ''' select the ROI of choice and convert it to dipimage '''
    myparas = paras.MyParameter()
    myparas.preset_40x(dz=experiments.dz)
    mysize_old = AllData.shape
    
    ''' define the area where to cut out the hologram '''
    cc_center = experiments.cc_center
    cc_size = experiments.cc_size  

    np.log(1+nip.ft2d(nip.DampEdge(np.squeeze(nip.image(AllData_aligned[0,:,:])), rwidth = .25, axes=(0,1))))
    
    AllAmp_roi_raw = qh.extractQPhaseCCterm(AllData_aligned, myparas.cc_center, myparas.cc_size)
    AllBright_roi_raw = qh.extractQPhaseCCterm(AllBright, myparas.cc_center, myparas.cc_size)
    AllBright_roi = np.mean(AllBright_roi_raw,0)
    
    ''' remove parabolic phase - Rainers MEthod doesnt work!'''
    allAmp_roi_sub = AllAmp_roi_raw/np.exp(1j*np.angle(AllBright_roi))*1/(np.mean(nip.abssqr(np.mean(AllBright_roi))));##_raw(:,:,0) remove background
    allAmp_roi_sub = allAmp_roi_sub*np.exp(-1j*nip.gaussf(np.angle(allAmp_roi_sub[0,:,:]),73)) ## remove some phase ramp - wrong center of CC?
    

#%%
''' Preprocess '''
#  0.) extract ROI for mean-calcualtion - NORMALIZE
#  1.) subtract the estimated background - not divide!
#  2.) normalize by dividing by its mean
Nz=allAmp_roi_sub.shape[0]
roi_size = 50
allAmp_clean = qh.QPhaseNormBackground(np.conj(allAmp_roi_sub),roi_center=(Nz//2,experiments.roi_center[0],experiments.roi_center[1]),roi_size=(Nz,roi_size,roi_size));

#%%
''' Extract ROi for later processing '''
subsample_fac_xy=1
sumbsampleZ=1
roisize = np.array((80, 50, 50))
mycenter_roi = np.array((allAmp_clean.shape[0]//2, 290,340))

if(0):
    roisize = np.array((300, 300, 80))
    mycenter_roi = np.array((allAmp_clean.shape[0]//2, 300, 300))

mysize_new = np.array(allAmp_clean.shape)
allAmp_clean_extract = nip.extract(allAmp_clean, roisize, mycenter_roi, checkComplex=False)

if(sumbsampleZ>1):
    allAmp_clean_extract = allAmp_clean_extract[:,:,0:sumbsampleZ:-1]
else:
    #allAmp_clean_extract = (ift(extract(ft(allAmp_clean_extract), [size(allAmp_clean_extract,1) size(allAmp_clean_extract,2) size(allAmp_clean_extract,3)/sumbsampleZ]))),
    print('Not implemented yet')
    allAmp_clean_extract = allAmp_clean_extract
    
tmp = subsample_fac_xy*mysize_old/mysize_new*np.array((myparas.dy_orig, myparas.dz_orig, myparas.dx_orig))
correctionfactor_xy = (30/25)
myparas.dx = tmp[1]*correctionfactor_xy
myparas.dy = tmp[2]*correctionfactor_xy
myparas.dz = tmp[0]*sumbsampleZ
print('Attention: Correction factor applied!')

# show butterfly 
allAmpFT = nip.ft((nip.DampEdge(allAmp_clean_extract, rwidth=.25, axes=(0,1,2))))
nip.view(allAmpFT[:,allAmpFT.shape[1]//2,:]**.1)
nip.view(allAmpFT[:,:,allAmpFT.shape[2]//2]**.1)


myparas.Nz, myparas.Nx, myparas.Ny = allAmp_clean_extract.shape

np.save('QPhase_Holo_Recon', 'myparas')
## Estimate the sphere for guided psf estimation 
# corresponds to:
# subsample_fac_xy=1;
# roisize = [100 100];
# mycenter_roi = [340 290];
# mysphere_pos(52,51,34) = 1;
#
#myshift = [25.5 26.5 30]; # XYZ
#myshift = myshift-size(allAmp_clean_extract)/2
#myradius= 3.8;
#mysphere = ((myParameter.dx*xx(allAmp_clean_extract)).^2+(myParameter.dy*yy(allAmp_clean_extract)).^2+(myParameter.dz*zz(allAmp_clean_extract)).^2)<myradius;
#mysphere = shift(mysphere, myshift);
#mysphere = mysphere>.4;
#cat(4, mysphere, allAmp_clean_extract)
##
#
#
##
#mysphere_mat = double(mysphere);
#
#
### save Data for Python
#allAmpSimu = double((allAmp_clean_extract));
#myfolder_save = '../../PYTHON/muScat/Data/cells/';
#
#save([myfolder_save, myFile,'myParameter.mat'], 'myParameter','-v7.3')
#save([myfolder_save, myFile, '_allAmp.mat'], 'allAmpSimu','-v7.3')
#save([myfolder_save, myFile, '_mysphere.mat'], 'mysphere_mat','-v7.3')
##
##allAmpSimu_ang = angle(allAmpSimu);
##([myfolder_save, myFile, '_allAmp_angle.mat'], 'allAmpSimu_ang','-v7.3')
#[myfolder_save, myFile,'myParameter.mat']
#[myfolder_save, myFile, '_allAmp.mat']
#[myfolder_save, myFile, '_mysphere.mat']
