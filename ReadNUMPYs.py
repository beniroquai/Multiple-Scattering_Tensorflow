#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 08:13:52 2019

@author: bene
"""

import numpy as np
import NanoImagingPack as nip
import tifffile as tif

#%%
mylist = []
myiter = 1
while True:
    try:
        myimage = np.load('/Volumes/ARCH_201703/Sequential/GREEN/ImgNo_'+str(myiter)+'_colorgreen.npy')
        myimage = nip.resample(myimage, (.25,.25))
        mylist.append(myimage)
        print(str(myiter))
        myiter+=1
    except:
        break
myarray = np.array(mylist)
tif.imsave('test.tif', myarray)
