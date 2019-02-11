#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 08:55:42 2019

@author: bene
"""

# phantom = abs(phantom3D(64));
# phantom(phantom==1) = .15;
# phantom(phantom>.0 & phantom<.1) = .1;
# phantom(phantom>.18 & phantom<.22) = .01;
# save('phantom.mat', 'phantom', '-v7.3')

import numpy as np
import h5py 
import matplotlib.pyplot as plt
filename = 'phantom_50_50_50'

myfile = h5py.File(filename+'.mat','r')
myphantom = np.array(myfile.get('phantom'))
myphantom[myphantom==1]=.2
#myphantom[myphantom==0]=.1
myphantom[myphantom==.2]=0.01


plt.subplot(131), plt.imshow(myphantom[:,:,25]), plt.colorbar()
plt.subplot(132), plt.imshow(myphantom[:,25,:]), plt.colorbar()
plt.subplot(133), plt.imshow(myphantom[25,:,:]), plt.colorbar(), plt.show()

np.save(filename + '.npy', myphantom)