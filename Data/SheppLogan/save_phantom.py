#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 09:44:52 2019

@author: bene
"""

import numpy as np
import h5py 
myfile = h5py.File('phantom_64_64_64.mat','r')
myphantom = np.array(myfile.get('phantom'))

np.save('phantom_64_64_64.npy', myphantom)
