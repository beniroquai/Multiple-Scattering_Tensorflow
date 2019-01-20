#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 10:08:51 2018

@author: bene
"""
import numpy as np
import src.data as data
mydata = np.load('/Users/bene/Dropbox/Dokumente/Promotion/PROJECTS/BOSTON/MUSCAT/PYTHON/muScat/tmp.npy')
data.export_realdatastack_h5('/Users/bene/Dropbox/Dokumente/Promotion/PROJECTS/BOSTON/MUSCAT/PYTHON/muScat/temp.h5', 'temp', mydata)
