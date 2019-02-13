#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 08:55:42 2019

@author: bene
"""
import numpy as np
import matplotlib.pyplot as plt
from mesh_vox import read_and_reshape_stl, voxelize

# path to the stl file
input_path = '/Users/bene/Dropbox/Dokumente/Promotion/PROJECTS/BOSTON/MUSCAT/PYTHON/muScat/Data/NEURON/neuron.stl'
# number of voxels used to represent the largest dimension of the 3D model
resolution = 32 

# read and rescale
mesh, bounding_box = read_and_reshape_stl(input_path, resolution)
# create voxel array
voxels, bounding_box = voxelize(mesh, bounding_box)

print(voxels)
np.save('/Users/bene/Dropbox/Dokumente/Promotion/PROJECTS/BOSTON/MUSCAT/PYTHON/muScat/Data/NEURON/myneuron_32_32_70.npy',voxels[0:70,35:32+35,35:32+35])