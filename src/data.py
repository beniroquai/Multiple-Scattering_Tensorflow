# -*- coding: utf-8 -*-
''' This file is just for loading previously generated data matrices
 from disk to be used as Numpy arrays '''
 
import numpy as np
import h5py 
import scipy.io
import scipy as scipy
import scipy.misc
import os
import tifffile as tif



def import_realdata_h5(filename = None, matname = None, is_complex = False):
    if filename==None:
        filename = 'img_noisev73.mat' # '/home/useradmin/Documents/Benedict /MATLAB/Kamilov-BPMSimulation/I_z_v73.mat'
    if matname==None: 
        matname = '/img_noise/data' # 'I_z'
            
    # readin HDF5 container and extract the data matrix
    img_mat_file  = h5py.File(filename ,'r') 
    img_data = img_mat_file.get(matname).get('data')
    if is_complex:
        raw_obj  = img_data.value.view(np.complex64)#.value.view(np.complex)
    else:
        raw_obj = np.array(img_data) # For converting to numpy array

    return raw_obj

def import_realdata_mat(filename = None, matname = None, is_complex = False):
    if filename==None:
        filename = 'img_noisev73.mat' # '/home/useradmin/Documents/Benedict /MATLAB/Kamilov-BPMSimulation/I_z_v73.mat'
    if matname==None: 
        matname = filename
                    
    # reading MAT files and extract the data matrix        
    img_mat_file = scipy.io.loadmat(filename)
    raw_obj = np.array(img_mat_file['obj'])
    #raw_obj = np.array(img_mat_file.values())[0] # For converting to numpy array

    return raw_obj

def import_parameters_mat(filename = None, matname = None):
    if filename==None:
        filename = './Data/BEADS/Beads_40x_100a_myParameter.mat'
    if matname==None: 
        matname = 'myParameter'

    ##load system data; new MATLAB v7.3 Format! 
    mat_matlab_data = h5py.File(filename)
    myParamter = mat_matlab_data['myParameter']
    
    return myParamter 

def save_as_tif(image, experiment_name, myfile_dir='./'):   
    """ Save images from results. """
    
    out_path_inputs = os.path.join(myfile_dir, experiment_name+'.tif')
    tif.imsave(out_path_inputs, image, append=True, bigtiff=True) #compression='lzw', 
    
