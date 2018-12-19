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
    ##load system data; new MATLAB v7.3 Format! 
    if filename==None:
        filename = 'img_noisev73.mat' 
    if matname==None: 
        matname = '/img_noise/data' 
        
    # readin HDF5 container and extract the data matrix
    mat_matlab_data = h5py.File(filename)
    myData = mat_matlab_data[matname]
    
    if is_complex:
#        raw_obj  = myData.value.view(np.complex64)#.value.view(np.complex)
        raw_obj = myData.value.view(np.double).reshape((myData.shape[0],myData.shape[1],myData.shape[1],2))
        raw_obj = raw_obj[:,:,:,0]+1j*raw_obj[:,:,:,1]
    else:
        raw_obj = np.array(myData) # For converting to numpy array

    return raw_obj

def import_realdata_mat(filename = None, matname = None, is_complex = False):
    if filename==None:
        filename = 'img_noisev73.mat' # '/home/useradmin/Documents/Benedict /MATLAB/Kamilov-BPMSimulation/I_z_v73.mat'
    if matname==None: 
        matname = '/img_noise/data' # 'I_z'
    else:
        matname = matname+'/data' # 'I_z'                
    # reading MAT files and extract the data matrix        
    img_mat_file = scipy.io.loadmat(filename)
    raw_obj = np.array(img_mat_file['obj'])
    #raw_obj = np.array(img_mat_file.values())[0] # For converting to numpy array

    return raw_obj

def import_parameters_mat(filename = None, matname = None):
    if matname==None: 
        matname = 'myParameter'

    ##load system data; new MATLAB v7.3 Format! 
    mat_matlab_data = h5py.File(filename)
    myParamter = mat_matlab_data[matname]
    
    return myParamter 

def save_as_tif(image, experiment_name, myfile_dir='./'):   
    """ Save images from results. """
    
    out_path_inputs = os.path.join(myfile_dir, experiment_name+'.tif')
    tif.imsave(out_path_inputs, image, append=True, bigtiff=True) #compression='lzw', 
    
def save_timeseries(image, experiment_name, myfile_dir='./'):
    out_path_inputs = os.path.join(myfile_dir, experiment_name+'.tif')
    for i in range(image.shape[0]):
        tif.imsave(out_path_inputs, image[i,:,:], append=True, bigtiff=True) #compression='lzw', 