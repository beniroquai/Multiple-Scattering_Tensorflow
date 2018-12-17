#!/usr/bin/env python
# coding: utf-8

# # This is the minimum Working Example to compute a multiple scattering experiment in the Q-Phase 

# In[3]:


#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 19:53:32 2017

@author: Bene

This file creates a fwd-model for the TESCAN Q-PHASE under 
multiple-scattering. It is majorly derived from  "LEarning approach for optical tomography"
U. S. Kamilov, BIG, EPFL, 2014.
"""
get_ipython().run_line_magic('load_ext', 'autoreload')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py 
import scipy.io
import scipy as scipy
import scipy.misc

# load own functions
from src import tf_helper as helper
from src import model as mus
from src import tf_generate_object as tf_go
from src import data as data


# In[ ]:





# In[ ]:





# # Define some optimization parameters 

# In[6]:



optimize = 0#0# want to optimize for the refractive index?

my_learningrate = 1e-3# learning rate
my_keep_prob = 1
tv_lambda = 1e-2
gr_lambda = 1e-2
obj_reg_lambda = 1e9

load_data = 0 # want to load previously stored data from disk (numpy) otherwise MATLAB experiment will be loaded    


# Define the location for the files; it is expected to be in the MATLAB -v7.3 format
# The parameter file has a struct-object called myParameter with all necessary variables
# The dataset file holds a complex matrix with Nx * Ny * Nz voxels called allAmpSimu
matlab_par_file = '/Users/bene/Dropbox/Dokumente/Promotion/PROJECTS/BOSTON/PYTHON/muScat/Data/EM-James/Parameter.mat'   # File with the experimental Parameters
# matlab_val_file = './Data/BEADS/Beads_40x_100a_allAmp.mat'   # File that stores the 3D (complex) Amplitude


# In[7]:


# This is the place to load data
''' File which stores the experimental parameters from the Q-PHASE setup 
    1.) Read in the parameters of the dataset ''' 
matlab_pars = data.import_parameters_mat(filename = matlab_par_file)
print
''' 2.) Read in the parameters of the dataset ''' 
# matlab_val = data.import_realdata_h5(filename = matlab_val_file, matname='allAmpSimu', is_complex=True)


# # Initiate the MuScat (Multiple Scattering) Object

# In[8]:


''' Create the Model'''
mm = mus.MuScatModel(matlab_pars, optimize, my_learningrate, my_keep_prob, tv_lambda, obj_reg_lambda, gr_lambda) # First initiliaze it


# ## Compute the System's properties (e.g. Pupil function/Illumination Source, K-vectors, etc.)

# In[ ]:


''' Compute the systems model'''
mm.computeSys()


# Now load data which has been saved previously (Optional!)

# In[ ]:


# Load Copute the systems model
#if(load_data):
#    mm.loadData()
#else:
#    mm.allSumAmp_mes = matlab_val
#

# ## Generate a phantom object in 3D 

# In[ ]:


''' Create a 3D Refractive Index Distributaton as a artificial sample'''
obj = tf_go.generateObject(mysize=mm.mysize, obj_dim=mm.dx, obj_type = 1, diameter = 1, dn = mm.dn)


# ## Create computational graph

# In[ ]:


''' Assign Object (3D sample) to Class'''
mm.obj_init = tf_go.generateInitObject(obj)

''' Create computational graph'''
mm.create_graph(obj, if_xla=0)


# ## Now evaluate the result - start inference

# In[ ]:


'''Compute result'''    
mm.eval_graph() # result will be stored in mm.allSumAmp

# mm.inference(obj, if_xla=0)


# 
# ## Visualize Results
# 

# In[ ]:


print(mm.allSumAmp.shape)
plt.imshow(np.abs(mm.allSumAmp[:,:,int(mm.Nx/2)]))
# mm.visualize_results()


# # Add some Regularizers

# In[ ]:


mm.regularizer(if_tvreg=True, if_posreg=True)


# In[ ]:





# In[ ]:




