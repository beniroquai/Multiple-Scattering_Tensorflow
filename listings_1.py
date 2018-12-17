#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 19:53:32 2017

@author: Bene

This file creates a fwd-model for the TESCAN Q-PHASE under 
multiple-scattering. It is majorly derived from  "LEarning approach for optical tomography"
U. S. Kamilov, BIG, EPFL, 2014.
"""
# %load_ext autoreload
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py 
import scipy.io
import scipy as scipy
import scipy.misc

# load own functions
import src.model as mus
import src.tf_helper as tf_helper
import src.tf_generate_object as tf_go
import src.data as data


from src import tf_helper as tf_helper, tf_generate_object as tf_go, data as data, model as mus

# Define some optimization parameters 
optimize = 0#0# want to optimize for the refractive index?

my_learningrate = 1e-3# learning rate
my_keep_prob = 1
tv_lambda = 1e-2
gr_lambda = 1e-2
obj_reg_lambda = 1e9

load_data = 0 # want to load previously stored data from disk (numpy) otherwise MATLAB experiment will be loaded
 
''' File which stores the experimental parameters from the Q-PHASE setup 
    1.) Read in the parameters of the dataset ''' 
matlab_par_file = './Data/BEADS/Beads_40x_100a_myParameter.mat'   
matlab_pars = data.import_parameters_mat(filename = matlab_par_file)

''' 2.) Read in the parameters of the dataset ''' 
matlab_val_file = './Data/BEADS/Beads_40x_100a_allAmp.mat'   
matlab_val = data.import_realdata_h5(filename = matlab_val_file, matname='allAmpSimu', is_complex=True)
# np_allAmpSimu = data.import_realdata_mat(filename = matlab_val_file)

''' Create the Model'''
muscat = mus.MuScatModel(matlab_pars, optimize, my_learningrate, my_keep_prob, tv_lambda, obj_reg_lambda, gr_lambda) # First initiliaze it

''' Adjust some parameters to fit it in the memory '''
#mm.Nz, mm.Nx, mm.Ny = np.shape(np_allAmpSimu)
muscat.Nz=100#int( np.double(np.array(self.myParamter.get('Nz'))))
muscat.Nx=100#np.int(np.floor((2*self.Rsim)/self.dx)+1);
muscat.Ny=100#np.int(np.floor((2*self.Rsim)/self.dy)+1)
muscat.NAc = 0.1

''' Compute the systems model'''
muscat.computeSys()

# Load Copute the systems model
if(load_data):
    muscat.loadData()
else:
    muscat.allSumAmp_mes = matlab_val

''' Create a 3D Refractive Index Distributaton as a artificial sample'''
obj = tf_go.generateObject(mysize=muscat.mysize, obj_dim=muscat.dx, obj_type ='sphere', diameter = 1, dn = muscat.dn)

''' Create computational graph'''
muscat.inference(obj, if_xla=0)

''' visualize Results'''
muscat.visResult()

'''Add Regularizers'''
muscat.regularizer({'RegTV':1,'RegGr':1,'RegL1':1})

'''define Cost-function'''
muscat.loss(loss_type = 4)

# initialize variables
muscat.compileGraph()

if(muscat.if_optimize):
    # feed tensor externally
    if(1):
        results = muscat.allSumAmp_mes
        results = np.angle(results)
        results = results - np.min(results)
        results = results/np.max(results)*(muscat.dn - 0.01)
#        results = np.flip(results, 0)
#        plt.imshow(results[:,mm.Nx/2,:]), plt.colorbar(), plt.show()
        

        tf_helper.saveHDF5(results, 'Obj_Guess')
        
    muscat.sess.run(tf.assign(muscat.TF_obj, results))
    
else:
    import sys; 
    print('sample was generated successfully')
    sys.exit(0)
    
    
print('Start optimizing')
tv_lambda = 1e-1
obj_reg_lambda = 1e6
my_learningrate =1e-4
gr_lambda = 1e-2

for iterx in range(1,1000):
    
          
#        if((current_error - last_error)<0):
#            my_learningrate = 1e-6
#        else:
#            my_learningrate = 1e-4
                
            

 #   last_error = current_error


    # dynamically change Variables
    #! 1/np.sqrt(iterx+100)*1e-2
    if(not np.mod(int(iterx/10),1)):
        my_keep_prob = .5#.95
    else:
        my_keep_prob = 1.
    
    if(not np.mod(iterx, 10)):
        if_evaluate = True
        print('Relative Error: ' + str(tf_helper.relative_error(muscat.obj, muscat.Obj_stack)))

    else:
        if_evaluate = False
    

    muscat.optimize(if_evaluate, iterx, my_learningrate, my_keep_prob, tv_lambda, obj_reg_lambda, gr_lambda)
       
        

tf_helper.saveHDF5(obj, 'Obj_Orig')
tf_helper.saveHDF5(muscat.Obj_stack, 'Obj_Stack')

if(0):
    print("Compare the measured data with the reconstructed intensity")
    plt.imshow((np.abs(TF_allSumAmp_mes.eval()[:, Nx/2, :])), cmap = 'gray')
    plt.show()
    plt.imshow((np.abs(TF_allSumAmp.eval()[:, Nx/2, :])), cmap = 'gray')
    plt.show()



print('Execute: ')
print('tensorboard --logdir:/tmp/tensorflow_logs --port 6006')
print(' and visit this page: http://localhost:6006/')


 
    
    
### 3D viewr stuff
    
from skimage import data
from skimage.viewer import CollectionViewer


import scipy.misc

max_val = np.max(Obj_stack)
for ii in range(0, Obj_stack.shape[0]):
    
    scipy.misc.toimage(Obj_stack[ii, :,:], cmin=0.0, cmax=max_val).save('./results/iter_real-obj'+str(ii)+'.jpg')
    print(str(ii))


scipy.misc.imsave('./results/iter_rec-obj'+str(ii)+'.tiff', Obj_stack)

from skimage import data
from skimage.viewer import CollectionViewer

#Obj_stack = np.angle(np_allAmpSimu)
#np.angle(np_allSumAmp)/np.max( np.angle(np_allSumAmp))

results = muscat.Obj_stack
#results = obj


#results = np.angle(np_allSumAmp)
results = results - np.min(results)
results = results/np.max(results)
img_collection = tuple(results)

view = CollectionViewer(img_collection)
view.show()


plt.imshow(np.angle(np_allAmpSimu[:,Nx/2, :])); plt.colorbar(); plt.show()
plt.imshow(np.abs(np_allAmpSimu[:,Nx/2, :])); plt.colorbar(); plt.show()


plt.imshow(np.angle(np_allSumAmp[:,Nx/2, :])); plt.colorbar(); plt.show()
plt.imshow(np.abs(np_allSumAmp[:,Nx/2, :])); plt.colorbar(); plt.show()


