import numpy as np 

# Parameters for optimization control
is_padding= False 
is_optimization= True
is_absorption= False
mysubsamplingIC= 0
NspikeLR= 25000 # try to get the system out of some local minima


'''Define Optimization Parameters'''
# these are hyperparameters
my_learningrate = 1e-1  # learning rate
NreduceLR = 500 # when should we reduce the Learningrate? 

lambda_tv = ((1e-2))##, 1e-2, 1e-2, 1e-3)) # lambda for Total variation - 1e-1
eps_tv = ((1e-12))##, 1e-12, 1e-8, 1e-6)) # - 1e-1 # smaller == more blocky
# these are fixed parameters
lambda_neg = 10000
Niter = 1200

Noptpsf = 1
Nsave = 100 # write info to disk
Ndisplay = Nsave
is_display =  False 

is_compute_psf='corr'


basepath = './'#'/projectnb/cislidt/diederich/muScat/Multiple-Scattering_Tensorflow/'
resultpath = 'Data/DROPLETS/RESULTS/'

dn = .051
myfac = 1e0# 0*dn*1e-3
myabsnorm = 1e5#myfac

np_global_phase = 0.
np_global_abs = 0.

''' microscope parameters '''
NAc = .52
shiftIcY = 0*.8 # has influence on the YZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
shiftIcX = 0*1 # has influence on the XZ-Plot - negative values shifts the input wave (coming from 0..end) to the left
zernikefactors = np.array((0,0,0,0,0,0,-.01,-.5001,0.01,0.01,.010))  # 7: ComaX, 8: ComaY, 11: Spherical Aberration
zernikefactors = np.array(( 0.001, 0.001, 0.01, 0, 0, 0., -3.4e-03,  2.2e-03, 0.001, .001, -1.0e+00))
zernikemask = np.array(np.abs(zernikefactors)>0)*1#!= np.array((0, 0, 0, 0, 0, 0, , 1, 1, 1, 1))# mask which factors should be updated



if(0):
    # 10mum bead
    # data files for parameters and measuremets 
    matlab_val_file = './Data/cells/Beads_40x_100a_100-250.tif_allAmp.mat'
    matlab_par_file = './Data/cells/Beads_40x_100a_100-250.tif_myParameter.mat'
    matlab_par_name = 'myParameter' 
    matlab_val_name = 'allAmpSimu' 
    mybackgroundval = -.95j  
elif(1):
    # data files for parameters and measuremets 
    matlab_val_file = './Data/cells/cross_section_10x0.3_hologram_full.tif_allAmp.mat'
    matlab_par_file = './Data/cells/cross_section_10x0.3_hologram_full.tif_myParameter.mat'
    matlab_par_name = 'myParameter' 
    matlab_val_name = 'allAmpSimu'
    mybackgroundval = -1j
elif(0):
    # data files for parameters and measuremets 
    matlab_val_file = './Data/cells/Cell_20x_100a_150-250.tif_allAmp.mat'
    matlab_par_file = './Data/cells/Cell_20x_100a_150-250.tif_myParameter.mat'
    matlab_par_name = 'myParameter' 
    matlab_val_name = 'allAmpSimu'   
    mybackgroundval = -.85j  
elif(0):
    # data files for parameters and measuremets 
    matlab_val_file = './Data/cells/S0019-2a_zstack_dz0-02um_751planes_40x_every8thslice.tif_allAmp.mat'
    matlab_par_file = './Data/cells/S0019-2a_zstack_dz0-02um_751planes_40x_every8thslice.tif_myParameter.mat'
    matlab_par_name = 'myParameter' 
    matlab_val_name = 'allAmpSimu'
else:
    matlab_par_file = './Data/DROPLETS/S19_multiple/Parameter.mat'; matlab_par_name='myParameter'
    matlab_val_file = './Data/DROPLETS/RESULTS/allAmp_simu.npy'
    mybackgroundval = -1j

 