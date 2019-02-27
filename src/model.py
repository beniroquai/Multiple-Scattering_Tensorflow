# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 09:52:05 2017
 
@author: useradmin
 
This 
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy as scipy
import scipy.misc
import yaml
from skimage.transform import warp, AffineTransform

# own functions
from src import tf_helper as tf_helper
from src import zernike as zern
from src import data as data
from src import tf_regularizers as tf_reg

class MuScatModel(object):
    def __init__(self, my_mat_paras, is_optimization = False):
 
        ''' Create Multiple Scattering Class;
        INPUTS:
        my_mat_paras - parameter file from MATLAB containing the experimental details - has to be load previously!
        '''
        # new MATLAB v7.3 Format!
        self.myParamter = my_mat_paras
        self.is_optimization = is_optimization
                 
        # ASsign variables from Experiment
        self.lambda0 = np.squeeze(np.array(self.myParamter.get('lambda0')))                 # free space wavelength (µm)
        self.NAo= np.squeeze(np.array(self.myParamter.get('NAo'))); # Numerical aperture objective
        self.NAc= np.squeeze(np.array(self.myParamter.get('NAc'))); # Numerical aperture condenser
        self.NAci = np.squeeze(np.array(self.myParamter.get('NAci'))); # Numerical aperture condenser
         
        # eventually decenter the illumination source - only integer!
        self.shiftIcX = 0#int(np.squeeze(np.array(self.myParamter.get('shiftIcX'))))
        self.shiftIcY = 0#int(np.squeeze(np.array(self.myParamter.get('shiftIcY'))))
         
        self.nEmbb = np.squeeze(np.array(self.myParamter.get('nEmbb'))) 
        self.dn=.1; # self.nImm - self.nEmbb
        print('Assigned some value for dn which is not good!')
         
        # calculate pixelsize
        self.dx = np.double(np.squeeze(np.array(self.myParamter.get('dx'))))
        self.dy = np.double(np.array(self.myParamter.get('dy')))
        self.dz = np.double(np.array(self.myParamter.get('dz')))
             
        # Sampling coordinates
        self.Rsim= 0.5*np.double(np.array(self.myParamter.get('Nx')))*self.dx; # Radius over which simulation is performed.
         
        self.Nz=int(np.double(np.array(self.myParamter.get('Nz'))))
        self.Nx=np.int(np.floor((2*self.Rsim)/self.dx)+1);
        self.Ny=np.int(np.floor((2*self.Rsim)/self.dy)+1)
         
        # create the first guess of the initial obj 
        self.obj = np.ones((self.Nz, self.Nx, self.Ny))
         
        # add a vector of zernike factors
        self.nzernikes = 9
        self.zernikefactors = np.zeros((1,self.nzernikes))
        self.zernikemask = np.zeros((1,self.nzernikes))
        # kamilov uses 420 z-planes with an overall size of 30µm; dx=72nm
 
        # refractive index immersion and embedding
        self.lambdaM = self.lambda0/self.nEmbb; # wavelength in the medium
 
    #@define_scope
    def computesys(self, obj=None, is_padding=False, is_tomo = False, dropout_prob=1, mysubsamplingIC=0):
        """ This computes the FWD-graph of the Q-PHASE microscope;
        1.) Compute the physical dimensions
        2.) Compute the sampling for the waves
        3.) Create the illumination waves depending on System's properties
 
        ##### IMPORTANT! ##### 
        The ordering of the channels is as follows:
            Nillu, Nz, Nx, Ny
        """
        # define whether we want to pad the experiment 
        self.is_padding = is_padding
        self.is_tomo = is_tomo
        self.mysubsamplingIC = mysubsamplingIC
        self.dropout_prob = dropout_prob
        
        if(is_padding):
            print('--------->WARNING: Padding is not yet working correctly!!!!!!!!')
            # add padding in X/Y to avoid wrap-arounds
            self.mysize_old = np.array((self.Nz, self.Nx, self.Ny))            
            self.Nx=self.Nx*2
            self.Ny=self.Ny*2
            self.mysize = np.array((self.Nz, self.Nx, self.Ny))
            self.obj = obj
            self.dx=self.dx
            self.dy=self.dy
        else:
            self.mysize=np.array((self.Nz, self.Nx, self.Ny))
            self.mysize_old = self.mysize
            
        # Allocate memory for the object 
        if obj is not None:
            self.obj = obj
        else:
            self.obj = np.zeros(self.mysize)
            
        # eventually add some dropout to the model?
        if(self.dropout_prob<1):
            self.tf_dropout_prob = tf.placeholder(tf.float32,[])
        else:
            self.tf_dropout_prob = tf.constant(self.dropout_prob)
 
        # Decide whether we wan'T to optimize or simply execute the model
        if (self.is_optimization==1):
            #self.TF_obj = tf.Variable(np.real(self.obj), dtype=tf.float32, name='Object_Variable')
            #self.TF_obj_absorption = tf.Variable(np.imag(self.obj), dtype=tf.float32, name='Object_Variable')                
            with tf.variable_scope("Complex_Object"):
                self.TF_obj = tf.get_variable('Object_Variable_Real', dtype=tf.float32, initializer=np.float32(np.real(self.obj)))
                self.TF_obj_absorption = tf.get_variable('Object_Variable_Imag', dtype=tf.float32, initializer=np.float32(np.imag(self.obj)))
                #set reuse flag to True
                tf.get_variable_scope().reuse_variables()
                #just an assertion!
                assert tf.get_variable_scope().reuse==True             
             
            # assign training variables 
            self.tf_lambda_tv = tf.placeholder(tf.float32, [])
            self.tf_eps = tf.placeholder(tf.float32, [])
            self.tf_meas = tf.placeholder(dtype=tf.complex64, shape=self.mysize_old)
            self.tf_learningrate = tf.placeholder(tf.float32, []) 
 
        elif(self.is_optimization==0):
            # Variables of the computational graph
            self.TF_obj = tf.constant(np.real(self.obj), dtype=tf.float32, name='Object_const')
            self.TF_obj_absorption = tf.constant(np.imag(self.obj), dtype=tf.float32, name='Object_const')
         
        elif(self.is_optimization==-1):
            # THis is for the case that we want to train the resnet
            self.tf_meas = tf.placeholder(dtype=tf.complex64, shape=self.mysize_old)
            # in case one wants to use this as a fwd-model for an inverse problem            
 
            #self.TF_obj = tf.Variable(np.real(self.obj), dtype=tf.float32, name='Object_Variable')
            #self.TF_obj_absorption = tf.Variable(np.imag(self.obj), dtype=tf.float32, name='Object_Variable')                
            self.TF_obj = tf.placeholder(dtype=tf.float32, shape=self.obj.shape, name='Object_Variable_Real')
            self.TF_obj_absorption = tf.placeholder(dtype=tf.float32, shape=self.obj.shape, name='Object_Variable_Imag')
 
            # assign training variables 
            self.tf_lambda_tv = tf.placeholder(tf.float32, [])
            self.tf_eps = tf.placeholder(tf.float32, [])
 
            self.tf_learningrate = tf.placeholder(tf.float32, [])

         
        ## Establish normalized coordinates.
        #-----------------------------------------
        vxx= tf_helper.xx((self.mysize[1], self.mysize[2]),'freq') * self.lambdaM * self.nEmbb / (self.dx * self.NAo);    # normalized optical coordinates in X
        vyy= tf_helper.yy((self.mysize[1], self.mysize[2]),'freq') * self.lambdaM * self.nEmbb / (self.dy * self.NAo);    # normalized optical coordinates in Y
         
        # AbbeLimit=lambda0/NAo;  # Rainer's Method
        # RelFreq = rr(mysize,'freq')*AbbeLimit/dx;  # Is not generally right (dx and dy)
        self.RelFreq = np.sqrt(tf_helper.abssqr(vxx) + tf_helper.abssqr(vyy));    # spanns the frequency grid of normalized pupil coordinates
        self.Po=self.RelFreq < 1.0;   # Create the pupil of the objective lens        
         
        # Precomputing the first 9 zernike coefficients 
        self.nzernikes = np.squeeze(self.zernikefactors.shape)
        self.myzernikes = np.zeros((self.Po.shape[0],self.Po.shape[1],self.nzernikes))+ 1j*np.zeros((self.Po.shape[0],self.Po.shape[1],self.nzernikes))
        r, theta = zern.cart2pol(vxx, vyy)        
        for i in range(0,self.nzernikes):
            self.myzernikes[:,:,i] = np.fft.fftshift(zern.zernike(r, theta, i+1, norm=False)) # or 8 in X-direction
             
        # eventually introduce a phase factor to approximate the experimental data better
        self.Po = np.fft.fftshift(self.Po)# Need to shift it before using as a low-pass filter    Po=np.ones((np.shape(Po)))
        print('----------> Be aware: We are taking aberrations into account!')
        # Assuming: System has coma along X-direction
        self.myaberration = np.sum(self.zernikefactors * self.myzernikes, axis=2)
        self.Po = 1.*self.Po#*np.exp(1j*self.myaberration)
         
        # Prepare the normalized spatial-frequency grid.
        self.S = self.NAc/self.NAo;   # Coherence factor
        self.Ic = self.RelFreq <= self.S
         
        # Take Darkfield into account
        if hasattr(self, 'NAci'):
            if self.NAci != None and self.NAci > 0:
                #print('I detected a darkfield illumination aperture!')
                self.S_o = self.NAc/self.NAo;   # Coherence factor
                self.S_i = self.NAci/self.NAo;   # Coherence factor
                self.Ic = (1.*(self.RelFreq < self.S_o) * 1.*(self.RelFreq > self.S_i))>0 # Create the pupil of the condenser plane
 
        # weigh the illumination source with some cos^2 intensity weight?!
        if(0):
            myIntensityFactor = 70
            self.Ic_map = np.cos((myIntensityFactor *tf_helper.xx((self.Nx, self.Ny), mode='freq')**2+myIntensityFactor *tf_helper.yy((self.Nx, self.Ny), mode='freq')**2))**2
            print('We are taking the cosine illuminatino shape!')
        elif(0):
            print('We are taking the gaussian illuminatino shape!')
            myIntensityFactor = 0.01
            self.Ic_map = np.exp(-tf_helper.rr((self.Nx, self.Ny),mode='freq')**2/myIntensityFactor)
        else:
            print('We are not weighing our illumination!')
            self.Ic_map = np.ones((self.Nx, self.Ny))
            
        
        # This is experimental
        if(self.mysubsamplingIC>0):
            self.checkerboard = np.zeros((self.mysubsamplingIC,self.mysubsamplingIC))# ((1,0),(0,0))  # testing for sparse illumination?!
            self.checkerboard[0,0] = 1
            print('-------> ATTENTION: WE have a CHECKeRBOArD  MASK IN THE PUPIL PLANE!!!!')
            self.checkerboard = np.matlib.repmat(self.checkerboard,self.Ic_map.shape[0]//self.mysubsamplingIC+1,self.Ic_map.shape[1]//self.mysubsamplingIC+1)
            self.checkerboard = self.checkerboard[0:self.Ic_map.shape[0], 0:self.Ic_map.shape[1]]
        else:
            self.checkerboard = np.ones(self.Ic.shape)

        self.Ic = self.Ic * self.Ic_map  # weight the intensity in the condenser aperture, unlikely to be uniform
        # print('--------> ATTENTION! - We are not weighing the Intensity int the illu-pupil!')
 
 
        # Shift the pupil in X-direction (optical missalignment)
        if hasattr(self, 'shiftIcX'):
            if self.shiftIcX != None:
                if(is_padding): self.shiftIcX=self.shiftIcX*2
                print('Shifting the illumination in X by: ' + str(self.shiftIcX) + ' Pixel')
                if(0):
                    self.Ic = np.roll(self.Ic, self.shiftIcX, axis=1)
                elif(1):
                    tform = AffineTransform(scale=(1, 1), rotation=0, shear=0, translation=(self.shiftIcX, 0))
                    self.Ic = warp(self.Ic, tform.inverse, output_shape=self.Ic.shape)
                elif(0):
                    # We apply a phase-factor to shift the source in realspace - so make it trainable
                    self.shift_xx = tf_helper.xx((self.mysize[1], self.mysize[2]),'freq')
                    self.Ic = np.abs(np.fft.ifft2(np.fft.fft2(self.Ic)*np.exp(1j*2*np.pi*self.shift_xx*self.shiftIcX))) 


        # Shift the pupil in Y-direction (optical missalignment)
        if hasattr(self, 'shiftIcY'):
            if self.shiftIcY != None:
                if(is_padding): self.shiftIcY=self.shiftIcY*2
                print('Shifting the illumination in Y by: ' + str(self.shiftIcY) + ' Pixel')
                if(0):
                    self.Ic = np.roll(self.Ic, self.shiftIcY, axis=0)
                elif(1):
                    tform = AffineTransform(scale=(1, 1), rotation=0, shear=0, translation=(0, self.shiftIcY))
                    self.Ic = warp(self.Ic, tform.inverse, output_shape=self.Ic.shape)
                elif(0):
                    # We apply a phase-factor to shift the source in realspace - so make it trainable
                    self.shift_yy = tf_helper.yy((self.mysize[1], self.mysize[2]),'freq')
                    self.Ic = np.abs(np.fft.ifft2(np.fft.fft2(self.Ic)*np.exp(1j*self.shift_yy*self.shiftIcY))) 

        self.Ic = self.Ic * self.checkerboard
 
        ## Forward propagator  (Ewald sphere based) DO NOT USE NORMALIZED COORDINATES HERE
        self.kxysqr= (tf_helper.abssqr(tf_helper.xx((self.mysize[1], self.mysize[2]), 'freq') / self.dx) + tf_helper.abssqr(
            tf_helper.yy((self.mysize[1], self.mysize[2]), 'freq') / self.dy)) + 0j;
        self.k0=1/self.lambdaM;
        self.kzsqr= tf_helper.abssqr(self.k0) - self.kxysqr;
        self.kz=np.sqrt(self.kzsqr);
        self.kz[self.kzsqr < 0]=0;
        self.dphi = 2*np.pi*self.kz*self.dz;  # exp(1i*kz*dz) would be the propagator for one slice
 
        ## Get a list of vector coordinates corresponding to the pixels in the mask
        xfreq= tf_helper.xx((self.mysize[1], self.mysize[2]),'freq');
        yfreq= tf_helper.yy((self.mysize[1], self.mysize[2]),'freq');
        self.Nc=np.sum(self.Ic>0); 
        print('Number of Illumination Angles / Plane waves: '+str(self.Nc))
         
        # Calculate the computatonal grid/sampling
        self.kxcoord = np.reshape(xfreq[self.Ic>0],[1, 1, 1, self.Nc]);    # NA-positions in condenser aperture plane in x-direction
        self.kycoord = np.reshape(yfreq[self.Ic>0],[1, 1, 1, self.Nc]);    # NA-positions in condenser aperture plane in y-direction
        self.RefrCos = np.reshape(self.k0/self.kz[self.Ic>0],[1, 1, 1, self.Nc]);   # 1/cosine used for the application of the refractive index steps to acount for longer OPD in medium under an oblique illumination angle
         
        ## Generate the illumination amplitudes
        self.intensityweights = self.Ic[self.Ic>0]
        self.A_input = self.intensityweights *np.exp((2*np.pi*1j) *
            (self.kxcoord * tf_helper.repmat4d(tf_helper.xx((self.mysize[1], self.mysize[2])), self.Nc) 
           + self.kycoord * tf_helper.repmat4d(tf_helper.yy((self.mysize[1], self.mysize[2])), self.Nc))) # Corresponds to a plane wave under many oblique illumination angles - bfxfun
         
        ## propagate field to z-stack and sum over all illumination angles
        self.Alldphi = -(np.reshape(np.arange(0, self.mysize[0], 1), [1, 1, self.mysize[0]])-self.Nz/2)*np.repeat(np.fft.fftshift(self.dphi)[:, :, np.newaxis], self.mysize[0], axis=2)
          
        # Ordinary backpropagation. This is NOT what we are interested in:
        self.myAllSlicePropagator=np.transpose(np.exp(1j*self.Alldphi) * (np.repeat(np.fft.fftshift(self.dphi)[:, :, np.newaxis], self.mysize[0], axis=2) >0), [2, 0, 1]);  # Propagates a single end result backwards to all slices
 
         
    #@define_scope
    def computemodel(self, is_resnet=False, is_forcepos=False, is_compute_psf=False):
        ''' Perform Multiple Scattering here
        1.) Multiple Scattering is perfomed by slice-wise propagation the E-Field throught the sample
        2.) Each Field has to be backprojected to the BFP
        3.) Last step is to ceate a focus-stack and sum over all angles
 
        This is done for all illumination angles (coming from illumination NA
        simultaneasly)'''
        self.is_resnet = is_resnet
        self.is_forcepos = is_forcepos
        self.is_compute_psf = is_compute_psf
 
        print("Buildup Q-PHASE Model ")
        ###### make sure, that the first dimension is "batch"-size; in this case it is the illumination number
        # @beniroquai It's common to have to batch dimensions first and not last.!!!!!
        # the following loop propagates the field sequentially to all different Z-planes
 
        ## propagate the field through the entire object for all angles simultaneously
        A_prop = np.transpose(self.A_input,[3, 0, 1, 2])  # ??????? what the hack is happening with transpose?!
         
        if(self.is_tomo):
            print('Experimentally using the tomographic scheme!')
            A_prop = np.conj(A_prop)
             
        myprop = np.exp(1j * self.dphi) * (self.dphi > 0);  # excludes the near field components in each step
        myprop = tf_helper.repmat4d(np.fft.fftshift(myprop), self.Nc)
        myprop = np.transpose(myprop, [3, 0, 1, 2])  # ??????? what the hack is happening with transpose?!
 
        print('--------> ATTENTION: I added a pi factor - is this correct?!')
        self.RefrEffect = np.squeeze(1j * np.pi* self.dz * self.k0 * self.RefrCos);  # Precalculate the oblique effect on OPD to speed it up
         
        # for now orientate the dimensions as (alpha_illu, x, y, z) - because tensorflow takes the first dimension as batch size
        with tf.name_scope('Variable_assignment'):
            self.TF_A_input = tf.constant(A_prop, dtype=tf.complex64)
            self.TF_RefrEffect = tf.reshape(tf.constant(self.RefrEffect, dtype=tf.complex64), [self.Nc, 1, 1])
            self.TF_myprop = tf.constant(np.squeeze(myprop), dtype=tf.complex64)
            self.TF_Po = tf.cast(tf.constant(self.Po), tf.complex64)
            self.TF_Zernikes = tf.constant(self.myzernikes, dtype=tf.float32)
            self.TF_myAllSlicePropagator = tf.constant(self.myAllSlicePropagator, dtype=tf.complex64)

           
            # Only update those Factors which are really necesarry (e.g. Defocus is not very likely!)
            self.TF_zernikefactors = tf.Variable(self.zernikefactors, dtype = tf.float32, name='var_zernikes')
            #indexes = tf.constant([[4], [5], [6], [7], [8], [9]])
            indexes = tf.cast(tf.where(tf.constant(self.zernikemask)>0), tf.int32)
            updates = tf.gather_nd(self.TF_zernikefactors,indexes)
            # Take slice
            # Build tensor with "filtered" gradient
            part_X = tf.scatter_nd(indexes, updates, tf.shape(self.TF_zernikefactors))
            self.TF_zernikefactors_filtered = part_X + tf.stop_gradient(-part_X + self.TF_zernikefactors)
            
        # TODO: Introduce the averraged RI along Z - MWeigert
        self.TF_A_prop = tf.squeeze(self.TF_A_input);
        self.U_z_list = []

        # Initiliaze memory
        self.allSumAmp = 0
        self.TF_allSumAmp = tf.zeros([self.mysize[0], self.Nx, self.Ny], dtype=tf.complex64)
        self.TF_allASF = []
        
        if not self.is_compute_psf:
            # only consider object scattering if we want to use it
        
            ''' Eventually add a RESNET-layer between RI and Microscope to correct for model discrepancy?'''
            if(self.is_resnet):
                with tf.variable_scope('res_real', reuse=False):
                    TF_real_3D = self.residual_block(tf.expand_dims(tf.expand_dims(self.TF_obj,3),0),1,True)
                    TF_real_3D = tf.squeeze(TF_real_3D)
                with tf.variable_scope('res_imag', reuse=False):
                    TF_imag_3D = self.residual_block(tf.expand_dims(tf.expand_dims(self.TF_obj_absorption,3),0),1,True)
                    TF_imag_3D = tf.squeeze(TF_imag_3D)
            else:
                TF_real_3D = self.TF_obj
                TF_imag_3D = self.TF_obj_absorption     
                
            # Eventually add dropout
            if(0):#self.dropout_prob<1):
                TF_real_3D = tf.layers.dropout(TF_real_3D, self.tf_dropout_prob)
                TF_imag_3D = tf.layers.dropout(TF_imag_3D, self.tf_dropout_prob)
                print('We add dropout if necessary')
    
            # wrapper for force-positivity on the RI-instead of penalizing it
            if(self.is_forcepos):
                print('----> ATTENTION: We add the PreMonotonicPos' )
                TF_real_3D = tf_reg.PreMonotonicPos(TF_real_3D)
                TF_imag_3D = tf_reg.PreMonotonicPos(TF_imag_3D)
    
            # simulate multiple scattering through object
            with tf.name_scope('Fwd_Propagate'):
                #print('---------ATTENTION: We are inverting the RI!')
                for pz in range(0, self.mysize[0]):
                    #self.TF_A_prop = tf.Print(self.TF_A_prop, [self.tf_iterator], 'Prpagation step: ')
                    with tf.name_scope('Refract'):
                        # beware the "i" is in TF_RefrEffect already!
                        if(self.is_padding):
                            tf_paddings = tf.constant([[self.mysize_old[1]//2, self.mysize_old[1]//2], [self.mysize_old[2]//2, self.mysize_old[2]//2]])
                            TF_real = tf.pad(TF_real_3D[-pz,:,:], tf_paddings, mode='CONSTANT', name='TF_obj_real_pad')
                            TF_imag = tf.pad(TF_imag_3D[-pz,:,:], tf_paddings, mode='CONSTANT', name='TF_obj_imag_pad')
                        else:
                            TF_real = (TF_real_3D[-pz,:,:])
                            TF_imag = (TF_imag_3D[-pz,:,:])
                            
    
                        self.TF_f = tf.exp(self.TF_RefrEffect*tf.complex(TF_real, TF_imag))
                        self.TF_A_prop = self.TF_A_prop * self.TF_f  # refraction step
                    with tf.name_scope('Propagate'):
                        self.TF_A_prop = tf.ifft2d(tf.fft2d(self.TF_A_prop) * self.TF_myprop) # diffraction step
     
 
        # in a final step limit this to the detection NA:
        self.TF_Po_aberr = tf.exp(1j*tf.cast(tf.reduce_sum(self.TF_zernikefactors_filtered*self.TF_Zernikes, axis=2), tf.complex64)) * self.TF_Po
        if not self.is_compute_psf:
            self.TF_A_prop = tf.ifft2d(tf.fft2d(self.TF_A_prop)*self.TF_Po * self.TF_Po_aberr)
        else:
            # in case we only want to compute the PSF, we don't need the input field
            self.TF_A_prop = tf_helper.my_ift2d(tf.ones_like(self.TF_A_prop)*tf_helper.ifftshift2d(self.TF_Po * self.TF_Po_aberr)) # propagate in real-space->fftshift!; tf_ones: need for broadcasting!
            self.TF_myAllSlicePropagator = self.TF_myAllSlicePropagator*tf.exp(-1j*tf.cast(self.Nz/2, tf.complex64))
            
            
        # Experimenting with pseudo tomographic data? No backpropgation necessary!
        if self.is_tomo:
            print('Only Experimental! Tomographic data?!')
            # Bring the slice back to focus - does this make any sense?! 
            print('----------> Bringing field back to focus')
            TF_centerprop = tf.exp(-1j*tf.cast(self.Nz/2*tf.angle(self.TF_myprop), tf.complex64))
            self.TF_A_prop = tf.ifft2d(tf.fft2d(self.TF_A_prop) * TF_centerprop) # diffraction step
            return self.TF_A_prop
         
        # create Z-Stack by backpropagating Information in BFP to Z-Position
        self.kzcoord = np.reshape(self.kz[self.Ic>0], [1, 1, 1, self.Nc]);
 
        # self.mid3D = ([np.int(np.ceil(self.A_input.shape[0] / 2) + 1), np.int(np.ceil(self.A_input.shape[1] / 2) + 1), np.int(np.ceil(self.mysize[0] / 2) + 1)])
        self.mid3D = ([np.int(self.mysize[0]//2), np.int(self.A_input.shape[0] // 2), np.int(self.A_input.shape[1]//2)])
        with tf.name_scope('Back_Propagate'):
            print('----------> ATTENTION: PHASE SCRAMBLING!')
            for pillu in range(0, self.Nc):
                with tf.name_scope('Back_Propagate_Step'):
                    with tf.name_scope('Adjust'):
                        #    fprintf('BackpropaAngle no: #d\n',pillu);
                        OneAmp = tf.expand_dims(self.TF_A_prop[pillu, :, :], 0)

                        # Fancy backpropagation assuming what would be measured if the sample was moved under oblique illumination:
                        # The trick is: First use conceptually the normal way
                        # and then apply the XYZ shift using the Fourier shift theorem (corresponds to physically shifting the object volume, scattered field stays the same):
                        self.TF_AdjustKXY = tf.squeeze(tf.conj(self.TF_A_input[pillu,:,:,])) # Maybe a bit of a dirty hack, but we first need to shift the zero coordinate to the center
                        self.TF_AdjustKZ = tf.cast(tf.transpose(np.exp(
                            2 * np.pi * 1j * self.dz * np.reshape(np.arange(0, self.mysize[0], 1), # We want to start from first Z-slice then go to last which faces the objective lens
                                  [1, 1, self.mysize[0]]) * self.kzcoord[:, :, :,pillu]), [2, 1, 0]), tf.complex64)
                        self.TF_allAmp = tf.ifft2d(tf.fft2d(OneAmp) * self.TF_myAllSlicePropagator) * self.TF_AdjustKZ * self.TF_AdjustKXY  # * (TF_AdjustKZ);  # 2x bfxfun.  Propagates a single amplitude pattern back to the whole stack
                        #tf_global_phase = tf.cast(tf.angle(self.TF_allAmp[self.mid3D[0],self.mid3D[1],self.mid3D[2]]), tf.complex64)
                        #tf_global_phase = tf.cast(np.random.randn(1)*np.pi,tf.complex64)
                        #self.TF_allAmp = self.TF_allAmp * tf.exp(1j*tf_global_phase) # Global Phases need to be adjusted at this step!  Use the zero frequency
                         
                    if (1):
                        with tf.name_scope('Propagate'):
                            self.TF_allAmp_3dft = tf.fft3d(tf.expand_dims(self.TF_allAmp, axis=0))
                            tf_global_phase = tf.angle(self.TF_allAmp_3dft[0,0,0,0])#tf.angle(self.TF_allAmp_3dft[0, self.mid3D[2], self.mid3D[1], self.mid3D[0]])
                            tf_global_phase = tf.cast(tf_global_phase, tf.complex64)

                            self.TF_allAmp = self.TF_allAmp * tf.exp(-1j * tf_global_phase);  # Global Phases need to be adjusted at this step!  Use the zero frequency
                    #print('Global phase: '+str(tf.exp(1j*tf.cast(tf.angle(self.TF_allAmp[self.mid3D[0],self.mid3D[1],self.mid3D[2]]), tf.complex64).eval()))
 
                    with tf.name_scope('Sum_Amps'): # Normalize amplitude by condenser intensity
                        self.TF_allSumAmp = self.TF_allSumAmp + self.TF_allAmp #/ self.intensityweights[pillu];  # Superpose the Amplitudes
                        if self.is_compute_psf:
                            self.TF_allASF.append(self.TF_allAmp)
                             
                    # print('Current illumination angle # is: ' + str(pillu))
 
 
        # Normalize the image such that the values do not depend on the fineness of
        # the source grid.
        self.TF_allSumAmp = self.TF_allSumAmp/tf.cast(np.sum(self.Ic), tf.complex64) # tf.cast(tf.reduce_max(tf.abs(self.TF_allSumAmp)), tf.complex64) # self.Nc #/
        # Following is the normalization according to Martin's book. It ensures
        # that a transparent specimen is imaged with unit intensity.
        # normfactor=abs(Po).^2.*abs(Ic); We do not use it, because it leads to
        # divide by zero for dark-field system. Instead, through normalizations
        # perfomed above, we ensure that image of a point under matched
        # illumination is unity. The brightness of all the other configurations is
        # relative to this benchmark.
 
        if self.is_compute_psf:
            self.TF_allASF = tf.stack(self.TF_allASF)
            self.TF_ATF = tf.squeeze(tf_helper.my_ft3d(tf.reduce_sum(self.TF_allASF, 0)/np.sum(self.Ic)))
            self.TF_ATF = self.TF_ATF/self.TF_ATF[0,0,0] # normalize ATF
            
        # negate padding        
        if self.is_padding:
            self.TF_allSumAmp = self.TF_allSumAmp[:,self.Nx//2-self.Nx//4:self.Nx//2+self.Nx//4, self.Ny//2-self.Ny//4:self.Ny//2+self.Ny//4]
             
        return self.TF_allSumAmp
     
     
    def computeconvolution(self):
        # We want to compute the born-fwd model
        print('Computing the fwd model in born approximation')
        
        TF_real_3D = self.TF_obj
        TF_imag_3D = self.TF_obj_absorption    
        
        # wrapper for force-positivity on the RI-instead of penalizing it
        if(self.is_forcepos):
            print('----> ATTENTION: We add the PreMonotonicPos' )
            TF_real_3D = tf_reg.PreMonotonicPos(TF_real_3D)
            TF_imag_3D = tf_reg.PreMonotonicPos(TF_imag_3D)

        
        ''' MATLAB
        dn = .1+0*.1i;
        mysphere = rr([mysize, Nz])<5;
        f = zeros(size(mysphere))+nEmbb;
        f(mysphere) = dn;
        myN = 1i*nEmbb+dz*k0*f; %RefrCos
        RefrEffect=-1i*dz*k0;RefrCos;  % To speed it up
        myN = exp(RefrEffect*f);
        
        myres = squeeze(ift3d(ft3d(myN)*ft3d(allSumAmp)));
        myres = squeeze(sum(myres, [], 4))
        '''
        TF_f = tf.complex(TF_real_3D, TF_imag_3D)
        TF_myN = tf.exp(tf.expand_dims(self.TF_RefrEffect,-1)*TF_f)

        self.TF_ATF_placeholder = tf.placeholder(tf.complex64, shape=TF_f.shape, name='TF_ATF_placeholder')
        # convolve object with ASF
        TF_myN = tf.squeeze(tf.reduce_mean(TF_myN, 0))
        
        TF_res = tf_helper.my_ift3d(tf_helper.my_ft3d(TF_myN)*self.TF_ATF_placeholder)
        return tf.squeeze(TF_res)
     
     
    def addRegularizer(self, is_tv, is_gr, is_pos):
        print('Do stuff')
 
    def defineWriter(self, logs_path = '/tmp/tensorflow_logs/example/'):
        # create writer object
        self.logs_path = logs_path 
        self.writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
         
     
    def writeParameterFile(self, mylr, mytv, myeps, filepath = '/myparameters.yml'):
        ''' Write out all parameters to a yaml file in case we need it later'''
        mydata = dict(
                shiftIcX = float(self.shiftIcX),
                shiftIcY = float(self.shiftIcY),                
                NAc = float(self.NAc),
                NAo = float(self.NAo), 
                Nc = float(self.Nc), 
                Nx = float(self.Nx), 
                Ny = float(self.Ny),
                Nz = float(self.Nz),
                dx = float(self.dx),
                dy = float(self.dy),
                dz = float(self.dz),
                dn = float(self.dn),
                lambda0 = float(self.lambda0),
                lambdaM = float(self.lambdaM),
                learningrate = mylr, 
                lambda_tv = mytv, 
                eps_tv = myeps) 
                #zernikfactors = float(self.zernikefactors))
 
        with open(filepath, 'w') as outfile:
                yaml.dump(mydata, outfile, default_flow_style=False)
                 
      
    def saveFigures(self, sess, savepath, tf_fwd, np_meas, mylosslist, myfidelitylist, myneglosslist, mytvlosslist, globalphaselist, globalabslist, 
                    result_phaselist=None, result_absorptionlist=None, init_guess=None, figsuffix=''):
        # This is the reconstruction
        if(init_guess is not None):
            myfwd, mymeas, my_res, my_res_absorption, myzernikes = sess.run([tf_fwd, self.tf_meas, self.TF_obj, self.TF_obj_absorption, self.TF_zernikefactors], 
                    feed_dict={self.tf_meas:np_meas, self.TF_obj:np.real(init_guess), self.TF_obj_absorption:np.imag(init_guess), self.tf_dropout_prob:1})
        else:
            myfwd, mymeas, my_res, my_res_absorption, myzernikes = sess.run([tf_fwd, self.tf_meas, self.TF_obj, self.TF_obj_absorption, self.TF_zernikefactors], feed_dict={self.tf_meas:np_meas})
             
        plt.figure()
        plt.subplot(231), plt.title('ABS XZ'),plt.imshow(np.abs(myfwd)[:,myfwd.shape[1]//2,:]), plt.colorbar()#, plt.show()
        plt.subplot(232), plt.title('ABS YZ'),plt.imshow(np.abs(myfwd)[:,:,myfwd.shape[2]//2]), plt.colorbar()#, plt.show()
        plt.subplot(233), plt.title('ABS XY'),plt.imshow(np.abs(myfwd)[myfwd.shape[0]//2,:,:]), plt.colorbar()#, plt.show()
        plt.subplot(234), plt.title('Angle XZ'),plt.imshow(np.angle(myfwd)[:,myfwd.shape[1]//2,:]), plt.colorbar()#, plt.show()
        plt.subplot(235), plt.title('Angle XZ'),plt.imshow(np.angle(myfwd)[:,:,myfwd.shape[2]//2]), plt.colorbar()#, plt.show()
        plt.subplot(236), plt.title('Angle XY'),plt.imshow(np.angle(myfwd)[myfwd.shape[0]//2,:,:]), plt.colorbar()#, plt.show()
        plt.savefig(savepath+'/res_myfwd'+figsuffix+'.png'), plt.show()
     
        # This is the measurment
        plt.figure()
        plt.subplot(231), plt.title('ABS XZ'),plt.imshow(np.abs(mymeas)[:,mymeas.shape[1]//2,:]), plt.colorbar()#, plt.show()
        plt.subplot(232), plt.title('ABS YZ'),plt.imshow(np.abs(mymeas)[:,:,mymeas.shape[2]//2]), plt.colorbar()#, plt.show()
        plt.subplot(233), plt.title('ABS XY'),plt.imshow(np.abs(mymeas)[mymeas.shape[0]//2,:,:]), plt.colorbar()#, plt.show()
        plt.subplot(234), plt.title('Angle XZ'),plt.imshow(np.angle(mymeas)[:,mymeas.shape[1]//2,:]), plt.colorbar()#, plt.show()
        plt.subplot(235), plt.title('Angle XZ'),plt.imshow(np.angle(mymeas)[:,:,mymeas.shape[2]//2]), plt.colorbar()#, plt.show()
        plt.subplot(236), plt.title('Angle XY'),plt.imshow(np.angle(mymeas)[mymeas.shape[0]//2,:,:]), plt.colorbar()#, plt.show()
        plt.savefig(savepath+'/res_mymeas'+figsuffix+'.png'), plt.show()
     
        # This is the residual
        plt.figure()
        plt.subplot(231), plt.title('Residual ABS XZ'),plt.imshow((np.abs(myfwd)-np.abs(mymeas))[:,myfwd.shape[1]//2,:]), plt.colorbar()#, plt.show()
        plt.subplot(232), plt.title('Residual ABS YZ'),plt.imshow((np.abs(myfwd)-np.abs(mymeas))[:,:,myfwd.shape[2]//2]), plt.colorbar()#, plt.show()
        plt.subplot(233), plt.title('Residual ABS XY'),plt.imshow((np.abs(myfwd)-np.abs(mymeas))[myfwd.shape[0]//2,:,:]), plt.colorbar()#, plt.show()
        plt.subplot(234), plt.title('Residual Angle XZ'),plt.imshow((np.angle(myfwd)-np.angle(mymeas))[:,myfwd.shape[1]//2,:]), plt.colorbar()#, plt.show()
        plt.subplot(235), plt.title('Residual Angle XZ'),plt.imshow((np.angle(myfwd)-np.angle(mymeas))[:,:,myfwd.shape[2]//2]), plt.colorbar()#, plt.show()
        plt.subplot(236), plt.title('Residual Angle XY'),plt.imshow((np.angle(myfwd)-np.angle(mymeas))[myfwd.shape[0]//2,:,:]), plt.colorbar()#, plt.show()
        plt.savefig(savepath+'/res_myresidual'+figsuffix+'.png'), plt.show()
         
        # diplay the error over time
        plt.figure()
        plt.subplot(231), plt.title('Error/Cost-function'), plt.semilogy((np.array(mylosslist)))#, plt.show()
        plt.subplot(232), plt.title('Fidelity-function'), plt.semilogy((np.array(myfidelitylist)))#, plt.show()
        plt.subplot(233), plt.title('Neg-loss-function'), plt.plot(np.array(myneglosslist))#, plt.show()
        plt.subplot(234), plt.title('TV-loss-function'), plt.semilogy(np.array(mytvlosslist))#, plt.show()
        plt.subplot(235), plt.title('Global Phase'), plt.plot(np.array(globalphaselist))#, plt.show()
        plt.subplot(236), plt.title('Global ABS'), plt.plot(np.array(globalabslist))#, plt.show()
        plt.savefig(savepath+'/myplots'+figsuffix+'.png'), plt.show()
         
        # Display RI result
        plt.figure()
        plt.subplot(231), plt.title('Result Phase: XZ'),plt.imshow(my_res[:,my_res.shape[1]//2,:]), plt.colorbar()#, plt.show()
        plt.subplot(232), plt.title('Result Phase: XZ'),plt.imshow(my_res[:,:,my_res.shape[2]//2]), plt.colorbar()#, plt.show()
        plt.subplot(233), plt.title('Result Phase: XY'),plt.imshow(my_res[my_res.shape[0]//2,:,:]), plt.colorbar()
        plt.subplot(234), plt.title('Result Abs: XZ'),plt.imshow(my_res_absorption[:,my_res.shape[1]//2,:]), plt.colorbar()#, plt.show()
        plt.subplot(235), plt.title('Result abs: XZ'),plt.imshow(my_res_absorption[:,:,my_res.shape[2]//2]), plt.colorbar()#, plt.show()
        plt.subplot(236), plt.title('Result abs: XY'),plt.imshow(my_res_absorption[my_res.shape[0]//2,:,:]), plt.colorbar()
        plt.savefig(savepath+'/RI_abs_result'+figsuffix+'.png'), plt.show()
         
        # Display recovered Pupil
        plt.figure()
        plt.subplot(131), plt.title('Po Phase'), plt.imshow(np.fft.fftshift(np.angle(sess.run(self.TF_Po_aberr)))), plt.colorbar()
        plt.subplot(132), plt.title('Po abs'), plt.imshow(np.fft.fftshift(np.abs(sess.run(self.TF_Po_aberr)))), plt.colorbar()
        plt.subplot(133), plt.bar(np.linspace(1, np.squeeze(myzernikes.shape), np.squeeze(myzernikes.shape)), myzernikes, align='center', alpha=0.5)
        plt.ylabel('Zernike Values')
        plt.title('Zernike Coefficients (Noll)')
        plt.show()
        plt.savefig(savepath+'/recovered_pupil'+figsuffix+'.png'), plt.show()

        # Eventually write H5 stacks to disc
        if(result_phaselist is not None): data.export_realdatastack_h5(savepath+'/myrefractiveindex'+figsuffix+'.h5', 'temp', np.array(result_phaselist))
        if(result_absorptionlist is not None): data.export_realdatastack_h5(savepath+'/myrefractiveindex_absorption'+figsuffix+'.h5', 'temp', np.array(result_absorptionlist))
        #myobj = my_res+1j*my_res_absorption
        #np.save('my_res_cmplx', myobj)         
                 
    def initObject():
        print('To Be done')
                    
        # ReSNET Stuff               
        # https://github.com/wenxinxu/resnet-in-tensorflow/blob/master/hyper_parameters.py
    def create_variables(self, name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
        '''
        :param name: A string. The name of the new variable
        :param shape: A list of dimensions
        :param initializer: User Xavier as default.
        :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
        layers.
        :return: The created variable
        '''
         
        ## TODO: to allow different weight decay to fully connected layer and conv layer
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.0002)
     
        new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                        regularizer=regularizer)
        return new_variables
     
     
    def batch_normalization_layer(self, input_layer, dimension):
        '''
        Helper function to do batch normalziation
        :param input_layer: 4D tensor
        :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
        :return: the 4D tensor after being normalized
        '''
        mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
        beta = tf.get_variable('beta', dimension, tf.float32,
                                   initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable('gamma', dimension, tf.float32,
                                    initializer=tf.constant_initializer(1.0, tf.float32))
        bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, 0.001)
     
        return bn_layer
 
    def bn_relu_conv_layer(self, input_layer, filter_shape, stride):
        '''
        A helper function to batch normalize, relu and conv the input layer sequentially
        :param input_layer: 4D tensor
        :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
        :param stride: stride size for conv
        :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
        '''
     
        in_channel = input_layer.get_shape().as_list()[-1]
     
        bn_layer = self.batch_normalization_layer(input_layer, in_channel)
        relu_layer = tf.nn.relu(bn_layer)
     
        filter = self.create_variables(name='conv', shape=filter_shape)
        conv_layer = tf.nn.conv3d(relu_layer, filter, strides=[1, stride, stride, 1, 1], padding='SAME')
        return conv_layer
     
    def residual_block(self, input_layer, output_channel=1, first_block=True):
        '''
        Defines a residual block in ResNet
        :param input_layer: 4D tensor
        :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
        :param first_block: if this is the first residual block of the whole network
        :return: 4D tensor.
        '''
        input_channel = input_layer.get_shape().as_list()[-1]
     
        # When it's time to "shrink" the image size, we use stride = 2
        if input_channel * 2 == output_channel:
            increase_dim = True
            stride = 2
        elif input_channel == output_channel:
            increase_dim = False
            stride = 1
        else:
            raise ValueError('Output and input channel does not match in residual blocks!!!')
     
        # The first conv layer of the first residual block does not need to be normalized and relu-ed.
        with tf.variable_scope('conv1_in_block'):
            if first_block:
                filter = self.create_variables(name='conv', shape=[3, 3, input_channel, output_channel, 1])
                conv1 = tf.nn.conv3d(input_layer, filter=filter, strides=[1, 1, 1, 1, 1], padding='SAME')
            else:
                conv1 = self.bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride)
     
        with tf.variable_scope('conv2_in_block'):
            conv2 = self.bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel, 1], 1)
     
        # When the channels of input layer and conv2 does not match, we add zero pads to increase the
        #  depth of input layers
        if increase_dim is True:
            pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='VALID')
            padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                         input_channel // 2]])
        else:
            padded_input = input_layer
     
        output = conv2 + padded_input
        return output