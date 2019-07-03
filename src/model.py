# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 09:52:05 2017
 
@author: useradmin
 
This 
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import yaml
from skimage.transform import warp, AffineTransform

# own functions
from src import tf_helper as tf_helper
from src import zernike as zern
from src import data as data
from src import tf_regularizers as tf_reg
from src import poisson_disk as pd
 
is_debug = False 

class MuScatModel(object):
    def __init__(self, my_mat_paras, is_optimization = False):
 
        ''' Create Multiple Scattering Class;
        INPUTS:
        my_mat_paras - parameter file from MATLAB containing the experimental details - has to be load previously!
        '''
        # new MATLAB v7.3 Format!
        self.params = my_mat_paras
        self.is_optimization = is_optimization
        
        self.dn = .1; # self.nImm - self.params.nEmbb
        print('Assigned some value for dn which is not good!')
         
        # add a vector of zernike factors
        self.params.Nzernikes = 11
        self.zernikefactors = np.zeros((self.params.Nzernikes,))
        self.zernikemask = np.zeros((self.params.Nzernikes,))
        # kamilov uses 420 z-planes with an overall size of 30µm; dx=72nm
         
        # refractive index immersion and embedding
        self.lambdaM = self.params.lambda0/self.params.nEmbb; # wavelength in the medium
 
    #@define_scope
    def computesys(self, obj=None, is_padding=False, is_tomo = False, mysubsamplingIC=0, is_compute_psf='BORN', is_dampic=True, is_mictype='BF'):
        # is_compute_psf: 'BORN', 'sep'
        """ This computes the FWD-graph of the Q-PHASE microscope;
        1.) Compute the physical dimensions
        2.) Compute the sampling for the waves
        3.) Create the illumination waves depending on System's properties
 
        # 'is_mictype' gives the possibility to switch between different microscope types like Brightfield, Phasecontrast etc. 
        BF - Brightfield 
        PC - Phase-Contrast (Zernike)
        DIC - Differential Interference Contrast
        DPC - Differential Phase Contrast (Oblique Illumination)
        
        
        
        
        ##### IMPORTANT! ##### 
        The ordering of the channels is as follows:
            Nillu, Nz, Nx, Ny
        """
        # define whether we want to pad the experiment 
        self.is_mictype = is_mictype
        self.is_padding = is_padding
        self.is_tomo = is_tomo
        self.mysubsamplingIC = mysubsamplingIC
        self.is_compute_psf = is_compute_psf
        self.is_dampic = is_dampic # want to damp the intensity at the edges of the illumination?
        
        if(is_padding):
            print('--------->WARNING: Padding is not yet working correctly!!!!!!!!')
            # add padding in X/Y to avoid wrap-arounds
            self.mysize_old = np.array((self.params.Nz, self.params.Nx, self.params.Ny))            
            self.params.Nx=self.params.Nx*2
            self.params.Ny=self.params.Ny*2
            self.mysize = np.array((self.params.Nz, self.params.Nx, self.params.Ny))
            self.obj = obj
            self.params.dx=self.params.dx
            self.params.dy=self.params.dy
        else:
            self.mysize=np.array((self.params.Nz, self.params.Nx, self.params.Ny))
            self.mysize_old = self.mysize
            
        # Allocate memory for the object 
        if obj is None:
            self.obj = np.zeros(self.mysize)
        else:
            self.obj = obj
            
            
        # Decide whether we wan'T to optimize or simply execute the model
        if (self.is_optimization):
            # assign training variables 
            self.tf_lambda_tv = tf.placeholder(tf.float32, [])
            self.tf_eps = tf.placeholder(tf.float32, [])
            self.tf_meas = tf.placeholder(dtype=tf.complex64, shape=self.mysize_old)
            self.tf_learningrate = tf.placeholder(tf.float32, []) 

        #else:
            # Variables of the computational graph
        #    self.TF_obj = tf.constant(np.real(self.obj), dtype=tf.float32, name='Object_const')
        #    self.TF_obj_absorption = tf.constant(np.imag(self.obj), dtype=tf.float32, name='Object_const')

        ''' Assign the object '''
        #self.TF_obj = tf.Variable(np.real(self.obj), dtype=tf.float32, name='Object_Variable')
        #self.TF_obj_absorption = tf.Variable(np.imag(self.obj), dtype=tf.float32, name='Object_Variable')
        if type(obj)==tf.Tensor:
            # In this case we use the model to let another object "pass-through"
            # obj is a tensor (probalby Variable) which holds both real and imaginary part
            self.TF_obj = tf.real(self.obj)
            self.TF_obj_absorption = tf.imag(self.obj)
        else:
            with tf.variable_scope("Complex_Object"):
                self.TF_obj = tf.get_variable('Object_Variable_Real', dtype=tf.float32, initializer=np.float32(np.real(self.obj)))
                self.TF_obj_absorption = tf.get_variable('Object_Variable_Imag', dtype=tf.float32, initializer=np.float32(np.imag(self.obj)))
                #set reuse flag to True
                tf.get_variable_scope().reuse_variables()
                #just an assertion!
                assert tf.get_variable_scope().reuse==True             
                
        ## Establish normalized coordinates.
        #-----------------------------------------
        vxx = tf_helper.xx((self.mysize[1], self.mysize[2]),'freq') * self.lambdaM * self.params.nEmbb / (self.params.dx * self.params.NAo);    # normalized optical coordinates in X
        vyy = tf_helper.yy((self.mysize[1], self.mysize[2]),'freq') * self.lambdaM * self.params.nEmbb / (self.params.dy * self.params.NAo);    # normalized optical coordinates in Y
         
        # AbbeLimit=lambda0/NAo;  # Rainer's Method
        # RelFreq = rr(mysize,'freq')*AbbeLimit/dx;  # Is not generally right (dx and dy)
        self.RelFreq = np.sqrt(tf_helper.abssqr(vxx) + tf_helper.abssqr(vyy));    # spanns the frequency grid of normalized pupil coordinates
        self.Po=np.complex128(self.RelFreq < 1.0);   # Create the pupil of the objective lens        
        
        # Prepare the normalized spatial-frequency grid.
        self.S = self.params.NAc/self.params.NAo;   # Coherence factor
 
        # Precomputing the first 9 zernike coefficients 
        self.params.Nzernikes = np.squeeze(self.zernikefactors.shape)
        self.myzernikes = np.zeros((self.Po.shape[0],self.Po.shape[1],self.params.Nzernikes))+ 1j*np.zeros((self.Po.shape[0],self.Po.shape[1],self.params.Nzernikes))
        r, theta = zern.cart2pol(vxx, vyy)        
        for i in range(0,self.params.Nzernikes):
            self.myzernikes[:,:,i] = np.fft.fftshift(zern.zernike(r, theta, i+1, norm=False)) # or 8 in X-direction
             
        # eventually introduce a phase factor to approximate the experimental data better
        print('----------> Be aware: We are taking aberrations into account!')
        # Assuming: System has coma along X-direction
        self.myaberration = np.sum(self.zernikefactors * self.myzernikes, axis=2)
        self.Po *= np.exp(1j*self.myaberration)
        
        # do individual pupil functoins according to is_mictype
        if(self.is_mictype=='BF' or self.is_mictype=='DF'):
            # Brightfield/Generic case
            self.Po = self.Po
        elif(self.is_mictype=='PC'):
            # Anullar Phase-plate with generic absorption
            print('We are taking a phase-ring in the Pupil plane into account!')
            p_o = (self.params.NAc*.9)/self.params.NAo;   # Coherence factor
            p_i = (self.params.NAci*1.1)/self.params.NAo;   # Coherence factor
            self.myphaseplate = (1.*(self.RelFreq < p_o) * 1.*(self.RelFreq > p_i))>0 # Create the pupil of the condenser plane
            self.Po *= np.exp(1j*np.pi/2*self.myphaseplate)
        elif(self.is_mictype=='DIC'):
            # DIC Phase mask from: https://github.com/mattersoflight/microlith/blob/master/%40microlith/computesys.m
            shearangle = 45
            shear = .48/2
            bias = 25
            freqGridShear=vxx*np.cos(shearangle*np.pi/180)+vyy*np.sin(shearangle*np.pi/180);
            halfshear=shear/2;
            halfbias=0.5*bias*np.pi/180;
            self.Po *= 1j*np.sin(2*np.pi*freqGridShear*halfshear-halfbias)
                    
        self.Po = np.fft.fftshift(self.Po)# Need to shift it before using as a low-pass filter    Po=np.ones((np.shape(Po)))
        # do individual illumination sources according to is_mictype
        if(self.is_mictype=='BF' or self.is_mictype == 'DIC'):
            # Brightfield/Generic case
            self.Ic = self.RelFreq <= self.S
        elif(self.is_mictype=='PC' or self.is_mictype=='DF'):
        #if hasattr(self, 'NAci'):
            # Anullar illumination e.g. for PC or DF 
            if self.params.NAci == None or self.params.NAci < 0:
                self.params.NAci = self.params.NAc - .1
                print('I detected a darkfield illumination aperture, but value was not set! ')
            self.S_o = self.params.NAc/self.params.NAo;   # Coherence factor
            self.S_i = self.params.NAci/self.params.NAo;   # Coherence factor
            self.Ic = (1.*(self.RelFreq < self.S_o) * 1.*(self.RelFreq > self.S_i))>0 # Create the pupil of the condenser plane
     
        # weigh the illumination source with some cos^2 intensity weight?!
        if(0):
            myIntensityFactor = 70
            self.Ic_map = np.cos((myIntensityFactor *tf_helper.xx((self.params.Nx, self.params.Ny), mode='freq')**2+myIntensityFactor *tf_helper.yy((self.params.Nx, self.params.Ny), mode='freq')**2))**2
            print('We are taking the cosine illuminatino shape!')
           
        if(self.is_dampic>0):
            print('We are taking the gaussian illuminatino shape!')
            myIntensityFactor = self.is_dampic
            self.Ic_map = np.exp(-tf_helper.rr((self.params.Nx, self.params.Ny),mode='freq')**2/myIntensityFactor)
        else:
            print('We are not weighing our illumination!')
            self.Ic_map = np.ones((self.params.Nx, self.params.Ny))
            

        
        # This is experimental
        if(self.mysubsamplingIC>0):
            self.checkerboard = np.zeros((self.mysubsamplingIC,self.mysubsamplingIC))# ((1,0),(0,0))  # testing for sparse illumination?!
            self.checkerboard[0,0] = 1
            print('-------> ATTENTION: WE have a CHECKeRBOArD  MASK IN THE PUPIL PLANE!!!!')
            #self.checkerboard = np.matlib.repmat(self.checkerboard,self.Ic_map.shape[0]//self.mysubsamplingIC+1,self.Ic_map.shape[1]//self.mysubsamplingIC+1)
            #self.checkerboard = self.checkerboard[0:self.Ic_map.shape[0], 0:self.Ic_map.shape[1]]
            
       
            print('Create a new random disk - This can take some time!')
            self.checkerboard = np.fft.fftshift(self.Po)*pd.get_poisson_disk(self.Ic.shape[0],self.Ic.shape[1],self.mysubsamplingIC)
            print('Done!')
        else:
            self.checkerboard = np.ones(self.Ic.shape)
        
        self.Ic = self.Ic * self.Ic_map  # weight the intensity in the condenser aperture, unlikely to be uniform
        # print('--------> ATTENTION! - We are not weighing the Intensity int the illu-pupil!')

 
        # Shift the pupil in X-direction (optical missalignment)
        if hasattr(self, 'shiftIcX') and self.is_compute_psf is None:
            if self.params.shiftIcX != None:
                if(is_padding): self.params.shiftIcX=self.params.shiftIcX*2
                print('Shifting the illumination in X by: ' + str(self.params.shiftIcX) + ' Pixel')
                if(0):
                    self.Ic = np.roll(self.Ic, self.params.shiftIcX, axis=1)
                elif(1):
                    tform = AffineTransform(scale=(1, 1), rotation=0, shear=0, translation=(self.params.shiftIcX, 0))
                    self.Ic = warp(self.Ic, tform.inverse, output_shape=self.Ic.shape)
                elif(0):
                    # We apply a phase-factor to shift the source in realspace - so make it trainable
                    self.shift_xx = tf_helper.xx((self.mysize[1], self.mysize[2]),'freq')
                    self.Ic = np.abs(np.fft.ifft2(np.fft.fft2(self.Ic)*np.exp(1j*2*np.pi*self.shift_xx*self.params.shiftIcX))) 

        # Shift the pupil in Y-direction (optical missalignment)
        if hasattr(self, 'shiftIcY') and self.is_compute_psf is None:
            if self.params.shiftIcY != None:
                if(is_padding): self.params.shiftIcY=self.params.shiftIcY*2
                print('Shifting the illumination in Y by: ' + str(self.params.shiftIcY) + ' Pixel')
                if(0):
                    self.Ic = np.roll(self.Ic, self.params.shiftIcY, axis=0)
                elif(1):
                    # Rainer suggests to normalize everyhing to be within range of one - so why not using NA coordinates?
                    self.TF_xx = tf_helper.xx((self.mysize[1],self.mysize[2])) * 2 * np.pi * self.TF_shiftIcX /( self.lambdaM*self.params.nEmbb/self.params.dx)
                    self.TF_yy = tf_helper.yy((self.mysize[1],self.mysize[2])) * 2 * np.pi * self.TF_shiftIcY /( self.lambdaM*self.params.nEmbb/self.params.dy)
                
                
                    tform = AffineTransform(scale=(1, 1), rotation=0, shear=0, translation=(0, self.params.shiftIcY))
                    self.Ic = warp(self.Ic, tform.inverse, output_shape=self.Ic.shape)
                elif(0):
                    # We apply a phase-factor to shift the source in realspace - so make it trainable
                    self.shift_yy = tf_helper.yy((self.mysize[1], self.mysize[2]),'freq')
                    self.Ic = np.abs(np.fft.ifft2(np.fft.fft2(self.Ic)*np.exp(1j*self.shift_yy*self.params.shiftIcY))) 

        # Reduce the number of illumination source point by filtering it with the poissson random disk or alike
        self.Ic = self.Ic * self.checkerboard
        
        
        ## Forward propagator  (Ewald sphere based) DO NOT USE NORMALIZED COORDINATES HERE
        self.kxysqr= (tf_helper.abssqr(tf_helper.xx((self.mysize[1], self.mysize[2]), 'freq') / self.params.dx) + 
                      tf_helper.abssqr(tf_helper.yy((self.mysize[1], self.mysize[2]), 'freq') / self.params.dy)) + 0j;
        self.k0=1/self.lambdaM;
        self.kzsqr= tf_helper.abssqr(self.k0) - self.kxysqr;
        self.kz=np.sqrt(self.kzsqr);
        self.kz[self.kzsqr < 0]=0;
        self.dphi = 2*np.pi*self.kz*self.params.dz;  # exp(1i*kz*dz) would be the propagator for one slice

        self.Nc=np.sum(self.Ic>0); 
        print('Number of Illumination Angles / Plane waves: '+str(self.Nc))
        
        if not self.is_compute_psf=='BORN' or is_compute_psf=='sep':
        
            ## Get a list of vector coordinates corresponding to the pixels in the mask
            xfreq= tf_helper.xx((self.mysize[1], self.mysize[2]),'freq');
            yfreq= tf_helper.yy((self.mysize[1], self.mysize[2]),'freq');

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
            self.Alldphi = -(np.reshape(np.arange(0, self.mysize[0], 1), [1, 1, self.mysize[0]]))*np.repeat(np.fft.fftshift(self.dphi)[:, :, np.newaxis], self.mysize[0], axis=2)
              
            # Ordinary backpropagation. This is NOT what we are interested in:
            self.myAllSlicePropagator = np.transpose(np.exp(1j*self.Alldphi) * (np.repeat(np.fft.fftshift(self.dphi)[:, :, np.newaxis], self.mysize[0], axis=2) >0), [2, 0, 1]);  # Propagates a single end result backwards to all slices

            ## propagate the field through the entire object for all angles simultaneously
            self.A_prop = np.transpose(self.A_input,[3, 0, 1, 2])  # ??????? what the hack is happening with transpose?!

            # Precalculate the oblique effect on OPD to speed it up
            print('--------> ATTENTION: I added no pi factor - is this correct?!')
            self.RefrEffect = np.reshape(np.squeeze(1j * self.params.dz * self.k0 * self.RefrCos), [self.Nc, 1, 1]) # pi-factor
            
            myprop = np.exp(1j * self.dphi) * (self.dphi > 0);  # excludes the near field components in each step
            myprop = tf_helper.repmat4d(np.fft.fftshift(myprop), self.Nc)
            self.myprop = np.transpose(myprop, [3, 0, 1, 2])  # ??????? what the hack is happening with transpose?!
            

            self.myAllSlicePropagator_psf = 0*self.myAllSlicePropagator # dummy variable to make the algorithm happy
        
        if is_compute_psf=='sep':
            self.Alldphi_psf = -(np.reshape(np.arange(0, self.mysize[0], 1), [1, 1, self.mysize[0]])-self.params.Nz/2)*np.repeat(np.fft.fftshift(self.dphi)[:, :, np.newaxis], self.mysize[0], axis=2)
            self.myAllSlicePropagator_psf = np.transpose(np.exp(1j*self.Alldphi_psf) * (np.repeat(np.fft.fftshift(self.dphi)[:, :, np.newaxis], self.mysize[0], axis=2) >0), [2, 0, 1]);  # Propagates a single end result backwards to all slices
            #self.myAllSlicePropagator_psf = np.transpose(np.exp(1j*self.Alldphi_psf) * (np.repeat(self.dphi[:, :, np.newaxis], self.mysize[0], axis=2) >0), [2, 0, 1]);  # Propagates a single end result backwards to all slices
            
            
        if (self.is_compute_psf=='BORN'):
            self.Alldphi_psf = -(np.reshape(np.arange(0, self.mysize[0], 1), [1, 1, self.mysize[0]])-self.params.Nz/2)*np.repeat(self.dphi[:, :, np.newaxis], self.mysize[0], axis=2)
            self.myAllSlicePropagator_psf = np.transpose(np.exp(-1j*self.Alldphi_psf) * (np.repeat(self.dphi[:, :, np.newaxis], self.mysize[0], axis=2) >0), [2, 0, 1]);  # Propagates a single end result backwards to all slices
            self.myAllSlicePropagator = 0*self.myAllSlicePropagator_psf # dummy variable to make the algorithm happy
            self.A_prop = 0*self.Alldphi_psf # dummy variable to make the algorithm happy
            self.RefrEffect = 0 # dummy variable to make the algorithm happy
            self.myprop = 0*self.Alldphi_psf # dummy variable to make the algorithm happy
            
            
    #@define_scope
    def computemodel(self, is_forcepos=False):
        ''' Perform Multiple Scattering here
        1.) Multiple Scattering is perfomed by slice-wise propagation the E-Field throught the sample
        2.) Each Field has to be backprojected to the BFP
        3.) Last step is to ceate a focus-stack and sum over all angles
 
        This is done for all illumination angles (coming from illumination NA
        simultaneasly)'''
        self.is_forcepos = is_forcepos
 
        print("Buildup Q-PHASE Model ")
        ###### make sure, that the first dimension is "batch"-size; in this case it is the illumination number
        # @beniroquai It's common to have to batch dimensions first and not last.!!!!!
        # the following loop propagates the field sequentially to all different Z-planes
 
         
        if(self.is_tomo):
            print('Experimentally using the tomographic scheme!')
            self.A_prop = np.conj(self.A_prop)
             
         
        # for now orientate the dimensions as (alpha_illu, x, y, z) - because tensorflow takes the first dimension as batch size
        with tf.name_scope('Variable_assignment_general'):
            self.TF_Ic = tf.cast(tf.constant(self.Ic), tf.complex64)
            self.TF_Ic_shift = self.TF_Ic # make the algorithm happy
            self.TF_Po = tf.cast(tf.constant(self.Po), tf.complex64)
            if(is_debug): self.TF_Po = tf.Print(self.TF_Po, [], 'Casting TF_Po')    
            
            self.TF_Zernikes = tf.convert_to_tensor(self.myzernikes, np.float32)
            self.TF_dphi = tf.cast(tf.constant(self.dphi), tf.complex64)
            
            #tf.constant(self.myzernikes, dtype=tf.float32) # TODO: HERE A CONVERSION IS HAPPENING complex to float?!
            # TODO: The following operation takes super long - WHY?
            self.TF_myAllSlicePropagator_psf =  tf.cast(tf.complex(np.float32(np.real(self.myAllSlicePropagator_psf)),np.float32(np.imag(self.myAllSlicePropagator_psf))), dtype=np.complex64)
            
                    
            # Only update those Factors which are really necesarry (e.g. Defocus is not very likely!)
            self.TF_zernikefactors = tf.Variable(self.zernikefactors, dtype = tf.float32, name='var_zernikes')
            self.TF_shiftIcX = tf.Variable(self.params.shiftIcX, dtype=tf.float32, name='tf_shiftIcX')
            self.TF_shiftIcY = tf.Variable(self.params.shiftIcY, dtype=tf.float32, name='tf_shiftIcY')        
            

            #indexes = tf.constant([[4], [5], [6], [7], [8], [9]])
            indexes = tf.cast(tf.where(tf.constant(self.zernikemask)>0), tf.int32)
            updates = tf.gather_nd(self.TF_zernikefactors,indexes)
            # Take slice
            # Build tensor with "filtered" gradient
            part_X = tf.scatter_nd(indexes, updates, tf.shape(self.TF_zernikefactors))
            self.TF_zernikefactors_filtered = part_X + tf.stop_gradient(-part_X + self.TF_zernikefactors)
          
        if self.is_compute_psf is 'BPM':
            with tf.name_scope('Variable_assignment_BPM'):
                
                # Define slice-wise propagator (i.e. Fresnel kernel)
                self.TF_myprop = tf.cast(tf.complex(np.real(np.squeeze(self.myprop)),np.imag(np.squeeze(self.myprop))), dtype=tf.complex64)
                # A propagator for all slices (2D->3D)
                self.TF_myAllSlicePropagator = tf.cast(tf.complex(np.real(self.myAllSlicePropagator), np.imag(self.myAllSlicePropagator)), tf.complex64)
                
                # This corresponds to the input illumination modes
                self.TF_A_input =  tf.cast(tf.complex(np.real(self.A_prop),np.imag(self.A_prop)), dtype=tf.complex64)
                self.TF_RefrEffect = tf.constant(self.RefrEffect, dtype=tf.complex64)
                
                if(is_debug): self.TF_A_input = tf.Print(self.TF_A_input, [], 'Casting TF_A_input')     

                # make illuminatino decentering "learnable"
                self.TF_xx = tf_helper.xx((self.mysize[1],self.mysize[2])) * 2 * np.pi * self.TF_shiftIcX /( self.lambdaM*self.params.nEmbb/self.params.dx)
                self.TF_yy = tf_helper.yy((self.mysize[1],self.mysize[2])) * 2 * np.pi * self.TF_shiftIcY /( self.lambdaM*self.params.nEmbb/self.params.dy)
                
                TF_icshift = tf.exp(1j*tf.cast((self.TF_xx+self.TF_yy), tf.complex64))
                self.TF_A_input  *= tf.expand_dims(TF_icshift,-1) # add global phase ramp in x/y  to decenter the source
                
                # TODO: Introduce the averraged RI along Z - MWeigert
                self.TF_A_prop = tf.squeeze(self.TF_A_input);
                self.U_z_list = []
                
        # Initiliaze memory
        self.allSumAmp = 0
        
        # compute multiple scattering only in higher Born order        
        if self.is_compute_psf is 'BPM':
            self.propslices()
        
        # in a final step limit this to the detection NA:
        self.TF_Po_aberr = tf.exp(1j*tf.cast(tf.reduce_sum(self.TF_zernikefactors_filtered*self.TF_Zernikes, axis=2), tf.complex64)) * self.TF_Po
        if self.is_compute_psf=='BPM':
            # The input field of the PSF calculation is the BFP of the objective lens
            self.TF_A_prop = tf.ifft2d(tf.fft2d(self.TF_A_prop)*self.TF_Po_aberr)
            
        # Experimenting with pseudo tomographic data? No backpropgation necessary!
        if self.is_tomo:
            return self.computetomo()         
                
        if self.is_compute_psf is 'BPM':
            # now backpropagate the multiple scattered slice to a focus stack - i.e. convolve with PTF
            self.TF_allSumAmp = self.propangles(self.TF_A_prop)
            self.TF_allSumAmp = self.TF_allSumAmp*tf.exp(tf.cast(-1j*np.pi/2, tf.complex64)) # align it with the born model
        else:
            if self.is_compute_psf=='BORN': 
                self.TF_ASF, self.TF_ATF = self.computepsf()
            elif self.is_compute_psf=='sep': #
                self.TF_ASF, self.TF_ATF = self.computepsf_sepang()

            # ASsign dummy variable- not used
            self.TF_allSumAmp = None 
            
            
        # negate padding        
        if self.is_padding:
            self.TF_allSumAmp = self.TF_allSumAmp[:,self.params.Nx//2-self.params.Nx//4:self.params.Nx//2+self.params.Nx//4, self.params.Ny//2-self.params.Ny//4:self.params.Ny//2+self.params.Ny//4]
        
        return self.TF_allSumAmp

    
    def compute_bpm(self, obj, is_padding=False, mysubsamplingIC=0, is_dampic=True):
        # this funciton is a wrapper for the bpm forward model
        
        if obj is None:
            # Compute System Function
            self.computesys(obj=None, is_padding=is_padding, mysubsamplingIC=mysubsamplingIC, is_compute_psf='BPM', is_dampic=is_dampic)
        
        else:
            # We need to subtract the background as BPM assumes dn as per-slice refractive index
            obj = obj - self.params.nEmbb
            
            # Compute System Function
            self.computesys(obj, is_padding=is_padding, mysubsamplingIC=mysubsamplingIC, is_compute_psf='BPM', is_dampic=is_dampic)
        
        # Compute Model
        return self.computemodel()

    
    def compute_born(self, obj, is_padding=False, mysubsamplingIC=0, is_precompute_psf=True, is_dampic=True):
        # This function is a wrapper to compute the Born fwd-model (convolution)
        self.computesys(obj, is_padding=False, mysubsamplingIC=mysubsamplingIC, is_compute_psf='BORN',is_dampic=is_dampic)
        
        # Create Model Instance
        self.computemodel()
        
        # PreCompute the ATF
        if is_precompute_psf:
            # Start a temporary session
            sess = tf.Session()#config=tf.ConfigProto(log_device_placement=True))
            sess.run(tf.global_variables_initializer())

            print('We are now precomputing the ASF')
            self.myATF = sess.run(self.TF_ATF)
            self.myASF = sess.run(self.TF_ASF)
            TF_ASF = self.myASF
        else:
            # for optimizing the ASF during training
            TF_ASF = self.TF_ASF

        # Define Fwd operator
        return self.computeconvolution(TF_ASF=TF_ASF, is_padding=is_padding)
    
        
    
    def computepsf_sepang(self):
        # here we only want to gather the partially coherent transfer function PTF
        # which is the correlation of the PTF_c and PTF_i 
        # we compute it as the slice propagation of the pupil function
        
        # 1. + 2.) Define the input field as the BFP and effective pupil plane of the condenser 
        # Compute the ASF for the Condenser and Imaging Pupil
        
        # in case we only want to compute the PSF, we don't need the input field
        self.TF_myAllSlicePropagator = self.myAllSlicePropagator_psf
        TF_A_prop =  tf_helper.my_ift2d(tf.ones_like(self.TF_A_prop)*tf_helper.ifftshift2d(self.TF_Po * self.TF_Po_aberr)) # propagate in real-space->fftshift!; tf_ones: need for broadcasting!
        self.TF_allSumAmp = self.propangles(TF_A_prop)
        
        TF_ASF = self.TF_allSumAmp
        TF_ASF = TF_ASF/tf.cast(tf.sqrt(tf.reduce_sum(tf_helper.tf_abssqr(TF_ASF))), tf.complex64)
        TF_ATF = tf_helper.my_ft3d(TF_ASF)#/tf.cast(tf.sqrt(1.*np.prod(self.mysize)), tf.complex64)
        
        tf_global_phase = tf.angle(TF_ATF[self.mysize[0]//2,self.mysize[1]//2,self.mysize[2]//2])#tf.angle(self.TF_allAmp_3dft[0, self.mid3D[2], self.mid3D[1], self.mid3D[0]])
        tf_global_phase = tf.cast(tf_global_phase, tf.complex64)
        TF_ATF = TF_ATF * tf.exp(1j * tf_global_phase);  # normalize ATF
          
        return TF_ASF, TF_ATF 



    def computepsf_working(self):

        # here we only want to gather the partially coherent transfer function PTF
        # which is the correlation of the PTF_c and PTF_i 
        # we compute it as the slice propagation of the pupil function

        # 1. + 2.) Define the input field as the BFP and effective pupil plane of the condenser 
        # Compute the ASF for the Condenser and Imaging Pupil
        # Rainer suggests to normalize everyhing to be within range of one - so why not using NA coordinates?
        self.TF_xx = tf_helper.xx((self.mysize[1],self.mysize[2])) * 2 * np.pi * self.TF_shiftIcX /( self.lambdaM*self.params.nEmbb/self.params.dx)
        self.TF_yy = tf_helper.yy((self.mysize[1],self.mysize[2])) * 2 * np.pi * self.TF_shiftIcY /( self.lambdaM*self.params.nEmbb/self.params.dy)
    
        TF_icshift = tf.exp(1j*tf.cast((self.TF_xx+self.TF_yy), tf.complex64))
        self.TF_Ic_shift = tf_helper.my_ift2d(tf_helper.my_ft2d(self.TF_Ic)*TF_icshift)

        
        TF_ASF_po = tf_helper.my_ift2d(self.TF_myAllSlicePropagator_psf * tf_helper.fftshift2d(self.TF_Po_aberr*1j))#* self.TF_Po_aberr))
        TF_ASF_ic = tf_helper.my_ift2d(self.TF_myAllSlicePropagator_psf * self.TF_Ic_shift * 1j)#self.TF_Ic*1j)
        
        # does not make any sense, but this way at least roughly same values compared to BPM appear
        #TF_ASF_ic = TF_ASF_ic/tf.complex(tf.sqrt(tf.reduce_sum(tf_helper.tf_abssqr(TF_ASF_ic))),0.) 

        # 3.) correlation of the pupil function to get the APTF
        TF_ASF = tf.conj(TF_ASF_po)*TF_ASF_ic #tf_helper.my_ift3d(TF_ATF_po,myfftfac)*tf.conj(tf_helper.my_ift3d(TF_ATF_ic,myfftfac))

        # I wish I could use this here - but not a good idea!
        #self.normfac = tf.sqrt(tf.reduce_sum(tf.abs(TF_ASF[self.mysize[0]//2,:,:])))
        #self.normfac = tf.sqrt(tf.reduce_sum(tf.abs(TF_ASF)))
        #self.normfac = tf.sqrt(tf.reduce_max(tf.abs(TF_ASF)))
        #self.normfac = tf.sqrt(tf.reduce_sum(tf.abs(TF_ASF),0))
        
        self.normfac = 1.
        TF_ASF = TF_ASF/tf.complex(self.normfac,0.) # TODO: norm Tensorflow?! 

        # 4.) precompute ATF - just im case
        TF_ATF = tf_helper.my_ft3d(TF_ASF)
        #normfac = tf.sqrt(tf.reduce_sum(tf_helper.tf_abssqr(TF_ATF[self.mysize[0]//2,:,:])))
        return TF_ASF, TF_ATF 
    
    
    def computepsf(self):

        # here we only want to gather the partially coherent transfer function PTF
        # which is the correlation of the PTF_c and PTF_i 
        # we compute it as the slice propagation of the pupil function

        # 1. + 2.) Define the input field as the BFP and effective pupil plane of the condenser 
        # Compute the ASF for the Condenser and Imaging Pupil
        # Rainer suggests to normalize everyhing to be within range of one - so why not using NA coordinates?
        self.TF_xx = tf_helper.xx((self.mysize[1],self.mysize[2])) * 2 * np.pi * self.TF_shiftIcX /( self.lambdaM*self.params.nEmbb/self.params.dx)
        self.TF_yy = tf_helper.yy((self.mysize[1],self.mysize[2])) * 2 * np.pi * self.TF_shiftIcY /( self.lambdaM*self.params.nEmbb/self.params.dy)
    
        TF_icshift = tf.exp(1j*tf.cast((self.TF_xx+self.TF_yy), tf.complex64))
        self.TF_Ic_shift = tf_helper.my_ift2d(tf_helper.my_ft2d(self.TF_Ic)*TF_icshift)

        
        h_det = tf_helper.my_ift2d(self.TF_myAllSlicePropagator_psf * tf_helper.fftshift2d(self.TF_Po_aberr*1j))#* self.TF_Po_aberr))
        h_illu = tf_helper.my_ift2d(self.TF_myAllSlicePropagator_psf * self.TF_Ic_shift * 1j)#self.TF_Ic*1j)
        
        # does not make any sense, but this way at least roughly same values compared to BPM appear
        h_res = tf.conj(h_det) * h_illu 

        # 3.) correlation of the pupil function to get the APTF
        dfx = 1/self.mysize[1]/self.params.dx
        dfy = 1/self.mysize[2]/self.params.dy
        
        TF_ATF = (2.0*tf_helper.my_ft2d(1.0j*h_res*dfx*dfy))
        ax = 2
        TF_ATF = tf.transpose(TF_ATF, [1,2,0])
        TF_ATF = tf.spectral.fft(TF_ATF)
        TF_ATF = tf.transpose(TF_ATF, [2,0,1])*self.params.dz
        
        total_source = tf.reduce_sum(self.TF_Ic_shift*self.TF_Po_aberr*tf.conj(self.TF_Po_aberr))*dfx*dfy
        TF_ATF = TF_ATF/total_source
        TF_ASF = tf_helper.my_ift3d(TF_ATF)
        
        
        # 4.) precompute ATF - just im case
        
        #normfac = tf.sqrt(tf.reduce_sum(tf_helper.tf_abssqr(TF_ATF[self.mysize[0]//2,:,:])))
        return TF_ASF, TF_ATF 

    def computeconvolution(self, TF_ASF=None, is_padding=False, border_region=(10,10,10)):
        # We want to compute the born-fwd model
        # TF_ATF - is the tensorflow node holding the ATF - alternatively use numpy arry!
        print('Computing the fwd model in born approximation')

        # compute the scattering potential according to 1st Born
        self.TF_nr = tf.complex(self.TF_obj, self.TF_obj_absorption)
        self.TF_no = tf.cast(self.params.nEmbb+0j, tf.complex64)
        k02 = (2*np.pi*self.params.nEmbb/self.params.lambda0)**2
        self.TF_V = (k02/(4*np.pi))*(self.TF_nr**2-self.TF_no**2)

        # We need to have a placeholder because the ATF is computed afterwards...
        if (TF_ASF is None):
            # Here we allow sideloading from 3rd party sources
            self.TF_ASF_placeholder = tf.placeholder(dtype=tf.complex64, shape=self.mysize, name='TF_ASF_placeholder')
        elif type(TF_ASF) is np.ndarray:
            # Here we load a precomputed PSF (e.g. coming from numpy)
            self.TF_ASF_placeholder = tf.cast(tf.constant(TF_ASF, name='TF_ASF_placeholder'), tf.complex64)
        else:
            # Here we leave the ASF open to be optimized - it's a pass-through
            self.TF_ASF_placeholder = tf.cast(TF_ASF, tf.complex64)
        
        # convolve object with ASF
        if is_padding==True or is_padding=='pad':
            # to avoid wrap-around artifacts
            self.TF_V_tmp = tf_helper.extract(self.TF_V, self.mysize*2)
            self.TF_ASF_placeholder_tmp = tf_helper.extract(self.TF_ASF_placeholder, self.mysize*2)
            TF_res = tf_helper.my_ift3d(tf_helper.my_ft3d(self.TF_V_tmp)*tf_helper.my_ft3d(self.TF_ASF_placeholder_tmp))
            TF_res = tf_helper.extract(TF_res, self.mysize)
        elif is_padding=='border':
            self.TF_ASF_placeholder_tmp = tf_helper.extract(self.TF_ASF_placeholder, self.mysize+2*border_region)
            TF_res = tf_helper.my_ift3d(tf_helper.my_ft3d(self.TF_V)*tf_helper.my_ft3d(self.TF_ASF_placeholder_tmp))
            TF_res = tf_helper.extract(TF_res, self.mysize)
        else:
            TF_res = tf_helper.my_ift3d(tf_helper.my_ft3d(self.TF_V)*tf_helper.my_ft3d(self.TF_ASF_placeholder))
            

            
            
        print('ATTENTION: WEIRD MAGIC NUMBER for background field!!')
        #return tf.squeeze(TF_res+(myfac-1j*myfac))/np.sqrt(2)
        return tf.squeeze(TF_res)#-1j) # add the background
           
    
    def computedeconv(self, ain, alpha = 5e-2):
        # thinkonov regularized deconvolution 
        self.TF_alpha = tf.placeholder_with_default(alpha, shape=[])
        return tf_helper.my_ift3d((tf.conj(self.TF_ATF)*tf_helper.my_ft3d(ain))/(tf.complex(tf.abs(self.TF_ATF)**2+self.TF_alpha,0.)))
    
    
    def propslices(self):
        ''' Here we do the multiple scattering e.g. the forward propagation of 
        all light modes through the 3D sample. It's basically the split-step 
        fourier method (SSFM) beam propagation method (BPM). The last slice 
        holds all scattering of the sample which can be used to generate a 
        focus stack'''
        
        TF_real_3D = self.TF_obj
        TF_imag_3D = self.TF_obj_absorption     
        
        # wrapper for force-positivity on the RI-instead of penalizing it
        if(self.is_forcepos):
            print('----> ATTENTION: We add the PreMonotonicPos' )
            TF_real_3D = tf_reg.PreMonotonicPos(TF_real_3D)
            TF_imag_3D = tf_reg.PreMonotonicPos(TF_imag_3D)

        # simulate multiple scattering through object
        with tf.name_scope('Fwd_Propagate'):
            #print('---------ATTENTION: We are inverting the RI!')
            for pz in range(0, self.mysize[0]):
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
                    if(is_debug): self.TF_A_prop = tf.Print(self.TF_A_prop, [], 'Performing Slice Propagation')     

 

        
    def propangles(self, TF_A_prop):
        ''' Here we generate a focus stack for all illumination modes independtly. 
        it follows the routine "propslices". '''
        # TF_A_prop - Input field to propagate
        TF_allSumAmp = tf.zeros([self.mysize[0], self.params.Nx, self.params.Ny], dtype=tf.complex64)
        
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
                        OneAmp = tf.expand_dims(TF_A_prop[pillu, :, :], 0)
                        # Fancy backpropagation assuming what would be measured if the sample was moved under oblique illumination:
                        # The trick is: First use conceptually the normal way
                        # and then apply the XYZ shift using the Fourier shift theorem (corresponds to physically shifting the object volume, scattered field stays the same):
                        self.TF_AdjustKXY = tf.squeeze(tf.conj(self.TF_A_input[pillu,:,:,])) # Maybe a bit of a dirty hack, but we first need to shift the zero coordinate to the center
                        self.TF_AdjustKZ = tf.cast(tf.transpose(np.exp(
                            2 * np.pi * 1j * self.params.dz * np.reshape(np.arange(0, self.mysize[0], 1), # We want to start from first Z-slice then go to last which faces the objective lens
                                  [1, 1, self.mysize[0]]) * self.kzcoord[:, :, :,pillu]), [2, 1, 0]), tf.complex64)
                        self.TF_allAmp = tf.ifft2d(tf.fft2d(OneAmp) * self.TF_myAllSlicePropagator) * self.TF_AdjustKZ * self.TF_AdjustKXY  # * (TF_AdjustKZ);  # 2x bfxfun.  Propagates a single amplitude pattern back to the whole stack
                        #tf_global_phase = tf.cast(tf.angle(self.TF_allAmp[self.mid3D[0],self.mid3D[1],self.mid3D[2]]), tf.complex64)
                        #tf_global_phase = tf.cast(np.random.randn(1)*np.pi,tf.complex64)
                        #self.TF_allAmp = self.TF_allAmp * tf.exp(1j*tf_global_phase) # Global Phases need to be adjusted at this step!  Use the zero frequency
                        
                        
                    if(1):
                        with tf.name_scope('Propagate'):
                            self.TF_allAmp_3dft = tf.fft3d(tf.expand_dims(self.TF_allAmp, axis=0))
                            #tf_global_phase = tf.angle(self.TF_allAmp_3dft[0,0,0,0])#tf.angle(self.TF_allAmp_3dft[0, self.mid3D[2], self.mid3D[1], self.mid3D[0]])
                            tf_global_phase = tf_helper.angle(self.TF_allAmp_3dft[0,0,0,0])#tf.angle(self.TF_allAmp_3dft[0, self.mid3D[2], self.mid3D[1], self.mid3D[0]])
                            tf_global_phase = tf.cast(tf_global_phase, tf.complex64)

                            self.TF_allAmp = self.TF_allAmp * tf.exp(-1j * tf_global_phase);  # Global Phases need to be adjusted at this step!  Use the zero frequency
                            #print('Global phase: '+str(tf.exp(1j*tf.cast(tf.angle(self.TF_allAmp[self.mid3D[0],self.mid3D[1],self.mid3D[2]]), tf.complex64).eval()))
                    else:
                        print('ATTENTION: WE are not using the global phase factor - angle is not implemented on GPU!')
 
                    with tf.name_scope('Sum_Amps'): # Normalize amplitude by condenser intensity
                        TF_allSumAmp = TF_allSumAmp + self.TF_allAmp #/ self.intensityweights[pillu];  # Superpose the Amplitudes
                        if(is_debug): TF_allSumAmp = tf.Print(TF_allSumAmp, [], 'Performing Angle Propagation')     
  
                    # print('Current illumination angle # is: ' + str(pillu))
 
 
            # Normalize the image such that the values do not depend on the fineness of
            # the source grid.
            #TF_allSumAmp = TF_allSumAmp/tf.cast(np.sum(self.Ic), tf.complex64) # tf.cast(tf.reduce_max(tf.abs(self.TF_allSumAmp)), tf.complex64) # self.Nc #/

            # Normalize along Z to account for energy conservation
            #TF_mynorm = tf.cast(tf.sqrt(tf_helper.tf_abssqr(tf.reduce_sum(TF_allSumAmp , axis=(1,2))))/np.prod(self.mysize[1:3]),tf.complex64)
            TF_mynorm = tf.cast(tf.sqrt(tf_helper.tf_abssqr(tf.reduce_mean(TF_allSumAmp , axis=(1,2)))),tf.complex64)
            TF_mynorm = tf.expand_dims(tf.expand_dims(TF_mynorm,1),1)
            TF_allSumAmp = TF_allSumAmp/TF_mynorm;
            print('BPM Normalization accounts for ENERGY conservation!!')

            # Following is the normalization according to Martin's book. It ensures
            # that a transparent specimen is imaged with unit intensity.
            # normfactor=abs(Po).^2.*abs(Ic); We do not use it, because it leads to
            # divide by zero for dark-field system. Instead, through normalizations
            # perfomed above, we ensure that image of a point under matched
            # illumination is unity. The brightness of all the other configurations is
            # relative to this benchmark.
            return TF_allSumAmp
     
    def computetomo(self):
        print('Only Experimental! Tomographic data?!')
        # Bring the slice back to focus - does this make any sense?! 
        print('----------> Bringing field back to focus')
        TF_centerprop = tf.exp(-1j*tf.cast(self.params.Nz/2*tf.angle(self.TF_myprop), tf.complex64))
        self.TF_A_prop = tf.ifft2d(tf.fft2d(self.TF_A_prop) * TF_centerprop) # diffraction step
        return self.TF_A_prop


    def addRegularizer(self, is_tv, is_gr, is_pos):
        print('Do stuff')
 
    def defineWriter(self, logs_path = '/tmp/tensorflow_logs/example/'):
        # create writer object
        self.logs_path = logs_path 
        self.writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
         
     
    def writeParameterFile(self, mylr, mytv, myeps, filepath = '/myparameters.yml'):
        ''' Write out all parameters to a yaml file in case we need it later'''
        mydata = dict(
                shiftIcX = float(self.params.shiftIcX),
                shiftIcY = float(self.params.shiftIcY),                
                NAc = float(self.params.NAc),
                NAo = float(self.params.NAo), 
                Nc = float(self.Nc), 
                Nx = float(self.params.Nx), 
                Ny = float(self.params.Ny),
                Nz = float(self.params.Nz),
                dx = float(self.params.dx),
                dy = float(self.params.dy),
                dz = float(self.params.dz),
                dn = float(self.dn),
                lambda0 = float(self.params.lambda0),
                lambdaM = float(self.lambdaM),
                learningrate = mylr, 
                lambda_tv = mytv, 
                eps_tv = myeps) 
                #zernikfactors = float(self.zernikefactors))
 
        with open(filepath, 'w') as outfile:
                yaml.dump(mydata, outfile, default_flow_style=False)
                 
    
            
            
         
    def saveFigures_list(self, savepath, myfwdlist, mylosslist, myfidelitylist, myneglosslist, mytvlosslist, result_phaselist, result_absorptionlist, 
                              globalphaselist, globalabslist, mymeas, figsuffix):
        ''' We want to save some figures for debugging purposes'''

        # get latest resutl from lists
        myfwd = myfwdlist[-1]
        my_res = result_phaselist[-1]
        my_res_absorption = result_absorptionlist[-1]
        
             
        plt.figure()
        plt.subplot(231), plt.title('REAL XZ'),plt.imshow(np.real(myfwd)[:,myfwd.shape[1]//2,:]), plt.colorbar()#, plt.show()
        plt.subplot(232), plt.title('REAL YZ'),plt.imshow(np.real(myfwd)[:,:,myfwd.shape[2]//2]), plt.colorbar()#, plt.show()
        plt.subplot(233), plt.title('REAL XY'),plt.imshow(np.real(myfwd)[myfwd.shape[0]//2,:,:]), plt.colorbar()#, plt.show()
        plt.subplot(234), plt.title('Imag XZ'),plt.imshow(np.imag(myfwd)[:,myfwd.shape[1]//2,:]), plt.colorbar()#, plt.show()
        plt.subplot(235), plt.title('Imag XZ'),plt.imshow(np.imag(myfwd)[:,:,myfwd.shape[2]//2]), plt.colorbar()#, plt.show()
        plt.subplot(236), plt.title('Imag XY'),plt.imshow(np.imag(myfwd)[myfwd.shape[0]//2,:,:]), plt.colorbar()#, plt.show()
        plt.savefig(savepath+'/res_myfwd_realimag'+figsuffix+'.png'), plt.show()

        if(0):
            plt.figure()    
            plt.subplot(231), plt.title('ABS XZ'),plt.imshow(np.abs(myfwd)[:,myfwd.shape[1]//2,:]), plt.colorbar()#, plt.show()
            plt.subplot(232), plt.title('ABS YZ'),plt.imshow(np.abs(myfwd)[:,:,myfwd.shape[2]//2]), plt.colorbar()#, plt.show()
            plt.subplot(233), plt.title('ABS XY'),plt.imshow(np.abs(myfwd)[myfwd.shape[0]//2 ,:,:]), plt.colorbar()# plt.show()
            #myfwd=myfwd*np.exp(1j*2)
            plt.subplot(234), plt.title('Angle XZ'),plt.imshow(np.angle(myfwd)[:,myfwd.shape[1]//2,:]), plt.colorbar()#, plt.show()
            plt.subplot(235), plt.title('Angle YZ'),plt.imshow(np.angle(myfwd)[:,:,myfwd.shape[2]//2]), plt.colorbar()#, plt.show()
            plt.subplot(236), plt.title('Angle XY'),plt.imshow(np.angle(myfwd)[myfwd.shape[0]//2 ,:,:]), plt.colorbar(), plt.show()
            plt.savefig(savepath+'/res_myfwd_ampphase'+figsuffix+'.png'), plt.show()
     
        # This is the measurment
        plt.figure()
        plt.subplot(231), plt.title('REAL XZ'),plt.imshow(np.real(mymeas)[:,mymeas.shape[1]//2,:]), plt.colorbar()#, plt.show()
        plt.subplot(232), plt.title('REAL YZ'),plt.imshow(np.real(mymeas)[:,:,mymeas.shape[2]//2]), plt.colorbar()#, plt.show()
        plt.subplot(233), plt.title('REAL XY'),plt.imshow(np.real(mymeas)[mymeas.shape[0]//2,:,:]), plt.colorbar()#, plt.show()
        plt.subplot(234), plt.title('Imag XZ'),plt.imshow(np.imag(mymeas)[:,mymeas.shape[1]//2,:]), plt.colorbar()#, plt.show()
        plt.subplot(235), plt.title('Imag XZ'),plt.imshow(np.imag(mymeas)[:,:,mymeas.shape[2]//2]), plt.colorbar()#, plt.show()
        plt.subplot(236), plt.title('Imag XY'),plt.imshow(np.imag(mymeas)[mymeas.shape[0]//2,:,:]), plt.colorbar()#, plt.show()
        plt.savefig(savepath+'/res_mymeas'+figsuffix+'.png'), plt.show()
     
        # This is the residual
        myresi = tf_helper.abssqr(myfwd-mymeas)
        plt.figure()
        plt.subplot(331), plt.title('Residual REAL XZ'),plt.imshow((np.real(myfwd)-np.real(mymeas))[:,myfwd.shape[1]//2,:]), plt.colorbar()#, plt.show()
        plt.subplot(332), plt.title('Residual REAL YZ'),plt.imshow((np.real(myfwd)-np.real(mymeas))[:,:,myfwd.shape[2]//2]), plt.colorbar()#, plt.show()
        plt.subplot(333), plt.title('Residual REAL XY'),plt.imshow((np.real(myfwd)-np.real(mymeas))[myfwd.shape[0]//2,:,:]), plt.colorbar()#, plt.show()
        plt.subplot(334), plt.title('Residual Imag XZ'),plt.imshow((np.imag(myfwd)-np.imag(mymeas))[:,myfwd.shape[1]//2,:]), plt.colorbar()#, plt.show()
        plt.subplot(335), plt.title('Residual Imag XZ'),plt.imshow((np.imag(myfwd)-np.imag(mymeas))[:,:,myfwd.shape[2]//2]), plt.colorbar()#, plt.show()
        plt.subplot(336), plt.title('Residual Imag XY'),plt.imshow((np.imag(myfwd)-np.imag(mymeas))[myfwd.shape[0]//2,:,:]), plt.colorbar()#, plt.show()
        plt.subplot(337), plt.title('Residual abssqr XZ'), plt.imshow((((myresi))**.2)[:,myresi.shape[1]//2,:]), plt.colorbar()#, plt.show()
        plt.subplot(338), plt.title('Residual abssqr XY'), plt.imshow((((myresi))**.2)[:,:,myresi.shape[2]//2]), plt.colorbar()#, plt.show()    
        plt.subplot(339), plt.title('Residual abssqr Yz'), plt.imshow((((myresi))**.2)[myresi.shape[0]//2,:,:]), plt.colorbar()#, plt.show()    
        

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
         
                 
    def saveFigures(self, sess, savepath, tf_fwd, np_meas, mylosslist, myfidelitylist, myneglosslist, mytvlosslist, globalphaselist, globalabslist, 
                    result_phaselist=None, result_absorptionlist=None, init_guess=None, figsuffix=''):
        ''' We want to save some figures for debugging purposes'''
        # This is the reconstruction
        if(init_guess is not None):
            myfwd, mymeas, my_res, my_res_absorption, myzernikes = sess.run([tf_fwd, self.tf_meas, self.TF_obj, self.TF_obj_absorption, self.TF_zernikefactors], 
                    feed_dict={self.tf_meas:np_meas, self.TF_obj:np.real(init_guess), self.TF_obj_absorption:np.imag(init_guess)})
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
        plt.subplot(232), plt.title('ABS YZ'),plt.imshow(np.abs(mymeas)[:,:,mymeasomcputeshape[2]//2]), plt.colorbar()#, plt.show()
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
        myshiftX = sess.run(self.TF_shiftIcX)
        myshiftY = sess.run(self.TF_shiftIcY)
        plt.subplot(131), plt.title('Po Phase'), plt.imshow(np.fft.fftshift(np.angle(sess.run(self.TF_Po_aberr)))), plt.colorbar()
        plt.subplot(132), plt.title('Ic, shiftX: '+str(myshiftX)+' myShiftY: '+str(myshiftY)), plt.imshow(np.fft.fftshift(np.abs(sess.run(self.TF_Po_aberr)))), plt.colorbar()
        plt.subplot(133), plt.bar(np.linspace(1, np.squeeze(myzernikes.shape), np.squeeze(myzernikes.shape)), myzernikes, align='center', alpha=0.5)
        plt.ylabel('Zernike Values')
        plt.title('Zernike Coefficients (Noll)')
        plt.savefig(savepath+'/recovered_pupil'+figsuffix+'.png'), plt.show()

        # Eventually write H5 stacks to disc
        if(result_phaselist is not None): data.export_realdatastack_h5(savepath+'/myrefractiveindex'+figsuffix+'.h5', 'temp', np.array(result_phaselist))
        if(result_absorptionlist is not None): data.export_realdatastack_h5(savepath+'/myrefractiveindex_absorption'+figsuffix+'.h5', 'temp', np.array(result_absorptionlist))
        #myobj = my_res+1j*my_res_absorption
        #np.save('my_res_cmplx', myobj)         
                 
    def initObject():
        print('To Be done')
                    
