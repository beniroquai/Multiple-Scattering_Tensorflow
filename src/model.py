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

# own functions
from src import tf_helper as tf_helper



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


        # kamilov uses 420 z-planes with an overall size of 30µm; dx=72nm

    #@define_scope
    def computesys(self, obj):
        """ This computes the FWD-graph of the Q-PHASE microscope;
        1.) Compute the physical dimensions
        2.) Compute the sampling for the waves
        3.) Create the illumination waves depending on System's properties

        ##### IMPORTANT! ##### 
        The ordering of the channels is as follows:
            Nillu, Nz, Nx, Ny
        """

        # Decide whether we wan'T to optimize or simply execute the model
        if (self.is_optimization):
            # in case one wants to use this as a fwd-model for an inverse problem
            self.TF_obj = tf.Variable(obj, dtype=tf.float32, name='Object_Variable')

        else:
            # Variables of the computational graph
            self.TF_obj = tf.placeholder(dtype=tf.float32, shape=obj.shape)

        # refractive index immersion and embedding
        self.lambdaM = self.lambda0/self.nEmbb; # wavelength in the medium
        
        ## Establish normalized coordinates.
        #-----------------------------------------
        self.mysize=np.array((self.Nz, self.Nx, self.Ny))
        vxx= tf_helper.xx_freq(self.mysize[1], self.mysize[2]) * self.lambdaM * self.nEmbb / (self.dx * self.NAo);    # normalized optical coordinates in X
        vyy= tf_helper.yy_freq(self.mysize[1], self.mysize[2]) * self.lambdaM * self.nEmbb / (self.dy * self.NAo);    # normalized optical coordinates in Y
        
        # AbbeLimit=lambda0/NAo;  # Rainer's Method
        # RelFreq = rr(mysize,'freq')*AbbeLimit/dx;  # Is not generally right (dx and dy)
        self.RelFreq = np.sqrt(tf_helper.abssqr(vxx) + tf_helper.abssqr(vyy));    # spanns the frequency grid of normalized pupil coordinates
        self.Po=self.RelFreq < 1.0;   # Create the pupil of the objective lens
        self.Po = np.fft.fftshift(self.Po*1.)# Need to shift it before using as a low-pass filter    Po=np.ones((np.shape(Po)))
        
        # Prepare the normalized spatial-frequency grid.
        self.S=self.NAc/self.NAo;   # Coherence factor
        self.Ic = self.RelFreq <= self.S
        #self.Ic = np.roll(self.Ic, -1, 1)
        if hasattr(self, 'NAci'):
            if self.NAci != None:
                #print('I detected a darkfield illumination aperture!')
                self.S_o = self.NAc/self.NAo;   # Coherence factor
                self.S_i = self.NAci/self.NAo;   # Coherence factor
                self.Ic = (1.*(self.RelFreq < self.S_o) * 1.*(self.RelFreq > self.S_i))>0 # Create the pupil of the condenser plane
            

        ## Forward propagator  (Ewald sphere based) DO NOT USE NORMALIZED COORDINATES HERE
        self.kxysqr= (tf_helper.abssqr(tf_helper.xx_freq(self.mysize[1], self.mysize[2]) / self.dx) + tf_helper.abssqr(
            tf_helper.yy_freq(self.mysize[1], self.mysize[2]) / self.dy)) + 0j;
        self.k0=1/self.lambdaM;
        self.kzsqr= tf_helper.abssqr(self.k0) - self.kxysqr;
        self.kz=np.sqrt(self.kzsqr);
        self.kz[self.kzsqr < 0]=0;
        self.dphi = 2*np.pi*self.kz*self.dz;  # exp(1i*kz*dz) would be the propagator for one slice

        ## Get a list of vector coordinates corresponding to the pixels in the mask
        xfreq= tf_helper.xx_freq(self.mysize[1], self.mysize[2]);
        yfreq= tf_helper.yy_freq(self.mysize[1], self.mysize[2]);
        self.Nc=np.sum(self.Ic); 
        print('Number of Illumination Angles / Plane waves: '+str(self.Nc))
        
        # Calculate the computatonal grid/sampling
        self.kxcoord = np.reshape(xfreq[self.Ic],[1, 1, 1, self.Nc]);    # NA-positions in condenser aperture plane in x-direction
        self.kycoord = np.reshape(yfreq[self.Ic],[1, 1, 1, self.Nc]);    # NA-positions in condenser aperture plane in y-direction
        self.RefrCos = np.reshape(self.k0/self.kz[self.Ic],[1, 1, 1, self.Nc]);   # 1/cosine used for the application of the refractive index steps to acount for longer OPD in medium under an oblique illumination angle
        
        ## Generate the illumination amplitudes
        self.A_input = np.exp((2*np.pi*1j) * (self.kxcoord * tf_helper.repmat4d(
            tf_helper.xx(self.mysize[1], self.mysize[2]), self.Nc) + self.kycoord * tf_helper.repmat4d(
            tf_helper.yy(self.mysize[1], self.mysize[2]), self.Nc))) # Corresponds to a plane wave under many oblique illumination angles - bfxfun
        
        ## propagate field to z-stack and sum over all illumination angles
        self.Alldphi = -np.reshape(np.arange(0, self.mysize[0], 1), [1, 1, self.mysize[0]])*np.repeat(np.fft.fftshift(self.dphi)[:, :, np.newaxis], self.mysize[0], axis=2)
         
        # Ordinary backpropagation. This is NOT what we are interested in:
        self.myAllSlicePropagator=np.transpose(np.exp(1j*self.Alldphi) * (np.repeat(np.fft.fftshift(self.dphi)[:, :, np.newaxis], self.mysize[0], axis=2) >0), [2, 0, 1]);  # Propagates a single end result backwards to all slices

        
    #@define_scope
    def computemodel(self):
        ''' Perform Multiple Scattering here
        1.) Multiple Scattering is perfomed by slice-wise propagation the E-Field throught the sample
        2.) Each Field has to be backprojected to the BFP
        3.) Last step is to ceate a focus-stack and sum over all angles

        This is done for all illumination angles (coming from illumination NA
        simultaneasly)'''


        print("Buildup Q-PHASE Model ")
        ###### make sure, that the first dimension is "batch"-size; in this case it is the illumination number
        # @beniroquai It's common to have to batch dimensions first and not last.!!!!!
        # the following loop propagates the field sequentially to all different Z-planes

        ## propagate the field through the entire object for all angles simultaneously
        A_prop = np.transpose(self.A_input,[3, 0, 1, 2])  # ??????? what the hack is happening with transpose?!

        myprop = np.exp(1j * self.dphi) * (self.dphi > 0);  # excludes the near field components in each step
        myprop = tf_helper.repmat4d(np.fft.fftshift(myprop), self.Nc)
        myprop = np.transpose(myprop, [3, 0, 1, 2])  # ??????? what the hack is happening with transpose?!

        RefrEffect = 1j * self.dz * self.k0 * self.RefrCos;  # Precalculate the oblique effect on OPD to speed it up
        RefrEffect = np.transpose(RefrEffect, [3,0,1,2])

        # for now orientate the dimensions as (alpha_illu, x, y, z) - because tensorflow takes the first dimension as batch size
        with tf.name_scope('Variable_assignment'):
            self.TF_A_input = tf.constant(A_prop, dtype=tf.complex64)
            self.TF_RefrEffect = tf.reshape(tf.constant(RefrEffect, dtype=tf.complex64), [self.Nc, 1, 1])
            self.TF_myprop = tf.squeeze(tf.constant(myprop, dtype=tf.complex64))
            self.TF_Po = tf.cast(tf.constant(self.Po + 1j * 0 * self.Po), tf.complex64)
        # TODO: Introduce the averraged RI along Z - MWeigert

        self.TF_A_prop = tf.squeeze(self.TF_A_input);
        self.U_z_list = []

        # Initiliaze memory
        self.allInt = 0;
        self.allSumAmp = 0;
        self.TF_allSumAmp = tf.zeros([self.mysize[0], self.Nx, self.Ny], dtype=tf.complex64)
        self.TF_allSumInt = tf.zeros([self.mysize[0], self.Nx, self.Ny], dtype=tf.float32)

        self.tf_iterator = tf.Variable(1)
        # simulate multiple scattering through object
        with tf.name_scope('Fwd_Propagate'):
            for pz in range(0, self.mysize[0]):
                self.tf_iterator += self.tf_iterator
                #self.TF_A_prop = tf.Print(self.TF_A_prop, [self.tf_iterator], 'Prpagation step: ')
                with tf.name_scope('Refract'):
                    self.TF_f = tf.complex(self.TF_obj[pz,:,:], self.TF_obj[pz,:,:] * 0)
                    self.TF_A_prop = self.TF_A_prop * tf.exp(self.TF_f * self.TF_RefrEffect);  # refraction step - bfxfun

                with tf.name_scope('Propagate'):
                    self.TF_A_prop = tf.ifft2d(tf.fft2d(self.TF_A_prop) * self.TF_myprop)

                # U_z_list.append(TF_A_prop)
                # print('Current Z-Slice # is: ' + str(pz))

        # in a final step limit this to the detection NA:
        self.TF_A_prop = tf.ifft2d(tf.fft2d(self.TF_A_prop) * self.TF_Po)

        self.TF_myAllSlicePropagator = tf.constant(self.myAllSlicePropagator, dtype=tf.complex64)
        self.kzcoord = np.reshape(self.kz[self.Ic], [1, 1, 1, self.Nc]);

        # create Z-Stack by backpropagating Information in BFP to Z-Position
        # self.mid3D = ([np.int(np.ceil(self.A_input.shape[0] / 2) + 1), np.int(np.ceil(self.A_input.shape[1] / 2) + 1), np.int(np.ceil(self.mysize[0] / 2) + 1)])
        self.mid3D = ([np.int(self.A_input.shape[0] // 2), np.int(self.A_input.shape[1]//2), np.int(self.mysize[0]//2)])

        with tf.name_scope('Back_Propagate'):
            for pillu in range(0, self.Nc):
                with tf.name_scope('Back_Propagate_Step'):
                    with tf.name_scope('Adjust'):
                        #    fprintf('BackpropaAngle no: #d\n',pillu);
                        OneAmp = tf.expand_dims(self.TF_A_prop[pillu, :, :, ], 0)

                        # Fancy backpropagation assuming what would be measured if the sample was moved under oblique illumination:
                        # The trick is: First use conceptually the normal way
                        # and then apply the XYZ shift using the Fourier shift theorem (corresponds to physically shifting the object volume, scattered field stays the same):
                        self.TF_AdjustKXY = tf.squeeze(tf.conj(self.TF_A_input[pillu, :,
                                                               :, ]))  # tf.transpose(tf.conj(TF_A_input[pillu, :,:,]), [2, 1, 0]) # Maybe a bit of a dirty hack, but we first need to shift the zero coordinate to the center
                        self.TF_AdjustKZ = tf.transpose(tf.constant(np.exp(
                            2 * np.pi * 1j * self.dz * np.reshape(np.arange(0, self.mysize[0], 1),
                                  [1, 1, self.mysize[0]]) * self.kzcoord[:, :, :,pillu]),dtype=tf.complex64), [2, 1, 0]);
                        self.TF_allAmp = tf.ifft2d(tf.fft2d(OneAmp) * self.TF_myAllSlicePropagator) * self.TF_AdjustKZ * self.TF_AdjustKXY  # * (TF_AdjustKZ);  # 2x bfxfun.  Propagates a single amplitude pattern back to the whole stack
                        self.TF_allAmp = self.TF_allAmp * tf.exp(1j*tf.cast(tf.angle(self.TF_allAmp[self.mid3D[0],self.mid3D[1],self.mid3D[2]]), tf.complex64)) # Global Phases need to be adjusted at this step!  Use the zero frequency
                        
                    if (0):
                        with tf.name_scope('Propagate'):
                            self.TF_allAmp_3dft = tf.fft3d(tf.expand_dims(self.TF_allAmp, axis=0))
                            self.TF_allAmp = self.TF_allAmp * tf.exp(-1j * tf.cast(
                                tf_helper.tf_angle(self.TF_allAmp_3dft[self.mid3D[2], self.mid3D[1], self.mid3D[0]]),
                                tf.complex64));  # Global Phases need to be adjusted at this step!  Use the zero frequency


                    # print(tf.exp(-1j*tf.cast(angle(TF_allAmp[self.mid3D[2], self.mid3D[0], self.mid3D[2]]), tf.complex64)).eval())

                    with tf.name_scope('Sum_Amps'):
                        self.TF_allSumAmp = self.TF_allSumAmp + self.TF_allAmp;  # Superpose the Amplitudes
                        self.TF_allSumInt = self.TF_allSumInt + tf_helper.tf_abssqr(self.TF_allAmp)

                    # print('Current illumination angle # is: ' + str(pillu))

        # Normalize amplitude
        self.TF_allSumAmp = self.TF_allSumAmp / self.Nc  # tf.reduce_max(TF_allSumAmp)
        return self.TF_allSumAmp














    def loadData(self, filename_Amp = 'allSumAmp.npy', filename_Int='allSumInt.npy'):
        # load the simulation data acquired with the fwd model 
        self.allSumInt_mes = np.load(filename_Int)
        self.allSumAmp_mes = np.load(filename_Amp)

    def defineWriter(self, logs_path = '/tmp/tensorflow_logs/example/'):
        # create writer object
        self.logs_path = logs_path 
        self.writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
       
        
