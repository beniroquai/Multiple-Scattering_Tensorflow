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
from src import zernike as zern

class MuScatModel(object):
    def __init__(self, my_mat_paras, is_optimization = False, is_optimization_psf = False):

        ''' Create Multiple Scattering Class;
        INPUTS:
        my_mat_paras - parameter file from MATLAB containing the experimental details - has to be load previously!
        '''
        # new MATLAB v7.3 Format!
        self.myParamter = my_mat_paras
        self.is_optimization = is_optimization
        self.is_optimization_psf = is_optimization_psf 
                
        # ASsign variables from Experiment
        self.lambda0 = np.squeeze(np.array(self.myParamter.get('lambda0')))                 # free space wavelength (µm)
        self.NAo= np.squeeze(np.array(self.myParamter.get('NAo'))); # Numerical aperture objective
        self.NAc= np.squeeze(np.array(self.myParamter.get('NAc'))); # Numerical aperture condenser
        self.NAci = np.squeeze(np.array(self.myParamter.get('NAci'))); # Numerical aperture condenser
        
        # eventually decenter the illumination source - only integer!
        self.shiftIcX = int(np.squeeze(np.array(self.myParamter.get('shiftIcX'))))
        self.shiftIcY = int(np.squeeze(np.array(self.myParamter.get('shiftIcY'))))
        
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
        self.obj = self.obj + 1j*self.obj*0
        
        # add a vector of zernike factors
        self.nzernikes = 9
        self.zernikefactors = np.zeros((1,self.nzernikes))
        # kamilov uses 420 z-planes with an overall size of 30µm; dx=72nm

        # refractive index immersion and embedding
        self.lambdaM = self.lambda0/self.nEmbb; # wavelength in the medium

    #@define_scope
    def computesys(self, obj, is_zernike=False, is_padding=False):
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
        
        if(is_padding):
            # add padding in X/Y to avoid wrap-arounds
            self.Nx=self.Nx*2
            self.Ny=self.Ny*2
            self.mysize=np.array((self.Nz, self.Nx, self.Ny))
            #self.dx=self.dx/2
            #self.dy=self.dy/2
            
            # Pad object with zeros along X/Y
            obj_tmp = np.zeros(self.mysize)
            obj_tmp[:,self.Nx//2-self.Nx//4:self.Nx//2+self.Nx//4, self.Ny//2-self.Ny//4:self.Ny//2+self.Ny//4] = obj
            self.obj = obj_tmp
        else:
            self.mysize=np.array((self.Nz, self.Nx, self.Ny))
            self.obj = obj + 1j*0*obj
    
            

        # Decide whether we wan'T to optimize or simply execute the model
        if (self.is_optimization):
            # in case one wants to use this as a fwd-model for an inverse problem
            self.TF_obj_abs = tf.Variable(np.real(self.obj), dtype=tf.float32, name='Object_Variable_real')
            self.TF_obj_phase = tf.Variable(np.imag(self.obj), dtype=tf.float32, name='Object_Variable_imag')
        else:
            # Variables of the computational graph
            self.TF_obj_abs = tf.constant(np.real(self.obj), dtype=tf.float32, name='Object_const_real')
            self.TF_obj_phase = tf.constant(np.imag(self.obj), dtype=tf.float32, name='Object_const_imag')
        
        ## Establish normalized coordinates.
        #-----------------------------------------
        vxx= tf_helper.xx((self.mysize[1], self.mysize[2]),'freq')* self.lambdaM * self.nEmbb / (self.dx * self.NAo);    # normalized optical coordinates in X
        vyy= tf_helper.yy((self.mysize[1], self.mysize[2]),'freq') * self.lambdaM * self.nEmbb / (self.dy * self.NAo);    # normalized optical coordinates in Y
        
        # AbbeLimit=lambda0/NAo;  # Rainer's Method
        # RelFreq = rr(mysize,'freq')*AbbeLimit/dx;  # Is not generally right (dx and dy)
        self.RelFreq = np.sqrt(tf_helper.abssqr(vxx) + tf_helper.abssqr(vyy));    # spanns the frequency grid of normalized pupil coordinates
        self.Po=self.RelFreq < 1.0;   # Create the pupil of the objective lens        
        
        # Precomputing the first 9 zernike coefficients 
        self.myzernikes = np.zeros((self.Po.shape[0],self.Po.shape[1],self.nzernikes))+ 1j*np.zeros((self.Po.shape[0],self.Po.shape[1],self.nzernikes))
        r, theta = zern.cart2pol(vxx, vyy)        
        for i in range(0,self.nzernikes):
            self.myzernikes[:,:,i] = np.fft.fftshift(zern.zernike(r, theta, i+1, norm=False)) # or 8 in X-direction
            
        # eventually introduce a phase factor to approximate the experimental data better
        self.Po = np.fft.fftshift(self.Po)# Need to shift it before using as a low-pass filter    Po=np.ones((np.shape(Po)))
        if is_zernike:
            print('----------> Be aware: We are taking aberrations into account!')
            # Assuming: System has coma along X-direction
            self.myaberration = np.sum(self.zernikefactors * self.myzernikes, axis=2)
            self.Po = 1.*self.Po
        
        # Prepare the normalized spatial-frequency grid.
        self.S=self.NAc/self.NAo;   # Coherence factor
        self.Ic = self.RelFreq <= self.S
        #self.Ic = np.roll(self.Ic, -1, 1)
        if hasattr(self, 'NAci'):
            if self.NAci != None and self.NAci > 0:
                #print('I detected a darkfield illumination aperture!')
                self.S_o = self.NAc/self.NAo;   # Coherence factor
                self.S_i = self.NAci/self.NAo;   # Coherence factor
                self.Ic = (1.*(self.RelFreq < self.S_o) * 1.*(self.RelFreq > self.S_i))>0 # Create the pupil of the condenser plane
        
        # Shift the pupil in X-direction (optical missalignment)
        if hasattr(self, 'shiftIcX'):
            if self.shiftIcX != None:
                print('Shifting the illumination in X by: ' + str(self.shiftIcX) + ' Pixel')
                self.Ic = np.roll(self.Ic, self.shiftIcX, axis=1)

        # Shift the pupil in Y-direction (optical missalignment)
        if hasattr(self, 'shiftIcY'):
            if self.shiftIcY != None:
                print('Shifting the illumination in Y by: ' + str(self.shiftIcY) + ' Pixel')
                self.Ic = np.roll(self.Ic, self.shiftIcY, axis=0)

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
        self.Nc=np.sum(self.Ic); 
        print('Number of Illumination Angles / Plane waves: '+str(self.Nc))
        
        # Calculate the computatonal grid/sampling
        self.kxcoord = np.reshape(xfreq[self.Ic],[1, 1, 1, self.Nc]);    # NA-positions in condenser aperture plane in x-direction
        self.kycoord = np.reshape(yfreq[self.Ic],[1, 1, 1, self.Nc]);    # NA-positions in condenser aperture plane in y-direction
        self.RefrCos = np.reshape(self.k0/self.kz[self.Ic],[1, 1, 1, self.Nc]);   # 1/cosine used for the application of the refractive index steps to acount for longer OPD in medium under an oblique illumination angle
        
        ## Generate the illumination amplitudes
        self.A_input = np.exp((2*np.pi*1j) * (self.kxcoord * tf_helper.repmat4d(
            tf_helper.xx((self.mysize[1], self.mysize[2])), self.Nc) + self.kycoord * tf_helper.repmat4d(
            tf_helper.yy((self.mysize[1], self.mysize[2])), self.Nc))) # Corresponds to a plane wave under many oblique illumination angles - bfxfun
        
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
            self.TF_Po = tf.cast(tf.constant(self.Po), tf.complex64)
            self.TF_Zernikes = tf.constant(self.myzernikes, dtype=tf.float32)
            
            if(self.is_optimization_psf):
                self.TF_zernikefactors = tf.Variable(self.zernikefactors, dtype = tf.float32, name='var_zernikes')
            else:
                self.TF_zernikefactors = tf.constant(self.zernikefactors, dtype = tf.float32, name='const_zernikes')

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
                    a = tf.cast(self.TF_obj_phase[pz,:,:], tf.complex64)
                    b = self.TF_RefrEffect
                    c = tf.cast(self.TF_obj_abs[pz,:,:], tf.complex64)
                    print(a)
                    print(b)
                    print(c)
                    print(1j)
                    self.TF_f = c*tf.exp(1j*a*b)
                    self.TF_A_prop = self.TF_A_prop * self.TF_f  # refraction step - bfxfun

                with tf.name_scope('Propagate'):
                    self.TF_A_prop = tf.ifft2d(tf.fft2d(self.TF_A_prop) * self.TF_myprop)

                # U_z_list.append(TF_A_prop)
                # print('Current Z-Slice # is: ' + str(pz))

        # in a final step limit this to the detection NA:
        self.TF_Po_aberr = tf.exp(1j*tf.cast(tf.reduce_sum(self.TF_zernikefactors*self.TF_Zernikes, axis=2), tf.complex64)) * self.TF_Po
        self.TF_A_prop = tf.ifft2d(tf.fft2d(self.TF_A_prop) * self.TF_Po_aberr)

        self.TF_myAllSlicePropagator = tf.constant(self.myAllSlicePropagator, dtype=tf.complex64)
        self.kzcoord = np.reshape(self.kz[self.Ic], [1, 1, 1, self.Nc]);

        # create Z-Stack by backpropagating Information in BFP to Z-Position
        # self.mid3D = ([np.int(np.ceil(self.A_input.shape[0] / 2) + 1), np.int(np.ceil(self.A_input.shape[1] / 2) + 1), np.int(np.ceil(self.mysize[0] / 2) + 1)])
        self.mid3D = ([np.int(self.mysize[0]//2), np.int(self.A_input.shape[0] // 2), np.int(self.A_input.shape[1]//2)])

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

        # negate padding        
        if self.is_padding:
            self.TF_allSumAmp = self.TF_allSumAmp[:,self.Nx//2-self.Nx//4:self.Nx//2+self.Nx//4, self.Ny//2-self.Ny//4:self.Ny//2+self.Ny//4]
            
        return self.TF_allSumAmp
    
    
    def computetrans(self):
        # compute the transferfunction
        # does this makes any sense?
        ''' Create a 3D Refractive Index Distributaton as a artificial sample'''
        print('Computing the APSF/ATF of the system')
        pointscat = np.zeros(self.mysize)
        pointscat[pointscat.shape[0]//2, pointscat.shape[1]//2, pointscat.shape[2]//2] = .1
        # introduce zernike factors here
        
        ''' Compute the systems model'''
        self.computesys(pointscat, is_zernike=True)
        
        ''' Evaluate the model '''
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        apsf = sess.run(self.computemodel())
        atf = np.fft.fftshift(np.fft.fftn(apsf))
        

        return atf, apsf 
    














    def loadData(self, filename_Amp = 'allSumAmp.npy', filename_Int='allSumInt.npy'):
        # load the simulation data acquired with the fwd model 
        self.allSumInt_mes = np.load(filename_Int)
        self.allSumAmp_mes = np.load(filename_Amp)

    def defineWriter(self, logs_path = '/tmp/tensorflow_logs/example/'):
        # create writer object
        self.logs_path = logs_path 
        self.writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
       
        
